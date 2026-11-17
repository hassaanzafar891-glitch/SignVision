"""Fail-closed comparison of evidence-backed evaluations. Standard library only."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re


DATASET_MD5 = "326d65539174dffdad54606bb2271e3c"
FEATURES = [coordinate for i in range(1, 22) for coordinate in (f"x{i}", f"y{i}")]
TABLES = {"alphabet": "alphabetDataset", "word": "wordDataset"}
ARTIFACTS = ("model", "scaler", "label_encoder")
PROTOCOL = "saved-scaler-encoder-softmax-v1"


class GateError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise GateError(message)


def mapping(value, name):
    require(isinstance(value, dict), f"{name} must be an object")
    return value


def digest(value, name, size=64):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % size, value) is not None,
            f"{name} must be a lowercase {size}-character hex digest")
    return value


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise GateError(f"Non-finite JSON value: {value}")

    def finite_float(value):
        number = float(value)
        require(math.isfinite(number), "Non-finite JSON number")
        return number

    try:
        data = Path(path).read_bytes()
        require(len(data) <= 10_000_000, "JSON evidence exceeds size limit")
        return json.loads(data.decode("utf-8-sig"), object_pairs_hook=reject_duplicates,
                          parse_constant=reject_constant, parse_float=finite_float)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise GateError("Cannot read valid JSON evaluation evidence") from error


def validate_dataset(dataset, task):
    dataset = mapping(dataset, "dataset")
    require(dataset.get("dvc_md5") == DATASET_MD5, "Dataset DVC identity is not the reviewed version")
    digest(dataset.get("sha256"), "dataset.sha256")
    digest(dataset.get("ordered_rows_sha256"), "dataset.ordered_rows_sha256")
    require(dataset.get("table") == TABLES[task], "Dataset table/task mismatch")
    require(dataset.get("feature_columns") == FEATURES, "Feature schema mismatch")


def validate_record(record, role):
    record = mapping(record, "evaluation record")
    require(type(record.get("schema_version")) is int and record["schema_version"] == 1,
            "Unsupported evaluation schema")
    require(record.get("role") == role, "Evaluation role mismatch")
    task = record.get("task")
    require(isinstance(task, str) and task in TABLES, "Unknown model task")
    require(record.get("protocol") == PROTOCOL, "Evaluation protocol mismatch")
    try:
        evaluated_at = datetime.fromisoformat(record["evaluated_at"])
        require(evaluated_at.utcoffset() is not None, "Evaluation timestamp needs a timezone")
    except (KeyError, TypeError, ValueError) as error:
        raise GateError("Invalid evaluation timestamp") from error
    validate_dataset(record.get("dataset"), task)
    dataset = record["dataset"]
    require(type(dataset.get("rows")) is int and dataset["rows"] > 0, "Invalid dataset row count")
    model = mapping(record.get("model"), "model")
    require(model.get("task") == task and model.get("family") == "keras_dense", "Model/task identity mismatch")
    hashes = mapping(model.get("artifacts"), "model.artifacts")
    require(set(hashes) == set(ARTIFACTS), "A complete model/scaler/encoder bundle is required")
    for key in ARTIFACTS:
        digest(hashes[key], f"model.artifacts.{key}")
    classes = record.get("classes")
    require(isinstance(classes, list) and len(classes) >= 2
            and all(isinstance(label, str) and label.strip() for label in classes), "Invalid class labels")
    require(len(set(classes)) == len(classes), "Duplicate class labels")
    held_out = mapping(record.get("held_out"), "held_out")
    require(held_out.get("provenance_status") == "reviewed_training_time_evidence",
            "Original held-out provenance is not verified")
    require(held_out.get("duplicate_disjoint") is True, "Held-out leakage checks did not pass")
    for key in ("manifest_sha256", "test_indices_sha256", "test_rows_sha256"):
        digest(held_out.get(key), f"held_out.{key}")
    counts = mapping(held_out.get("partition_counts"), "partition_counts")
    require(set(counts) == {"train", "validation", "test"}
            and all(type(value) is int and value > 0 for value in counts.values()), "Invalid partition counts")
    require(sum(counts.values()) == dataset["rows"], "Partition coverage does not match dataset")
    require(type(held_out.get("sample_count")) is int
            and held_out["sample_count"] == counts["test"], "Held-out sample count mismatch")
    metrics = mapping(record.get("metrics"), "metrics")
    for key in ("test_accuracy", "test_loss"):
        value = metrics.get(key)
        require(type(value) in (int, float) and math.isfinite(value), f"Invalid {key}")
        require(value >= 0 and (key != "test_accuracy" or value <= 1), f"Out-of-range {key}")
    return record


def compare_evaluations(production, candidate):
    production = validate_record(production, "production")
    candidate = validate_record(candidate, "candidate")
    require(production["task"] == candidate["task"], "Production/candidate task mismatch")
    require(production["dataset"] == candidate["dataset"], "Production/candidate dataset mismatch")
    require(production["classes"] == candidate["classes"], "Production/candidate class identity mismatch")
    for key in ("test_indices_sha256", "test_rows_sha256", "sample_count"):
        require(production["held_out"][key] == candidate["held_out"][key], "Held-out split identity mismatch")
    accuracy = candidate["metrics"]["test_accuracy"]
    require(accuracy >= production["metrics"]["test_accuracy"], "Candidate accuracy regression")
    return {
        "approved": True, "task": candidate["task"], "policy": "no_accuracy_regression_v1",
        "production_record_sha256": canonical_sha256(production),
        "candidate_record_sha256": canonical_sha256(candidate),
        "production_artifacts": production["model"]["artifacts"],
        "candidate_artifacts": candidate["model"]["artifacts"],
        "dataset_sha256": candidate["dataset"]["sha256"],
        "test_indices_sha256": candidate["held_out"]["test_indices_sha256"],
        "test_accuracy_delta": accuracy - production["metrics"]["test_accuracy"],
        "production_metrics": production["metrics"], "candidate_metrics": candidate["metrics"],
        "deployment_performed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = compare_evaluations(read_json(args.production), read_json(args.candidate))
    except GateError as error:
        print(json.dumps({"approved": False, "reason": str(error)}))
        raise SystemExit(1)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
