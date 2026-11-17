"""Evaluate saved bundles only with reviewed training-time partition evidence.

Never creates a replacement split, trains, logs to MLflow, or deploys anything.
Pickles must come from trusted, reviewed bundles: deserializing them runs code.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import pickle
import sqlite3

from tracking.evaluation_gate import (
    ARTIFACTS, DATASET_MD5, FEATURES, PROTOCOL, TABLES, GateError, canonical_sha256,
    compare_evaluations, digest, mapping, read_json, require, validate_dataset,
)
from tracking.import_baselines import sha256


SUFFIXES = {"model": "model.h5", "scaler": "scaler.pkl", "label_encoder": "label_encoder.pkl"}


def read_bundle(directory, task, expected_manifest_sha256=None):
    directory = Path(directory).resolve()
    path = directory / f"{task}_provenance.json"
    require(path.is_file(), f"Missing {task} training-time provenance; evaluation blocked")
    require(path.resolve().parent == directory, "External provenance file is not allowed")
    manifest = mapping(read_json(path), "provenance manifest")
    require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
            "Unsupported provenance schema")
    require(manifest.get("task") == task, "Provenance task mismatch")
    validate_dataset(manifest.get("dataset"), task)
    provenance = mapping(manifest.get("provenance"), "provenance")
    require(provenance.get("status") == "reviewed_training_time_evidence", "Unverified historical split")
    for key in ("reviewed_by", "evidence_reference"):
        require(isinstance(provenance.get(key), str) and provenance[key].strip(), f"Missing provenance {key}")
    hashes = mapping(manifest.get("artifacts"), "provenance.artifacts")
    require(set(hashes) == set(ARTIFACTS), "Incomplete provenance artifact bundle")
    for key, suffix in SUFFIXES.items():
        digest(hashes[key], f"artifacts.{key}")
        artifact = directory / f"{task}_{suffix}"
        require(artifact.is_file() and artifact.resolve().parent == directory, "Missing or external bundle artifact")
        require(sha256(artifact) == hashes[key], f"{task} {key} artifact hash mismatch")
    partitions = mapping(manifest.get("partitions"), "partitions")
    require(set(partitions) == {"train", "validation", "test"}, "All training-time partitions are required")
    for indices in partitions.values():
        require(isinstance(indices, list) and indices
                and all(type(index) is int and index >= 0 for index in indices), "Invalid recorded partition indices")
        require(len(set(indices)) == len(indices), "Duplicate partition indices")
    flattened = [index for indices in partitions.values() for index in indices]
    require(len(set(flattened)) == len(flattened), "Partition indices overlap")
    manifest_hash = canonical_sha256(manifest)
    if expected_manifest_sha256 is not None:
        digest(expected_manifest_sha256, "reviewed production manifest hash")
        require(manifest_hash == expected_manifest_sha256, "Production manifest does not match reviewed policy")
    return manifest, manifest_hash


def read_rows(database, task):
    database = Path(database).resolve()
    # Fail before reading rows if the exact processed database is not established.
    require(database.is_file(), "DVC dataset is missing")
    require(not any(Path(str(database) + suffix).exists() for suffix in ("-wal", "-journal")),
            "SQLite has sidecar changes; a stable reviewed dataset snapshot is required")
    raw = database.read_bytes()
    require(hashlib.md5(raw).hexdigest() == DATASET_MD5, "DVC dataset MD5 mismatch")
    database_sha256 = hashlib.sha256(raw).hexdigest()
    connection = sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        require(connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)], "SQLite integrity failed")
        # Identical feature/label selection and natural row order as the trainers.
        rows = connection.execute(f"SELECT {','.join(FEATURES)},label FROM {TABLES[task]}").fetchall()
    finally:
        connection.close()
    require(bool(rows), "Empty evaluation dataset")
    for row in rows:
        require(all(isinstance(value, Real) and math.isfinite(value) for value in row[:-1]), "Invalid features")
        require(isinstance(row[-1], str) and row[-1].strip(), "Invalid label")
    require(sha256(database) == database_sha256, "Dataset changed while being read")
    return rows, database_sha256


def verify_partitions(manifest, rows, database_sha256):
    require(manifest["dataset"]["sha256"] == database_sha256, "Historical dataset SHA-256 mismatch")
    require(manifest["dataset"]["ordered_rows_sha256"] == canonical_sha256(rows),
            "Historical SQL row ordering/content does not match")
    parts = manifest["partitions"]
    flattened = [index for indices in parts.values() for index in indices]
    require(sorted(flattened) == list(range(len(rows))), "Training/validation/test indices do not partition this dataset")
    # Check exact feature-vector separation, including duplicates with different labels.
    groups = {name: {tuple(float(value) for value in rows[i][:-1]) for i in ids} for name, ids in parts.items()}
    require(not (groups["train"] & groups["validation"] or groups["train"] & groups["test"]
                 or groups["validation"] & groups["test"]), "Duplicate-vector leakage across partitions")
    classes = {row[-1] for row in rows}
    require(all({rows[i][-1] for i in ids} == classes for ids in parts.values()), "Incomplete partition class coverage")
    return {
        "provenance_status": "reviewed_training_time_evidence", "duplicate_disjoint": True,
        "test_indices_sha256": canonical_sha256(parts["test"]),
        "test_rows_sha256": canonical_sha256([rows[i] for i in parts["test"]]),
        "partition_counts": {name: len(ids) for name, ids in parts.items()}, "sample_count": len(parts["test"]),
    }


def load_objects(directory, task):
    from tensorflow.keras.models import load_model

    model = load_model(str(directory / f"{task}_model.h5"), compile=False)
    with (directory / f"{task}_scaler.pkl").open("rb") as stream:
        scaler = pickle.load(stream)
    with (directory / f"{task}_label_encoder.pkl").open("rb") as stream:
        encoder = pickle.load(stream)
    return model, scaler, encoder


def score_bundle(objects, rows, test_indices):
    import numpy as np

    model, scaler, encoder = objects
    require(getattr(scaler, "n_features_in_", None) == 42, "Saved scaler has the wrong feature count")
    classes = [str(label) for label in encoder.classes_]
    require(set(classes) == {row[-1] for row in rows}, "Saved encoder classes do not match the dataset")
    x = np.asarray([rows[i][:-1] for i in test_indices], dtype=np.float64)
    # Only transform the verified held-out rows. Never fit/refit preprocessing.
    x = scaler.transform(x)
    require(np.isfinite(x).all(), "Scaler produced non-finite features")
    y = encoder.transform([rows[i][-1] for i in test_indices])
    probabilities = np.asarray(model.predict(x, batch_size=256, verbose=0))
    require(probabilities.shape == (len(test_indices), len(classes)), "Invalid prediction shape")
    require(np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all()
            and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5), "Expected finite softmax probabilities")
    # Categorical cross entropy on saved softmax outputs, with Keras' default epsilon.
    loss = -np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-7, 1 - 1e-7)).mean()
    return classes, {"test_accuracy": float((probabilities.argmax(axis=1) == y).mean()), "test_loss": float(loss)}


def evaluate_bundle(directory, task, role, bundle, rows, database_sha256):
    manifest, manifest_sha = bundle
    held_out = verify_partitions(manifest, rows, database_sha256)
    held_out["manifest_sha256"] = manifest_sha
    # Recheck identity immediately before deserializing trusted artifacts.
    rechecked = read_bundle(directory, task, manifest_sha)
    require(rechecked[0] == manifest, "Provenance changed during evaluation")
    classes, metrics = score_bundle(load_objects(Path(directory).resolve(), task), rows, manifest["partitions"]["test"])
    read_bundle(directory, task, manifest_sha)
    return {
        "schema_version": 1, "role": role, "task": task, "protocol": PROTOCOL,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {**manifest["dataset"], "rows": len(rows)},
        "model": {"task": task, "family": "keras_dense", "artifacts": manifest["artifacts"]},
        "classes": classes, "held_out": held_out, "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[1] / "data/db/main_dataset.db")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-evidence", action="store_true")
    parser.add_argument("--workspace", type=Path, help="Restrict CI bundle paths to the checked-out workspace")
    parser.add_argument("--candidate-root", type=Path, help="Restrict candidate files to their checkout")
    args = parser.parse_args()
    try:
        if args.workspace:
            for directory in (args.production_dir, args.candidate_dir):
                require(directory.resolve().is_relative_to(args.workspace.resolve()), "Bundle path is outside the workspace")
        if args.candidate_root:
            require(args.candidate_dir.resolve().is_relative_to(args.candidate_root.resolve()),
                    "Candidate bundle is outside its checkout")
        bundles = {}
        for task in TABLES:
            pin = os.environ.get(f"SIGNVISION_PRODUCTION_{task.upper()}_MANIFEST_SHA256", "")
            bundles[(task, "production")] = read_bundle(args.production_dir, task, pin)
            bundles[(task, "candidate")] = read_bundle(args.candidate_dir, task)
        if args.check_evidence:
            print("Bundle hashes and reviewed provenance passed preflight; no evaluation or training performed.")
            return
        require(args.output_dir is not None, "A new output directory is required")
        output = args.output_dir.resolve()
        require(not output.exists(), "Refusing to overwrite evaluation outputs")
        # Complete dataset/provenance checks for BOTH tasks and roles before loading any model.
        datasets = {task: read_rows(args.dataset, task) for task in TABLES}
        for (task, role), (manifest, _) in bundles.items():
            verify_partitions(manifest, *datasets[task])
        for task in TABLES:
            require(bundles[(task, "production")][0]["partitions"]["test"] ==
                    bundles[(task, "candidate")][0]["partitions"]["test"], "Candidate and production holdouts differ")
        output.mkdir(parents=True, exist_ok=False)
        approvals = {}
        for task in TABLES:
            records = {}
            for role, directory in (("production", args.production_dir), ("candidate", args.candidate_dir)):
                record = evaluate_bundle(directory, task, role, bundles[(task, role)], *datasets[task])
                records[role] = record
                (output / f"{task}_{role}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            approvals[task] = compare_evaluations(records["production"], records["candidate"])
        approval = {"approved": True, "tasks": approvals, "workflow_commit": os.environ.get("GITHUB_SHA"),
                    "candidate_commit": os.environ.get("CANDIDATE_GIT_COMMIT"),
                    "deployment_performed": False}
        # This file exists ONLY if both task gates passed. An intermediate record is not approval.
        (output / "approval.json").write_text(json.dumps(approval, indent=2) + "\n", encoding="utf-8")
        print("Both task gates passed. Candidate evidence generated; no deployment performed.")
    except (GateError, OSError, sqlite3.DatabaseError, ValueError, TypeError, AttributeError) as error:
        # Never print deserializer/HTTP exception details or credential-bearing URLs.
        print(f"Evaluation blocked ({type(error).__name__}). No candidate approval issued.")
        if isinstance(error, GateError):
            print(str(error))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
