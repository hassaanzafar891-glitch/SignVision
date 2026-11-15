"""Separate classical baselines. Running this module trains; --help does not.

No neural-network trainer is imported; production models and historical runs
are never overwritten.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from numbers import Real
import os
from pathlib import Path
import pickle
import shutil
import sqlite3
import uuid

from tracking.import_baselines import BACKEND, git_metadata, sha256
from tracking.training import dataset_metadata
from tracking.data_validation import validate_training_database


TABLES = {"alphabet": "alphabetDataset", "word": "wordDataset"}
FEATURES = [coordinate for i in range(1, 22) for coordinate in (f"x{i}", f"y{i}")]
RF_PARAMS = {
    "n_estimators": 100, "max_depth": 20, "min_samples_leaf": 1,
    "max_features": "sqrt", "random_state": 42, "n_jobs": 2,
}
RECORDED_NN_ACCURACY = {"alphabet": 0.9837, "word": 1.0}
PARTITIONS = ("train", "validation", "test")


def load_dataset(database, kind):
    table = TABLES[kind]  # Only known table names may enter SQL.
    database = Path(database).resolve()
    validate_training_database(database, table)
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        # Match the existing trainers' selected features and natural SQL order.
        # Do not add ORDER BY or select the id column.
        rows = connection.execute(f"SELECT {','.join(FEATURES)},label FROM {table}").fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"No training rows in {table}")
    if any(not isinstance(value, Real) for row in rows for value in row[:-1]):
        raise ValueError("All 42 features must be numerical")
    if any(row[-1] is None or not str(row[-1]).strip() for row in rows):
        raise ValueError("Labels must not be empty")
    return [list(row[:-1]) for row in rows], [str(row[-1]) for row in rows]


def partition_indices(features, labels):
    from tracking.data_splits import split_three_way

    indices = split_three_way(features, labels, seed=42)
    return {name: selected.tolist() for name, selected in zip(PARTITIONS, indices)}


def split_fingerprints(indices):
    return {name: hashlib.sha256(json.dumps(indices[name], separators=(",", ":")).encode()).hexdigest()
            for name in PARTITIONS}


def build_classifier():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(**RF_PARAMS)


def evaluate_predictions(expected, predicted, classes):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, labels=classes,
                                   average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=classes).tolist(),
    }


def comparison_report(kind, accuracy, dataset, fingerprints, neural_metadata=None):
    """Historical scores alone, or a matching seed/count, do not prove fairness."""
    evidence = neural_metadata or {}
    reference = RECORDED_NN_ACCURACY[kind]
    checks = {
        "model_kind": evidence.get("model_name") == kind,
        "ordered_dataset_sha256": evidence.get("dataset", {}).get("sha256") == dataset["sha256"],
        "exact_partition_indices": evidence.get("split_indices_sha256") == fingerprints,
        "recorded_score": round(evidence.get("test_metrics", {}).get("test_accuracy", -1) * 100, 2)
                          == round(reference * 100, 2),
    }
    verified = all(checks.values())
    result = {
        "random_forest_test_accuracy": accuracy,
        "previously_recorded_neural_network_test_accuracy": reference,
        "split_comparability": "verified" if verified else "not_verified",
        "checks": checks,
        "note": "Historical NN scores were not recomputed. Matching dataset order and all three "
                "partition-index hashes are required for a like-for-like held-out comparison. "
                "Signer-independent performance is not established; the word test dataset is small.",
    }
    if verified:
        result["accuracy_difference_percentage_points"] = (
            accuracy - evidence["test_metrics"]["test_accuracy"]
        ) * 100
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def train_task(kind, database, mlflow, *, backend=BACKEND, neural_metadata=None):
    features, labels = load_dataset(database, kind)
    indices = partition_indices(features, labels)
    fingerprints = split_fingerprints(indices)
    dataset = dataset_metadata(features, labels, database, TABLES[kind])
    dataset["partition_samples"] = {name: len(selected) for name, selected in indices.items()}
    classifier = build_classifier()
    classes = sorted(set(labels))
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex
    output = Path(backend) / "data" / "training_runs" / "random_forest" / kind / identifier
    output.mkdir(parents=True, exist_ok=False)
    git = git_metadata(Path(backend).parent)
    # start_run contacts the configured server BEFORE any fitting. No fallback
    # may silently train an untracked baseline when tracking is unavailable.
    with mlflow.start_run(run_name=f"{kind}-classical-random-forest-{identifier}", tags={
        "baseline.type": "classical_random_forest", "model.family": "classical_ml",
        "model.kind": kind, "training.git.commit": git["import_commit"] or "unavailable",
        "training.git.dirty": str(git["import_checkout_dirty"]).lower(),
    }) as run:
        mlflow.log_params({
            **classifier.get_params(), "model_name": f"{kind}_random_forest",
            "dataset.table": dataset["table"], "dataset.samples": dataset["samples"],
            "dataset.features": 42, "dataset.class_count": dataset["class_count"],
            "dataset.sha256": dataset["sha256"], "split_seed": 42,
            "split_method": "split_three_way", "duplicate_disjoint": True,
            "scaling": "none (tree classifier)",
            **{f"split.{name}_samples": len(indices[name]) for name in PARTITIONS},
            f"previously_recorded_nn.{kind}.test_accuracy": RECORDED_NN_ACCURACY[kind],
        })
        def selected(name):
            return [features[i] for i in indices[name]], [labels[i] for i in indices[name]]

        classifier.fit(*selected("train"))
        results = {}
        for name in PARTITIONS:
            x, y = selected(name)
            results[name] = evaluate_predictions(y, classifier.predict(x), classes)
            mlflow.log_metrics({f"{name}_accuracy": results[name]["accuracy"],
                                f"{name}_macro_f1": results[name]["macro_f1"]})
            with (output / f"{name}_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["actual/predicted", *classes])
                writer.writerows([label, *row] for label, row in zip(classes, results[name]["confusion_matrix"]))
        comparison = comparison_report(kind, results["test"]["accuracy"], dataset,
                                       fingerprints, neural_metadata)
        with (output / "random_forest.pkl").open("wb") as stream:
            pickle.dump(classifier, stream)
        write_json(output / "classes.json", classes)
        write_json(output / "splits.json", {"seed": 42, "indices": indices, "sha256": fingerprints})
        write_json(output / "evaluation.json", results)
        write_json(output / "comparison.json", comparison)
        shutil.copyfile(__file__, output / "training_script.py")
        shutil.copyfile(Path(__file__).with_name("data_splits.py"), output / "data_splits.py")
        versions = {}
        for package in ("scikit-learn", "numpy", "mlflow"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        metadata = {
            "model_name": f"{kind}_random_forest", "model_kind": kind,
            "baseline_type": "classical_random_forest", "mlflow_run_id": run.info.run_id,
            "hyperparameters": classifier.get_params(), "dataset": dataset,
            "split_indices_sha256": fingerprints, "git": git, "versions": versions,
            "artifacts": {path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
                          for path in output.iterdir() if path.is_file()},
        }
        write_json(output / "run_metadata.json", metadata)
        mlflow.set_tag("comparison.split_comparability", comparison["split_comparability"])
        mlflow.log_artifacts(str(output), artifact_path="random_forest_baseline")
    print(f"{kind}: {output}; split comparability: {comparison['split_comparability']}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["both", *TABLES], default="both")
    parser.add_argument("--database", type=Path, default=BACKEND / "data/db/main_dataset.db")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--experiment", default="SignVision Classical Baselines")
    for kind in TABLES:
        parser.add_argument(f"--{kind}-nn-metadata", type=Path,
                            help="Optional historical NN metadata with exact partition-index hashes")
    args = parser.parse_args()
    import mlflow

    if mlflow.active_run() is not None or os.environ.get("MLFLOW_RUN_ID"):
        parser.error("Refusing to resume or modify an existing MLflow run")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    for kind in TABLES if args.model == "both" else (args.model,):
        evidence = getattr(args, f"{kind}_nn_metadata")
        evidence = json.loads(evidence.read_text(encoding="utf-8")) if evidence else None
        train_task(kind, args.database, mlflow, neural_metadata=evidence)


if __name__ == "__main__":
    main()
