"""Explicit local retraining with frozen partitions. No promotion or deployment."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid

from tracking.data_validation import validate_training_database
from tracking.evaluate_models import read_rows, verify_partitions, SUFFIXES
from tracking.evaluation_gate import (
    DATASET_MD5, FEATURES, TABLES, canonical_sha256, read_json, require,
)
from tracking.import_baselines import BACKEND, git_metadata, sha256, script_path
from tracking.train_all import tracking_environment
from tracking.training import CONTROLLED_SESSION


def write_new(path, value):
    """Never replace an existing evidence record."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sources():
    paths = [script_path(BACKEND, task) for task in TABLES]
    paths += [BACKEND / "tracking" / name for name in (
        "retraining.py", "training.py", "data_splits.py", "data_validation.py",
        "evaluate_models.py", "evaluation_gate.py", "import_baselines.py")]
    paths += [BACKEND / "requirements.txt", BACKEND / "requirements-tracking.txt"]
    return {str(path.relative_to(BACKEND)): sha256(path) for path in paths}


def prepare_plan(database):
    # Lazy imports: lifecycle/approval utilities and basic CI need no ML packages.
    from tracking.data_splits import split_three_way

    database = Path(database).resolve()
    dvc = database.with_suffix(database.suffix + ".dvc")
    require(dvc.is_file() and f"md5: {DATASET_MD5}" in dvc.read_text(encoding="utf-8"),
            "Reviewed DVC pointer is missing or changed")
    tasks = {}
    for task, table in TABLES.items():
        validate_training_database(database, table)
        rows, database_hash = read_rows(database, task)
        parts = split_three_way([row[:-1] for row in rows], [row[-1] for row in rows], seed=42)
        dataset = {"dvc_md5": DATASET_MD5, "sha256": database_hash,
                   "ordered_rows_sha256": canonical_sha256(rows), "table": table,
                   "feature_columns": FEATURES}
        record = {"dataset": dataset, "partitions": {
            name: [int(i) for i in indices]
            for name, indices in zip(("train", "validation", "test"), parts)}}
        verify_partitions(record, rows, database_hash)
        tasks[task] = record
    return {"schema_version": 1, "seed": 42,
            "holdout_policy": "frozen_test_indices_never_used_for_fit_or_validation",
            "historical_production_comparability": "unverified_promotion_blocked_until_reviewed_evidence",
            "dvc_pointer_sha256": sha256(dvc),
            "sources": sources(), "tasks": tasks}


class Session:
    def __init__(self, plan, root):
        self.plan, self.root, self.runs = plan, root, {}

    def started(self, run):
        require(run.name in TABLES and run.name not in self.runs, "Unexpected training task")
        run.metadata["controlled_training"] = {
            "plan_sha256": canonical_sha256(self.plan), "seed": 42,
            "dataset": self.plan["tasks"][run.name]["dataset"],
            "source_sha256": self.plan["sources"],
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "packages": dict(sorted((d.metadata["Name"], d.version)
                                                    for d in metadata.distributions() if d.metadata["Name"]))},
            "determinism": "tensorflow_op_determinism_requested_not_cross_platform_bitwise_guaranteed",
        }
        self.runs[run.name] = run

    def partitions(self, run, features, labels, partitions):
        # Compare the rows actually loaded by the trainer, not only the preflight DB.
        normalized = []
        for x, label in zip(features, labels):
            if isinstance(label, (tuple, list)):
                label = label[0]
            normalized.append([*[float(v) for v in x], str(label)])
        expected = self.plan["tasks"][run.name]
        require(canonical_sha256(normalized) == expected["dataset"]["ordered_rows_sha256"],
                "Trainer rows differ from frozen data")
        require(partitions == expected["partitions"], "Trainer split differs from frozen holdout")
        # Written before scaler fitting/model construction.
        write_new(run.output_dir / "partition_plan.json", expected)

    def finished(self, run):
        require((run.output_dir / "partition_plan.json").is_file(), "Partition capture missing")
        metrics = run.metadata.get("test_metrics", {})
        import math
        for key in ("test_accuracy", "test_loss"):
            value = metrics.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0
                    and (key != "test_accuracy" or value <= 1), "Invalid final training metrics")
        record = {"schema_version": 1, "task": run.name,
                  **self.plan["tasks"][run.name],
                  "artifacts": {key: sha256(run.output_dir / f"{run.name}_{suffix}")
                                for key, suffix in SUFFIXES.items()},
                  "provenance": {"status": "captured_training_time_unreviewed",
                                 "training_plan_sha256": canonical_sha256(self.plan),
                                 "mlflow_run_id": run.metadata["mlflow_run_id"],
                                 "training_git": run.metadata["git"],
                                 "captured_at": datetime.now(timezone.utc).isoformat()}}
        write_new(run.output_dir / f"{run.name}_provenance.json", record)


def execute(plan, expected_sha256, database, output_root, tracking_uri):
    require(canonical_sha256(plan) == expected_sha256, "Frozen plan hash mismatch")
    require(prepare_plan(database) == plan, "Dataset, source, or deterministic split changed; review a new plan")
    output_root = Path(output_root).resolve()
    # Bundles may only be created under isolated training outputs, never production.
    require(output_root.is_relative_to((BACKEND / "data/training_runs").resolve()),
            "Candidate root must be within backend/data/training_runs")
    require(not os.environ.get("MLFLOW_RUN_ID"), "Cannot resume a run")
    require("tensorflow" not in sys.modules, "Controlled training requires a fresh CLI process")
    import mlflow
    require(mlflow.active_run() is None, "Cannot resume a run")
    mlflow.tracking.MlflowClient(tracking_uri=tracking_uri).search_experiments(max_results=1)
    root = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    write_new(root / "plan.json", plan)
    for name, expected in plan["sources"].items():
        destination = root / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BACKEND / name, destination)
        require(sha256(destination) == expected, "Source snapshot changed")
    shutil.copyfile(Path(database).with_suffix(".db.dvc"), root / "dataset.dvc")
    session = Session(plan, root)
    with tracking_environment({
        "SIGNVISION_TRACKING_ENABLED": "true", "MLFLOW_TRACKING_URI": tracking_uri,
        "MLFLOW_EXPERIMENT_NAME": "SignVision Controlled Retraining",
        "TF_DETERMINISTIC_OPS": "1", "CUDA_VISIBLE_DEVICES": "-1", "MPLBACKEND": "Agg",
    }):
        # Must be a fresh CLI process, not an already-running inference server.
        import tensorflow as tf
        tf.config.experimental.enable_op_determinism()
        from PSL.alphabet_recognition.train_alphabet_model import train_alphabets
        from PSL.word_recognition.train_word_model import train_words
        token = CONTROLLED_SESSION.set(session)
        try:
            for trainer in (train_alphabets, train_words):
                tf.keras.utils.set_random_seed(42)
                trainer()
        finally:
            CONTROLLED_SESSION.reset(token)
    require(prepare_plan(database) == plan, "Dataset/source changed during training; bundle blocked")
    bundle = root / "bundle"
    bundle.mkdir(exist_ok=False)
    write_new(bundle / "plan.json", plan)
    for task, run in session.runs.items():
        require(run.metadata["training_status"] == "FINISHED" and not run.metadata["tracking_errors"],
                "Incomplete MLflow tracking; candidate bundle blocked")
        for suffix in (*SUFFIXES.values(), "provenance.json"):
            name = f"{task}_{suffix}"
            shutil.copyfile(run.output_dir / name, bundle / name)
        shutil.copyfile(run.output_dir / "run_metadata.json", bundle / f"{task}_run_metadata.json")
    require(set(session.runs) == set(TABLES), "Both tasks are required")
    files = {p.name: sha256(p) for p in bundle.iterdir()}
    version = {"schema_version": 1, "version": root.name,
               "plan_sha256": canonical_sha256(plan), "git": git_metadata(BACKEND.parent),
               "files": files, "promotion_approved": False}
    write_new(bundle / "version.json", version)
    # Separate audit run binds both task runs to the complete bundle; failures leave
    # local outputs but no completion receipt. Never modify an existing MLflow run.
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("SignVision Controlled Retraining")
    with mlflow.start_run(run_name=f"candidate-bundle-{root.name}") as audit:
        mlflow.log_artifacts(str(root), artifact_path="candidate")
        mlflow.set_tags({"candidate.version": root.name, "candidate.review": "unreviewed",
                         "candidate.plan_sha256": canonical_sha256(plan)})
        audit_id = audit.info.run_id
    write_new(root / "complete.json", {"bundle_sha256": canonical_sha256(version),
                                       "mlflow_audit_run_id": audit_id, "deployment_performed": False})
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Validate and freeze a plan; never fits models")
    prepare.add_argument("--output", required=True, type=Path)
    train = sub.add_parser("train", help="Explicit CPU-only training; no deployment")
    train.add_argument("--plan", required=True, type=Path)
    train.add_argument("--plan-sha256", required=True)
    train.add_argument("--execute", action="store_true")
    train.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    train.add_argument("--output-root", type=Path, default=BACKEND / "data/training_runs/candidates")
    args = parser.parse_args()
    database = BACKEND / "data/db/main_dataset.db"
    if args.command == "prepare":
        plan = prepare_plan(database)
        write_new(args.output, plan)
        print(f"Frozen plan SHA-256: {canonical_sha256(plan)}")
    else:
        require(args.execute, "Training requires explicit --execute")
        print(execute(read_json(args.plan), args.plan_sha256, database, args.output_root, args.tracking_uri))


if __name__ == "__main__":
    main()
