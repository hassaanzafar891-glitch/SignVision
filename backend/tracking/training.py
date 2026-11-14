"""Optional MLflow logging and isolated outputs for future training runs.

This module has no TensorFlow, NumPy, or MLflow import-time dependencies.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import uuid
import warnings

from tracking.import_baselines import BACKEND, git_metadata, sha256, tracking_uri


DAGSHUB_TRACKING_URI = "https://dagshub.com/hassaanzafar891-glitch/SignVision.mlflow"
EPOCH_METRICS = ("accuracy", "val_accuracy", "loss", "val_loss")


def json_value(value):
    """Convert NumPy/TensorFlow values without importing either library."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def dataset_metadata(features, labels, source, table):
    """Fingerprint the ordered, selected rows actually supplied to the split.

    This is independent of SQLite pages and includes labels. It remains correct
    if the source database has WAL data or is changed after the rows are read.
    """
    if len(features) != len(labels):
        raise ValueError("Feature and label counts differ")
    digest = hashlib.sha256()
    counts = Counter()
    for feature, label in zip(features, labels):
        # Alphabet SQL returns one-element label tuples; word SQL returns strings.
        if isinstance(label, (list, tuple)) and len(label) == 1:
            label = label[0]
        row = json.dumps([feature, label], default=json_value,
                         ensure_ascii=False, separators=(",", ":"))
        digest.update(row.encode("utf-8") + b"\n")
        counts[str(label)] += 1
    return {
        "source": Path(source).name,
        "table": table,
        "samples": len(features),
        "features": len(features[0]) if len(features) else 0,
        "class_count": len(counts),
        "class_counts": dict(sorted(counts.items())),
        "sha256": digest.hexdigest(),
        "fingerprint_method": "sha256-ordered-json-feature-label-rows-v1",
    }


class TrainingRun:
    def __init__(self, name, script, *, backend=BACKEND, environ=None):
        self.name = name
        self.script = Path(script)
        self.backend = Path(backend).resolve()
        self.environ = os.environ if environ is None else environ
        self.enabled = self.environ.get("SIGNVISION_TRACKING_ENABLED", "").lower() in (
            "1", "true", "yes", "on",
        )
        identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex
        self.output_dir = self.backend / "data" / "training_runs" / name / identifier
        self.client = None
        self.owns_run = False
        self.metadata = {
            "model_name": name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_status": "RUNNING",
            "tracking_requested": self.enabled,
            "tracking_errors": [],
            "params": {},
            "epoch_metrics": [],
            "test_metrics": {},
        }

    def _tracking_error(self, operation, exception):
        # Exception messages can contain authenticated URLs; never persist them.
        self.metadata["tracking_errors"].append({
            "operation": operation, "exception_type": type(exception).__name__,
        })
        warnings.warn(f"MLflow {operation} failed; local training outputs are retained.",
                      RuntimeWarning, stacklevel=2)

    def _send(self, operation, *args, **kwargs):
        if self.client is None or self.metadata["tracking_errors"]:
            return
        try:
            return getattr(self.client, operation)(*args, **kwargs)
        except Exception as error:
            self._tracking_error(operation, error)

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=False)
        print(f"Training outputs: {self.output_dir}")
        git = git_metadata(self.backend.parent)
        self.metadata["git"] = {
            "commit": git["import_commit"], "checkout_dirty": git["import_checkout_dirty"],
        }
        shutil.copyfile(self.script, self.output_dir / "training_script.py")
        shutil.copyfile(Path(__file__), self.output_dir / "tracking_helper.py")
        if self.enabled:
            try:
                self.client = importlib.import_module("mlflow")
                if self.client.active_run() is not None or self.environ.get("MLFLOW_RUN_ID"):
                    raise RuntimeError("A training run must not resume an existing MLflow run")
                self.client.set_tracking_uri(tracking_uri(self.backend, self.environ))
                self.client.set_experiment(
                    self.environ.get("MLFLOW_EXPERIMENT_NAME") or "SignVision Training"
                )
                active = self.client.start_run(run_name=f"{self.name}-{self.output_dir.name}", tags={
                    "model.name": self.name,
                    "training.performed": "true",
                    "training.git.commit": git["import_commit"] or "unavailable",
                    "training.git.dirty": str(git["import_checkout_dirty"]).lower(),
                })
                self.owns_run = True
                self.metadata["mlflow_run_id"] = active.info.run_id
            except Exception as error:
                self._tracking_error("initialization", error)
        return self

    def log_params(self, params):
        self.metadata["params"].update(params)
        self._send("log_params", params)

    def record_dataset(self, features, labels, source, table, *, train_samples, test_samples):
        dataset = dataset_metadata(features, labels, source, table)
        dataset.update(train_samples=train_samples, test_samples=test_samples)
        self.metadata["dataset"] = dataset
        self.log_params({f"dataset.{key}": value for key, value in dataset.items()
                         if key != "class_counts"})

    def record_model(self, model, *, epochs, batch_size, split):
        architecture = json.loads(model.to_json())
        (self.output_dir / "model_architecture.json").write_text(
            json.dumps(architecture, indent=2), encoding="utf-8",
        )
        optimizer = model.optimizer
        learning_rate = optimizer.learning_rate
        if hasattr(learning_rate, "numpy"):
            learning_rate = learning_rate.numpy()
        layers = []
        for layer in model.layers:
            config = layer.get_config()
            layers.append({"type": type(layer).__name__, **{
                key: config[key] for key in ("units", "activation", "rate") if key in config
            }})
        self.metadata["optimizer_config"] = optimizer.get_config()
        self.log_params({
            "model_name": self.name,
            "architecture": json.dumps(layers, separators=(",", ":")),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": float(learning_rate),
            "optimizer": type(optimizer).__name__,
            **split,
        })

    def callback(self, callback_class):
        run = self

        class EpochTracking(callback_class):
            def on_epoch_end(self, epoch, logs=None):
                values = {key: float(value) for key, value in (logs or {}).items()
                          if key in EPOCH_METRICS and value is not None}
                run.metadata["epoch_metrics"].append({"epoch": epoch + 1, **values})
                if values:
                    run._send("log_metrics", values, step=epoch + 1)

        return EpochTracking()

    def record_test(self, *, loss, accuracy):
        self.metadata["test_metrics"] = {"test_loss": float(loss), "test_accuracy": float(accuracy)}
        self._send("log_metrics", self.metadata["test_metrics"])

    def _save_metadata(self):
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(self.metadata, indent=2, default=json_value, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def __exit__(self, exception_type, exception, traceback):
        self.metadata["training_status"] = "FAILED" if exception_type else "FINISHED"
        try:
            self.metadata["artifacts"] = {
                path.name: {"sha256": sha256(path), "size_bytes": path.stat().st_size}
                for path in self.output_dir.iterdir() if path.is_file()
            }
            self._save_metadata()
            self._send("log_artifacts", str(self.output_dir), artifact_path="training")
        finally:
            if self.owns_run:
                try:
                    self.client.end_run(status="FAILED" if exception_type or
                                         self.metadata["tracking_errors"] else "FINISHED")
                except Exception as error:
                    self._tracking_error("end_run", error)
            # Include upload/end-run errors in the retained local record as well.
            self._save_metadata()
        return False
