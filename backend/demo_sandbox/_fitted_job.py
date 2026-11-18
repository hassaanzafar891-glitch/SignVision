"""Actual scikit-learn fitting on sandbox synthetic train rows; never test scoring."""

import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import pickle
import sys

from demo_sandbox import foundation, _synthetic_job
from tracking.evaluation_gate import canonical_sha256, require


PARAMETERS = {"criterion": "gini", "max_depth": 3, "min_samples_leaf": 1, "random_state": 42}


def request_for(trigger_id, alert_id):
    request = _synthetic_job.request_for(trigger_id, alert_id)
    data = foundation._dataset(foundation._config())
    validation = data["partitions"]["validation"]
    return {**request, "kind": "fitted-synthetic-training-request-v1",
            "workflow_simulation": False, "real_model_training": True,
            "validation_indices": validation, "validation_rows": [data["rows"][i] for i in validation],
            "split_indices_sha256": {name: canonical_sha256(ids) for name, ids in data["partitions"].items()},
            "model_configuration": {"algorithm": "sklearn.tree.DecisionTreeClassifier", "parameters": PARAMETERS}}


def _metrics(model, request):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    rows = request["validation_rows"]
    actual = [row["label"] for row in rows]
    predicted = model.predict([row["features"] for row in rows])
    labels = foundation.POLICY["class_labels"]
    return {"validation_accuracy": float(accuracy_score(actual, predicted)),
            "validation_macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
            "validation_confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
            "validation_samples": len(rows)}


def train(request):
    import sklearn
    from sklearn.tree import DecisionTreeClassifier

    require(canonical_sha256(request) == canonical_sha256(request_for(request.get("trigger_id"), request.get("alert_id"))),
            "Sandbox training request differs from frozen train/validation partitions")
    started = datetime.now(timezone.utc).isoformat()
    model = DecisionTreeClassifier(**PARAMETERS)
    rows = request["train_rows"]
    model.fit([row["features"] for row in rows], [row["label"] for row in rows])
    metrics = _metrics(model, request)  # Fixed configuration; no search/checkpoint selection.
    artifact = pickle.dumps(model, protocol=4)
    require(len(artifact) <= 8192, "Fitted sandbox model exceeds size limit")
    receipt = {"schema_version": 1, "scope": foundation.SCOPE, "status": "TRAINED",
               "trigger_id": request["trigger_id"], "alert_id": request["alert_id"],
               "dataset_sha256": request["dataset_sha256"], "split_indices_sha256": request["split_indices_sha256"],
               "train_indices": request["train_indices"], "validation_indices": request["validation_indices"],
               "configuration": request["model_configuration"], "estimator_parameters": model.get_params(),
               "sklearn_version": sklearn.__version__,
               "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
               "model_sha256": hashlib.sha256(artifact).hexdigest(), "metrics": metrics,
               "training_samples": len(rows), "fitted": True, "test_used": False,
               "hyperparameter_selection": "fixed_before_training", "workflow_simulation": False,
               "real_model_training": True, "production_evidence": False, "promotion_approved": False}
    return {"receipt": receipt, "model_base64": base64.b64encode(artifact).decode("ascii")}


def validate(payload, request):
    """Validate trusted child output without fitting again or scoring test rows.

    Pickle is executable: only this fixed worker's local, bounded output is accepted.
    This does not authenticate files against a malicious privileged host owner.
    """
    import sklearn
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.utils.validation import check_is_fitted

    require(isinstance(payload, dict) and set(payload) == {"receipt", "model_base64"}, "Invalid fitted job output")
    encoded = payload["model_base64"]
    require(isinstance(encoded, str) and len(encoded) <= 11000, "Invalid fitted model encoding")
    artifact = base64.b64decode(encoded, validate=True)
    require(len(artifact) <= 8192, "Fitted model exceeds size limit")
    receipt = payload["receipt"]
    require(isinstance(receipt, dict) and hashlib.sha256(artifact).hexdigest() == receipt.get("model_sha256"),
            "Fitted model checksum mismatch")
    stream = io.BytesIO(artifact)
    model = pickle.load(stream)
    require(stream.read() == b"" and type(model) is DecisionTreeClassifier, "Unexpected sandbox estimator")
    check_is_fitted(model)
    require(model.n_features_in_ == 42 and list(model.classes_) == foundation.POLICY["class_labels"],
            "Fitted model feature/class mismatch")
    require(all(model.get_params()[k] == v for k, v in PARAMETERS.items()), "Fitted model configuration mismatch")
    # Non-configured defaults are fixed by this installed sklearn version.
    require(model.get_params() == DecisionTreeClassifier(**PARAMETERS).get_params(), "Unexpected estimator settings")
    started, finished = receipt.get("started_at"), receipt.get("finished_at")
    require(isinstance(started, str) and isinstance(finished, str), "Missing training timestamps")
    first, last = datetime.fromisoformat(started), datetime.fromisoformat(finished)
    require(first.utcoffset() is not None and last.utcoffset() is not None and last >= first,
            "Invalid training timestamps")
    expected = {"schema_version": 1, "scope": foundation.SCOPE, "status": "TRAINED",
               "trigger_id": request["trigger_id"], "alert_id": request["alert_id"],
               "dataset_sha256": request["dataset_sha256"], "split_indices_sha256": request["split_indices_sha256"],
               "train_indices": request["train_indices"], "validation_indices": request["validation_indices"],
               "configuration": request["model_configuration"], "estimator_parameters": model.get_params(),
               "sklearn_version": sklearn.__version__,
               "started_at": started, "finished_at": finished, "model_sha256": hashlib.sha256(artifact).hexdigest(),
               "metrics": _metrics(model, request), "training_samples": len(request["train_rows"]),
               "fitted": True, "test_used": False, "hyperparameter_selection": "fixed_before_training",
               "workflow_simulation": False, "real_model_training": True,
               "production_evidence": False, "promotion_approved": False}
    require(canonical_sha256(receipt) == canonical_sha256(expected), "Training receipt does not match fitted artifact")
    return artifact


def main():
    try:
        data = sys.stdin.buffer.read(65537)
        require(len(data) <= 65536, "Fitted job request exceeds limit")
        print(json.dumps(train(_synthetic_job.decode(data)), allow_nan=False, sort_keys=True))
    except Exception as error:
        # Never issue a successful receipt or echo potentially sensitive exception text.
        print(json.dumps({"scope": foundation.SCOPE, "status": "FAILED", "error_type": type(error).__name__}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
