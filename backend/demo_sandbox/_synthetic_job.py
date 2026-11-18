"""Private, fixed synthetic workflow worker; stdin/stdout only, no model training."""

import json
import sys

from demo_sandbox import foundation
from tracking.evaluation_gate import GateError, canonical_sha256, digest, require


def decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate simulation JSON field")
            result[key] = value
        return result
    def constant(value):
        raise GateError("Non-finite simulation JSON")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def request_for(trigger_id, alert_id):
    digest(trigger_id, "simulation trigger identity")
    digest(alert_id, "simulation alert identity")
    config = foundation._config()
    dataset = foundation._dataset(config)
    indices = dataset["partitions"]["train"]
    return {"schema_version": 1, "scope": foundation.SCOPE,
            "kind": "synthetic-retraining-request-v1", "trigger_id": trigger_id, "alert_id": alert_id,
            "dataset_sha256": canonical_sha256(dataset), "train_indices": indices,
            "train_rows": [dataset["rows"][i] for i in indices],
            "workflow_simulation": True, "real_model_training": False, "production_evidence": False}


def simulate(request):
    """Recompute one toy threshold from known synthetic training rows only.

    This arithmetic operation is a workflow simulation, not a trained PSL model.
    It produces no test/validation predictions, accuracy, loss or approvals.
    """
    require(isinstance(request, dict), "Invalid synthetic job request")
    expected = request_for(request.get("trigger_id"), request.get("alert_id"))
    require(canonical_sha256(request) == canonical_sha256(expected), "Synthetic training request differs")
    labels = foundation.POLICY["class_labels"]
    lower = max(row["features"][0] for row in request["train_rows"] if row["label"] == labels[0])
    upper = min(row["features"][0] for row in request["train_rows"] if row["label"] == labels[1])
    require(lower < upper, "Synthetic class boundary is invalid")
    model = {"schema_version": 1, "scope": foundation.SCOPE,
             "artifact_type": "synthetic-threshold-retraining-simulation-v1",
             "version": "demo-job-" + request["trigger_id"][:16],
             "trigger_id": request["trigger_id"], "alert_id": request["alert_id"],
             "feature_count": 42, "class_labels": labels,
             "rule": {"feature_index": 0, "threshold": (lower + upper) / 2},
             "dataset_sha256": request["dataset_sha256"],
             "train_indices_sha256": canonical_sha256(request["train_indices"]),
             "workflow_simulation": True, "real_model_training": False,
             "production_evidence": False, "promotion_approved": False}
    return {"scope": foundation.SCOPE, "trigger_id": request["trigger_id"],
            "operation": "synthetic-threshold-recomputation", "used_partitions": ["train"],
            "training_rows": len(request["train_rows"]), "validation_used": False, "test_used": False,
            "workflow_simulation": True, "real_model_training": False,
            "quality_metrics_computed": False, "model": model}


def main():
    try:
        data = sys.stdin.buffer.read(32769)
        require(len(data) <= 32768, "Synthetic request exceeds size limit")
        result = simulate(decode(data))
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (GateError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"scope": foundation.SCOPE, "status": "FAILED",
                          "workflow_simulation": True, "error_type": type(error).__name__}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
