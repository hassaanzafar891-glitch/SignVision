"""One-time, train-only fitted baseline. Initial evaluation role, never promotion."""

from datetime import datetime, timezone
import hashlib
import pickle

from demo_sandbox import foundation
from tracking.evaluation_gate import canonical_sha256, read_json, require
from tracking.import_baselines import sha256


VERSION = "fitted-evaluation-champion-v1"
PARAMETERS = {"criterion": "gini", "max_depth": 1, "min_samples_leaf": 1, "random_state": 42}
FILES = ("champion/model.pkl", "champion/training_receipt.json", "champion_pointer.json")


def path(name):
    require(name in FILES, "Invalid evaluation champion path")
    p = foundation._path("evaluation/" + name)
    if p.exists():
        require(p.stat().st_size <= 8192, "Champion artifact exceeds bound")
    return p


def dataset_identity(dataset):
    rows, parts = dataset["rows"], dataset["partitions"]
    require(sorted(i for ids in parts.values() for i in ids) == list(range(len(rows))), "Partition coverage mismatch")
    groups = {name: {tuple(rows[i]["features"]) for i in ids} for name, ids in parts.items()}
    require(not (groups["train"] & groups["validation"] or groups["train"] & groups["test"]
                 or groups["validation"] & groups["test"]), "Duplicate-vector leakage")
    return {"dataset_sha256": canonical_sha256(dataset),
            "split_indices_sha256": {k: canonical_sha256(v) for k, v in parts.items()},
            "train_indices": parts["train"], "validation_indices": parts["validation"],
            "preprocessing": "identity-42-features-v1"}


def _receipt(dataset, model, artifact, started, finished):
    import sklearn
    return {"schema_version": 1, "scope": foundation.SCOPE, "version": VERSION,
            **dataset_identity(dataset), "status": "TRAINED", "algorithm": "sklearn.tree.DecisionTreeClassifier",
            "parameters": model.get_params(), "sklearn_version": sklearn.__version__,
            "model_sha256": hashlib.sha256(artifact).hexdigest(), "started_at": started, "finished_at": finished,
            "training_samples": len(dataset["partitions"]["train"]), "fitted": True,
            "validation_used": False, "test_used": False, "hyperparameter_selection": "fixed_before_training",
            "production_evidence": False, "promotion_performed": False}


def _pointer(receipt):
    return {"schema_version": 1, "scope": foundation.SCOPE, "version": VERSION,
            "state": "initial-evaluation-champion-only", "model_sha256": receipt["model_sha256"],
            "training_receipt_sha256": canonical_sha256(receipt),
            "foundation_pointer_sha256": sha256(foundation._path("active.json")),
            "production_evidence": False, "promotion_performed": False}


def prepare():
    """Caller holds the evaluation write lock. No overwrite, repair or test scoring."""
    from sklearn.tree import DecisionTreeClassifier
    dataset = foundation.read_dataset()
    dataset_identity(dataset)
    if any(path(name).exists() for name in FILES):
        load()  # Complete existing initialization is read-only; partial initialization blocks.
        return
    started = datetime.now(timezone.utc).isoformat()
    model = DecisionTreeClassifier(**PARAMETERS)
    train = [dataset["rows"][i] for i in dataset["partitions"]["train"]]
    model.fit([r["features"] for r in train], [r["label"] for r in train])
    artifact = pickle.dumps(model, protocol=4)
    receipt = _receipt(dataset, model, artifact, started, datetime.now(timezone.utc).isoformat())
    payloads = (artifact, foundation._encode(receipt), foundation._encode(_pointer(receipt)))
    for name, payload in zip(FILES, payloads):
        require(len(payload) <= 8192, "Champion artifact too large")
        p = path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with path(name).open("xb") as stream:
            stream.write(payload)
    load()


def load():
    """Load only trusted local sandbox artifacts after receipt/pointer hash checks."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.utils.validation import check_is_fitted
    dataset = foundation.read_dataset()
    require(all(path(name).is_file() for name in FILES), "Missing fitted champion or training receipt; prepare-champion required")
    receipt, pointer = read_json(path(FILES[1])), read_json(path(FILES[2]))
    artifact = path(FILES[0]).read_bytes()
    require(isinstance(receipt, dict) and receipt.get("model_sha256") == hashlib.sha256(artifact).hexdigest(),
            "Champion model hash mismatch")
    require(canonical_sha256(pointer) == canonical_sha256(_pointer(receipt)), "Evaluation champion pointer mismatch")
    model = pickle.loads(artifact)
    require(type(model) is DecisionTreeClassifier and model.get_params() == DecisionTreeClassifier(**PARAMETERS).get_params(),
            "Unexpected baseline estimator/configuration")
    check_is_fitted(model)
    require(model.n_features_in_ == 42 and list(model.classes_) == dataset["class_labels"]
            and int(model.tree_.n_node_samples[0]) == len(dataset["partitions"]["train"]), "Baseline fit shape mismatch")
    first, last = receipt.get("started_at"), receipt.get("finished_at")
    require(isinstance(first, str) and isinstance(last, str), "Missing baseline timestamps")
    start, end = datetime.fromisoformat(first), datetime.fromisoformat(last)
    require(start.utcoffset() is not None and end.utcoffset() is not None and end >= start, "Invalid baseline timestamps")
    expected = _receipt(dataset, model, artifact, first, last)
    require(canonical_sha256(receipt) == canonical_sha256(expected), "Baseline training receipt identity mismatch")
    return model, receipt, pointer
