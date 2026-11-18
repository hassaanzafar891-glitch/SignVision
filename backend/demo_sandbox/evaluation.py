"""Genuine fitted-model evaluation on frozen synthetic test rows; no activation."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import pickle

from demo_sandbox import foundation, drift, orchestrator, evaluation_champion as champion
from tracking.evaluation_gate import (
    GateError, canonical_sha256, digest, read_json, require, require_no_accuracy_regression,
)
from tracking.import_baselines import sha256


CASES = ("challenger", "stricter-control")
MAX_RECEIPTS = 2  # One immutable, deduplicated result per fixed case; no eviction.
PROTOCOL = "fitted-synthetic-frozen-test-v1"


def _path(name):
    require(name in ("journal.json", ".journal.tmp", ".write.lock"), "Invalid demo evaluation path")
    path = foundation._path("evaluation/" + name)
    if path.exists():
        require(path.stat().st_size <= foundation.EXTENSION_LIMITS["evaluation/" + name],
                "Demo evaluation file exceeds size limit")
    return path


def _candidate(job_slot):
    require(isinstance(job_slot, str) and job_slot in tuple(f"job-{i:03d}" for i in range(1, 11)),
            "Invalid fixed sandbox job slot")
    report = orchestrator.inspect()  # Full event, request, receipt and artifact checks, no refitting.
    job = next((j for j in report["jobs"] if j["slot"] == job_slot), None)
    require(job is not None and job["kind"] == "fitted-synthetic-trigger-v1" and job["status"] == "FINISHED",
            "A successful genuinely fitted sandbox job is required")
    root = "retraining/" + job_slot + "/"
    receipt = read_json(foundation._path(root + "training_receipt.json"))
    identity = champion.dataset_identity(foundation.read_dataset())
    require(receipt["dataset_sha256"] == identity["dataset_sha256"]
            and receipt["split_indices_sha256"] == identity["split_indices_sha256"]
            and receipt["train_indices"] == identity["train_indices"]
            and receipt["validation_indices"] == identity["validation_indices"]
            and receipt["test_used"] is False and receipt["hyperparameter_selection"] == "fixed_before_training",
            "Challenger split/selection provenance mismatch")
    model_path = foundation._path(root + "model.pkl")
    require(sha256(model_path) == receipt["model_sha256"] == job["artifacts"]["model.pkl"],
            "Challenger model hash mismatch")
    model = pickle.loads(model_path.read_bytes())  # Only the verified trusted local worker artifact.
    return model, {"version": job_slot, "trigger_id": job["trigger_id"], "model_sha256": receipt["model_sha256"],
                   "training_receipt_sha256": canonical_sha256(receipt)}


def _context(job_slot="job-002"):
    foundation.status()
    dataset = foundation.read_dataset()
    parts, rows = dataset["partitions"], dataset["rows"]
    champion.dataset_identity(dataset)
    test = parts["test"]
    identity = {"dataset_sha256": canonical_sha256(dataset), "partition": "test",
                "test_indices_sha256": canonical_sha256(test),
                "test_rows_sha256": canonical_sha256([rows[i] for i in test]),
                "sample_count": len(test), "classes": dataset["class_labels"],
                "seed": dataset["seed"], "production_independence_verified": False}
    candidate_model, candidate_id = _candidate(job_slot)
    champion_model, receipt, pointer = champion.load()
    return {"dataset": dataset, "test_identity": identity, "job_slot": job_slot,
            "champion": (champion_model, {"version": champion.VERSION, "model_sha256": receipt["model_sha256"],
                "training_receipt_sha256": canonical_sha256(receipt)}),
            "candidate": (candidate_model, candidate_id),
            "active_pointer_sha256": sha256(champion.path("champion_pointer.json")),
            "foundation_pointer_sha256": pointer["foundation_pointer_sha256"]}


def _score(context, model):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    dataset = context["dataset"]
    labels = dataset["class_labels"]
    test = [dataset["rows"][i] for i in dataset["partitions"]["test"]]
    actual = [r["label"] for r in test]
    predicted = model.predict([r["features"] for r in test])  # Identity preprocessing for BOTH models.
    require(set(predicted) <= set(labels), "Invalid fitted-model predictions")
    return {"test_accuracy": float(accuracy_score(actual, predicted)),
            "test_macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
            "sample_count": len(test), "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist()}


def _result(context, case):
    require(isinstance(case, str) and case in CASES, "Unknown synthetic evaluation case")
    control = case == "stricter-control"
    minimum_delta = 0.05 if control else 0.0  # Declared policy control, never tuned using test results.
    reference_metrics = _score(context, context["champion"][0])
    candidate_metrics = _score(context, context["candidate"][0])
    delta = candidate_metrics["test_accuracy"] - reference_metrics["test_accuracy"]
    try:
        require_no_accuracy_regression(reference_metrics["test_accuracy"], candidate_metrics["test_accuracy"])
        require(delta >= minimum_delta, "Stricter demo policy requires an accuracy gain of at least 0.05")
        decision, reason = "PASS", "Candidate accuracy meets or exceeds reference accuracy"
    except GateError as error:
        decision, reason = "BLOCK", str(error)
    return {"schema_version": 1, "scope": foundation.SCOPE, "protocol": PROTOCOL,
            "case": case, "comparison_only_control": control, "job_slot": context["job_slot"],
            "policy": "no_accuracy_regression_v1", "thresholds": {"minimum_accuracy_delta": minimum_delta},
            "additional_policy": "demo_minimum_gain_0.05" if control else None,
            "preprocessing": "identity-42-features-v1",
            "test_identity": context["test_identity"],
            "test_identity_sha256": canonical_sha256(context["test_identity"]),
            "active_pointer_sha256": context["active_pointer_sha256"],
            "foundation_pointer_sha256": context["foundation_pointer_sha256"],
            "reference": {**context["champion"][1], "metrics": reference_metrics},
            "candidate": {**context["candidate"][1], "metrics": candidate_metrics},
            "test_accuracy_delta": delta, "decision": decision, "reason": reason,
            "eligible_for_future_sandbox_promotion": decision == "PASS" and not control,
            "production_evidence": False, "production_approval": False, "promotion_performed": False,
            "notice": "Genuinely fitted synthetic models only; no independent production validation"}


def _journal(context):
    require(not _path(".journal.tmp").exists(), "Incomplete evaluation write; inspect manually")
    path = _path("journal.json")
    empty = {"schema_version": 1, "scope": foundation.SCOPE,
             "kind": "synthetic-evaluation-journal-v1", "max_receipts": MAX_RECEIPTS, "receipts": []}
    journal = read_json(path) if path.exists() else drift._seal_journal(empty)
    require(isinstance(journal, dict) and set(journal) == set(empty) | {"journal_sha256"},
            "Invalid evaluation journal fields")
    require(drift._seal_journal(journal) == journal, "Evaluation journal checksum mismatch")
    require(all(journal[k] == v for k, v in empty.items() if k != "receipts"),
            "Invalid evaluation journal identity")
    require(isinstance(journal["receipts"], list) and len(journal["receipts"]) <= MAX_RECEIPTS,
            "Evaluation journal capacity exceeded")
    seen = set()
    for receipt in journal["receipts"]:
        require(isinstance(receipt, dict), "Invalid evaluation receipt")
        expected = _result(context, receipt.get("case"))
        require(set(receipt) == set(expected) | {"receipt_id", "evaluated_at", "receipt_sha256"},
                "Invalid evaluation receipt fields")
        require(all(canonical_sha256(receipt[k]) == canonical_sha256(v) for k, v in expected.items()),
                "Evaluation receipt differs from frozen test replay")
        require(receipt["case"] not in seen, "Duplicate evaluation case")
        seen.add(receipt["case"])
        require(receipt["receipt_id"] == canonical_sha256(expected), "Evaluation receipt identity mismatch")
        require(receipt["receipt_sha256"] == canonical_sha256({
            k: v for k, v in receipt.items() if k != "receipt_sha256"}), "Evaluation receipt checksum mismatch")
        stamp = receipt["evaluated_at"]
        require(isinstance(stamp, str) and len(stamp) <= 64, "Invalid evaluation timestamp")
        try:
            require(datetime.fromisoformat(stamp).utcoffset() is not None, "Timestamp needs timezone")
        except ValueError as error:
            raise GateError("Invalid evaluation timestamp") from error
    return journal


@contextmanager
def _lock():
    path = _path(".write.lock")
    path.parent.mkdir(exist_ok=True)
    with _path(".write.lock").open("xb") as stream:
        stream.write(b"synthetic-demo-only evaluation writer\n")
        info = os.fstat(stream.fileno())
    try:
        yield
    finally:
        drift._owned_remove(_path(".write.lock"), (info.st_dev, info.st_ino))


def _save(journal):
    payload = foundation._encode(drift._seal_journal(journal))
    require(len(payload) <= 65536, "Evaluation journal exceeds size limit")
    temporary = _path(".journal.tmp")
    stream = temporary.open("xb")
    info = os.fstat(stream.fileno())
    try:
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, _path("journal.json"))
    except BaseException:
        drift._owned_remove(temporary, (info.st_dev, info.st_ino))
        raise


def run(case="challenger", *, job_slot="job-002"):
    context = _context(job_slot)
    _result(context, case)  # Reject invalid requests before any writes.
    with _lock():
        context = _context(job_slot)
        journal = _journal(context)
        receipt = next((r for r in journal["receipts"] if r["case"] == case), None)
        deduplicated = receipt is not None
        if receipt is None:
            require(len(journal["receipts"]) < MAX_RECEIPTS, "Evaluation capacity reached")
            result = _result(context, case)
            receipt = {**result, "receipt_id": canonical_sha256(result),
                       "evaluated_at": datetime.now(timezone.utc).isoformat()}
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            journal["receipts"].append(receipt)
            _save(journal)
    return {"scope": foundation.SCOPE, "deduplicated": deduplicated, "receipt": receipt}


def inspect(*, job_slot="job-002"):
    context = _context(job_slot)
    require(not _path(".write.lock").exists(), "Evaluation writer busy; inspect when complete")
    journal = _journal(context)
    return {"scope": foundation.SCOPE, "integrity_verified": True,
            "max_receipts": MAX_RECEIPTS, "journal_sha256": journal["journal_sha256"],
            "receipts": journal["receipts"], "promotion_performed": False}


def require_receipt(receipt_id, *, candidate_version, candidate_model_sha256, test_identity_sha256):
    """Mandatory receipt preflight for a FUTURE sandbox activator; never activates.

    Checks current pointer/data/fixture identity and recomputes both scores.
    No production gate evidence or approval is implied by this demo-only receipt.
    """
    for value in (receipt_id, candidate_model_sha256, test_identity_sha256):
        digest(value, "demo promotion receipt binding")
    receipt = next((r for r in inspect(job_slot=candidate_version)["receipts"] if r["receipt_id"] == receipt_id), None)
    require(receipt is not None, "A recorded valid sandbox evaluation receipt is required")
    require(receipt["decision"] == "PASS" and receipt["eligible_for_future_sandbox_promotion"] is True
            and receipt["case"] == "challenger", "Sandbox evaluation receipt does not permit promotion")
    require(receipt["candidate"]["version"] == candidate_version
            and receipt["candidate"]["model_sha256"] == candidate_model_sha256
            and receipt["test_identity_sha256"] == test_identity_sha256,
            "Sandbox evaluation receipt target/test identity mismatch")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-champion", help="One-time train-only baseline initialization, no promotion")
    prepare.add_argument("--job-slot", default="job-002")
    evaluate = sub.add_parser("run", help="Evaluate fitted champion/challenger; no training or activation")
    evaluate.add_argument("--case", choices=CASES, default="challenger")
    evaluate.add_argument("--job-slot", default="job-002")
    inspection = sub.add_parser("inspect", help="Read-only receipt verification and replay")
    inspection.add_argument("--job-slot", default="job-002")
    args = parser.parse_args()
    try:
        if args.command == "prepare-champion":
            _candidate(args.job_slot)
            with _lock():
                if _path("journal.json").exists():
                    champion.load()  # Never fit/recreate a champion behind existing test results.
                else:
                    champion.prepare()
            report = {"scope": foundation.SCOPE, "champion": champion.load()[1], "promotion_performed": False}
        else:
            report = run(args.case, job_slot=args.job_slot) if args.command == "run" else inspect(job_slot=args.job_slot)
        print(json.dumps(report, indent=2))
        if args.command == "run" and report["receipt"]["decision"] == "BLOCK":
            raise SystemExit(1)
    except (GateError, OSError, ImportError, AttributeError, pickle.PickleError, EOFError) as error:
        parser.exit(1, f"Synthetic evaluation blocked: {error}\n")


if __name__ == "__main__":
    main()
