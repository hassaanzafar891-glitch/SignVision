"""Manual sandbox-only activation transactions. Never routes production inference."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os

from demo_sandbox import foundation, drift, evaluation, evaluation_champion as champion
from tracking.evaluation_gate import GateError, canonical_sha256, digest, read_json, require
from tracking.import_baselines import sha256


CANDIDATE = "job-002"
MAX_EVENTS = 20
PENDING = ("PENDING", "RECOVERY_FAILED")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _path(name):
    require(name in ("active.json", "journal.json", ".active.tmp", ".journal.tmp", ".write.lock"),
            "Unsupported activation path")
    path = foundation._path("activation/" + name)
    if path.exists():
        require(path.stat().st_size <= foundation.EXTENSION_LIMITS["activation/" + name], "Activation file too large")
    return path


@contextmanager
def _lock():
    foundation.status()
    path = _path(".write.lock")
    path.parent.mkdir(exist_ok=True)
    with _path(".write.lock").open("xb") as stream:
        stream.write(f"sandbox-only activation pid={os.getpid()}\n".encode())
        info = os.fstat(stream.fileno())
    try:
        yield
    finally:
        drift._owned_remove(_path(".write.lock"), (info.st_dev, info.st_ino))


def _atomic(name, value):
    require(name in ("active.json", "journal.json"), "Invalid atomic target")
    data = foundation._encode(value)
    require(len(data) <= foundation.EXTENSION_LIMITS["activation/" + name], "Activation record too large")
    temporary = _path(".active.tmp" if name == "active.json" else ".journal.tmp")
    with temporary.open("xb") as stream:
        info = os.fstat(stream.fileno())
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            drift._owned_remove(temporary, (info.st_dev, info.st_ino))
            raise
    try:
        os.replace(temporary, _path(name))
    except BaseException:
        drift._owned_remove(temporary, (info.st_dev, info.st_ino))
        raise


def _save(journal):
    _atomic("journal.json", drift._seal_journal(journal))


def _descriptor(version):
    if version == champion.VERSION:
        model, receipt, _ = champion.load()
        return model, {"version": version, "model_sha256": receipt["model_sha256"],
                       "training_receipt_sha256": canonical_sha256(receipt)}
    require(version == CANDIDATE, "Only the verified sandbox challenger is supported")
    model, identity = evaluation._candidate(CANDIDATE)
    return model, {key: identity[key] for key in ("version", "model_sha256", "training_receipt_sha256")}


def _pointer(active, previous=None):
    dataset = foundation.read_dataset()
    return {"schema_version": 1, "scope": foundation.SCOPE, "active": active, "previous": previous,
            "dataset_identity": champion.dataset_identity(dataset),
            "evaluation_champion_pointer_sha256": sha256(champion.path("champion_pointer.json")),
            "foundation_pointer_sha256": sha256(foundation._path("active.json")),
            "production_evidence": False, "production_activation": False}


def _validate_pointer(pointer, binding=None):
    require(isinstance(pointer, dict), "Invalid activation pointer")
    for descriptor in (pointer.get("active"), pointer.get("previous")):
        if descriptor is None:
            continue
        require(isinstance(descriptor, dict) and set(descriptor) == {
            "version", "model_sha256", "training_receipt_sha256"}, "Invalid model descriptor")
        require(descriptor["version"] in (champion.VERSION, CANDIDATE), "External model version forbidden")
        for key in ("model_sha256", "training_receipt_sha256"):
            digest(descriptor[key], "activation artifact identity")
    require(pointer.get("active") is not None, "Active descriptor required")
    expected = _pointer(pointer["active"], pointer["previous"]) if binding is None else {
        **binding, "active": pointer["active"], "previous": pointer["previous"]}
    require(canonical_sha256(pointer) == canonical_sha256(expected),
            "Pointer dataset/split/configuration identity mismatch")
    require((pointer["active"]["version"] == champion.VERSION and pointer["previous"] is None)
            or (pointer["active"]["version"] == CANDIDATE and pointer["previous"] is not None
                and pointer["previous"]["version"] == champion.VERSION), "Invalid previous champion binding")


def _health(descriptor):
    model, expected = _descriptor(descriptor["version"])
    require(expected == descriptor, "Active model/receipt hash mismatch")
    dataset = foundation.read_dataset()
    index = dataset["partitions"]["train"][0]
    row = dataset["rows"][index]
    predictions = model.predict([row["features"]])
    probabilities = model.predict_proba([row["features"]])
    require(len(predictions) == 1 and predictions[0] == row["label"], "Health inference label mismatch")
    require(probabilities.shape == (1, len(dataset["class_labels"]))
            and all(0 <= float(v) <= 1 for v in probabilities[0])
            and abs(float(sum(probabilities[0])) - 1) < 1e-6, "Health probabilities invalid")
    return {"healthy": True, "model_sha256": expected["model_sha256"], "version": descriptor["version"],
            "sample_partition": "train", "sample_index": index, "prediction": str(predictions[0]),
            "timestamp": _now(), "test_used": False}


def _pointer_health(expected):
    actual = read_json(_path("active.json"))
    require(actual == expected, "Post-activation pointer differs from intended identity")
    _validate_pointer(actual)
    return _health(actual["active"])


def _load():
    foundation.status()
    # One verified immutable data/pointer binding per journal read, not per historical event.
    binding = _pointer(None)
    require(not any(_path(name).exists() for name in (".active.tmp", ".journal.tmp")),
            "Interrupted temporary write; inspect manually before recovery")
    empty = {"schema_version": 1, "scope": foundation.SCOPE, "kind": "sandbox-activation-journal-v1",
             "max_events": MAX_EVENTS, "events": []}
    journal = read_json(_path("journal.json")) if _path("journal.json").exists() else drift._seal_journal(empty)
    require(isinstance(journal, dict) and set(journal) == set(empty) | {"journal_sha256"}, "Invalid activation journal")
    require(drift._seal_journal(journal) == journal, "Activation journal checksum mismatch")
    require(canonical_sha256({k: journal[k] for k in empty if k != "events"}) ==
            canonical_sha256({k: v for k, v in empty.items() if k != "events"}), "Activation journal policy mismatch")
    require(isinstance(journal["events"], list) and len(journal["events"]) <= MAX_EVENTS, "Activation audit capacity exceeded")
    current, pending = None, None
    fields = {"sequence", "operation_id", "action", "receipt_id", "before", "target", "result", "status",
              "started_at", "finished_at", "health_checks", "reason", "event_sha256"}
    for number, event in enumerate(journal["events"], 1):
        require(isinstance(event, dict) and set(event) == fields, "Invalid activation event")
        require(type(event["sequence"]) is int and event["sequence"] == number, "Invalid activation sequence")
        require(event["action"] in ("init", "promote", "rollback"), "Invalid activation action")
        require(event["before"] == current, "Activation history continuity mismatch")
        require(event["event_sha256"] == canonical_sha256({k: v for k, v in event.items() if k != "event_sha256"}),
                "Activation event checksum mismatch")
        digest(event["operation_id"], "activation operation identity")
        if event["receipt_id"] is not None:
            digest(event["receipt_id"], "activation evaluation receipt identity")
        for name in ("before", "target", "result"):
            if event[name] is not None:
                _validate_pointer(event[name], binding)
        require(isinstance(event["health_checks"], list) and len(event["health_checks"]) <= 5, "Invalid health audit")
        require(isinstance(event["reason"], str) and len(event["reason"]) <= 120, "Invalid reason code")
        start = datetime.fromisoformat(event["started_at"])
        require(start.utcoffset() is not None, "Activation timestamp lacks timezone")
        if event["status"] in PENDING:
            require(number == len(journal["events"]) and event["result"] is None, "Invalid incomplete transaction")
            pending = event
        else:
            require(event["status"] in ("SUCCEEDED", "REJECTED", "RECOVERED"), "Invalid activation status")
            finish = datetime.fromisoformat(event["finished_at"])
            require(finish.utcoffset() is not None and finish >= start, "Invalid completion timestamp")
            expected = event["target"] if event["status"] == "SUCCEEDED" else event["before"]
            if event["status"] == "RECOVERED" and event["action"] in ("init", "rollback"):
                expected = event["target"]
            require(event["result"] == expected, "Invalid activation result binding")
            current = event["result"]
    actual = read_json(_path("active.json")) if _path("active.json").exists() else None
    if pending:
        require(actual in (current, pending["target"], pending["before"]), "Unexpected pointer during recovery")
    else:
        require(actual == current, "Active pointer differs from committed history")
    return journal, current, pending


def _seal_event(event):
    event["event_sha256"] = canonical_sha256({k: v for k, v in event.items() if k != "event_sha256"})


def _finish(journal, event, status, result, reason):
    event.update(status=status, result=result, reason=reason, finished_at=_now())
    _seal_event(event)
    _save(journal)


def _restore(journal, event, reason):
    fallback = event["before"] if event["action"] == "promote" else event["target"]
    try:
        require(fallback is not None, "No verified fallback pointer")
        event["health_checks"] = event["health_checks"][:3]  # Bound repeated recovery diagnostics.
        event["health_checks"].append(_health(fallback["active"]))
        _atomic("active.json", fallback)
        event["health_checks"].append(_pointer_health(fallback))
        _finish(journal, event, "RECOVERED", fallback, reason)
        return {"status": "RECOVERED", "active": fallback["active"], "operation_id": event["operation_id"]}
    except BaseException as error:
        event.update(status="RECOVERY_FAILED", result=None, reason="recovery_failed_" + type(error).__name__, finished_at=None)
        _seal_event(event)
        _save(journal)
        raise GateError("Recovery failed; active reads blocked; operator inspection required") from error


def _authorize(receipt_id, before):
    digest(receipt_id, "sandbox PASS receipt")
    context = evaluation._context(CANDIDATE)
    normal = evaluation._result(context, "challenger")
    receipt = evaluation.require_receipt(receipt_id, candidate_version=CANDIDATE,
        candidate_model_sha256=context["candidate"][1]["model_sha256"],
        test_identity_sha256=normal["test_identity_sha256"])
    require(receipt["policy"] == "no_accuracy_regression_v1" and receipt["thresholds"] == {"minimum_accuracy_delta": 0.0}
            and receipt["additional_policy"] is None and receipt["protocol"] == evaluation.PROTOCOL,
            "Activation policy mismatch")
    require(before["active"]["version"] == champion.VERSION
            and before["active"]["model_sha256"] == receipt["reference"]["model_sha256"]
            and before["evaluation_champion_pointer_sha256"] == receipt["active_pointer_sha256"],
            "Evaluation champion differs from active baseline")
    return _pointer(_descriptor(CANDIDATE)[1], before["active"])


def execute(action, *, receipt_id=None):
    require(action in ("init", "promote", "rollback", "recover"), "Unsupported sandbox action")
    if receipt_id is not None:
        digest(receipt_id, "sandbox evaluation receipt")
    with _lock():
        journal, before, pending = _load()
        if pending:
            result = _restore(journal, pending, "interrupted_operation_restored_previous_champion")
            return {"scope": foundation.SCOPE, **result, "requested_action_executed": False}
        if action == "recover":
            return {"scope": foundation.SCOPE, "status": "NO_RECOVERY_REQUIRED"}
        require(len(journal["events"]) < MAX_EVENTS, "Activation audit full; no further mutations allowed")
        event = {"sequence": len(journal["events"]) + 1, "action": action, "receipt_id": receipt_id,
                 "before": before, "target": None, "result": None, "status": "PENDING", "started_at": _now(),
                 "finished_at": None, "health_checks": [], "reason": "pending"}
        event["operation_id"] = canonical_sha256(event)
        try:
            if action == "init":
                require(before is None, "Activation already initialized")
                target = _pointer(_descriptor(champion.VERSION)[1])
            elif action == "promote":
                require(before is not None, "Initialize sandbox activation first")
                target = _authorize(receipt_id, before)
                event["health_checks"].append(_health(before["active"]))
            else:
                require(before is not None and before["previous"] is not None, "No previous champion available")
                # Allows rollback even if the currently active challenger is corrupt.
                target = _pointer(before["previous"])
            event["target"] = target
            event["health_checks"].append(_health(target["active"]))
        except Exception as error:
            event["target"] = None
            journal["events"].append(event)
            reason = str(error)[:120] if isinstance(error, GateError) else "preflight_rejected_" + type(error).__name__
            _finish(journal, event, "REJECTED", before, reason)
            return {"scope": foundation.SCOPE, "status": "REJECTED", "reason": event["reason"],
                    "operation_id": event["operation_id"]}
        journal["events"].append(event)
        _seal_event(event)
        _save(journal)  # Write-ahead record BEFORE pointer replacement.
        try:
            _atomic("active.json", target)
            event["health_checks"].append(_pointer_health(target))
            _finish(journal, event, "SUCCEEDED", target, "sandbox_" + action + "_healthy")
            return {"scope": foundation.SCOPE, "status": "SUCCEEDED", "action": action,
                    "active": target["active"], "health": event["health_checks"][-1], "operation_id": event["operation_id"]}
        except BaseException as error:
            event["health_checks"].append({"healthy": False, "phase": "activation_or_commit",
                "error_type": type(error).__name__, "timestamp": _now(), "test_used": False})
            result = _restore(journal, event, "activation_failed_" + type(error).__name__)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return {"scope": foundation.SCOPE, **result, "requested_action_executed": False}


def audit():
    require(not _path(".write.lock").exists(), "Activation writer busy")
    journal, current, pending = _load()
    return {"scope": foundation.SCOPE, "integrity_verified": True, "recovery_required": pending is not None,
            "committed_pointer": current, "journal": journal, "production_activation": False}


def status():
    report = audit()
    require(not report["recovery_required"], "Pending activation: recover before active model reads")
    pointer = report["committed_pointer"]
    return {"scope": foundation.SCOPE, "initialized": pointer is not None, "pointer": pointer,
            "health": _health(pointer["active"]) if pointer else None, "production_activation": False}


def read_active():
    report = status()  # Committed state + verified model + health, no implicit recovery.
    require(report["initialized"], "Sandbox activation not initialized")
    return _descriptor(report["pointer"]["active"]["version"])[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "rollback", "recover", "status", "audit"):
        sub.add_parser(command)
    promotion = sub.add_parser("promote", help="Explicit sandbox-only promotion; requires exact recorded PASS receipt")
    promotion.add_argument("--receipt-id", required=True)
    args = parser.parse_args()
    try:
        report = status() if args.command == "status" else audit() if args.command == "audit" else execute(
            args.command, receipt_id=getattr(args, "receipt_id", None))
        print(json.dumps(report, indent=2))
        if report.get("status") in ("REJECTED", "RECOVERED"):
            raise SystemExit(1)
    except (GateError, OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"Sandbox activation blocked ({type(error).__name__}); inspect sandbox audit\n")
    except KeyboardInterrupt:
        parser.exit(130, "Sandbox activation interrupted; inspect audit/recovery state\n")


if __name__ == "__main__":
    main()
