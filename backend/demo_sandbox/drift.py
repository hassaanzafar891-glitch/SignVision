"""Synthetic prediction-distribution drift and local console/file alerts only.

No production telemetry, training, approval, activation, registry or network calls.
Normal traffic replays validation inputs; shifted traffic cycles training inputs
whose actual fixture prediction is demo_class_0. Neither uses test predictions.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

from demo_sandbox import foundation
from monitoring import js_divergence
from tracking.evaluation_gate import GateError, canonical_sha256, digest, read_json, require


CONFIG_PATH = Path(__file__).with_name("drift_config.json")
DEFAULTS = {"schema_version": 1, "scope": foundation.SCOPE, "threshold": 0.1,
            "min_samples": 12, "batch_size": 24, "max_batch_size": 240, "max_events": 20}
MAX_SEQUENCE = 2 ** 63 - 1


def _threshold(value):
    require(type(value) in (int, float) and 0 < value <= 1 and math.isfinite(value),
            "Threshold must be finite and in (0,1]")
    return float(value)


def _settings():
    config = read_json(CONFIG_PATH)
    require(isinstance(config, dict), "Invalid demo drift configuration")
    threshold = _threshold(config.get("threshold"))
    require(canonical_sha256({**config, "threshold": 0.1}) == canonical_sha256(DEFAULTS),
            "Unsupported demo drift settings; only threshold is configurable")
    return {**config, "threshold": threshold}


def _path(name):
    require(name in ("reference.json", "events.json", ".reference.tmp", ".events.tmp", ".write.lock"),
            "Unsupported demo drift file")
    path = foundation._path("drift/" + name)
    if path.exists():
        require(path.stat().st_size <= foundation.EXTENSION_LIMITS["drift/" + name],
                "Demo drift file exceeds size limit")
    return path


def _counts(labels, predictions):
    result = dict.fromkeys(labels, 0)
    for prediction in predictions:
        require(prediction in result, "Unexpected synthetic prediction")
        result[prediction] += 1
    return result


def _context():
    settings = _settings()
    state = foundation.status()
    dataset = foundation.read_dataset()
    model = foundation.read_active()
    # No predictions are made for test rows, including during audit replay.
    predictions = {i: foundation._predict(model, dataset["rows"][i]["features"])
                   for name in ("train", "validation") for i in dataset["partitions"][name]}
    validation = dataset["partitions"]["validation"]
    reference = {
        "schema_version": 1, "scope": foundation.SCOPE,
        "kind": "synthetic-validation-predictions-v1",
        "model_version": state["active_version"], "model_manifest_sha256": state["active_manifest_sha256"],
        "dataset_sha256": canonical_sha256(dataset), "partition": "validation",
        "partition_indices_sha256": canonical_sha256(validation),
        "sample_count": len(validation),
        "counts": _counts(dataset["class_labels"], (predictions[i] for i in validation)),
        "production_evidence": False, "training_performed": False,
    }
    require(reference["sample_count"] >= settings["min_samples"], "Insufficient demo reference samples")
    shifted_pool = [i for i in dataset["partitions"]["train"]
                    if predictions[i] == dataset["class_labels"][0]]
    require(bool(shifted_pool), "No non-test shifted inputs are available")
    return {"settings": settings, "reference": reference, "predictions": predictions,
            "labels": dataset["class_labels"], "pools": {"normal": validation, "shifted": shifted_pool}}


def _load_reference(context):
    path = _path("reference.json")
    require(path.is_file(), "Synthetic reference unavailable; run the reference command first")
    reference = read_json(path)
    require(canonical_sha256(reference) == canonical_sha256(context["reference"]),
            "Synthetic reference integrity or model/dataset identity mismatch")
    return reference


def _analyze(context, scenario, batch_size, threshold):
    require(isinstance(scenario, str) and scenario in context["pools"], "Unknown demo traffic scenario")
    require(type(batch_size) is int and 1 <= batch_size <= context["settings"]["max_batch_size"],
            "Demo batch size must be an integer between 1 and 240")
    threshold = _threshold(threshold)
    pool = context["pools"][scenario]
    indices = [pool[i % len(pool)] for i in range(batch_size)]
    observed = _counts(context["labels"], (context["predictions"][i] for i in indices))
    score = None
    status = "insufficient_current_samples"
    if batch_size >= context["settings"]["min_samples"]:
        score = js_divergence(context["reference"]["counts"], observed)
        status = "drift_detected" if score > threshold else "no_alert"
    return {
        "schema_version": 1, "scope": foundation.SCOPE, "kind": "synthetic-drift-observation-v1",
        "reference_sha256": canonical_sha256(context["reference"]),
        "model_manifest_sha256": context["reference"]["model_manifest_sha256"],
        "dataset_sha256": context["reference"]["dataset_sha256"],
        "scenario": scenario, "batch_size": batch_size,
        "min_samples": context["settings"]["min_samples"],
        "observed": {"sample_count": batch_size, "counts": observed,
                     "row_indices_sha256": canonical_sha256(indices)},
        "divergence_score": score, "threshold": threshold, "comparison": "strictly_greater_than",
        "status": status, "alert": status == "drift_detected",
        "production_evidence": False, "training_triggered": False,
    }


def _seal_journal(journal):
    body = {k: v for k, v in journal.items() if k != "journal_sha256"}
    return {**body, "journal_sha256": canonical_sha256(body)}


def verify_event(context, event):
    """Replay-check an event, including one retained by a later sandbox consumer."""
    require(isinstance(event, dict), "Invalid demo event")
    core = _analyze(context, event.get("scenario"), event.get("batch_size"), event.get("threshold"))
    require(set(event) == set(core) | {"event_id", "timestamp", "sequence", "event_sha256"},
            "Invalid demo event fields")
    require(all(canonical_sha256(event[k]) == canonical_sha256(v) for k, v in core.items()),
            "Demo event does not match replayed predictions or reference")
    require(event["event_id"] == canonical_sha256(core), "Invalid demo event identity")
    require(type(event["sequence"]) is int and 1 <= event["sequence"] <= MAX_SEQUENCE,
            "Invalid demo event sequence")
    require(isinstance(event["timestamp"], str) and len(event["timestamp"]) <= 64,
            "Invalid demo timestamp")
    try:
        require(datetime.fromisoformat(event["timestamp"]).utcoffset() is not None,
                "Demo timestamp needs a timezone")
    except ValueError as error:
        raise GateError("Invalid demo timestamp") from error
    digest(event["event_sha256"], "demo event checksum")
    require(event["event_sha256"] == canonical_sha256({
        k: v for k, v in event.items() if k != "event_sha256"}), "Demo event checksum mismatch")
    return core


def _load_journal(context):
    path = _path("events.json")
    if not path.exists():
        return _seal_journal({"schema_version": 1, "scope": foundation.SCOPE, "max_events": 20,
                              "discarded_events": 0, "events": []})
    journal = read_json(path)
    require(isinstance(journal, dict) and set(journal) == {
        "schema_version", "scope", "max_events", "discarded_events", "events", "journal_sha256"},
        "Invalid demo event journal schema")
    digest(journal["journal_sha256"], "journal identity")
    require(_seal_journal(journal) == journal, "Demo journal checksum mismatch")
    identity = {k: journal[k] for k in ("schema_version", "scope", "max_events")}
    require(canonical_sha256(identity) == canonical_sha256({
        "schema_version": 1, "scope": foundation.SCOPE, "max_events": 20}), "Invalid demo journal identity")
    discarded, events = journal["discarded_events"], journal["events"]
    require(type(discarded) is int and 0 <= discarded <= MAX_SEQUENCE, "Invalid demo eviction counter")
    require(isinstance(events, list) and len(events) <= context["settings"]["max_events"]
            and (not discarded or len(events) == context["settings"]["max_events"]),
            "Invalid demo event retention window")
    ids = set()
    for position, event in enumerate(events, start=discarded + 1):
        verify_event(context, event)
        require(event["sequence"] == position, "Invalid demo event sequence")
        require(event["event_id"] not in ids, "Invalid or duplicate demo event identity")
        ids.add(event["event_id"])
    return journal


def _owned_remove(path, identity):
    foundation._plain(path, directory=False)
    current = path.lstat()
    require((current.st_dev, current.st_ino) == identity, "Sandbox temporary file was replaced; inspect manually")
    path.unlink()


@contextmanager
def _write_lock():
    lock = _path(".write.lock")
    lock.parent.mkdir(exist_ok=True)
    lock = _path(".write.lock")
    with lock.open("xb") as stream:
        stream.write(f"synthetic-demo-only writer pid={os.getpid()}\n".encode())
        info = os.fstat(stream.fileno())
        identity = (info.st_dev, info.st_ino)
    try:
        yield
    finally:
        _owned_remove(_path(".write.lock"), identity)


def _atomic_write(name, value):
    require(name in ("reference.json", "events.json"), "Unsupported demo write target")
    payload = foundation._encode(value)
    require(len(payload) <= foundation.EXTENSION_LIMITS["drift/" + name], "Demo record exceeds size limit")
    temporary = _path(".reference.tmp" if name == "reference.json" else ".events.tmp")
    stream = temporary.open("xb")
    identity_info = os.fstat(stream.fileno())
    identity = (identity_info.st_dev, identity_info.st_ino)
    try:
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        destination = _path(name)
        os.replace(temporary, destination)  # Only the two fixed drift record paths.
    except BaseException:
        _owned_remove(temporary, identity)
        raise


def create_reference():
    context = _context()
    with _write_lock():
        if _path("reference.json").exists():
            reference = _load_reference(context)
        else:
            require(not _path("events.json").exists(), "Cannot replace a reference behind existing events")
            reference = context["reference"]
            _atomic_write("reference.json", reference)
    return {"scope": foundation.SCOPE, "reference_sha256": canonical_sha256(reference),
            "reference": reference, "training_triggered": False}


def run(scenario="shifted", *, batch_size=None, threshold=None):
    context = _context()
    _load_reference(context)  # Missing/mismatched references fail before taking a write lock.
    batch_size = context["settings"]["batch_size"] if batch_size is None else batch_size
    threshold = context["settings"]["threshold"] if threshold is None else threshold
    core = _analyze(context, scenario, batch_size, threshold)
    with _write_lock():
        _load_reference(context)
        journal = _load_journal(context)
        event = next((e for e in journal["events"] if e["event_id"] == canonical_sha256(core)), None)
        deduplicated = event is not None
        if event is None:
            sequence = journal["discarded_events"] + len(journal["events"]) + 1
            require(sequence <= MAX_SEQUENCE, "Demo sequence counter is saturated")
            event = {**core, "event_id": canonical_sha256(core), "sequence": sequence,
                     "timestamp": datetime.now(timezone.utc).isoformat()}
            event["event_sha256"] = canonical_sha256(event)
            journal["events"].append(event)
            if len(journal["events"]) > context["settings"]["max_events"]:
                journal["events"].pop(0)
                journal["discarded_events"] += 1
            journal = _seal_journal(journal)
            _atomic_write("events.json", journal)
    message = "SYNTHETIC DEMO DRIFT ALERT" if event["alert"] else "SYNTHETIC DEMO: " + event["status"]
    return {"message": message, "event": event, "deduplicated": deduplicated,
            "retained_events": len(journal["events"]), "discarded_events": journal["discarded_events"],
            "event_file": "drift/events.json", "training_triggered": False}


def inspect():
    context = _context()
    require(not _path(".write.lock").exists(), "Demo drift writer is busy; inspect again after it completes")
    require(not any(_path(name).exists() for name in (".reference.tmp", ".events.tmp")),
            "Unfinished demo write found; inspect temporary files manually")
    reference = _load_reference(context)
    journal = _load_journal(context)
    return {"scope": foundation.SCOPE, "reference_verified": True,
            "reference_sha256": canonical_sha256(reference), "reference_counts": reference["counts"],
            "journal_integrity_verified": True, "journal_sha256": journal["journal_sha256"],
            "retained_events": len(journal["events"]),
            "discarded_events": journal["discarded_events"], "events": journal["events"],
            "production_evidence": False, "training_triggered": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("reference", help="Create/recheck an explicit validation-only synthetic reference")
    replay = sub.add_parser("run", help="Replay synthetic predictions and record a local observation/alert")
    replay.add_argument("--scenario", choices=("normal", "shifted"), default="shifted")
    replay.add_argument("--batch-size", type=int)
    replay.add_argument("--threshold", type=float)
    sub.add_parser("inspect", help="Read-only reference, replay and journal integrity checks")
    args = parser.parse_args()
    try:
        if args.command == "reference":
            result = create_reference()
        elif args.command == "run":
            result = run(args.scenario, batch_size=args.batch_size, threshold=args.threshold)
        else:
            result = inspect()
        print(json.dumps(result, indent=2))
    except (GateError, OSError) as error:
        parser.exit(1, f"Synthetic demo drift blocked: {error}\n")


if __name__ == "__main__":
    main()
