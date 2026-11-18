"""Explicitly armed sandbox jobs: legacy simulation or genuine synthetic-data fitting."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import sysconfig
import threading
import time

from demo_sandbox import foundation, drift, _synthetic_job as worker, _fitted_job as fitted
from tracking.evaluation_gate import GateError, canonical_sha256, digest, read_json, require
from tracking.import_baselines import sha256


MAX_JOBS = 10
OUTPUT_LIMIT = 16384
TERMINAL = ("FINISHED", "FAILED", "TIMED_OUT")
ERRORS = (None, "job_exit_failed", "job_timeout", "output_limit", "job_launch_failed",
          "invalid_job_output", "interrupted")


class ConsumerBusy(GateError):
    pass


def _now():
    return datetime.now(timezone.utc)


def _date(value):
    require(isinstance(value, str) and len(value) <= 64, "Invalid simulation timestamp")
    try:
        result = datetime.fromisoformat(value)
        require(result.utcoffset() is not None, "Simulation timestamp needs a timezone")
        return result
    except ValueError as error:
        raise GateError("Invalid simulation timestamp") from error


def _integer(value, low, high, name):
    require(type(value) is int and low <= value <= high, f"Invalid {name}")
    return value


def _path(name):
    relative = "retraining/" + name
    require(relative in foundation.EXTENSION_LIMITS, "Unsupported simulation path")
    path = foundation._path(relative)
    if path.exists():
        require(path.stat().st_size <= foundation.EXTENSION_LIMITS[relative], "Simulation file exceeds size limit")
    return path


def _new_file(name, value):
    payload = value if isinstance(value, bytes) else foundation._encode(value)
    require(len(payload) <= foundation.EXTENSION_LIMITS["retraining/" + name], "Simulation artifact too large")
    path = _path(name)
    path.parent.mkdir(exist_ok=True)
    with _path(name).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(_path(name))


@contextmanager
def _lock():
    path = _path(".write.lock")
    path.parent.mkdir(exist_ok=True)
    path = _path(".write.lock")
    try:
        stream = path.open("xb")
    except FileExistsError as error:
        raise ConsumerBusy("Synthetic consumer is already running") from error
    with stream:
        stream.write(f"synthetic-demo-only consumer pid={os.getpid()}\n".encode())
        info = os.fstat(stream.fileno())
    try:
        yield
    finally:
        drift._owned_remove(_path(".write.lock"), (info.st_dev, info.st_ino))


def _save(journal):
    journal = drift._seal_journal(journal)
    payload = foundation._encode(journal)
    require(len(payload) <= 131072, "Simulation journal exceeds size limit")
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


def _snapshot():
    report = drift.inspect()  # Full reference, replay, checksum and sequence validation.
    context = drift._context()
    require(report["reference_sha256"] == canonical_sha256(context["reference"]), "Drift reference changed")
    return report, context


def _base(alert, number, source_hash, timeout, cooldown, job_kind="simulation"):
    trigger_id = canonical_sha256({"scope": foundation.SCOPE, "sequence": number,
                                   "alert_id": alert["event_id"], "alert_sha256": alert["event_sha256"]})
    return {"schema_version": 1, "scope": foundation.SCOPE,
            "kind": "fitted-synthetic-trigger-v1" if job_kind == "fitted" else "synthetic-trigger-v1",
            "sequence": number, "slot": f"job-{number:03d}", "trigger_id": trigger_id,
            "alert_id": alert["event_id"], "alert_event": alert,
            "source_journal_sha256": source_hash, "timeout_seconds": timeout, "cooldown_seconds": cooldown,
            "workflow_simulation": job_kind != "fitted", "real_model_training": job_kind == "fitted",
            "promotion_performed": False}


def _history(context):
    require(not _path(".journal.tmp").exists(), "Unfinished simulation journal write; inspect manually")
    path = _path("journal.json")
    if path.exists():
        journal = read_json(path)
    else:
        journal = drift._seal_journal({"schema_version": 1, "scope": foundation.SCOPE,
                                      "kind": "synthetic-trigger-journal-v1", "max_jobs": MAX_JOBS, "jobs": []})
    require(isinstance(journal, dict) and set(journal) == {
        "schema_version", "scope", "kind", "max_jobs", "jobs", "journal_sha256"}, "Invalid simulation journal")
    digest(journal["journal_sha256"], "simulation journal checksum")
    require(drift._seal_journal(journal) == journal, "Simulation journal checksum mismatch")
    identity = {k: journal[k] for k in ("schema_version", "scope", "kind", "max_jobs")}
    require(canonical_sha256(identity) == canonical_sha256({"schema_version": 1, "scope": foundation.SCOPE,
        "kind": "synthetic-trigger-journal-v1", "max_jobs": MAX_JOBS}), "Invalid simulation journal identity")
    require(isinstance(journal["jobs"], list) and len(journal["jobs"]) <= MAX_JOBS, "Simulation capacity exceeded")
    seen, expected_files = set(), set()
    for number, record in enumerate(journal["jobs"], 1):
        require(isinstance(record, dict), "Invalid simulation record")
        alert = record.get("alert_event")
        drift.verify_event(context, alert)
        require(alert["alert"] is True and alert["status"] == "drift_detected", "Trigger is not a drift alert")
        timeout = _integer(record.get("timeout_seconds"), 1, 30, "job timeout")
        cooldown = _integer(record.get("cooldown_seconds"), 0, 3600, "job cooldown")
        source_hash = digest(record.get("source_journal_sha256"), "source journal identity")
        job_kind = "fitted" if record.get("kind") == "fitted-synthetic-trigger-v1" else "simulation"
        engine = fitted if job_kind == "fitted" else worker
        base = _base(alert, number, source_hash, timeout, cooldown, job_kind)
        require(set(record) == set(base) | {"status", "started_at", "finished_at", "cooldown_until",
                                            "job_output", "artifacts"}, "Invalid simulation record fields")
        require(all(canonical_sha256(record[k]) == canonical_sha256(v) for k, v in base.items()),
                "Simulation trigger identity mismatch")
        require(record["alert_id"] not in seen, "Duplicate consumed alert")
        seen.add(record["alert_id"])
        started = _date(record["started_at"])
        require(record["status"] in ("RUNNING", *TERMINAL), "Invalid simulation status")
        request = engine.request_for(record["trigger_id"], record["alert_id"])
        names = {"request.json"}
        if record["status"] == "RUNNING":
            require(record["finished_at"] is None and record["cooldown_until"] is None
                    and record["job_output"] is None, "Invalid incomplete simulation")
        else:
            finished = _date(record["finished_at"])
            require(finished >= started and _date(record["cooldown_until"]) == finished + timedelta(seconds=cooldown),
                    "Invalid simulation completion/cooldown times")
            output = record["job_output"]
            require(isinstance(output, dict) and set(output) == {
                "status", "exit_code", "error_code", "stdout_sha256", "stderr_sha256", "stdout_bytes",
                "stderr_bytes", "output_truncated", "payload"}, "Invalid simulation job output")
            require(output["status"] == record["status"] and output["error_code"] in ERRORS,
                    "Inconsistent simulation output status")
            require(output["exit_code"] is None or type(output["exit_code"]) is int, "Invalid child exit code")
            for stream in ("stdout", "stderr"):
                digest(output[stream + "_sha256"], "captured output checksum")
                _integer(output[stream + "_bytes"], 0, OUTPUT_LIMIT, "captured output length")
            require(type(output["output_truncated"]) is bool, "Invalid truncation flag")
            names.add("output.json")
            if record["status"] == "FINISHED":
                require(output["exit_code"] == 0 and output["error_code"] is None
                        and output["output_truncated"] is False,
                        "Invalid simulation result")
                if job_kind == "fitted":
                    fitted.validate(output["payload"], request)
                    names.update(("model.pkl", "training_receipt.json"))
                else:
                    require(canonical_sha256(output["payload"]) == canonical_sha256(worker.simulate(request)),
                            "Invalid simulation result")
                    names.add("model.json")
            else:
                require(output["payload"] is None and output["error_code"] is not None, "Invalid failed simulation")
                require(record["status"] != "TIMED_OUT" or output["error_code"] == "job_timeout", "Invalid timeout record")
        require(isinstance(record["artifacts"], dict) and set(record["artifacts"]) == names,
                "Incomplete simulation artifact inventory")
        for name in names:
            relative = record["slot"] + "/" + name  # slot and filenames were validated above.
            artifact = _path(relative)
            require(artifact.is_file() and sha256(artifact) == record["artifacts"][name], "Simulation artifact changed")
            if name == "model.pkl":
                require(artifact.read_bytes() == fitted.validate(record["job_output"]["payload"], request),
                        "Fitted model artifact differs")
            else:
                expected = request if name == "request.json" else record["job_output"] if name == "output.json" else record["job_output"]["payload"]["receipt" if name == "training_receipt.json" else "model"]
                require(canonical_sha256(read_json(artifact)) == canonical_sha256(expected), "Simulation artifact contents differ")
            expected_files.add(relative)
    actual_files = {p.relative_to(_path("journal.json").parent).as_posix()
                    for number in range(1, MAX_JOBS + 1)
                    for p in _path(f"job-{number:03d}/request.json").parent.glob("*")}
    require(actual_files == expected_files, "Orphan/incomplete simulation artifacts; inspect manually")
    return journal


def _launch(request, timeout):
    """Run one fixed child with bounded capture, execution and cleanup."""
    executable = getattr(sys, "_base_executable", sys.executable)  # Avoid Windows venv launcher children.
    environment = {key: value for key, value in os.environ.items()
                   if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "TMPDIR"}}
    environment.update(PYTHONPATH=str(Path(__file__).resolve().parents[1]), PYTHONUTF8="1")
    command = [executable, "-S", "-B", "-m", "demo_sandbox._synthetic_job"]
    if request.get("kind") == "fitted-synthetic-training-request-v1":
        # Use the direct base executable (killable on Windows), with only this
        # interpreter's dependency directory. No user site or .pth startup hooks.
        bootstrap = "import sys,runpy;sys.path.append(" + repr(sysconfig.get_paths()["purelib"]) + ");runpy.run_module('demo_sandbox._fitted_job',run_name='__main__')"
        command = [executable, "-S", "-B", "-c", bootstrap]
        environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    process = subprocess.Popen(command,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=foundation._root(), env=environment, shell=False)
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    overflow, mutex = threading.Event(), threading.Lock()
    def collect(name):
        try:
            while chunk := getattr(process, name).read(1024):
                with mutex:
                    room = OUTPUT_LIMIT - sum(map(len, buffers.values()))
                    buffers[name].extend(chunk[:room])
                    if len(chunk) > room:
                        overflow.set()
                        break
        except (OSError, ValueError):
            pass
    def send():
        try:
            process.stdin.write(foundation._encode(request))
            process.stdin.close()
        except (OSError, ValueError):
            pass
    threads = [threading.Thread(target=collect, args=(name,), daemon=True) for name in buffers]
    threads.append(threading.Thread(target=send, daemon=True))
    error = None
    try:
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if overflow.is_set():
                error = "output_limit"
                break
            if time.monotonic() >= deadline:
                error = "job_timeout"
                break
            time.sleep(0.01)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=1)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
    error = error or ("output_limit" if overflow.is_set() else None if process.returncode == 0 else "job_exit_failed")
    return {"error_code": error, "exit_code": process.returncode,
            "stdout": bytes(buffers["stdout"]), "stderr": bytes(buffers["stderr"]),
            "output_truncated": overflow.is_set()}


def _job_output(request, timeout):
    interrupted = False
    try:
        result = _launch(request, timeout)
    except KeyboardInterrupt:
        result = {"error_code": "interrupted", "exit_code": None, "stdout": b"", "stderr": b"", "output_truncated": False}
        interrupted = True
    except (OSError, subprocess.SubprocessError):
        result = {"error_code": "job_launch_failed", "exit_code": None, "stdout": b"", "stderr": b"", "output_truncated": False}
    payload = None
    if result["error_code"] is None:
        try:
            payload = worker.decode(result["stdout"])
            if request.get("kind") == "fitted-synthetic-training-request-v1":
                fitted.validate(payload, request)
            else:
                require(canonical_sha256(payload) == canonical_sha256(worker.simulate(request)), "Unexpected simulation output")
        except (ValueError, TypeError, KeyError, RecursionError, ImportError, AttributeError, EOFError, pickle.PickleError):
            result["error_code"], payload = "invalid_job_output", None
    return {"status": "TIMED_OUT" if result["error_code"] == "job_timeout" else "FAILED" if result["error_code"] else "FINISHED",
            "exit_code": result["exit_code"], "error_code": result["error_code"], "payload": payload,
            "stdout_sha256": hashlib.sha256(result["stdout"]).hexdigest(), "stderr_sha256": hashlib.sha256(result["stderr"]).hexdigest(),
            "stdout_bytes": len(result["stdout"]), "stderr_bytes": len(result["stderr"]),
            "output_truncated": result["output_truncated"]}, interrupted


def _summary(status, **values):
    return {"scope": foundation.SCOPE, "status": status, "workflow_simulation": True,
            "real_model_training": False, "promotion_performed": False, "jobs_started": 0, **values}


def _once(armed, timeout, cooldown, job_kind="simulation"):
    report, context = _snapshot()
    if _path(".write.lock").exists():
        return _summary("busy")
    history = _history(context)
    if any(job["status"] == "RUNNING" for job in history["jobs"]):
        return _summary("blocked_incomplete_job")
    seen = {job["alert_id"] for job in history["jobs"]}
    pending = [event for event in report["events"] if event["alert"] and event["event_id"] not in seen]
    if not armed:
        return _summary("unarmed", pending_alerts=len(pending), processed_alerts=len(seen))
    if not pending:
        return _summary("no_unprocessed_alert", processed_alerts=len(seen))
    if len(history["jobs"]) >= MAX_JOBS:
        return _summary("capacity_reached", max_jobs=MAX_JOBS)
    if history["jobs"]:
        remaining = (_date(history["jobs"][-1]["cooldown_until"]) - _now()).total_seconds()
        if remaining > 0:
            return _summary("cooldown", remaining_seconds=round(remaining, 3))
    try:
        with _lock():
            # Recheck after obtaining exclusive ownership; a prior consumer may
            # have finished while this one was preparing its snapshot.
            report, context = _snapshot()
            fresh = _history(context)
            require(canonical_sha256(fresh) == canonical_sha256(history), "Simulation history changed; rescan")
            alert = next((e for e in report["events"] if e["event_id"] == pending[0]["event_id"]), None)
            require(alert is not None and alert["alert"], "Selected alert is no longer available")
            record = {**_base(alert, len(history["jobs"]) + 1, report["journal_sha256"], timeout, cooldown, job_kind),
                      "status": "RUNNING", "started_at": _now().isoformat(), "finished_at": None,
                      "cooldown_until": None, "job_output": None, "artifacts": {}}
            engine = fitted if job_kind == "fitted" else worker
            request = engine.request_for(record["trigger_id"], record["alert_id"])
            require(request["dataset_sha256"] == alert["dataset_sha256"], "Synthetic job dataset differs")
            record["artifacts"]["request.json"] = _new_file(record["slot"] + "/request.json", request)
            history["jobs"].append(record)
            _save(history)  # Persist the consumed alert BEFORE launching the child.
            output, interrupted = _job_output(request, timeout)
            record["artifacts"]["output.json"] = _new_file(record["slot"] + "/output.json", output)
            if output["status"] == "FINISHED":
                if job_kind == "fitted":
                    artifact = fitted.validate(output["payload"], request)
                    record["artifacts"]["model.pkl"] = _new_file(record["slot"] + "/model.pkl", artifact)
                    record["artifacts"]["training_receipt.json"] = _new_file(record["slot"] + "/training_receipt.json", output["payload"]["receipt"])
                else:
                    record["artifacts"]["model.json"] = _new_file(record["slot"] + "/model.json", output["payload"]["model"])
            finished = _now()
            record.update(status=output["status"], finished_at=finished.isoformat(),
                          cooldown_until=(finished + timedelta(seconds=cooldown)).isoformat(), job_output=output)
            _save(history)
            if interrupted:
                raise KeyboardInterrupt
            return _summary(record["status"], jobs_started=1, trigger=record,
                            workflow_simulation=job_kind != "fitted", real_model_training=job_kind == "fitted",
                            message="FITTED ON SYNTHETIC SANDBOX DATA; no production training or activation" if job_kind == "fitted"
                            else "SYNTHETIC RETRAINING WORKFLOW SIMULATION; no real model training or activation")
    except ConsumerBusy:
        return _summary("busy")


def consume(*, arm=False, timeout_seconds=10, cooldown_seconds=30, watch_seconds=0, job_kind="simulation"):
    require(type(arm) is bool, "Arming must be explicit boolean")
    require(isinstance(job_kind, str) and job_kind in ("simulation", "fitted"), "Unknown sandbox job kind")
    _integer(timeout_seconds, 1, 30, "job timeout")
    _integer(cooldown_seconds, 0, 3600, "job cooldown")
    _integer(watch_seconds, 0, 60, "watch duration")
    deadline = time.monotonic() + watch_seconds
    while True:
        result = _once(arm, timeout_seconds, cooldown_seconds, job_kind)
        if not arm or result["status"] not in ("no_unprocessed_alert", "cooldown") or not watch_seconds:
            return result
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {**result, "watch_expired": True}
        time.sleep(min(1, remaining))


def inspect():
    report, context = _snapshot()
    require(not _path(".write.lock").exists(), "Simulation consumer busy; inspect after it completes")
    history = _history(context)
    return _summary("verified", journal_integrity_verified=True, max_jobs=MAX_JOBS,
                    jobs=history["jobs"], drift_journal_sha256=report["journal_sha256"],
                    real_model_training=any(job["real_model_training"] for job in history["jobs"]),
                    workflow_simulation=all(job["workflow_simulation"] for job in history["jobs"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("consume", help="Read-only unless --arm; at most one sandbox job")
    run.add_argument("--arm", action="store_true")
    run.add_argument("--timeout-seconds", type=int, default=10)
    run.add_argument("--cooldown-seconds", type=int, default=30)
    run.add_argument("--watch-seconds", type=int, default=0)
    run.add_argument("--job-kind", choices=("simulation", "fitted"), default="simulation",
                     help="fitted launches genuine scikit-learn training on synthetic train rows")
    sub.add_parser("inspect", help="Read-only trigger/artifact integrity inspection")
    args = parser.parse_args()
    try:
        values = vars(args)
        command = values.pop("command")
        result = consume(**values) if command == "consume" else inspect()
        print(json.dumps(result, indent=2))
        if result["status"] in ("FAILED", "TIMED_OUT", "blocked_incomplete_job", "capacity_reached"):
            raise SystemExit(1)
    except (GateError, OSError) as error:
        parser.exit(1, f"Synthetic consumer blocked: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Synthetic consumer interrupted; inspect its persisted status\n")


if __name__ == "__main__":
    main()
