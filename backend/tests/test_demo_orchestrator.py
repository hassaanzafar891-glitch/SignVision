"""Synthetic workflow simulations and temporary sentinel files; no real models."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_sandbox import foundation, drift, orchestrator as runner, _synthetic_job as worker
from tracking.evaluation_gate import GateError
from tracking.import_baselines import sha256


class DemoOrchestratorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.addCleanup(patch.stopall)
        patch.object(foundation, "WORKSPACE", self.workspace).start()
        self.protected = {}
        for name in ("backend/data/models/alphabet_model.h5", "backend/data/db/main_dataset.db",
                     "backend/mlflow.db", "mlflow.db", "docker-compose.yml"):
            p = self.workspace / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(("protected:" + name).encode())
            self.protected[p] = p.read_bytes()
        foundation.initialize()
        drift.create_reference()
        self.root = self.workspace / ".signvision-demo"

    def alert(self, threshold=0.1):
        return drift.run("shifted", threshold=threshold)["event"]

    def snapshot(self):
        return {p.relative_to(self.root).as_posix(): p.read_bytes()
                for p in self.root.rglob("*") if p.is_file()}

    def assert_isolated(self):
        self.assertEqual(self.protected, {p: p.read_bytes() for p in self.protected})
        self.assertEqual(foundation.status()["active_version"], foundation.CHAMPION)

    def ok(self, request, timeout):
        return {"error_code": None, "exit_code": 0, "stdout": foundation._encode(worker.simulate(request)),
                "stderr": b"", "output_truncated": False}

    def test_default_unarmed_is_read_only_even_with_valid_alert(self):
        self.alert()
        before = self.snapshot()
        with patch.object(runner, "_launch") as launch:
            result = runner.consume()
        self.assertEqual(result["status"], "unarmed")
        self.assertEqual(result["pending_alerts"], 1)
        self.assertEqual(result["jobs_started"], 0)
        self.assertEqual(before, self.snapshot())
        launch.assert_not_called()
        self.assert_isolated()

    def test_no_alert_or_normal_observation_does_not_start_a_job(self):
        drift.run("normal")
        before = self.snapshot()
        with patch.object(runner, "_launch") as launch:
            result = runner.consume(arm=True)
        self.assertEqual(result["status"], "no_unprocessed_alert")
        self.assertEqual(before, self.snapshot())
        launch.assert_not_called()

    def test_actual_fixed_worker_runs_once_on_training_rows_only(self):
        event = self.alert()
        result = runner.consume(arm=True, cooldown_seconds=0)
        self.assertEqual(result["status"], "FINISHED")
        self.assertEqual(result["jobs_started"], 1)
        job = result["trigger"]
        self.assertEqual(job["alert_id"], event["event_id"])
        self.assertFalse(job["real_model_training"])
        self.assertFalse(job["promotion_performed"])
        output = job["job_output"]["payload"]
        self.assertEqual(output["used_partitions"], ["train"])
        self.assertEqual(output["training_rows"], 24)
        self.assertFalse(output["validation_used"])
        self.assertFalse(output["test_used"])
        self.assertFalse(output["quality_metrics_computed"])
        for name, digest in job["artifacts"].items():
            self.assertEqual(sha256(self.root / "retraining" / job["slot"] / name), digest)
        self.assertTrue(runner.inspect()["journal_integrity_verified"])
        self.assert_isolated()

    def test_duplicate_alert_does_not_launch_or_rewrite_history(self):
        self.alert()
        with patch.object(runner, "_launch", side_effect=self.ok) as launch:
            runner.consume(arm=True)
            before = self.snapshot()
            repeated = runner.consume(arm=True)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(repeated["status"], "no_unprocessed_alert")
        self.assertEqual(before, self.snapshot())

    def test_failed_alert_is_consumed_without_automatic_retry(self):
        self.alert()
        failed = {"error_code": "job_exit_failed", "exit_code": 2, "stdout": b"", "stderr": b"failure",
                  "output_truncated": False}
        with patch.object(runner, "_launch", return_value=failed) as launch:
            result = runner.consume(arm=True, cooldown_seconds=0)
            again = runner.consume(arm=True)
        self.assertEqual(result["status"], "FAILED")
        self.assertNotIn("model.json", result["trigger"]["artifacts"])
        self.assertEqual(again["status"], "no_unprocessed_alert")
        self.assertEqual(launch.call_count, 1)
        self.assertTrue(runner.inspect()["journal_integrity_verified"])

    def test_launch_failure_records_category_without_secret_exception_text(self):
        self.alert()
        with patch.object(runner, "_launch", side_effect=OSError("secret-do-not-store")):
            result = runner.consume(arm=True)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["trigger"]["job_output"]["error_code"], "job_launch_failed")
        self.assertNotIn("secret-do-not-store", json.dumps(runner.inspect()))
        self.assert_isolated()

    def test_invalid_child_output_cannot_publish_an_artifact(self):
        self.alert()
        invalid = {"error_code": None, "exit_code": 0, "stdout": b'{"promotion_approved":true}',
                   "stderr": b"", "output_truncated": False}
        with patch.object(runner, "_launch", return_value=invalid):
            result = runner.consume(arm=True)
        self.assertEqual(result["trigger"]["job_output"]["error_code"], "invalid_job_output")
        self.assertNotIn("model.json", result["trigger"]["artifacts"])

    def test_persisted_cooldown_cannot_be_bypassed_by_new_cli_value(self):
        clock = runner._now()
        self.alert()
        with patch.object(runner, "_now", return_value=clock), patch.object(runner, "_launch", side_effect=self.ok):
            runner.consume(arm=True, cooldown_seconds=30)
        self.alert(0.11)
        before = self.snapshot()
        with patch.object(runner, "_now", return_value=clock), patch.object(runner, "_launch") as launch:
            blocked = runner.consume(arm=True, cooldown_seconds=0)
        self.assertEqual(blocked["status"], "cooldown")
        launch.assert_not_called()
        self.assertEqual(before, self.snapshot())
        with patch.object(runner, "_now", return_value=clock + timedelta(seconds=31)), \
             patch.object(runner, "_launch", side_effect=self.ok):
            self.assertEqual(runner.consume(arm=True, cooldown_seconds=0)["status"], "FINISHED")

    def test_invalid_drift_event_blocks_before_any_trigger_write(self):
        self.alert()
        path = self.root / "drift/events.json"
        journal = json.loads(path.read_text())
        journal["events"][0]["divergence_score"] = 0.99
        event = journal["events"][0]
        event["event_sha256"] = runner.canonical_sha256({k: v for k, v in event.items() if k != "event_sha256"})
        path.write_text(json.dumps(drift._seal_journal(journal)))
        with patch.object(runner, "_launch") as launch, self.assertRaises(GateError):
            runner.consume(arm=True)
        launch.assert_not_called()
        self.assertFalse((self.root / "retraining").exists())
        self.assert_isolated()

    def test_two_consumers_cannot_execute_concurrent_jobs(self):
        self.alert()
        started, release = threading.Event(), threading.Event()
        def slow(request, timeout):
            started.set()
            self.assertTrue(release.wait(5))
            return self.ok(request, timeout)
        with patch.object(runner, "_launch", side_effect=slow) as launch, ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(runner.consume, arm=True, cooldown_seconds=0)
            self.assertTrue(started.wait(5))
            try:
                second = runner.consume(arm=True, cooldown_seconds=0)
                self.assertEqual(second["status"], "busy")
            finally:
                release.set()
            self.assertEqual(first.result(timeout=10)["status"], "FINISHED")
        self.assertEqual(launch.call_count, 1)

    def test_timeout_kills_the_actual_direct_child_and_records_failure(self):
        self.alert()
        popen = subprocess.Popen
        processes = []
        def sleeper(command, **kwargs):
            child = popen([command[0], "-S", "-B", "-c", "import time; time.sleep(5)"], **kwargs)
            processes.append(child)
            return child
        started = time.monotonic()
        with patch.object(runner.subprocess, "Popen", side_effect=sleeper):
            result = runner.consume(arm=True, timeout_seconds=1)
        self.assertEqual(result["status"], "TIMED_OUT")
        self.assertLess(time.monotonic() - started, 5)
        self.assertIsNotNone(processes[0].poll())
        self.assertEqual(result["trigger"]["job_output"]["error_code"], "job_timeout")
        self.assertFalse((self.root / "retraining/.write.lock").exists())

    def test_output_limit_stops_a_noisy_child_with_bounded_capture(self):
        self.alert()
        popen = subprocess.Popen
        def noisy(command, **kwargs):
            return popen([command[0], "-S", "-B", "-c",
                          "import sys; sys.stdout.write('x'*200000); sys.stdout.flush()"], **kwargs)
        with patch.object(runner.subprocess, "Popen", side_effect=noisy):
            result = runner.consume(arm=True)
        output = result["trigger"]["job_output"]
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(output["error_code"], "output_limit")
        self.assertTrue(output["output_truncated"])
        self.assertLessEqual(output["stdout_bytes"] + output["stderr_bytes"], runner.OUTPUT_LIMIT)

    def test_fixed_child_command_strips_production_credentials(self):
        self.alert()
        popen = subprocess.Popen
        def checked(command, **kwargs):
            self.assertEqual(command[1:], ["-S", "-B", "-m", "demo_sandbox._synthetic_job"])
            self.assertFalse(kwargs["shell"])
            self.assertEqual(kwargs["cwd"], self.root)
            for key in ("MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI", "AWS_SECRET_ACCESS_KEY", "SIGNVISION_TRACKING_ENABLED"):
                self.assertNotIn(key, kwargs["env"])
            return popen(command, **kwargs)
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "https://not-demo.invalid", "AWS_SECRET_ACCESS_KEY": "not-real"}), \
             patch.object(runner.subprocess, "Popen", side_effect=checked):
            self.assertEqual(runner.consume(arm=True)["status"], "FINISHED")
        self.assert_isolated()

    def test_capacity_preserves_dedup_history_without_eviction(self):
        with patch.object(runner, "_launch", side_effect=self.ok) as launch:
            for index in range(10):
                self.alert((index + 1) / 100)
                self.assertEqual(runner.consume(arm=True, cooldown_seconds=0)["status"], "FINISHED")
            self.alert(0.11)
            result = runner.consume(arm=True, cooldown_seconds=0)
        self.assertEqual(result["status"], "capacity_reached")
        self.assertEqual(launch.call_count, 10)
        history = runner.inspect()
        self.assertEqual(len(history["jobs"]), 10)
        self.assertLessEqual((self.root / "retraining/journal.json").stat().st_size, 131072)

    def test_incomplete_job_blocks_automatic_retry_after_crash(self):
        self.alert()
        with patch.object(runner, "_job_output", side_effect=RuntimeError("simulated parent crash")), self.assertRaises(RuntimeError):
            runner.consume(arm=True)
        self.assertEqual(runner.inspect()["jobs"][0]["status"], "RUNNING")
        with patch.object(runner, "_launch") as launch:
            self.assertEqual(runner.consume(arm=True)["status"], "blocked_incomplete_job")
        launch.assert_not_called()

    def test_worker_request_rejects_validation_or_test_training_inputs(self):
        request = worker.request_for("a" * 64, "b" * 64)
        request["train_indices"][0] = foundation.read_dataset()["partitions"]["test"][0]
        with self.assertRaisesRegex(GateError, "request differs"):
            worker.simulate(request)

    def test_artifact_tampering_is_not_hidden_by_recomputed_inventory_hash(self):
        self.alert()
        with patch.object(runner, "_launch", side_effect=self.ok):
            runner.consume(arm=True, cooldown_seconds=0)
        path = self.root / "retraining/job-001/model.json"
        model = json.loads(path.read_text())
        model["promotion_approved"] = True
        path.write_text(json.dumps(model))
        journal_path = self.root / "retraining/journal.json"
        journal = json.loads(journal_path.read_text())
        journal["jobs"][0]["artifacts"]["model.json"] = sha256(path)
        journal_path.write_text(json.dumps(drift._seal_journal(journal)))
        with self.assertRaises(GateError):
            runner.inspect()
        with patch.object(runner, "_launch") as launch, self.assertRaises(GateError):
            runner.consume(arm=True)
        launch.assert_not_called()

    def test_bounded_watch_automatically_consumes_new_alert_then_exits(self):
        emitted = []
        def emit(_):
            if not emitted:
                emitted.append(self.alert())
        with patch.object(runner.time, "sleep", side_effect=emit), patch.object(runner, "_launch", side_effect=self.ok) as launch:
            result = runner.consume(arm=True, watch_seconds=5, cooldown_seconds=0)
        self.assertEqual(result["status"], "FINISHED")
        self.assertEqual(result["jobs_started"], 1)
        self.assertEqual(launch.call_count, 1)

    def test_empty_watch_expires_and_unarmed_watch_never_waits(self):
        with patch.object(runner.time, "monotonic", side_effect=[0, 2]), patch.object(runner, "_launch") as launch:
            result = runner.consume(arm=True, watch_seconds=1)
        self.assertTrue(result["watch_expired"])
        launch.assert_not_called()
        with patch.object(runner.time, "sleep") as sleep:
            self.assertEqual(runner.consume(watch_seconds=60)["status"], "unarmed")
        sleep.assert_not_called()

    def test_inspect_and_unarmed_mode_do_not_change_job_files(self):
        self.alert()
        with patch.object(runner, "_launch", side_effect=self.ok):
            runner.consume(arm=True)
        before = self.snapshot()
        runner.inspect()
        runner.consume()
        self.assertEqual(before, self.snapshot())

    def test_invalid_cli_controls_and_missing_arm_do_not_launch(self):
        self.alert()
        for values in ({"arm": "true"}, {"timeout_seconds": 0}, {"timeout_seconds": 31},
                       {"cooldown_seconds": -1}, {"watch_seconds": 61}):
            with self.subTest(values=values), self.assertRaises(GateError):
                runner.consume(**values)
        output = io.StringIO()
        with patch.object(sys, "argv", ["orchestrator", "consume"]), redirect_stdout(output), patch.object(runner, "_launch") as launch:
            runner.main()
        self.assertEqual(json.loads(output.getvalue())["status"], "unarmed")
        launch.assert_not_called()
        with patch.object(sys, "argv", ["orchestrator", "consume", "--arm", "--job-command", "unsafe"]), \
             redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main()
        self.assertFalse((self.root / "retraining").exists())


if __name__ == "__main__":
    unittest.main()
