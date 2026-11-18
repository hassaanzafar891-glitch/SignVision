"""Promotion fault injection with actual fitted synthetic models in temporary roots."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_sandbox import foundation, drift, orchestrator, _fitted_job as fitted
from demo_sandbox import evaluation, evaluation_champion as champion, activation
from tracking.evaluation_gate import GateError
from tracking.import_baselines import sha256


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Optional sklearn activation dependency unavailable")
class DemoActivationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        root_patch = patch.object(foundation, "WORKSPACE", self.workspace)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.protected = {}
        for name in ("backend/data/models/alphabet_model.h5", "backend/data/db/main_dataset.db", "mlflow.db", "docker-compose.yml"):
            p = self.workspace / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"protected production sentinel")
            self.protected[p] = sha256(p)
        foundation.initialize()
        drift.create_reference()
        drift.run("shifted")
        orchestrator.consume(arm=True, cooldown_seconds=0)  # Preserve the legacy simulation as slot 1.
        drift.run("shifted", batch_size=25)
        def launch(request, timeout):
            return {"error_code": None, "exit_code": 0, "stdout": foundation._encode(fitted.train(request)),
                    "stderr": b"", "output_truncated": False}
        with patch.object(orchestrator, "_launch", side_effect=launch):
            orchestrator.consume(arm=True, job_kind="fitted", cooldown_seconds=0)
        with evaluation._lock():
            champion.prepare()
        self.passed = evaluation.run()["receipt"]
        self.blocked = evaluation.run("stricter-control")["receipt"]
        self.protected.update({p: sha256(p) for p in (foundation._path("active.json"),
            champion.path("champion_pointer.json"), champion.path("champion/model.pkl"),
            foundation._path("retraining/job-002/model.pkl"), evaluation._path("journal.json"))})
        self.assertEqual(activation.execute("init")["status"], "SUCCEEDED")

    def promote(self):
        return activation.execute("promote", receipt_id=self.passed["receipt_id"])

    def test_successful_promotion_loads_real_model_and_explicit_rollback_is_healthy(self):
        result = self.promote()
        self.assertEqual(result["status"], "SUCCEEDED")
        self.assertTrue(result["health"]["healthy"])
        self.assertEqual(result["health"]["sample_partition"], "train")
        self.assertEqual(result["active"]["model_sha256"], self.passed["candidate"]["model_sha256"])
        self.assertEqual(activation.read_active().max_depth, 3)
        state = activation.status()
        self.assertEqual(state["pointer"]["previous"]["version"], champion.VERSION)
        result = activation.execute("rollback")
        self.assertEqual(result["status"], "SUCCEEDED")
        self.assertTrue(result["health"]["healthy"])
        self.assertEqual(activation.read_active().max_depth, 1)
        self.assertEqual(self.protected, {p: sha256(p) for p in self.protected})

    def test_stricter_block_receipt_is_audited_and_cannot_authorize_promotion(self):
        before = sha256(activation._path("active.json"))
        result = activation.execute("promote", receipt_id=self.blocked["receipt_id"])
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("does not permit", result["reason"])
        self.assertEqual(before, sha256(activation._path("active.json")))
        self.assertEqual(activation.audit()["journal"]["events"][-1]["status"], "REJECTED")

    def test_missing_or_unknown_receipt_cannot_promote(self):
        for receipt_id in (None, "a" * 64):
            with self.subTest(receipt_id=receipt_id):
                self.assertEqual(activation.execute("promote", receipt_id=receipt_id)["status"], "REJECTED")
        self.assertEqual(activation.status()["pointer"]["active"]["version"], champion.VERSION)

    def test_tampered_candidate_rejected_before_pointer_update(self):
        p = foundation._path("retraining/job-002/model.pkl")
        before = sha256(activation._path("active.json"))
        p.write_bytes(b"tampered model")
        self.assertEqual(self.promote()["status"], "REJECTED")
        self.assertEqual(before, sha256(activation._path("active.json")))
        self.assertTrue(activation.status()["health"]["healthy"])

    def test_post_activation_health_failure_restores_previous_champion(self):
        original = activation._pointer_health
        def fail_candidate(expected):
            if expected["active"]["version"] == activation.CANDIDATE:
                raise GateError("injected post-activation health failure")
            return original(expected)
        with patch.object(activation, "_pointer_health", side_effect=fail_candidate):
            result = self.promote()
        self.assertEqual(result["status"], "RECOVERED")
        self.assertEqual(activation.read_active().max_depth, 1)
        event = activation.audit()["journal"]["events"][-1]
        self.assertTrue(any(not check["healthy"] for check in event["health_checks"]))
        self.assertTrue(event["health_checks"][-1]["healthy"])

    def test_keyboard_interrupt_after_pointer_replace_restores_and_records(self):
        original = activation._atomic
        fired = []
        def interrupt(name, value):
            original(name, value)
            if name == "active.json" and value["active"]["version"] == activation.CANDIDATE and not fired:
                fired.append(True)
                raise KeyboardInterrupt
        with patch.object(activation, "_atomic", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.promote()
        self.assertEqual(activation.read_active().max_depth, 1)
        self.assertEqual(activation.audit()["journal"]["events"][-1]["status"], "RECOVERED")

    def test_hard_crash_pending_record_blocks_reads_then_recovery_restores(self):
        self.promote()
        journal, _, _ = activation._load()
        event = journal["events"][-1]
        # Emulate a hard kill after pointer replace, before final journal commit.
        event.update(status="PENDING", result=None, finished_at=None, reason="pending")
        activation._seal_event(event)
        activation._save(journal)
        with self.assertRaises(GateError):
            activation.read_active()
        result = activation.execute("recover")
        self.assertEqual(result["status"], "RECOVERED")
        self.assertEqual(activation.read_active().max_depth, 1)

    def test_atomic_replace_failure_preserves_old_pointer_and_recovers(self):
        original = activation.os.replace
        fired = []
        def fail_once(source, target):
            if str(source).endswith(".active.tmp") and not fired:
                fired.append(True)
                raise OSError("injected atomic replace failure")
            return original(source, target)
        with patch.object(activation.os, "replace", side_effect=fail_once):
            result = self.promote()
        self.assertEqual(result["status"], "RECOVERED")
        self.assertFalse(activation._path(".active.tmp").exists())
        self.assertEqual(activation.read_active().max_depth, 1)

    def test_explicit_rollback_works_even_if_active_candidate_is_corrupt(self):
        self.promote()
        foundation._path("retraining/job-002/model.pkl").write_bytes(b"corrupted active candidate")
        with self.assertRaises(GateError):
            activation.status()
        self.assertEqual(activation.execute("rollback")["status"], "SUCCEEDED")
        self.assertEqual(activation.read_active().max_depth, 1)

    def test_failed_recovery_blocks_active_reads_without_claiming_health(self):
        with patch.object(activation, "_pointer_health", side_effect=GateError("injected persistent health failure")), \
             self.assertRaises(GateError):
            self.promote()
        self.assertTrue(activation.audit()["recovery_required"])
        with self.assertRaises(GateError):
            activation.read_active()
        self.assertEqual(activation.execute("recover")["status"], "RECOVERED")

    def test_tampered_pointer_and_journal_fail_closed(self):
        for name in ("active.json", "journal.json"):
            p = activation._path(name)
            original = p.read_bytes()
            value = json.loads(original)
            value["production_activation"] = True
            p.write_text(json.dumps(value))
            with self.subTest(name=name), self.assertRaises(GateError):
                activation.status()
            p.write_bytes(original)

    def test_concurrent_writer_cannot_switch_pointer(self):
        with activation._lock():
            with self.assertRaises(FileExistsError):
                self.promote()
            with self.assertRaises(GateError):
                activation.status()

    def test_audit_capacity_stops_mutations_without_eviction(self):
        with patch.object(activation, "MAX_EVENTS", 2):
            # Existing header uses 20, so a changed policy must fail closed.
            with self.assertRaises(GateError):
                self.promote()
        journal, _, _ = activation._load()
        # Fill with genuinely rejected rollback operations, preserving the initialized pointer.
        for _ in range(activation.MAX_EVENTS - len(journal["events"])):
            self.assertEqual(activation.execute("rollback")["status"], "REJECTED")
        before = sha256(activation._path("journal.json"))
        with self.assertRaises(GateError):
            self.promote()
        self.assertEqual(before, sha256(activation._path("journal.json")))
        self.assertEqual(len(activation.audit()["journal"]["events"]), activation.MAX_EVENTS)

    def test_cli_block_status_audit_and_no_path_override(self):
        with patch.object(sys, "argv", ["activation", "promote", "--receipt-id", self.blocked["receipt_id"]]), \
             redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            activation.main()
        self.assertEqual(error.exception.code, 1)
        for command in ("status", "audit"):
            with patch.object(sys, "argv", ["activation", command]), redirect_stdout(io.StringIO()):
                activation.main()


if __name__ == "__main__":
    unittest.main()
