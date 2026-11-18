"""Synthetic fixture replay and temporary files only. Never train or use telemetry."""

from copy import deepcopy
import builtins
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_sandbox import foundation, drift
from monitoring import js_divergence
from tracking.evaluation_gate import GateError, canonical_sha256


class DemoDriftTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.addCleanup(patch.stopall)
        patch.object(foundation, "WORKSPACE", self.workspace).start()
        self.protected = {}
        for name in ("backend/data/models/alphabet_model.h5", "backend/data/db/main_dataset.db",
                     "backend/mlflow.db", "mlflow.db", "docker-compose.yml"):
            path = self.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("protected:" + name).encode())
            self.protected[path] = path.read_bytes()
        foundation.initialize()
        self.root = self.workspace / ".signvision-demo"
        self.core = self.snapshot()

    def snapshot(self):
        return {p.relative_to(self.root).as_posix(): p.read_bytes()
                for p in self.root.rglob("*") if p.is_file()}

    def assert_isolated(self):
        self.assertEqual(self.protected, {p: p.read_bytes() for p in self.protected})
        for name, data in self.core.items():
            self.assertEqual((self.root / name).read_bytes(), data)
        self.assertEqual(foundation.status()["active_version"], foundation.CHAMPION)

    def journal(self):
        return json.loads((self.root / "drift/events.json").read_text(encoding="utf-8"))

    def write_journal(self, value):
        (self.root / "drift/events.json").write_text(json.dumps(value), encoding="utf-8")

    def reseal(self, journal):
        for event in journal["events"]:
            event["event_sha256"] = canonical_sha256({k: v for k, v in event.items() if k != "event_sha256"})
        return drift._seal_journal(journal)

    def test_reference_uses_validation_predictions_and_no_test_inputs(self):
        dataset = foundation.read_dataset()
        non_test = set(dataset["partitions"]["train"] + dataset["partitions"]["validation"])
        called = []
        original = foundation._predict
        def guard(model, features):
            self.assertIn(features[1], non_test)
            called.append(features[1])
            return original(model, features)
        with patch.object(foundation, "_predict", side_effect=guard):
            result = drift.create_reference()
            drift.run("shifted")
        reference = result["reference"]
        self.assertEqual(reference["partition"], "validation")
        self.assertEqual(reference["sample_count"], 24)
        self.assertEqual(reference["partition_indices_sha256"], canonical_sha256(dataset["partitions"]["validation"]))
        expected = dict.fromkeys(dataset["class_labels"], 0)
        model = foundation.read_active()
        for index in dataset["partitions"]["validation"]:
            expected[original(model, dataset["rows"][index]["features"])] += 1
        self.assertEqual(reference["counts"], expected)
        self.assertFalse(reference["production_evidence"])
        self.assertTrue(called)
        self.assert_isolated()

    def test_normal_replay_has_zero_score_and_no_alert(self):
        reference = drift.create_reference()["reference"]
        event = drift.run("normal")["event"]
        self.assertEqual(event["observed"]["counts"], reference["counts"])
        self.assertEqual(event["divergence_score"], 0.0)
        self.assertEqual(event["status"], "no_alert")
        self.assertFalse(event["alert"])

    def test_shifted_replay_emits_real_computed_synthetic_alert(self):
        reference = drift.create_reference()["reference"]
        result = drift.run("shifted")
        event = result["event"]
        self.assertEqual(event["observed"]["counts"], {"demo_class_0": 24, "demo_class_1": 0})
        self.assertEqual(event["divergence_score"], js_divergence(reference["counts"], event["observed"]["counts"]))
        self.assertGreater(event["divergence_score"], event["threshold"])
        self.assertTrue(event["alert"])
        self.assertIn("SYNTHETIC DEMO DRIFT ALERT", result["message"])
        self.assertFalse(result["training_triggered"])
        self.assertEqual(event["reference_sha256"], canonical_sha256(reference))
        self.assertIsNotNone(drift.datetime.fromisoformat(event["timestamp"]).utcoffset())
        self.assertEqual(event["event_sha256"], canonical_sha256({k: v for k, v in event.items() if k != "event_sha256"}))

    def test_threshold_is_strictly_exceeded_not_equal(self):
        drift.create_reference()
        score = drift.run("shifted")["event"]["divergence_score"]
        self.assertFalse(drift.run("shifted", threshold=score)["event"]["alert"])
        self.assertTrue(drift.run("shifted", threshold=score / 2)["event"]["alert"])
        self.assertFalse(drift.run("shifted", threshold=1)["event"]["alert"])

    def test_source_threshold_configuration_is_used_and_recorded(self):
        config = deepcopy(drift.DEFAULTS)
        config["threshold"] = 1
        path = self.workspace / "drift-config.json"
        path.write_text(json.dumps(config))
        with patch.object(drift, "CONFIG_PATH", path):
            drift.create_reference()
            event = drift.run("shifted")["event"]
        self.assertEqual(event["threshold"], 1.0)
        self.assertFalse(event["alert"])

    def test_invalid_thresholds_batches_and_config_do_not_write_events(self):
        drift.create_reference()
        before = self.snapshot()
        for value in (True, 0, -0.1, 1.1, float("nan"), float("inf"), "0.1"):
            with self.subTest(threshold=value), self.assertRaises(GateError):
                drift.run(threshold=value)
        for value in (True, 0, -1, 241, 1.5, "24"):
            with self.subTest(batch_size=value), self.assertRaises(GateError):
                drift.run(batch_size=value)
        path = self.workspace / "invalid.json"
        path.write_text(json.dumps({**drift.DEFAULTS, "registry": "production"}))
        with patch.object(drift, "CONFIG_PATH", path), self.assertRaises(GateError):
            drift.run()
        self.assertEqual(before, self.snapshot())

    def test_insufficient_batch_has_no_score_or_alert(self):
        drift.create_reference()
        event = drift.run(batch_size=1)["event"]
        self.assertEqual(event["status"], "insufficient_current_samples")
        self.assertIsNone(event["divergence_score"])
        self.assertFalse(event["alert"])

    def test_missing_reference_is_not_automatically_created(self):
        before = self.snapshot()
        with self.assertRaisesRegex(GateError, "reference unavailable"):
            drift.run()
        self.assertEqual(before, self.snapshot())

    def test_reference_is_deterministic_and_mismatches_fail_closed(self):
        result = drift.create_reference()
        path = self.root / "drift/reference.json"
        original = path.read_bytes()
        self.assertEqual(result, drift.create_reference())
        self.assertEqual(path.read_bytes(), original)
        for field, value in (("partition", "test"), ("model_manifest_sha256", "0" * 64),
                             ("counts", {"demo_class_0": 24, "demo_class_1": 0}),
                             ("production_evidence", True)):
            changed = json.loads(original)
            changed[field] = value
            path.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaisesRegex(GateError, "reference integrity"):
                drift.run()
            path.write_bytes(original)
        self.assertFalse((self.root / "drift/events.json").exists())

    def test_repeated_observation_deduplicates_without_changing_first_timestamp(self):
        drift.create_reference()
        first = drift.run()
        before = self.snapshot()
        repeated = drift.run()
        self.assertTrue(repeated["deduplicated"])
        self.assertEqual(first["event"], repeated["event"])
        self.assertEqual(before, self.snapshot())

    def test_observation_identity_and_score_match_another_workspace(self):
        drift.create_reference()
        first = drift.run()["event"]
        with tempfile.TemporaryDirectory() as other, patch.object(foundation, "WORKSPACE", Path(other).resolve()):
            foundation.initialize()
            drift.create_reference()
            second = drift.run()["event"]
        for key in first:
            if key not in ("timestamp", "event_sha256"):
                self.assertEqual(first[key], second[key], key)

    def test_journal_is_bounded_with_auditable_eviction_counts(self):
        drift.create_reference()
        for index in range(23):
            drift.run(threshold=(index + 1) / 100)
        result = drift.inspect()
        self.assertEqual(result["retained_events"], 20)
        self.assertEqual(result["discarded_events"], 3)
        self.assertEqual([e["sequence"] for e in result["events"]], list(range(4, 24)))
        self.assertLessEqual((self.root / "drift/events.json").stat().st_size, 131072)
        self.assert_isolated()

    def test_inspect_is_read_only_and_checks_all_retained_events(self):
        drift.create_reference()
        drift.run("normal")
        drift.run("shifted")
        before = self.snapshot()
        result = drift.inspect()
        self.assertTrue(result["journal_integrity_verified"])
        self.assertEqual(len(result["events"]), 2)
        self.assertEqual(before, self.snapshot())

    def test_recomputed_checksums_cannot_hide_forged_scores_counts_or_scope(self):
        drift.create_reference()
        drift.run()
        original = self.journal()
        for key, value in (("divergence_score", 0.99), ("production_evidence", True),
                           ("observed", {"sample_count": 24, "counts": {"demo_class_0": 0, "demo_class_1": 24},
                                         "row_indices_sha256": "0" * 64})):
            journal = deepcopy(original)
            journal["events"][0][key] = value
            self.write_journal(self.reseal(journal))
            before = (self.root / "drift/events.json").read_bytes()
            with self.subTest(key=key), self.assertRaises(GateError):
                drift.inspect()
            with self.assertRaises(GateError):
                drift.run()
            self.assertEqual(before, (self.root / "drift/events.json").read_bytes())
        self.assert_isolated()

    def test_corrupt_journal_timestamp_and_checksum_fail_closed(self):
        drift.create_reference()
        drift.run()
        original = self.journal()
        for timestamp in ("not-a-date", "2026-01-01T00:00:00", 123):
            value = deepcopy(original)
            value["events"][0]["timestamp"] = timestamp
            self.write_journal(self.reseal(value))
            with self.subTest(timestamp=timestamp), self.assertRaises(GateError):
                drift.inspect()
        value = deepcopy(original)
        value["journal_sha256"] = "0" * 64
        self.write_journal(value)
        with self.assertRaisesRegex(GateError, "checksum"):
            drift.inspect()
        (self.root / "drift/events.json").write_text('{"events": [')
        with self.assertRaises(GateError):
            drift.run()

    def test_reference_cannot_be_recreated_behind_existing_history(self):
        drift.create_reference()
        drift.run()
        before = (self.root / "drift/events.json").read_bytes()
        (self.root / "drift/reference.json").unlink()
        with self.assertRaisesRegex(GateError, "behind existing events"):
            drift.create_reference()
        self.assertEqual(before, (self.root / "drift/events.json").read_bytes())

    def test_concurrent_writer_and_unfinished_temp_files_block_inspection(self):
        drift.create_reference()
        with drift._write_lock():
            with self.assertRaises(FileExistsError):
                drift.run()
            with self.assertRaisesRegex(GateError, "busy"):
                drift.inspect()
        path = self.root / "drift/.events.tmp"
        path.write_bytes(b"unfinished")
        with self.assertRaisesRegex(GateError, "Unfinished"):
            drift.inspect()
        self.assertEqual(path.read_bytes(), b"unfinished")

    def test_atomic_write_failure_keeps_old_journal_and_cleans_owned_files(self):
        drift.create_reference()
        drift.run("normal")
        before = (self.root / "drift/events.json").read_bytes()
        with patch.object(drift.os, "replace", side_effect=OSError("synthetic failure")), self.assertRaises(OSError):
            drift.run("shifted")
        self.assertEqual(before, (self.root / "drift/events.json").read_bytes())
        self.assertFalse((self.root / "drift/.events.tmp").exists())
        self.assertFalse((self.root / "drift/.write.lock").exists())
        self.assertTrue(drift.inspect()["journal_integrity_verified"])

    def test_hardlinked_events_cannot_touch_a_production_store(self):
        drift.create_reference()
        try:
            os.link(self.workspace / "mlflow.db", self.root / "drift/events.json")
        except (OSError, NotImplementedError):
            self.skipTest("Hard links are unavailable")
        with self.assertRaisesRegex(GateError, "hard-linked"):
            drift.run()
        self.assertEqual(self.protected, {p: p.read_bytes() for p in self.protected})

    def test_production_environment_and_live_monitor_are_not_used(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("mlflow", "tensorflow", "keras", "sklearn", "mediapipe"):
                raise AssertionError("Demo tried to load production/ML libraries")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded), \
             patch("monitoring.get_monitor", side_effect=AssertionError("Live telemetry must not be used")), \
             patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "https://not-demo.invalid", "SIGNVISION_TRACKING_ENABLED": "true"}):
            drift.create_reference()
            drift.run()
            drift.inspect()
        self.assert_isolated()

    def test_cli_runs_and_inspects_but_rejects_arbitrary_event_paths(self):
        for args in (["reference"], ["run", "--scenario", "shifted"], ["inspect"]):
            output = io.StringIO()
            with patch.object(sys, "argv", ["demo_sandbox.drift", *args]), redirect_stdout(output):
                drift.main()
            self.assertEqual(json.loads(output.getvalue()).get("scope", foundation.SCOPE), foundation.SCOPE)
        with patch.object(sys, "argv", ["drift", "run", "--event-file", "backend/mlflow.db"]), \
             redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            drift.main()
        self.assert_isolated()

    def test_unknown_or_oversized_drift_files_are_not_accepted(self):
        drift.create_reference()
        extra = self.root / "drift/unknown.json"
        extra.write_text('{}')
        with self.assertRaisesRegex(GateError, "Unexpected"):
            drift.run()
        extra.unlink()
        (self.root / "drift/events.json").write_bytes(b' ' * 131073)
        with self.assertRaisesRegex(GateError, "size limit"):
            drift.inspect()


if __name__ == "__main__":
    unittest.main()
