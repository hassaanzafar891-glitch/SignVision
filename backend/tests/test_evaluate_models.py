"""Synthetic holdout evidence and mocked inference. No training/live model reads."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking import evaluate_models as evaluate
from tracking import evaluation_gate as gate


class ModelEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = [tuple([float(i)] * 42 + ["a" if i % 2 == 0 else "b"]) for i in range(6)]
        self.manifest = {
            "schema_version": 1, "task": "alphabet",
            "dataset": {"dvc_md5": gate.DATASET_MD5, "sha256": "a" * 64,
                        "ordered_rows_sha256": gate.canonical_sha256(self.rows),
                        "table": "alphabetDataset", "feature_columns": gate.FEATURES},
            "artifacts": {}, "partitions": {"train": [0, 1], "validation": [2, 3], "test": [4, 5]},
            "provenance": {"status": "reviewed_training_time_evidence", "reviewed_by": "synthetic-test",
                           "evidence_reference": "synthetic-fixture-only"},
        }
        for key, suffix in evaluate.SUFFIXES.items():
            path = self.root / f"alphabet_{suffix}"
            path.write_bytes(f"fake-{key}".encode())
            self.manifest["artifacts"][key] = evaluate.sha256(path)
        self.path = self.root / "alphabet_provenance.json"
        self.write_manifest()

    def write_manifest(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_bundle_preflight_hash_pinning_and_missing_evidence(self):
        manifest, digest = evaluate.read_bundle(self.root, "alphabet", gate.canonical_sha256(self.manifest))
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(digest, gate.canonical_sha256(self.manifest))
        with self.assertRaisesRegex(gate.GateError, "policy"):
            evaluate.read_bundle(self.root, "alphabet", "0" * 64)
        with self.assertRaisesRegex(gate.GateError, "Missing"):
            evaluate.read_bundle(self.root, "word", "0" * 64)
        (self.root / "alphabet_scaler.pkl").write_bytes(b"tampered")
        with self.assertRaisesRegex(gate.GateError, "hash mismatch"):
            evaluate.read_bundle(self.root, "alphabet")

    def test_unreviewed_or_incomplete_provenance_is_blocked(self):
        for section, key in (("provenance", "reviewed_by"), ("provenance", "evidence_reference"),
                             ("partitions", "train"), ("artifacts", "label_encoder")):
            manifest = deepcopy(self.manifest)
            del manifest[section][key]
            self.path.write_text(json.dumps(manifest))
            with self.subTest(section=section, key=key), self.assertRaises(gate.GateError):
                evaluate.read_bundle(self.root, "alphabet")

    def test_exact_partition_coverage_and_duplicate_disjointness(self):
        result = evaluate.verify_partitions(self.manifest, self.rows, "a" * 64)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["test_indices_sha256"], gate.canonical_sha256([4, 5]))
        with self.assertRaisesRegex(gate.GateError, "SHA-256"):
            evaluate.verify_partitions(self.manifest, self.rows, "b" * 64)
        for indices in ([0, 1], [4, 99], [4]):
            bad = deepcopy(self.manifest)
            bad["partitions"]["test"] = indices
            with self.subTest(indices=indices), self.assertRaises(gate.GateError):
                evaluate.verify_partitions(bad, self.rows, "a" * 64)
        leaked = list(self.rows)
        leaked[4] = self.rows[0]
        manifest = deepcopy(self.manifest)
        manifest["dataset"]["ordered_rows_sha256"] = gate.canonical_sha256(leaked)
        with self.assertRaisesRegex(gate.GateError, "leakage"):
            evaluate.verify_partitions(manifest, leaked, "a" * 64)
        with self.assertRaisesRegex(gate.GateError, "row ordering"):
            evaluate.verify_partitions(self.manifest, list(reversed(self.rows)), "a" * 64)

    def test_dataset_reader_pins_md5_and_refuses_sidecars(self):
        path = self.root / "fixture.db"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE alphabetDataset (" + ",".join(gate.FEATURES) + ",label)")
        connection.executemany("INSERT INTO alphabetDataset VALUES (" + ",".join(["?"] * 43) + ")", self.rows)
        connection.commit()
        connection.close()
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(gate.GateError, "MD5"):
            evaluate.read_rows(path, "alphabet")
        with patch.object(evaluate, "DATASET_MD5", md5):
            rows, digest = evaluate.read_rows(path, "alphabet")
            self.assertEqual(rows, self.rows)
            self.assertEqual(digest, evaluate.sha256(path))
            Path(str(path) + "-wal").write_bytes(b"pending")
            with self.assertRaisesRegex(gate.GateError, "sidecar"):
                evaluate.read_rows(path, "alphabet")

    def test_unproven_holdout_never_loads_models(self):
        manifest = deepcopy(self.manifest)
        manifest["partitions"]["test"] = [0, 1]
        with patch.object(evaluate, "load_objects") as load:
            with self.assertRaises(gate.GateError):
                evaluate.evaluate_bundle(self.root, "alphabet", "production", (manifest, "a" * 64), self.rows, "a" * 64)
        load.assert_not_called()

    def test_approval_is_written_only_after_both_task_gates_pass(self):
        from test_evaluation_gate import record

        for regression in (False, True):
            with self.subTest(word_regression=regression):
                output = self.root / ("blocked" if regression else "passing")
                arguments = ["evaluate_models", "--production-dir", str(self.root),
                             "--candidate-dir", str(self.root), "--output-dir", str(output)]
                def fake_bundle(directory, task, *args):
                    manifest = deepcopy(self.manifest)
                    manifest["task"] = task
                    return manifest, "a" * 64

                def fake_evaluation(directory, task, role, *args):
                    score = 0.7 if regression and task == "word" and role == "candidate" else 0.8
                    return record(role, task, score)

                with patch.object(sys, "argv", arguments), patch.object(evaluate, "read_bundle", side_effect=fake_bundle), \
                     patch.object(evaluate, "read_rows", return_value=(self.rows, "a" * 64)), \
                     patch.object(evaluate, "verify_partitions"), \
                     patch.object(evaluate, "evaluate_bundle", side_effect=fake_evaluation), patch("builtins.print"):
                    if regression:
                        with self.assertRaises(SystemExit) as failed:
                            evaluate.main()
                        self.assertEqual(failed.exception.code, 1)
                    else:
                        evaluate.main()
                self.assertEqual((output / "approval.json").exists(), not regression)
                if not regression:
                    approval = gate.read_json(output / "approval.json")
                    self.assertTrue(approval["approved"])
                    self.assertEqual(set(approval["tasks"]), {"alphabet", "word"})

    def test_missing_production_history_blocks_cli_before_data_or_model_loading(self):
        args = ["evaluate_models", "--check-evidence", "--production-dir", str(self.root),
                "--candidate-dir", str(self.root)]
        self.path.unlink()  # Synthetic fixture only.
        with patch.object(sys, "argv", args), patch.object(evaluate, "read_rows") as rows, \
             patch.object(evaluate, "load_objects") as load, patch("builtins.print"):
            with self.assertRaises(SystemExit) as failed:
                evaluate.main()
        self.assertEqual(failed.exception.code, 1)
        rows.assert_not_called()
        load.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is not installed")
    def test_saved_scaler_transform_only_on_test_rows_and_real_metric_calculation(self):
        import numpy as np

        model, scaler, encoder = MagicMock(), MagicMock(), MagicMock()
        scaler.n_features_in_ = 42
        scaler.transform.side_effect = lambda x: x
        encoder.classes_ = np.array(["a", "b"])
        encoder.transform.return_value = np.array([0, 1])
        model.predict.return_value = np.array([[0.8, 0.2], [0.1, 0.9]])
        classes, metrics = evaluate.score_bundle((model, scaler, encoder), self.rows, [4, 5])
        self.assertEqual(classes, ["a", "b"])
        self.assertEqual(metrics["test_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["test_loss"], -float(np.log([0.8, 0.9]).mean()))
        np.testing.assert_array_equal(scaler.transform.call_args.args[0], np.array([self.rows[4][:-1], self.rows[5][:-1]]))
        scaler.fit.assert_not_called()
        model.fit.assert_not_called()
        model.predict.return_value = np.array([[float("nan"), 0], [0, 1]])
        with self.assertRaises(gate.GateError):
            evaluate.score_bundle((model, scaler, encoder), self.rows, [4, 5])


if __name__ == "__main__":
    unittest.main()
