"""Synthetic records only: no historical metrics or model artifacts are edited."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking import evaluation_gate as gate


def record(role="production", task="alphabet", accuracy=0.8):
    return {
        "schema_version": 1, "role": role, "task": task, "protocol": gate.PROTOCOL,
        "evaluated_at": "2026-01-01T00:00:00+00:00",
        "dataset": {"dvc_md5": gate.DATASET_MD5, "sha256": "a" * 64,
                    "ordered_rows_sha256": "f" * 64,
                    "table": gate.TABLES[task], "feature_columns": gate.FEATURES, "rows": 6},
        "model": {"task": task, "family": "keras_dense", "artifacts": {key: "b" * 64 for key in gate.ARTIFACTS}},
        "classes": ["a", "b"],
        "held_out": {"provenance_status": "reviewed_training_time_evidence", "duplicate_disjoint": True,
                     "manifest_sha256": "c" * 64, "test_indices_sha256": "d" * 64,
                     "test_rows_sha256": "e" * 64, "sample_count": 2,
                     "partition_counts": {"train": 2, "validation": 2, "test": 2}},
        "metrics": {"test_accuracy": accuracy, "test_loss": 0.2},
    }


class EvaluationGateTests(unittest.TestCase):
    def test_non_regressing_candidates_pass_for_both_tasks_and_bind_artifacts(self):
        for task in gate.TABLES:
            for accuracy in (0.8, 0.9):
                with self.subTest(task=task, accuracy=accuracy):
                    production, candidate = record(task=task), record("candidate", task, accuracy)
                    candidate["model"]["artifacts"]["model"] = "f" * 64
                    result = gate.compare_evaluations(production, candidate)
                    self.assertTrue(result["approved"])
                    self.assertFalse(result["deployment_performed"])
                    self.assertEqual(result["candidate_artifacts"]["model"], "f" * 64)
                    self.assertEqual(result["candidate_record_sha256"], gate.canonical_sha256(candidate))

    def test_accuracy_regression_has_no_tolerance(self):
        with self.assertRaisesRegex(gate.GateError, "regression"):
            gate.compare_evaluations(record(), record("candidate", accuracy=0.7999999999))

    def test_missing_required_fields_fail_closed(self):
        original = record("candidate")
        paths = [(key,) for key in original] + [
            ("metrics", "test_accuracy"), ("metrics", "test_loss"),
            ("model", "artifacts", "scaler"), ("dataset", "sha256"),
            ("held_out", "test_rows_sha256"), ("held_out", "manifest_sha256"),
            ("held_out", "partition_counts"),
        ]
        for path in paths:
            with self.subTest(path=path):
                candidate = deepcopy(original)
                target = candidate
                for key in path[:-1]:
                    target = target[key]
                del target[path[-1]]
                with self.assertRaises(gate.GateError):
                    gate.compare_evaluations(record(), candidate)

    def test_malformed_objects_metric_ranges_and_nan(self):
        for malformed in (None, [], "not a record", 1, {"test_accuracy": 0.9}):
            with self.subTest(malformed=malformed), self.assertRaises(gate.GateError):
                gate.compare_evaluations(record(), malformed)
        for metric in ("test_accuracy", "test_loss"):
            values = [None, True, "0.9", float("nan"), float("inf"), -0.1]
            if metric == "test_accuracy":
                values.append(1.01)
            for value in values:
                with self.subTest(metric=metric, value=value):
                    candidate = record("candidate")
                    candidate["metrics"][metric] = value
                    with self.assertRaises(gate.GateError):
                        gate.compare_evaluations(record(), candidate)
        for key in ("model", "dataset", "held_out", "metrics"):
            candidate = record("candidate")
            candidate[key] = []
            with self.subTest(key=key), self.assertRaises(gate.GateError):
                gate.compare_evaluations(record(), candidate)

    def test_dataset_split_task_and_protocol_mismatches(self):
        changes = [
            ("dataset", "sha256", "f" * 64), ("dataset", "dvc_md5", "0" * 32),
            ("dataset", "table", "wordDataset"), ("dataset", "feature_columns", []),
            ("model", "task", "word"), ("model", "family", "random_forest"),
            ("held_out", "test_indices_sha256", "f" * 64),
            ("held_out", "test_rows_sha256", "f" * 64), ("held_out", "sample_count", 3),
            ("held_out", "duplicate_disjoint", False), ("held_out", "provenance_status", "assumed"),
        ]
        for section, key, value in changes:
            candidate = record("candidate")
            candidate[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(gate.GateError):
                gate.compare_evaluations(record(), candidate)
        for key, value in (("role", "production"), ("protocol", "legacy"), ("schema_version", True),
                           ("evaluated_at", "2026-01-01"), ("classes", ["a", "a"])):
            candidate = record("candidate")
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(gate.GateError):
                gate.compare_evaluations(record(), candidate)
        with self.assertRaises(gate.GateError):
            gate.compare_evaluations(record(), record("candidate", "word"))

    def test_invalid_production_record_also_blocks(self):
        production = record()
        production["metrics"]["test_accuracy"] = None
        with self.assertRaises(gate.GateError):
            gate.compare_evaluations(production, record("candidate"))

    def test_json_reader_rejects_duplicates_nonfinite_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            for data in ('{"test_accuracy": 0.1, "test_accuracy": 0.9}', '{"score": NaN}',
                         '{"score": 1e999}', '{'):
                path.write_text(data)
                with self.subTest(data=data), self.assertRaises(gate.GateError):
                    gate.read_json(path)
            with self.assertRaises(gate.GateError):
                gate.read_json(Path(directory) / "missing.json")


if __name__ == "__main__":
    unittest.main()
