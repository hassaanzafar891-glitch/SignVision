"""Temporary fixtures and fake estimators only; no Random Forest is trained."""

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking import train_random_forest as baseline


class FakeClassifier:
    """Recording stand-in; fit does not learn anything."""
    def get_params(self):
        return dict(baseline.RF_PARAMS)

    def fit(self, features, labels):
        self.seen = (features, labels)

    def predict(self, features):
        return ["a" if row[0] % 2 == 0 else "b" for row in features]


class RandomForestBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = Path(self.temp.name) / "backend"
        self.backend.mkdir()
        self.database = self.backend / "dataset.db"
        connection = sqlite3.connect(self.database)
        for table in baseline.TABLES.values():
            connection.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, " +
                               ",".join(f"{name} REAL" for name in baseline.FEATURES) + ", label TEXT)")
            for i in range(6):
                connection.execute(f"INSERT INTO {table} VALUES ({','.join(['?'] * 44)})",
                                   [1000 + i, *([float(i)] * 42), "a" if i % 2 == 0 else "b"])
        connection.commit()
        connection.close()
        self.indices = {"train": [0, 1], "validation": [2, 3], "test": [4, 5]}

    def test_sql_loader_excludes_id_uses_all_features_and_leaves_source_unchanged(self):
        original = baseline.sha256(self.database)
        for kind in baseline.TABLES:
            x, y = baseline.load_dataset(self.database, kind)
            self.assertEqual(len(x), 6)
            self.assertEqual(len(x[0]), 42)
            self.assertEqual(x[0], [0.0] * 42)
            self.assertEqual(x[1], [1.0] * 42)
            self.assertEqual(y[:2], ["a", "b"])
        self.assertEqual(original, baseline.sha256(self.database))
        with self.assertRaises(KeyError):
            baseline.load_dataset(self.database, "invalid_table")
        with self.assertRaises(sqlite3.OperationalError):
            baseline.load_dataset(self.backend / "missing.db", "alphabet")
        self.assertFalse((self.backend / "missing.db").exists())

    def test_uses_existing_split_helper_with_seed_42(self):
        helper = MagicMock()
        helper.split_three_way.return_value = [SimpleNamespace(tolist=lambda ids=ids: ids)
                                               for ids in self.indices.values()]
        with patch.dict(sys.modules, {"tracking.data_splits": helper}):
            self.assertEqual(baseline.partition_indices([[1.0] * 42], ["a"]), self.indices)
        helper.split_three_way.assert_called_once_with([[1.0] * 42], ["a"], seed=42)

    def test_only_training_rows_fit_and_both_tasks_log_separate_baselines(self):
        production = self.backend / "data/models/alphabet_model.h5"
        production.parent.mkdir(parents=True)
        production.write_bytes(b"do-not-touch")
        client = MagicMock()
        client.start_run.side_effect = [nullcontext(SimpleNamespace(info=SimpleNamespace(run_id=kind)))
                                        for kind in baseline.TABLES]
        outputs = []
        for kind in baseline.TABLES:
            classifier = FakeClassifier()
            with patch.object(baseline, "build_classifier", return_value=classifier), \
                 patch.object(baseline, "partition_indices", return_value=self.indices), \
                 patch.object(baseline, "evaluate_predictions", return_value={
                     "accuracy": 1.0, "macro_f1": 1.0, "confusion_matrix": [[1, 0], [0, 1]],
                 }), patch("builtins.print"):
                output = baseline.train_task(kind, self.database, client, backend=self.backend)
            self.assertEqual(classifier.seen, ([[0.0] * 42, [1.0] * 42], ["a", "b"]))
            self.assertTrue(output.is_relative_to(self.backend / "data/training_runs/random_forest"))
            outputs.append(output)
            manifest = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(manifest["baseline_type"], "classical_random_forest")
            self.assertEqual(manifest["dataset"]["partition_samples"], {"train": 2, "validation": 2, "test": 2})
            self.assertEqual(manifest["hyperparameters"]["random_state"], 42)
            for name, record in manifest["artifacts"].items():
                self.assertEqual(baseline.sha256(output / name), record["sha256"])
            self.assertTrue((output / "random_forest.pkl").exists())
            for partition in baseline.PARTITIONS:
                self.assertTrue((output / f"{partition}_confusion_matrix.csv").exists())
                client.log_metrics.assert_any_call({f"{partition}_accuracy": 1.0, f"{partition}_macro_f1": 1.0})
            comparison = json.loads((output / "comparison.json").read_text())
            self.assertEqual(comparison["split_comparability"], "not_verified")
            self.assertNotIn("accuracy_difference_percentage_points", comparison)
            client.log_artifacts.assert_any_call(str(output), artifact_path="random_forest_baseline")
        self.assertNotEqual(outputs[0], outputs[1])
        self.assertEqual(production.read_bytes(), b"do-not-touch")
        self.assertEqual(client.start_run.call_count, 2)
        for call in client.start_run.call_args_list:
            self.assertEqual(call.kwargs["tags"]["baseline.type"], "classical_random_forest")

    def test_server_failure_prevents_fitting(self):
        client = MagicMock()
        client.start_run.side_effect = ConnectionError("server unavailable")
        classifier = MagicMock()
        with patch.object(baseline, "build_classifier", return_value=classifier), \
             patch.object(baseline, "partition_indices", return_value=self.indices):
            with self.assertRaises(ConnectionError):
                baseline.train_task("alphabet", self.database, client, backend=self.backend)
        classifier.fit.assert_not_called()

    def test_historical_score_seed_and_counts_do_not_prove_comparability(self):
        dataset = {"sha256": "ordered-dataset-hash"}
        hashes = baseline.split_fingerprints(self.indices)
        evidence = {"model_name": "alphabet", "dataset": dataset,
                    "params": {"split_random_state": 42}, "test_metrics": {"test_accuracy": 0.98373985}}
        result = baseline.comparison_report("alphabet", 0.9, dataset, hashes, evidence)
        self.assertEqual(result["split_comparability"], "not_verified")
        evidence["split_indices_sha256"] = hashes
        self.assertEqual(baseline.comparison_report("alphabet", 0.9, dataset, hashes, evidence)
                         ["split_comparability"], "verified")
        self.assertAlmostEqual(
            baseline.comparison_report("alphabet", 0.9, dataset, hashes, evidence)
            ["accuracy_difference_percentage_points"], (0.9 - 0.98373985) * 100,
        )
        changed = {**self.indices, "test": [5, 4]}
        self.assertEqual(baseline.comparison_report("alphabet", 0.9, dataset,
                                                  baseline.split_fingerprints(changed), evidence)
                         ["split_comparability"], "not_verified")

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
    def test_accuracy_macro_f1_and_class_order_without_training(self):
        scores = baseline.evaluate_predictions(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
        self.assertEqual(scores["accuracy"], 0.75)
        self.assertAlmostEqual(scores["macro_f1"], 11 / 15)
        self.assertEqual(scores["confusion_matrix"], [[1, 1], [0, 2]])
        classifier = baseline.build_classifier()  # Construction only; never fit.
        self.assertEqual(classifier.random_state, 42)
        self.assertEqual(classifier.n_jobs, 2)


if __name__ == "__main__":
    unittest.main()
