"""No training, MLflow installation, or external services are used by these tests."""

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking import training
from tracking.import_baselines import current_script_config, script_path


class Adam:
    learning_rate = SimpleNamespace(numpy=lambda: 0.001)

    def get_config(self):
        return {"learning_rate": 0.001, "name": "Adam"}


class Dense:
    def get_config(self):
        return {"units": 120, "activation": "relu"}


class TrainingTrackingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.backend = Path(temporary.name) / "backend"
        self.backend.mkdir()
        self.script = self.backend / "train.py"
        self.script.write_text("# Training is never executed in this fixture.\n")
        self.client = MagicMock()
        self.client.active_run.return_value = None
        self.client.start_run.return_value.info.run_id = "mock-run-id"
        self.importer = MagicMock(return_value=self.client)
        patch.object(training, "importlib", SimpleNamespace(import_module=self.importer)).start()
        self.addCleanup(patch.stopall)
        patch.object(training, "git_metadata", return_value={
            "import_commit": "b" * 40, "import_checkout_dirty": True,
        }).start()
        patch("builtins.print").start()

    def run_context(self, enabled=False, **environment):
        env = {"SIGNVISION_TRACKING_ENABLED": "true" if enabled else "false", **environment}
        return training.TrainingRun("alphabet", self.script, backend=self.backend, environ=env)

    def read_record(self, run):
        return json.loads((run.output_dir / "run_metadata.json").read_text(encoding="utf-8"))

    def test_disabled_does_not_import_mlflow_and_outputs_are_unique(self):
        production = self.backend / "data/models/alphabet_model.h5"
        production.parent.mkdir(parents=True)
        production.write_bytes(b"existing-production-model")
        with self.run_context() as first:
            (first.output_dir / "alphabet_model.h5").write_bytes(b"new-output")
        with self.run_context() as second:
            pass
        self.assertNotEqual(first.output_dir, second.output_dir)
        self.assertTrue(first.output_dir.is_relative_to(self.backend / "data/training_runs"))
        self.assertEqual(production.read_bytes(), b"existing-production-model")
        self.importer.assert_not_called()
        record = self.read_record(first)
        self.assertEqual(record["training_status"], "FINISHED")
        self.assertFalse(record["tracking_requested"])
        self.assertEqual(record["artifacts"]["alphabet_model.h5"]["sha256"],
                         hashlib.sha256(b"new-output").hexdigest())

    def test_metrics_model_dataset_git_and_artifacts_are_logged(self):
        with self.run_context(enabled=True) as run:
            run.record_dataset([[1.0, 2.0], [3.0, 4.0]], [("a",), ("b",)],
                               "main_dataset.db", "alphabetDataset", train_samples=1, test_samples=1)
            model = SimpleNamespace(to_json=lambda: '{"class_name":"Sequential"}',
                                    optimizer=Adam(), layers=[Dense()])
            run.record_model(model, epochs=25, batch_size=1, split={"validation_split": 0.2})
            callback = run.callback(object)
            callback.on_epoch_end(0, {"accuracy": 0.6, "val_accuracy": 0.5,
                                      "loss": 0.8, "val_loss": 0.9})
            callback.on_epoch_end(1, {"accuracy": 0.7, "loss": 0.4})
            run.record_test(loss=0.3, accuracy=0.75)
            for suffix in ("model.h5", "scaler.pkl", "label_encoder.pkl"):
                (run.output_dir / f"alphabet_{suffix}").write_bytes(b"fixture")
        self.client.set_tracking_uri.assert_called_once_with((self.backend / "mlruns").as_uri())
        tags = self.client.start_run.call_args.kwargs["tags"]
        self.assertEqual(tags["training.git.commit"], "b" * 40)
        self.assertEqual(tags["training.git.dirty"], "true")
        params = self.read_record(run)["params"]
        self.assertEqual(params["learning_rate"], 0.001)
        self.assertEqual(params["optimizer"], "Adam")
        self.assertEqual(params["epochs"], 25)
        self.assertEqual(params["batch_size"], 1)
        self.assertEqual(params["model_name"], "alphabet")
        self.assertEqual(json.loads(params["architecture"])[0]["units"], 120)
        self.assertEqual(len(params["dataset.sha256"]), 64)
        self.client.log_metrics.assert_any_call({"accuracy": 0.6, "val_accuracy": 0.5,
                                                "loss": 0.8, "val_loss": 0.9}, step=1)
        # Missing validation metrics are omitted rather than fabricated as zero.
        self.client.log_metrics.assert_any_call({"accuracy": 0.7, "loss": 0.4}, step=2)
        self.client.log_metrics.assert_any_call({"test_loss": 0.3, "test_accuracy": 0.75})
        self.client.log_artifacts.assert_called_once_with(str(run.output_dir), artifact_path="training")
        self.client.end_run.assert_called_once_with(status="FINISHED")
        self.assertEqual(self.read_record(run)["mlflow_run_id"], "mock-run-id")

    def test_dagshub_uri_and_credentials_are_not_saved_in_metadata(self):
        with self.run_context(enabled=True, MLFLOW_TRACKING_URI=training.DAGSHUB_TRACKING_URI,
                              MLFLOW_TRACKING_PASSWORD="do-not-log", MLFLOW_EXPERIMENT_NAME="Test") as run:
            pass
        self.client.set_tracking_uri.assert_called_once_with(training.DAGSHUB_TRACKING_URI)
        self.client.set_experiment.assert_called_once_with("Test")
        self.assertNotIn("do-not-log", (run.output_dir / "run_metadata.json").read_text())

    def test_requested_but_missing_mlflow_falls_back_to_local(self):
        self.importer.side_effect = ModuleNotFoundError("mlflow")
        with self.assertWarns(RuntimeWarning):
            with self.run_context(enabled=True) as run:
                (run.output_dir / "alphabet_model.h5").write_bytes(b"kept")
        self.assertEqual(self.read_record(run)["training_status"], "FINISHED")
        self.assertEqual(self.read_record(run)["tracking_errors"][0]["operation"], "initialization")
        self.client.start_run.assert_not_called()

    def test_logging_failure_retains_metrics_and_marks_remote_run_failed(self):
        self.client.log_metrics.side_effect = ConnectionError("secret must not appear in metadata")
        with self.assertWarns(RuntimeWarning):
            with self.run_context(enabled=True) as run:
                run.callback(object).on_epoch_end(0, {"accuracy": 0.5})
                run.record_test(loss=0.2, accuracy=0.8)
        record = self.read_record(run)
        self.assertEqual(record["test_metrics"]["test_accuracy"], 0.8)
        self.assertEqual(record["training_status"], "FINISHED")
        self.assertNotIn("secret must not appear", json.dumps(record))
        self.client.end_run.assert_called_once_with(status="FAILED")

    def test_training_exception_is_not_suppressed(self):
        with self.assertRaisesRegex(ValueError, "training failed"):
            with self.run_context(enabled=True) as run:
                raise ValueError("training failed")
        self.assertEqual(self.read_record(run)["training_status"], "FAILED")
        self.client.end_run.assert_called_once_with(status="FAILED")

    def test_existing_active_mlflow_run_is_not_hijacked_or_closed(self):
        self.client.active_run.return_value = SimpleNamespace()
        with self.assertWarns(RuntimeWarning):
            with self.run_context(enabled=True):
                pass
        self.client.start_run.assert_not_called()
        self.client.end_run.assert_not_called()

    def test_dataset_fingerprint_covers_values_labels_and_order(self):
        def fingerprint(features, labels):
            return training.dataset_metadata(features, labels, "db", "table")["sha256"]
        base = fingerprint([[1, 2], [3, 4]], ["a", "b"])
        self.assertEqual(base, fingerprint([[1, 2], [3, 4]], [("a",), ("b",)]))
        self.assertNotEqual(base, fingerprint([[1, 9], [3, 4]], ["a", "b"]))
        self.assertNotEqual(base, fingerprint([[1, 2], [3, 4]], ["a", "c"]))
        self.assertNotEqual(base, fingerprint([[3, 4], [1, 2]], ["b", "a"]))


class TrainerContractTests(unittest.TestCase):
    """Read syntax only; never import or execute either TensorFlow trainer."""

    def test_original_architectures_splits_and_preprocessing_are_preserved(self):
        for name, units, epochs, batch in (("alphabet", [120, 64], 25, 1),
                                           ("word", [256, 128, 64], 50, 32)):
            path = script_path(training.BACKEND, name)
            config = current_script_config(path)
            dense = [layer for layer in config["layers"] if layer["type"] == "Dense"]
            self.assertEqual([layer["args"][0] for layer in dense[:-1]], units)
            self.assertTrue(all(layer["kwargs"]["activation"] == "relu" for layer in dense[:-1]))
            self.assertEqual(dense[0]["kwargs"]["input_shape"], (42,))
            self.assertEqual(dense[-1]["args"][0], {"expression": "num_classes"})
            self.assertEqual(dense[-1]["kwargs"]["activation"], "softmax")
            self.assertEqual([layer["args"][0] for layer in config["layers"]
                              if layer["type"] == "Dropout"], [0.3, 0.3])
            self.assertEqual(config["compile"], {"loss": "categorical_crossentropy",
                                                 "optimizer": "adam", "metrics": ["accuracy"]})
            self.assertEqual(config["fit"]["epochs"], epochs)
            self.assertEqual(config["fit"]["batch_size"], batch)
            self.assertEqual(config["split"]["method"], "split_three_way")
            self.assertEqual(config["split"]["test_size"], 0.2)
            self.assertEqual(config["split"]["validation_size"], 0.2)
            self.assertEqual(config["split"]["split_random_state"], 42)
            self.assertTrue(config["split"]["duplicate_disjoint"])
            self.assertNotIn("validation_split", config["fit"])
            self.assertEqual(config["fit"]["validation_data"], {"expression": "(X_val, y_val)"})
            if name == "alphabet":
                self.assertIn("EpochBar()", config["fit"]["callbacks"]["expression"])
            source = path.read_text(encoding="utf-8")
            self.assertIn("StandardScaler().fit(X_train)", source)
            self.assertIn("X_train = scaler.transform(X_train)", source)
            self.assertIn("X_val = scaler.transform(X_val)", source)
            self.assertIn("scaler.transform(X_test)", source)
            tree = ast.parse(source)
            # No scaler fit/fit_transform may consume validation/test/full data.
            scaler_fits = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute)
                           and node.func.attr in ("fit", "fit_transform", "partial_fit")
                           and (ast.unparse(node.func.value) == "StandardScaler()"
                                or ast.unparse(node.func.value) == "scaler")]
            self.assertEqual(len(scaler_fits), 1)
            self.assertEqual(scaler_fits[0].func.attr, "fit")
            self.assertEqual([ast.unparse(arg) for arg in scaler_fits[0].args], ["X_train"])
            assignments = {ast.unparse(node.targets[0]): ast.unparse(node.value)
                           for node in ast.walk(tree) if isinstance(node, ast.Assign)}
            self.assertEqual(assignments["(X_train, X_val, X_test)"],
                             "(X[train_idx], X[val_idx], X[test_idx])")
            self.assertEqual(assignments["(y_train, y_val, y_test)"],
                             "(label_encoded[train_idx], label_encoded[val_idx], label_encoded[test_idx])"
                             if name == "alphabet" else
                             "(label_cat[train_idx], label_cat[val_idx], label_cat[test_idx])")
            self.assertTrue(assignments["(train_idx, val_idx, test_idx)"].startswith("split_three_way("))
            evaluations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Attribute) and node.func.attr == "evaluate"]
            self.assertEqual(len(evaluations), 1)
            self.assertEqual([ast.unparse(arg) for arg in evaluations[0].args], ["X_test", "y_test"])

    def test_all_training_outputs_are_redirected_to_run_directory(self):
        for name in ("alphabet", "word"):
            source = script_path(training.BACKEND, name).read_text(encoding="utf-8")
            self.assertNotIn('data\\\\models', source)
            for suffix in ("model.h5", "scaler.pkl", "label_encoder.pkl"):
                self.assertIn(f'run.output_dir / "{name}_{suffix}"', source)
            self.assertIn("run.callback(Callback)", source)
            self.assertIn("run.record_test(loss=score[0], accuracy=score[1])", source)


if __name__ == "__main__":
    unittest.main()
