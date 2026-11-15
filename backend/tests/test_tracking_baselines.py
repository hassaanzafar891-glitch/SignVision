"""Baseline tests run with the standard library; a real MLflow test is optional."""

from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking import import_baselines as importer


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.backend = Path(self.temporary.name) / "backend"
        models = self.backend / "data" / "models"
        models.mkdir(parents=True)
        for name in importer.MODEL_NAMES:
            for suffix in ("model.h5", "scaler.pkl", "label_encoder.pkl"):
                # Deliberately not valid models/pickles: importer must not load them.
                (models / f"{name}_{suffix}").write_bytes(f"{name}/{suffix}".encode())
            target = importer.script_path(self.backend, name)
            target.parent.mkdir(parents=True)
            shutil.copyfile(importer.script_path(importer.BACKEND, name), target)
        self.client = MagicMock()
        self.client.start_run.side_effect = [
            nullcontext(SimpleNamespace(info=SimpleNamespace(run_id=name)))
            for name in importer.MODEL_NAMES
        ]
        self.logged = []
        self.client.log_artifacts.side_effect = self.capture_artifacts

    def capture_artifacts(self, directory, artifact_path):
        self.assertEqual(artifact_path, "baseline")
        self.logged.append({p.name: p.read_bytes() for p in Path(directory).iterdir()})

    def test_both_bundles_checksums_and_no_metrics(self):
        provenance = {"import_commit": "a" * 40, "import_checkout_dirty": True}
        with patch.object(importer, "git_metadata", return_value=provenance):
            result = importer.import_baselines(self.client, self.backend, {})
        self.assertEqual(result, {"alphabet": "alphabet", "word": "word"})
        self.assertEqual(len(self.logged), 2)
        self.client.log_metric.assert_not_called()
        self.client.log_metrics.assert_not_called()
        for name, logged in zip(importer.MODEL_NAMES, self.logged):
            manifest = json.loads(logged["manifest.json"])
            self.assertFalse(manifest["historical_metrics_available"])
            self.assertFalse(manifest["historical_training_provenance_verified"])
            self.assertEqual(manifest["provenance"], provenance)
            self.assertEqual(manifest["configuration_provenance"], "current_script_configuration")
            self.assertEqual(len(manifest["artifacts"]), 4)
            for artifact in manifest["artifacts"]:
                content = logged[artifact["artifact"]]
                self.assertEqual(artifact["sha256"], hashlib.sha256(content).hexdigest())
                self.assertEqual(artifact["size_bytes"], len(content))
                self.assertEqual(content, (self.backend / artifact["source"]).read_bytes())
            self.assertIn(f"{name}_model.h5", logged)
            self.assertIn(f"{name}_scaler.pkl", logged)
            self.assertIn(f"{name}_label_encoder.pkl", logged)
        for call in self.client.log_params.call_args_list:
            self.assertTrue(all(key.startswith("current_script_config.") for key in call.args[0]))

    def test_current_script_configuration_is_read_without_execution(self):
        path = importer.script_path(self.backend, "alphabet")
        path.write_text(path.read_text(encoding="utf-8") +
                        '\nraise RuntimeError("must not execute")\n', encoding="utf-8")
        alphabet = importer.current_script_config(path)
        word = importer.current_script_config(importer.script_path(self.backend, "word"))
        self.assertEqual(alphabet["fit"]["epochs"], 25)
        self.assertEqual(alphabet["fit"]["batch_size"], 1)
        for config in (alphabet, word):
            self.assertEqual(config["split"], {
                "method": "split_three_way", "test_size": 0.2,
                "validation_size": 0.2, "split_random_state": 42,
                "duplicate_disjoint": True,
            })
            self.assertNotIn("validation_split", config["fit"])
            self.assertEqual(config["fit"]["validation_data"], {"expression": "(X_val, y_val)"})
        self.assertEqual(word["fit"]["epochs"], 50)
        self.assertEqual(word["fit"]["batch_size"], 32)
        self.assertEqual(word["compile"]["optimizer"], "adam")
        self.assertEqual(word["compile"]["metrics"], ["accuracy"])

    def test_missing_second_bundle_fails_before_any_tracking(self):
        (self.backend / "data/models/word_scaler.pkl").unlink()
        with self.assertRaises(FileNotFoundError):
            importer.import_baselines(self.client, self.backend, {})
        self.assertEqual(self.client.mock_calls, [])

    def test_split_reader_supports_legacy_calls_and_explicit_helper_seeds(self):
        path = self.backend / "config_fixture.py"
        path.write_text("train_test_split(x, y, test_size=0.2, random_state=7)\n")
        self.assertEqual(importer.current_script_config(path)["split"],
                         {"test_size": 0.2, "random_state": 7})
        for argument in ("seed=7", "7"):
            with self.subTest(argument=argument):
                path.write_text(
                    f"split_three_way(features, labels, {argument})\n"
                    'run.record_model(model, split={"test_size": 0.2, '
                    '"validation_size": 0.2, "split_random_state": 42, '
                    '"duplicate_disjoint": True})\n'
                )
                config = importer.current_script_config(path)["split"]
                self.assertEqual(config["method"], "split_three_way")
                self.assertEqual(config["split_random_state"], 7)
                self.assertEqual(config["validation_size"], 0.2)

    def test_local_default_is_absolute_and_remote_is_optional(self):
        self.assertEqual(importer.tracking_uri(self.backend, {}),
                         (self.backend / "mlruns").as_uri())
        env = {"MLFLOW_TRACKING_URI": "https://dagshub.com/owner/repo.mlflow",
               "MLFLOW_EXPERIMENT_NAME": "Example Baselines",
               "MLFLOW_TRACKING_PASSWORD": "secret-not-to-log"}
        importer.import_baselines(self.client, self.backend, env)
        self.client.set_tracking_uri.assert_called_once_with(env["MLFLOW_TRACKING_URI"])
        self.client.set_experiment.assert_called_once_with("Example Baselines")
        self.assertNotIn("secret-not-to-log", repr(self.logged))
        self.assertNotIn("secret-not-to-log", repr(self.client.mock_calls))

    def test_git_unavailable_is_explicit(self):
        with patch.object(importer.subprocess, "run", side_effect=FileNotFoundError):
            self.assertEqual(importer.git_metadata(self.backend),
                             {"import_commit": None, "import_checkout_dirty": None})

    @unittest.skipUnless(importlib.util.find_spec("mlflow"), "MLflow is not installed")
    def test_real_local_mlflow_round_trip(self):
        import mlflow
        from mlflow.tracking import MlflowClient

        previous_uri = mlflow.get_tracking_uri()
        self.addCleanup(mlflow.set_tracking_uri, previous_uri)
        # Newer local MLflow installs require explicit opt-in for file stores.
        # This affects only this temporary fixture, never production storage.
        environment = patch.dict("os.environ", {"MLFLOW_ALLOW_FILE_STORE": "true"})
        environment.start()
        self.addCleanup(environment.stop)
        runs = importer.import_baselines(mlflow, self.backend, {})
        client = MlflowClient(tracking_uri=importer.tracking_uri(self.backend, {}))
        for name, run_id in runs.items():
            run = client.get_run(run_id)
            self.assertEqual(run.info.status, "FINISHED")
            self.assertEqual(run.data.metrics, {})
            self.assertEqual(run.data.tags["model.kind"], name)
            artifacts = client.list_artifacts(run_id, "baseline")
            self.assertEqual(len(artifacts), 5)
            destination = self.backend / f"download-{name}"
            destination.mkdir()
            manifest_path = client.download_artifacts(
                run_id, "baseline/manifest.json", str(destination)
            )
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            model = next(a for a in manifest["artifacts"] if a["artifact"].endswith(".h5"))
            model_path = client.download_artifacts(
                run_id, f"baseline/{model['artifact']}", str(destination)
            )
            self.assertEqual(importer.sha256(model_path), model["sha256"])


if __name__ == "__main__":
    unittest.main()
