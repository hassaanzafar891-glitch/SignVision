"""Actual small sklearn fits in isolated temporary workspaces; never PSL training."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_sandbox import foundation, drift, orchestrator, _fitted_job as fitted
from tracking.evaluation_gate import GateError, canonical_sha256
from tracking.import_baselines import sha256


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Optional sklearn training dependency unavailable")
class FittedSandboxTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        workspace_patch = patch.object(foundation, "WORKSPACE", self.workspace)
        workspace_patch.start()
        self.addCleanup(workspace_patch.stop)
        self.protected = {}
        for name in ("backend/data/models/alphabet_model.h5", "backend/data/db/main_dataset.db",
                     "mlflow.db", "docker-compose.yml"):
            path = self.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"protected production sentinel")
            self.protected[path] = sha256(path)
        foundation.initialize()
        drift.create_reference()
        drift.run("shifted", batch_size=25)
        self.request = fitted.request_for("a" * 64, "b" * 64)

    def result(self, request, timeout):
        return {"error_code": None, "exit_code": 0, "stdout": foundation._encode(fitted.train(request)),
                "stderr": b"", "output_truncated": False}

    def test_actual_fit_receives_only_training_rows_and_scores_only_validation(self):
        from sklearn.tree import DecisionTreeClassifier
        original_fit, original_predict = DecisionTreeClassifier.fit, DecisionTreeClassifier.predict
        seen_fit, seen_predict = [], []
        def fit(model, x, y, *args, **kwargs):
            seen_fit.append((x, y))
            return original_fit(model, x, y, *args, **kwargs)
        def predict(model, x, *args, **kwargs):
            seen_predict.append(x)
            return original_predict(model, x, *args, **kwargs)
        with patch.object(DecisionTreeClassifier, "fit", fit), patch.object(DecisionTreeClassifier, "predict", predict):
            payload = fitted.train(self.request)
        self.assertEqual(len(seen_fit), 1)
        self.assertEqual(seen_fit[0][0], [r["features"] for r in self.request["train_rows"]])
        self.assertEqual(seen_predict, [[r["features"] for r in self.request["validation_rows"]]])
        data = foundation.read_dataset()
        test = {tuple(data["rows"][i]["features"]) for i in data["partitions"]["test"]}
        self.assertFalse(test & {tuple(x) for x in seen_fit[0][0]})
        self.assertFalse(test & {tuple(x) for x in seen_predict[0]})
        self.assertFalse(payload["receipt"]["test_used"])
        self.assertEqual(payload["receipt"]["training_samples"], 24)
        self.assertEqual(payload["receipt"]["metrics"]["validation_samples"], 24)

    def test_actual_subprocess_artifact_loadable_receipt_and_duplicate_safe(self):
        result = orchestrator.consume(arm=True, job_kind="fitted", timeout_seconds=30, cooldown_seconds=0)
        self.assertEqual(result["status"], "FINISHED")
        self.assertTrue(result["real_model_training"])
        self.assertFalse(result["workflow_simulation"])
        job = result["trigger"]
        root = self.workspace / ".signvision-demo/retraining" / job["slot"]
        receipt = json.loads((root / "training_receipt.json").read_text())
        self.assertEqual(receipt["model_sha256"], sha256(root / "model.pkl"))
        with (root / "model.pkl").open("rb") as stream:
            model = pickle.load(stream)  # Only this test's fixed worker output.
        self.assertTrue(hasattr(model, "tree_"))
        self.assertEqual(model.n_features_in_, 42)
        self.assertEqual(receipt["status"], "TRAINED")
        self.assertEqual(receipt["dataset_sha256"], canonical_sha256(foundation.read_dataset()))
        with patch.object(orchestrator, "_launch") as launch:
            self.assertEqual(orchestrator.consume(arm=True, job_kind="fitted")["status"], "no_unprocessed_alert")
            orchestrator.inspect()
        launch.assert_not_called()
        self.assertEqual(self.protected, {p: sha256(p) for p in self.protected})
        self.assertEqual(foundation.status()["active_version"], foundation.CHAMPION)

    def test_validation_and_test_rows_cannot_be_injected_into_fit_request(self):
        data = foundation.read_dataset()
        for name in ("validation", "test"):
            request = deepcopy(self.request)
            index = data["partitions"][name][0]
            request["train_indices"][0] = index
            request["train_rows"][0] = data["rows"][index]
            with self.subTest(name=name), self.assertRaises(GateError):
                fitted.train(request)

    def test_unarmed_fitted_mode_does_not_fit_or_write(self):
        root = self.workspace / ".signvision-demo"
        before = {p: sha256(p) for p in root.rglob("*") if p.is_file()}
        with patch.object(orchestrator, "_launch") as launch:
            self.assertEqual(orchestrator.consume(job_kind="fitted")["status"], "unarmed")
        launch.assert_not_called()
        self.assertEqual(before, {p: sha256(p) for p in root.rglob("*") if p.is_file()})

    def test_failed_fit_creates_no_success_receipt_or_model_and_no_retry(self):
        from sklearn.tree import DecisionTreeClassifier
        with patch.object(DecisionTreeClassifier, "fit", side_effect=OSError("fit failed")), \
             patch.object(orchestrator, "_launch", side_effect=self.result):
            result = orchestrator.consume(arm=True, job_kind="fitted")
        self.assertEqual(result["status"], "FAILED")
        root = self.workspace / ".signvision-demo/retraining" / result["trigger"]["slot"]
        self.assertFalse((root / "model.pkl").exists())
        self.assertFalse((root / "training_receipt.json").exists())
        with patch.object(orchestrator, "_launch") as launch:
            self.assertEqual(orchestrator.consume(arm=True, job_kind="fitted")["status"], "no_unprocessed_alert")
        launch.assert_not_called()

    def test_invalid_metrics_from_child_cannot_issue_receipt(self):
        def invalid(request, timeout):
            result = self.result(request, timeout)
            payload = json.loads(result["stdout"])
            payload["receipt"]["metrics"]["validation_accuracy"] = -1
            result["stdout"] = foundation._encode(payload)
            return result
        with patch.object(orchestrator, "_launch", side_effect=invalid):
            result = orchestrator.consume(arm=True, job_kind="fitted")
        self.assertEqual(result["status"], "FAILED")
        self.assertNotIn("training_receipt.json", result["trigger"]["artifacts"])

    def test_fixed_seed_repeats_fitted_model_and_measured_metrics(self):
        first, second = fitted.train(self.request), fitted.train(self.request)
        self.assertEqual(first["model_base64"], second["model_base64"])
        self.assertEqual(first["receipt"]["metrics"], second["receipt"]["metrics"])
        fitted.validate(first, self.request)

    def test_fitted_and_simulation_modes_share_consumed_alert_ids(self):
        orchestrator.consume(arm=True, cooldown_seconds=0)
        with patch.object(orchestrator, "_launch") as launch:
            self.assertEqual(orchestrator.consume(arm=True, job_kind="fitted")["status"], "no_unprocessed_alert")
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
