"""Exercise the one-command workflow with mocked trainers, never real fits."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking import train_all
from tracking import train_random_forest


class WorkflowTests(unittest.TestCase):
    def test_default_validates_both_tables_without_training(self):
        with patch.object(train_all, "validate_training_database", return_value={"valid": True}) as validate, \
             patch.object(train_all, "train_neural_models") as neural:
            result = train_all.run_workflow()
        self.assertEqual(result["mode"], "validation_only")
        self.assertEqual([call.args[1] for call in validate.call_args_list], list(train_all.TABLES))
        neural.assert_not_called()

    def test_validation_failure_prevents_all_training(self):
        with patch.object(train_all, "validate_training_database", side_effect=ValueError("bad data")), \
             patch.object(train_all, "train_neural_models") as neural:
            with self.assertRaisesRegex(ValueError, "bad data"):
                train_all.run_workflow(train=True)
        neural.assert_not_called()

    def test_opt_in_sequence_uses_mocked_trainers_and_restores_environment(self):
        mlflow = MagicMock()
        mlflow.active_run.return_value = None
        with patch.dict(os.environ, {"MLFLOW_EXPERIMENT_NAME": "original"}), \
             patch.dict(sys.modules, {"mlflow": mlflow}), \
             patch.object(train_all, "validate_training_database", return_value={"valid": True}), \
             patch.object(train_all, "train_neural_models") as neural, \
             patch.object(train_random_forest, "train_task") as forest:
            os.environ.pop("MLFLOW_RUN_ID", None)
            result = train_all.run_workflow(train=True)
            self.assertEqual(os.environ["MLFLOW_EXPERIMENT_NAME"], "original")
        neural.assert_called_once()
        self.assertEqual([call.args[0] for call in forest.call_args_list], ["alphabet", "word"])
        mlflow.tracking.MlflowClient.assert_called_once_with(tracking_uri="http://127.0.0.1:5000")
        self.assertEqual(result["mode"], "training_requested")

    def test_server_failure_prevents_neural_training(self):
        mlflow = MagicMock()
        mlflow.active_run.return_value = None
        mlflow.tracking.MlflowClient.return_value.search_experiments.side_effect = ConnectionError("offline")
        with patch.dict(sys.modules, {"mlflow": mlflow}), \
             patch.object(train_all, "validate_training_database", return_value={"valid": True}), \
             patch.object(train_all, "train_neural_models") as neural:
            with self.assertRaises(ConnectionError):
                train_all.run_workflow(train=True)
        neural.assert_not_called()


if __name__ == "__main__":
    unittest.main()
