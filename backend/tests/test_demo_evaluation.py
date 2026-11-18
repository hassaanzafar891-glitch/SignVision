"""Actual fitted synthetic models; no production artifacts or metric fixtures."""

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
from demo_sandbox import evaluation, evaluation_champion as champion
from tracking.evaluation_gate import GateError, canonical_sha256
from tracking.import_baselines import sha256


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Optional sklearn evaluation dependency unavailable")
class DemoEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        p = patch.object(foundation, "WORKSPACE", self.workspace)
        p.start()
        self.addCleanup(p.stop)
        foundation.initialize()
        drift.create_reference()
        drift.run("shifted")
        def launch(request, timeout):
            return {"error_code": None, "exit_code": 0, "stdout": foundation._encode(fitted.train(request)),
                    "stderr": b"", "output_truncated": False}
        with patch.object(orchestrator, "_launch", side_effect=launch):
            orchestrator.consume(arm=True, job_kind="fitted", cooldown_seconds=0)
        with evaluation._lock():
            champion.prepare()
        self.slot = "job-001"

    def run_case(self, case="challenger"):
        return evaluation.run(case, job_slot=self.slot)["receipt"]

    def test_pass_has_actual_equal_test_rows_metrics_and_receipt_binding(self):
        result = self.run_case()
        self.assertEqual(result["decision"], "PASS")
        self.assertEqual(result["policy"], "no_accuracy_regression_v1")
        self.assertNotEqual(result["reference"]["model_sha256"], result["candidate"]["model_sha256"])
        context = evaluation._context(self.slot)
        for role, key in (("champion", "reference"), ("candidate", "candidate")):
            self.assertEqual(result[key]["metrics"], evaluation._score(context, context[role][0]))
        self.assertEqual(result["test_identity"]["sample_count"], 24)
        self.assertFalse(result["production_evidence"])
        self.assertFalse(result["promotion_performed"])
        self.assertEqual(evaluation.require_receipt(result["receipt_id"], candidate_version=self.slot,
            candidate_model_sha256=result["candidate"]["model_sha256"],
            test_identity_sha256=result["test_identity_sha256"]), result)

    def test_stricter_case_blocks_actual_measured_tie_and_is_not_promotion_receipt(self):
        result = self.run_case("stricter-control")
        self.assertEqual(result["decision"], "BLOCK")
        self.assertEqual(result["test_accuracy_delta"], 0)
        self.assertEqual(result["thresholds"]["minimum_accuracy_delta"], .05)
        self.assertIn("Stricter", result["reason"])
        with self.assertRaises(GateError):
            evaluation.require_receipt(result["receipt_id"], candidate_version=self.slot,
                candidate_model_sha256=result["candidate"]["model_sha256"],
                test_identity_sha256=result["test_identity_sha256"])

    def test_baseline_fits_training_only_and_never_predicts_or_reinitializes(self):
        from sklearn.tree import DecisionTreeClassifier
        # Use another temporary sandbox to observe the first baseline fit.
        with tempfile.TemporaryDirectory() as directory, patch.object(foundation, "WORKSPACE", Path(directory).resolve()):
            foundation.initialize()
            dataset = foundation.read_dataset()
            calls = []
            original = DecisionTreeClassifier.fit
            def fit(model, x, y, *args, **kwargs):
                calls.append(x)
                return original(model, x, y, *args, **kwargs)
            with patch.object(DecisionTreeClassifier, "fit", fit), \
                 patch.object(DecisionTreeClassifier, "predict", side_effect=AssertionError("No baseline scoring")):
                with evaluation._lock():
                    champion.prepare()
                    champion.prepare()
            self.assertEqual(calls, [[dataset["rows"][i]["features"] for i in dataset["partitions"]["train"]]])
            self.assertFalse(champion.load()[1]["test_used"])

    def test_scoring_uses_identical_test_features_and_never_fits(self):
        from sklearn.tree import DecisionTreeClassifier
        context = evaluation._context(self.slot)
        calls = []
        original = DecisionTreeClassifier.predict
        def predict(model, x, *args, **kwargs):
            calls.append(x)
            return original(model, x, *args, **kwargs)
        with patch.object(DecisionTreeClassifier, "predict", predict), \
             patch.object(DecisionTreeClassifier, "fit", side_effect=AssertionError("Evaluation cannot fit")):
            evaluation._result(context, "challenger")
        expected = [context["dataset"]["rows"][i]["features"] for i in context["dataset"]["partitions"]["test"]]
        self.assertEqual(calls, [expected, expected])

    def test_tampered_model_artifacts_fail_before_test_scoring(self):
        paths = [champion.path("champion/model.pkl"), foundation._path("retraining/job-001/model.pkl")]
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(b"tampered")
            with self.subTest(path=path), patch.object(evaluation, "_score") as score, self.assertRaises(GateError):
                self.run_case()
            score.assert_not_called()
            path.write_bytes(original)

    def test_missing_training_receipt_blocks_both_roles(self):
        for path in (champion.path("champion/training_receipt.json"),
                     foundation._path("retraining/job-001/training_receipt.json")):
            original = path.read_bytes()
            path.unlink()  # Only this temporary test workspace.
            with self.subTest(path=path), self.assertRaises(GateError):
                self.run_case()
            path.write_bytes(original)

    def test_modified_frozen_test_data_or_pointer_blocks(self):
        paths = [foundation._path("data/synthetic-v1.json"), foundation._path("active.json"),
                 champion.path("champion_pointer.json")]
        for path in paths:
            original = path.read_bytes()
            value = json.loads(original)
            if path.name == "synthetic-v1.json":
                value["partitions"]["test"][0] = value["partitions"]["train"][0]
            else:
                value["model_sha256"] = "0" * 64
            path.write_text(json.dumps(value))
            with self.subTest(path=path), self.assertRaises(GateError):
                self.run_case()
            path.write_bytes(original)

    def test_tampered_test_identity_even_with_recomputed_checksums_blocks(self):
        self.run_case()
        path = evaluation._path("journal.json")
        journal = json.loads(path.read_bytes())
        result = journal["receipts"][0]
        result["test_identity"]["test_indices_sha256"] = "0" * 64
        result["test_identity_sha256"] = canonical_sha256(result["test_identity"])
        result["receipt_id"] = canonical_sha256({k: v for k, v in result.items()
            if k not in ("receipt_id", "evaluated_at", "receipt_sha256")})
        result["receipt_sha256"] = canonical_sha256({k: v for k, v in result.items() if k != "receipt_sha256"})
        path.write_text(json.dumps(drift._seal_journal(journal)))
        with self.assertRaises(GateError):
            evaluation.inspect(job_slot=self.slot)

    def test_missing_evaluation_receipt_or_wrong_target_cannot_allow_future_promotion(self):
        with self.assertRaises(GateError):
            evaluation.require_receipt("a" * 64, candidate_version=self.slot,
                candidate_model_sha256="b" * 64, test_identity_sha256="c" * 64)
        result = self.run_case()
        with self.assertRaises(GateError):
            evaluation.require_receipt(result["receipt_id"], candidate_version=self.slot,
                candidate_model_sha256="b" * 64, test_identity_sha256=result["test_identity_sha256"])

    def test_journal_bounded_deduplicated_read_only_inspection_and_immutable_pointers(self):
        pointers = [foundation._path("active.json"), champion.path("champion_pointer.json")]
        before = [sha256(p) for p in pointers]
        first = self.run_case()
        self.run_case("stricter-control")
        journal_hash = sha256(evaluation._path("journal.json"))
        duplicate = evaluation.run(job_slot=self.slot)
        self.assertTrue(duplicate["deduplicated"])
        self.assertEqual(duplicate["receipt"], first)
        self.assertEqual(len(evaluation.inspect(job_slot=self.slot)["receipts"]), 2)
        self.assertEqual(journal_hash, sha256(evaluation._path("journal.json")))
        self.assertEqual(before, [sha256(p) for p in pointers])
        self.assertLessEqual(evaluation._path("journal.json").stat().st_size, 65536)

    def test_invalid_case_or_job_path_does_not_write_evaluation(self):
        for values in ({"case": "regression-control"}, {"job_slot": "../../backend/data/models"}):
            with self.subTest(values=values), self.assertRaises(GateError):
                evaluation.run(**values)
        self.assertFalse(evaluation._path("journal.json").exists())

    def test_cli_block_exits_nonzero_and_inspect_succeeds(self):
        with patch.object(sys, "argv", ["evaluation", "run", "--job-slot", self.slot, "--case", "stricter-control"]), \
             redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
            evaluation.main()
        self.assertEqual(raised.exception.code, 1)
        with patch.object(sys, "argv", ["evaluation", "inspect", "--job-slot", self.slot]), redirect_stdout(io.StringIO()):
            evaluation.main()


if __name__ == "__main__":
    unittest.main()
