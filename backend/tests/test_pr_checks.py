"""PR contracts: fixture unit tests are separate from the actual-dataset CI job."""

from contextlib import closing, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking import pr_dataset, pr_gate
from tracking.data_validation import FEATURE_COLUMNS, TABLES
from tracking.evaluation_gate import GateError


class PRDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database = self.root / "fixture.db"
        with closing(sqlite3.connect(self.database)) as connection, connection:
            for table in TABLES:
                connection.execute(f"CREATE TABLE {table} (id, {','.join(FEATURE_COLUMNS)}, label)")
                for i in range(18):
                    row = [i, *([float(i)] * 42), "label" + str(i % 2)]
                    connection.execute(f"INSERT INTO {table} VALUES ({','.join('?' for _ in row)})", row)
        self.pointer = self.root / "fixture.db.dvc"

    def policy(self):
        raw = self.database.read_bytes()
        md5 = hashlib.md5(raw).hexdigest()
        self.pointer.write_text(f"outs:\n- md5: {md5}\n  size: {len(raw)}\n  hash: md5\n  path: main_dataset.db\n")
        return patch.multiple(pr_dataset, SIZE=len(raw), SHA256=hashlib.sha256(raw).hexdigest(), DATASET_MD5=md5)

    def test_read_only_validates_both_tables_and_identity(self):
        before = self.database.read_bytes()
        with self.policy(), redirect_stdout(io.StringIO()):
            result = pr_dataset.validate(self.database, pointer=self.pointer)
        self.assertTrue(result["validated"])
        self.assertEqual(len(result["tables"]), 2)
        self.assertEqual(before, self.database.read_bytes())

    def test_missing_corrupt_wrong_identity_and_sidecars_fail_closed(self):
        with self.policy():
            with self.assertRaises(GateError):
                pr_dataset.validate(self.root / "missing", pointer=self.pointer)
            self.database.write_bytes(b"invalid data")
            with self.assertRaises(GateError):
                pr_dataset.validate(self.database, pointer=self.pointer)
        with self.policy():
            Path(str(self.database) + "-wal").write_bytes(b"changes")
            with self.assertRaises(GateError):
                pr_dataset.validate(self.database, pointer=self.pointer)

    def test_invalid_schema_labels_and_features_are_rejected_after_identity_check(self):
        for statement in ("ALTER TABLE alphabetDataset RENAME COLUMN x1 TO missing_x1",
                          "UPDATE wordDataset SET label=NULL", "UPDATE wordDataset SET x1='bad'"):
            original = self.database.read_bytes()
            with closing(sqlite3.connect(self.database)) as connection, connection:
                connection.execute(statement)
            with self.policy(), self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                pr_dataset.validate(self.database, pointer=self.pointer)
            self.database.write_bytes(original)

    def test_changed_or_duplicate_dvc_pointer_is_not_implicitly_approved(self):
        with self.policy():
            text = self.pointer.read_text()
            self.pointer.write_text(text.replace("main_dataset.db", "another.db"))
            with self.assertRaises(GateError):
                pr_dataset.verify_pointer(self.pointer)
            self.pointer.write_text(text + "  md5: " + "0" * 32)
            with self.assertRaises(GateError):
                pr_dataset.verify_pointer(self.pointer)

    def test_anonymous_fetch_does_not_use_auth_and_verifies_actual_bytes(self):
        data = self.database.read_bytes()
        opener = SimpleNamespace(open=lambda url, timeout: io.BytesIO(data))
        destination = self.root / "download.db"
        with self.policy(), patch.object(pr_dataset.urllib.request, "build_opener", return_value=opener) as build, \
             redirect_stdout(io.StringIO()):
            result = pr_dataset.fetch(destination, pointer=self.pointer)
        self.assertTrue(result["validated"])
        self.assertIsInstance(build.call_args.args[0], pr_dataset.urllib.request.ProxyHandler)
        self.assertEqual(build.call_args.args[0].proxies, {})
        self.assertEqual(destination.read_bytes(), data)
        with self.policy(), self.assertRaises(GateError):
            pr_dataset.fetch(destination, pointer=self.pointer)

    def test_unauthorized_remote_fails_without_credentials_or_fixture_fallback(self):
        opener = SimpleNamespace(open=unittest.mock.Mock(side_effect=urllib.error.HTTPError("fixed-public-url", 401, "unauthorized", {}, None)))
        with self.policy(), patch.object(pr_dataset.urllib.request, "build_opener", return_value=opener), \
             patch.object(pr_dataset, "validate") as validate, self.assertRaisesRegex(GateError, "HTTP 401"):
            pr_dataset.fetch(self.root / "download.db", pointer=self.pointer)
        validate.assert_not_called()
        self.assertEqual(opener.open.call_count, 1)
        self.assertFalse((self.root / "download.db").exists())

    def test_bad_download_is_removed_without_touching_existing_database(self):
        before = self.database.read_bytes()
        opener = SimpleNamespace(open=lambda url, timeout: io.BytesIO(b"malformed"))
        destination = self.root / "download.db"
        with self.policy(), patch.object(pr_dataset.urllib.request, "build_opener", return_value=opener), self.assertRaises(GateError):
            pr_dataset.fetch(destination, pointer=self.pointer)
        self.assertFalse(destination.exists())
        self.assertEqual(self.database.read_bytes(), before)


class PRProductionGuardTests(unittest.TestCase):
    def test_production_artifact_changes_block_before_any_synthetic_fitting(self):
        with patch.object(pr_gate, "production_changes", return_value=["backend/data/models/alphabet_model.h5"]), \
             patch("demo_sandbox.foundation.initialize") as initialize, self.assertRaisesRegex(GateError, "Production.*BLOCKED"):
            pr_gate.assess(base_sha="a" * 40)
        initialize.assert_not_called()

    def test_git_comparison_uses_base_history_and_fixed_protected_directory(self):
        with patch.object(pr_gate.subprocess, "run", return_value=SimpleNamespace(stdout=b"backend/data/models/word_scaler.pkl\0")) as run:
            self.assertEqual(pr_gate.production_changes("a" * 40), ["backend/data/models/word_scaler.pkl"])
        self.assertEqual(run.call_args.args[0][-2:], ["--", "backend/data/models"])
        with self.assertRaises(GateError):
            pr_gate.production_changes("--unsafe")

    def test_cli_failed_guard_returns_nonzero_and_never_approves_production(self):
        output = io.StringIO()
        with patch.object(pr_gate, "production_changes", return_value=["backend/data/models/new.pkl"]), \
             patch.object(sys, "argv", ["pr_gate", "--base-sha", "a" * 40]), \
             redirect_stdout(output), self.assertRaises(SystemExit) as error:
            pr_gate.main()
        self.assertEqual(error.exception.code, 1)
        report = json.loads(output.getvalue())
        self.assertEqual(report["decision"], "BLOCK")
        self.assertFalse(report["production_approval"])


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Optional CPU sandbox dependency unavailable")
class PRFittedGateTests(unittest.TestCase):
    def launch(self, request, timeout):
        from demo_sandbox import foundation, _fitted_job
        return {"error_code": None, "exit_code": 0, "stdout": foundation._encode(_fitted_job.train(request)),
                "stderr": b"", "output_truncated": False}

    def test_actual_normal_and_stricter_control_receipts_are_reproduced(self):
        report = pr_gate.assess()
        self.assertEqual(report["normal_case"]["decision"], "PASS")
        self.assertEqual(report["negative_control"]["decision"], "BLOCK")
        self.assertFalse(report["production_approval"])
        self.assertEqual(report["normal_case"]["test_identity"], report["negative_control"]["test_identity"])

    def test_genuinely_fitted_weaker_model_fails_required_gate(self):
        from demo_sandbox import _fitted_job, orchestrator
        weak = {**_fitted_job.PARAMETERS, "min_samples_leaf": 20}
        with patch.object(_fitted_job, "PARAMETERS", weak), patch.object(orchestrator, "_launch", side_effect=self.launch), \
             self.assertRaisesRegex(GateError, "Candidate accuracy regression"):
            pr_gate.assess()

    def test_tampered_evaluation_evidence_fails_required_gate(self):
        from demo_sandbox import evaluation, orchestrator
        original = evaluation.run
        def corrupt(case="challenger", **kwargs):
            result = original(case, **kwargs)
            if case == "challenger":
                path = evaluation._path("journal.json")
                value = json.loads(path.read_text())
                value["receipts"][0]["test_identity_sha256"] = "0" * 64
                path.write_text(json.dumps(value))
            return result
        with patch.object(evaluation, "run", side_effect=corrupt), patch.object(orchestrator, "_launch", side_effect=self.launch), \
             self.assertRaises(GateError):
            pr_gate.assess()


@unittest.skipUnless(importlib.util.find_spec("yaml"), "Optional YAML parser unavailable in stdlib job")
class PRWorkflowTests(unittest.TestCase):
    def test_pr_jobs_are_automatic_secret_free_and_not_allowed_to_ignore_failures(self):
        import yaml
        workflow = yaml.safe_load((Path(__file__).parents[2] / ".github/workflows/frontend-ci.yml").read_text())
        self.assertIn("pull_request", workflow["on"])
        self.assertNotIn("pull_request_target", workflow["on"])
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        for name in ("pr_model_gate",):
            job = workflow["jobs"][name]
            self.assertEqual(job["if"], "github.event_name == 'pull_request'")
            encoded = json.dumps(job)
            self.assertNotIn("secrets.", encoded)
            self.assertNotIn("continue-on-error", encoded)
            self.assertNotIn("candidate-approval", encoded)
            self.assertTrue(all(step.get("with", {}).get("persist-credentials") is False
                for step in job["steps"] if step.get("uses", "").startswith("actions/checkout")))
        jobs = workflow["jobs"]
        self.assertNotIn("pr_dataset", jobs)  # Private data moved to trusted workflow_run.
        self.assertIn("tracking.pr_gate --base-sha", json.dumps(jobs["pr_model_gate"]))
        self.assertIn("requirements-ci-light.txt", json.dumps(jobs["pr_model_gate"]))
        self.assertIn("workflow_dispatch", jobs["candidate_approval"]["if"])
        self.assertIn("unittest discover", json.dumps(jobs["backend"]))
        self.assertIn("npm ci", json.dumps(jobs["frontend"]))


if __name__ == "__main__":
    unittest.main()
