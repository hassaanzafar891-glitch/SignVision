"""Mock GitHub/DagsHub boundaries; never use credentials or publish a real check."""

import base64
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking import trusted_dataset_ci as ci, pr_dataset
from tracking.evaluation_gate import GateError, canonical_sha256


BASE, HEAD, MERGE = "a" * 40, "b" * 40, "c" * 40
APP_ID, CHECK_ID = 1234, 5678  # Test identities only; not configured real accounts.


class FakeGitHub:
    def __init__(self):
        self.repo = {"id": 100, "full_name": ci.REPOSITORY, "default_branch": "main"}
        self.source = {"id": 200, "full_name": "fork-owner/SignVision"}
        self.run = {"id": 300, "workflow_id": 400, "name": ci.SOURCE_NAME, "path": ci.SOURCE_PATH,
            "event": "pull_request", "status": "completed", "repository": self.repo,
            "head_sha": HEAD, "head_repository": self.source, "head_branch": "feature/test",
            "run_attempt": 1, "pull_requests": []}
        self.event = {"action": "completed", "repository": deepcopy(self.repo), "workflow_run": deepcopy(self.run)}
        self.pr = {"number": 500, "state": "open", "changed_files": 1, "merge_commit_sha": MERGE,
            "base": {"ref": "main", "sha": BASE, "repo": self.repo},
            "head": {"sha": HEAD, "ref": "feature/test", "repo": self.source}}
        self.changed = [{"filename": "README.md", "status": "modified"}]
        self.pin_bytes = (ci.ROOT / ci.PIN_PATH).read_bytes()
        self.branch_sha = BASE
        self.branch_protected = True
        self.associated = [{"number": 500}]
        self.merge_parents = [BASE, HEAD]
        self.calls, self.checks = [], {}

    def context(self):
        return ci.verify_origin(self.event, self, BASE)

    def request(self, method, path, body=None):
        self.calls.append((method, path, deepcopy(body)))
        prefix = "/repos/" + ci.REPOSITORY
        if path == prefix:
            return deepcopy(self.repo)
        if path == prefix + "/commits/main":
            return {"sha": self.branch_sha}
        if path == prefix + "/branches/main":
            return {"name": "main", "protected": self.branch_protected, "commit": {"sha": self.branch_sha}}
        if path == prefix + "/actions/workflows/frontend-ci.yml":
            return {"id": 400, "path": ci.SOURCE_PATH, "name": ci.SOURCE_NAME}
        if path == prefix + "/actions/runs/300":
            return deepcopy(self.run)
        if path.startswith(prefix + "/pulls?"):
            return deepcopy(self.associated)
        if path == prefix + "/pulls/500":
            return deepcopy(self.pr)
        if path.startswith(prefix + "/pulls/500/files?"):
            return deepcopy(self.changed)
        if path == prefix + "/git/commits/" + MERGE:
            return {"parents": [{"sha": value} for value in self.merge_parents]}
        if path.startswith(prefix + "/contents/" + ci.PIN_PATH):
            return {"type": "file", "encoding": "base64", "size": len(self.pin_bytes),
                    "content": base64.b64encode(self.pin_bytes).decode()}
        if path == prefix + "/check-runs" and method == "POST":
            result = {**deepcopy(body), "id": CHECK_ID, "app": {"id": APP_ID}, "pull_requests": []}
            self.checks[CHECK_ID] = result
            return deepcopy(result)
        if path == prefix + "/check-runs/" + str(CHECK_ID):
            if method == "PATCH":
                self.checks[CHECK_ID].update(deepcopy(body))
            return deepcopy(self.checks[CHECK_ID])
        raise AssertionError("Unexpected network boundary: " + method + " " + path)


class OriginSecurityTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeGitHub()

    def test_fork_with_empty_association_arrays_is_bound_to_exact_current_head(self):
        context = self.api.context()
        self.assertTrue(context["allowed"])
        self.assertEqual(context["head_sha"], HEAD)
        self.assertEqual(context["head_repository_id"], 200)
        self.assertEqual(context["pr_number"], 500)

    def test_merge_run_requires_both_exact_parents_not_stale_merge_metadata(self):
        self.api.run["head_sha"] = self.api.event["workflow_run"]["head_sha"] = MERGE
        self.assertTrue(self.api.context()["allowed"])
        self.api.merge_parents = [BASE, "d" * 40]
        with self.assertRaises(GateError):
            self.api.context()

    def test_wrong_repository_workflow_event_attempt_and_default_branch_fail(self):
        mutations = [lambda a: a.event["repository"].update(id=999),
            lambda a: a.run.update(workflow_id=999), lambda a: a.run.update(event="workflow_dispatch"),
            lambda a: a.run.update(run_attempt=2), lambda a: a.run.update(path="evil.yml"),
            lambda a: a.repo.update(default_branch="unprotected"), lambda a: a.event.update(action="requested"),
            lambda a: setattr(a, "branch_protected", False),
            lambda a: a.event["workflow_run"]["head_repository"].update(id=999),
            lambda a: a.event["workflow_run"].update(head_branch="another-branch")]
        for mutate in mutations:
            api = FakeGitHub()
            mutate(api)
            with self.subTest(mutate=mutate), self.assertRaises(GateError):
                api.context()

    def test_changed_closed_foreign_and_ambiguous_prs_fail(self):
        mutations = [lambda a: a.pr["head"].update(sha="d" * 40), lambda a: a.pr.update(state="closed"),
            lambda a: a.pr["head"]["repo"].update(id=999), lambda a: a.pr["base"]["repo"].update(id=999),
            lambda a: a.associated.append({"number": 500})]
        for mutate in mutations:
            api = FakeGitHub()
            mutate(api)
            with self.subTest(mutate=mutate), self.assertRaises(GateError):
                api.context()

    def test_changed_trusted_base_or_historical_association_fails(self):
        self.api.branch_sha = "d" * 40
        with self.assertRaises(GateError):
            self.api.context()
        self.api.branch_sha = BASE
        self.api.run["pull_requests"] = [{"number": 500, "head": {"sha": "d" * 40}, "base": {"sha": BASE}}]
        with self.assertRaises(GateError):
            self.api.context()

    def test_protected_changes_and_renames_are_rejected_before_reader_use(self):
        for name in (ci.PIN_PATH, ".dvc/config", ".github/workflows/dataset-validation.yml",
                     "backend/tracking/data_validation.py", "backend/tracking/trusted_dataset_ci.py",
                     "backend/json.py", "urllib/request.py", "backend/urllib/__init__.py",
                     "tracking/__init__.py", "backend/tracking/new_policy.py"):
            for renamed in (False, True):
                self.api.changed = [{"filename": "README.md", "previous_filename": name}] if renamed else [{"filename": name}]
                context = self.api.context()
                self.assertFalse(context["allowed"])
                ci.start_check(context, self.api, APP_ID)
                credentials = Mock(side_effect=AssertionError("Reader credential must not be read"))
                with self.assertRaises(GateError):
                    ci.validate_private(self.api.event, self.api, context, BASE, CHECK_ID, APP_ID, credentials)
                credentials.assert_not_called()

    def test_pin_blob_change_or_truncated_file_list_never_authorizes(self):
        self.api.pin_bytes = self.api.pin_bytes.replace(b"326d", b"0000")
        self.assertFalse(self.api.context()["allowed"])
        self.api.pr["changed_files"] = 2
        with self.assertRaises(GateError):
            self.api.context()

    def test_api_transport_refuses_other_repositories_and_authenticated_redirects(self):
        api = ci.GitHub("test-token")
        with self.assertRaises(GateError):
            api.request("GET", "/repos/attacker/repo")
        self.assertIsNone(ci.NoRedirect().redirect_request(None, None, 302, None, None, "https://evil.example"))


class CredentialAndPublicationTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeGitHub()
        self.context = self.api.context()
        ci.start_check(self.context, self.api, APP_ID)

    def report(self):
        return {"decision": "PASS", "context_sha256": canonical_sha256(self.context),
                "dataset_sha256": pr_dataset.SHA256, "dvc_md5": pr_dataset.DATASET_MD5,
                "bytes": pr_dataset.SIZE, "validated": True}

    def test_actual_validator_reused_only_after_lazy_reader_preconditions(self):
        credentials = Mock(return_value=("reader", "test-reader-token"))
        result = {"sha256": pr_dataset.SHA256, "dvc_md5": pr_dataset.DATASET_MD5,
                  "bytes": pr_dataset.SIZE, "validated": True}
        with patch.object(ci, "download_private") as download, patch.object(pr_dataset, "validate", return_value=result) as validate:
            report = ci.validate_private(self.api.event, self.api, self.context, BASE, CHECK_ID, APP_ID, credentials)
        credentials.assert_called_once()
        download.assert_called_once()
        validate.assert_called_once()
        self.assertEqual(report, self.report())

    def test_stale_head_or_wrong_check_issuer_blocks_before_reader_and_publisher(self):
        credentials = Mock()
        publisher = Mock()
        self.api.pr["head"]["sha"] = "d" * 40
        with self.assertRaises(GateError):
            ci.validate_private(self.api.event, self.api, self.context, BASE, CHECK_ID, APP_ID, credentials)
        with self.assertRaises(GateError):
            ci.finish_check(self.api.event, self.api, publisher, self.context, BASE, CHECK_ID, APP_ID, "success", self.report())
        credentials.assert_not_called()
        publisher.request.assert_not_called()
        self.api.pr["head"]["sha"] = HEAD
        self.api.checks[CHECK_ID]["app"]["id"] = 999
        with self.assertRaises(GateError):
            ci.validate_private(self.api.event, self.api, self.context, BASE, CHECK_ID, APP_ID, credentials)
        credentials.assert_not_called()

    def test_success_publishes_only_exact_sha_app_and_context(self):
        result = ci.finish_check(self.api.event, self.api, self.api, self.context, BASE, CHECK_ID, APP_ID, "success", self.report())
        self.assertEqual(result, "success")
        self.assertEqual(self.api.checks[CHECK_ID]["head_sha"], HEAD)
        self.assertEqual(self.api.checks[CHECK_ID]["app"]["id"], APP_ID)
        self.assertEqual(self.api.checks[CHECK_ID]["conclusion"], "success")

    def test_missing_failed_skipped_or_mismatched_evidence_is_failure(self):
        for outcome, report in (("failure", self.report()), ("skipped", None), ("cancelled", self.report()),
                                ("success", None), ("success", {**self.report(), "context_sha256": "0" * 64})):
            self.api.checks[CHECK_ID].update(status="in_progress", conclusion=None)
            with self.subTest(outcome=outcome):
                self.assertEqual(ci.finish_check(self.api.event, self.api, self.api, self.context, BASE,
                    CHECK_ID, APP_ID, outcome, report), "failure")

    def test_completed_failure_cannot_be_replayed_into_success(self):
        self.api.checks[CHECK_ID].update(status="completed", conclusion="failure")
        publisher = Mock()
        with self.assertRaises(GateError):
            ci.finish_check(self.api.event, self.api, publisher, self.context, BASE,
                            CHECK_ID, APP_ID, "success", self.report())
        publisher.request.assert_not_called()

    def test_changed_run_attempt_and_numeric_check_id_block_reader_use(self):
        credentials = Mock()
        self.api.run["run_attempt"] = 2
        with self.assertRaises(GateError):
            ci.validate_private(self.api.event, self.api, self.context, BASE, CHECK_ID, APP_ID, credentials)
        self.api.run["run_attempt"] = 1
        self.api.checks[CHECK_ID]["id"] = 999
        with self.assertRaises(GateError):
            ci.validate_private(self.api.event, self.api, self.context, BASE, CHECK_ID, APP_ID, credentials)
        credentials.assert_not_called()

    def test_wrong_head_external_id_or_app_cannot_receive_success(self):
        for field, value in (("head_sha", "d" * 40), ("external_id", "forged"), ("name", "forged")):
            original = self.api.checks[CHECK_ID][field]
            self.api.checks[CHECK_ID][field] = value
            publisher = Mock()
            with self.subTest(field=field), self.assertRaises(GateError):
                ci.finish_check(self.api.event, self.api, publisher, self.context, BASE, CHECK_ID, APP_ID, "success", self.report())
            publisher.request.assert_not_called()
            self.api.checks[CHECK_ID][field] = original

    def test_private_retrieval_auth_never_leaks_through_redirect_or_error(self):
        mock_open = Mock(side_effect=urllib.error.HTTPError("https://not-logged", 302, "secret-response", {}, None))
        transport = SimpleNamespace(open=mock_open)
        with patch.object(ci, "opener", return_value=transport), self.assertRaises(GateError) as error:
            ci.download_private(Path("never-created.db"), "reader", "test-secret")
        self.assertNotIn("test-secret", str(error.exception))
        self.assertNotIn("secret-response", str(error.exception))
        request = mock_open.call_args.args[0]
        self.assertTrue(request.full_url.startswith(pr_dataset.REMOTE + "/"))
        self.assertEqual(mock_open.call_count, 1)


@unittest.skipUnless(importlib.util.find_spec("yaml"), "Optional YAML parser unavailable")
class TrustedWorkflowTests(unittest.TestCase):
    def test_trusted_job_boundaries_and_default_branch_only_checkout(self):
        import yaml
        import json
        root = Path(__file__).parents[2]
        workflow = yaml.safe_load((root / ".github/workflows/dataset-validation.yml").read_text())
        self.assertEqual(workflow["on"], {"workflow_run": {"workflows": ["SignVision CI"], "types": ["completed"]}})
        self.assertTrue(all(value == "read" for value in workflow["permissions"].values()))
        for job in workflow["jobs"].values():
            encoded = json.dumps(job)
            self.assertNotIn("download-artifact", encoded)
            self.assertNotIn("upload-artifact", encoded)
            self.assertNotIn("cache:", encoded)
            self.assertNotIn("pip install", encoded)
            self.assertNotIn("continue-on-error", encoded)
            for step in job["steps"]:
                if "uses" in step:
                    self.assertRegex(step["uses"], r"@[0-9a-f]{40}$")
                if step.get("uses", "").startswith("actions/checkout"):
                    self.assertEqual(step["with"]["ref"], "${{ github.sha }}")
                    self.assertFalse(step["with"]["persist-credentials"])
                if "secrets." in json.dumps(step):
                    index = job["steps"].index(step)
                    self.assertTrue(any(" recheck" in previous.get("run", "") for previous in job["steps"][:index]))
        self.assertNotIn("DAGSHUB", json.dumps(workflow["jobs"]["preflight"]))
        self.assertNotIn("DAGSHUB", json.dumps(workflow["jobs"]["start_check"]))
        self.assertNotIn("DAGSHUB", json.dumps(workflow["jobs"]["finish_check"]))
        original = yaml.safe_load((root / ".github/workflows/frontend-ci.yml").read_text())
        self.assertNotIn("pr_dataset", original["jobs"])
        for name in ("backend", "frontend", "pr_model_gate"):
            self.assertNotIn("secrets.", json.dumps(original["jobs"][name]))


if __name__ == "__main__":
    unittest.main()
