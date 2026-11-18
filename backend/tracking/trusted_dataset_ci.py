"""Trusted-default-branch private dataset CI. Never checks out or executes PR code."""

import argparse
import base64
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

from tracking import pr_dataset
from tracking.evaluation_gate import GateError, canonical_sha256, read_json, require


REPOSITORY = "hassaanzafar891-glitch/SignVision"
DEFAULT_BRANCH = "main"
SOURCE_PATH = ".github/workflows/frontend-ci.yml"
SOURCE_NAME = "SignVision CI"
CHECK_NAME = "Trusted actual DVC dataset validation"
PIN_PATH = "backend/data/db/main_dataset.db.dvc"
ROOT = Path(__file__).resolve().parents[2]
PROTECTED_FILES = {PIN_PATH, "backend/data/db/main_dataset.db", "backend/__init__.py",
    "backend/tracking/__init__.py", "backend/tracking/trusted_dataset_ci.py",
    "backend/tracking/pr_dataset.py", "backend/tracking/data_validation.py",
    "backend/tracking/evaluation_gate.py", "backend/tracking/import_baselines.py",
    "sitecustomize.py", "usercustomize.py", "backend/sitecustomize.py", "backend/usercustomize.py"}


def positive(value):
    require(type(value) is int and value > 0, "Invalid GitHub numeric identity")
    return value


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value), "Invalid GitHub commit SHA")
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


class GitHub:
    def __init__(self, token):
        require(isinstance(token, str) and token and not any(c.isspace() for c in token), "GitHub token unavailable")
        self.token = token

    def request(self, method, path, body=None):
        require(path.startswith("/repos/" + REPOSITORY + "/") or path == "/repos/" + REPOSITORY,
                "External GitHub API destination forbidden")
        # Only the approved base repository is queried; fork pin bytes use the base's commit lookup.
        payload = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request("https://api.github.com" + path, data=payload, method=method,
            headers={"Authorization": "Bearer " + self.token, "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json"})
        try:
            with opener().open(request, timeout=30) as response:
                raw = response.read(2_000_001)
            require(len(raw) <= 2_000_000, "GitHub metadata response too large")
            return json.loads(raw)
        except urllib.error.HTTPError as error:
            code = error.code
            error.close()
            raise GateError(f"GitHub API request failed (HTTP {code})") from None
        except (urllib.error.URLError, ValueError):
            raise GateError("GitHub API metadata unavailable") from None


def repo_matches(value, repo_id):
    return isinstance(value, dict) and value.get("full_name") == REPOSITORY and value.get("id") == repo_id


def sensitive(name):
    require(isinstance(name, str) and len(name) <= 1024 and "\\" not in name
            and all(part not in ("", ".", "..") for part in name.split("/")), "Invalid changed-file path")
    name = name.lower()
    parts = name.split("/")
    module = parts[1] if len(parts) > 1 and parts[0] == "backend" else parts[0]
    # A future merge must not smuggle a new package/module that shadows a
    # trusted validator import on Python's cwd/PYTHONPATH search paths.
    shadows_stdlib = module.split(".")[0] in sys.stdlib_module_names
    startup_module = name.endswith(".py") and (len(parts) == 1 or (len(parts) == 2 and parts[0] == "backend"))
    return (name.startswith((".github/", ".dvc/", "backend/tracking/", "tracking/"))
            or name in PROTECTED_FILES or shadows_stdlib or startup_module)


def verify_origin(event, api, trusted_sha):
    """No DagsHub credential/App private key is read by this preflight."""
    sha(trusted_sha)
    require(event.get("action") == "completed", "Unexpected workflow_run activity")
    repo = api.request("GET", "/repos/" + REPOSITORY)
    repo_id = positive(repo["id"])
    require(repo_matches(event.get("repository"), repo_id) and repo.get("default_branch") == DEFAULT_BRANCH,
            "Repository/default-branch mismatch")
    current = api.request("GET", f"/repos/{REPOSITORY}/commits/{DEFAULT_BRANCH}")
    require(current["sha"] == trusted_sha, "Trusted default branch advanced; rerun source CI")
    branch = api.request("GET", f"/repos/{REPOSITORY}/branches/{DEFAULT_BRANCH}")
    require(branch.get("name") == DEFAULT_BRANCH and branch.get("protected") is True
            and branch.get("commit", {}).get("sha") == trusted_sha,
            "Default branch is not protected or its commit changed")
    workflow = api.request("GET", f"/repos/{REPOSITORY}/actions/workflows/frontend-ci.yml")
    require(workflow.get("path") == SOURCE_PATH and workflow.get("name") == SOURCE_NAME,
            "Unapproved originating workflow")
    submitted = event["workflow_run"]
    require(repo_matches(submitted.get("repository"), repo_id), "Event run repository mismatch")
    run_id = positive(submitted["id"])
    run = api.request("GET", f"/repos/{REPOSITORY}/actions/runs/{run_id}")
    require(run.get("id") == run_id and run.get("workflow_id") == positive(workflow["id"])
            and submitted.get("workflow_id") == workflow["id"] and run.get("path") == SOURCE_PATH
            and run.get("name") == SOURCE_NAME and run.get("event") == "pull_request"
            and run.get("status") == "completed" and repo_matches(run.get("repository"), repo_id),
            "Originating workflow run is not an approved PR run")
    run_sha = sha(run["head_sha"])
    require(submitted.get("head_sha") == run_sha and submitted.get("run_attempt") == positive(run["run_attempt"]),
            "Originating run/attempt mismatch")
    require(positive(run["head_repository"]["id"]) == submitted.get("head_repository", {}).get("id")
            and run["head_repository"].get("full_name") == submitted.get("head_repository", {}).get("full_name")
            and run.get("head_branch") == submitted.get("head_branch"), "Originating fork/branch identity changed")
    # Fork payloads can omit pull_requests. Discover through the base PR API,
    # never trust a PR-uploaded artifact or assume a check's pull_requests array is populated.
    fork_name = run["head_repository"]["full_name"]
    require(isinstance(fork_name, str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", fork_name), "Invalid source repository")
    query = urllib.parse.urlencode({"state": "open", "base": DEFAULT_BRANCH,
        "head": fork_name.split("/")[0] + ":" + run["head_branch"], "per_page": 100})
    associated = api.request("GET", f"/repos/{REPOSITORY}/pulls?{query}")
    require(isinstance(associated, list) and len(associated) < 100, "Ambiguous/truncated PR association")
    matches = []
    for item in associated:
        number = positive(item["number"])
        pr = api.request("GET", f"/repos/{REPOSITORY}/pulls/{number}")
        if pr.get("state") != "open" or pr.get("base", {}).get("ref") != DEFAULT_BRANCH:
            continue
        if run_sha not in (pr.get("head", {}).get("sha"), pr.get("merge_commit_sha")):
            continue
        if pr.get("head", {}).get("repo", {}).get("id") != run.get("head_repository", {}).get("id"):
            continue
        require(pr["number"] == number and repo_matches(pr["base"]["repo"], repo_id), "PR base repository mismatch")
        require(pr["head"]["ref"] == run["head_branch"]
                and pr["head"]["repo"].get("full_name") == fork_name, "PR source repository/branch mismatch")
        head_sha = sha(pr["head"]["sha"])
        require(pr["base"]["sha"] == trusted_sha, "PR base no longer matches trusted default branch")
        if run_sha != head_sha:
            merge = api.request("GET", f"/repos/{REPOSITORY}/git/commits/{run_sha}")
            require([parent.get("sha") for parent in merge.get("parents", [])] == [trusted_sha, head_sha],
                    "Source merge does not bind current PR head and base")
        # If GitHub supplies a run-time association, require its exact historical head/base too.
        refs = run.get("pull_requests", [])
        if refs:
            require(any(ref.get("number") == number and ref.get("head", {}).get("sha") == head_sha
                        and ref.get("base", {}).get("sha") == trusted_sha for ref in refs), "Stale PR run association")
        matches.append((pr, head_sha))
    require(len(matches) == 1, "No unique current PR head associated with originating run")
    pr, head_sha = matches[0]
    context = {"repository": REPOSITORY, "repository_id": repo_id, "workflow_id": workflow["id"],
               "run_id": run_id, "run_attempt": run["run_attempt"], "run_head_sha": run_sha,
               "pr_number": pr["number"], "head_sha": head_sha, "head_repository_id": pr["head"]["repo"]["id"],
               "base_sha": trusted_sha, "trusted_sha": trusted_sha,
               "dataset_sha256": pr_dataset.SHA256, "allowed": True, "reason": "approved_pin_and_inputs"}
    pr_dataset.verify_pointer()
    count = pr.get("changed_files")
    require(type(count) is int and 0 <= count <= 1000, "Changed-file list too large or unavailable")
    changed = []
    for page in range(1, (count + 99) // 100 + 1):
        batch = api.request("GET", f"/repos/{REPOSITORY}/pulls/{pr['number']}/files?per_page=100&page={page}")
        require(isinstance(batch, list), "Invalid PR file list")
        changed.extend(batch)
    require(len(changed) == count, "Truncated PR file list")
    for item in changed:
        if sensitive(item["filename"]) or (item.get("previous_filename") and sensitive(item["previous_filename"])):
            return {**context, "allowed": False, "reason": "protected_validation_inputs_changed"}
    pin = api.request("GET", f"/repos/{REPOSITORY}/contents/{PIN_PATH}?ref={head_sha}")
    require(pin.get("type") == "file" and pin.get("encoding") == "base64"
            and type(pin.get("size")) is int and pin["size"] <= 4096, "Invalid PR dataset pin blob")
    try:
        raw = base64.b64decode(pin["content"].replace("\n", ""), validate=True)
    except (ValueError, TypeError):
        raise GateError("Invalid encoded PR dataset pin") from None
    if raw != (ROOT / PIN_PATH).read_bytes() or len(raw) != pin["size"]:
        return {**context, "allowed": False, "reason": "unapproved_dataset_pin"}
    return context


def recheck(event, api, context, trusted_sha):
    fresh = verify_origin(event, api, trusted_sha)
    require(canonical_sha256(fresh) == canonical_sha256(context), "Stale or mismatched validation context; no credential use or success permitted")
    return fresh


def external_id(context):
    return f"dataset:{context['run_id']}:{context['run_attempt']}:{context['pr_number']}:{context['trusted_sha']}"


def verify_check(check, context, app_id):
    require(check.get("name") == CHECK_NAME and check.get("head_sha") == context["head_sha"]
            and check.get("external_id") == external_id(context)
            and check.get("app", {}).get("id") == positive(app_id), "Check identity/issuer mismatch")


def start_check(context, publisher, app_id):
    body = {"name": CHECK_NAME, "head_sha": context["head_sha"], "external_id": external_id(context),
            "status": "in_progress", "output": {"title": "Actual private dataset validation", "summary": "Trusted validation pending."}}
    if not context["allowed"]:
        body.update(status="completed", conclusion="failure")
        body["output"]["summary"] = "Rejected: " + context["reason"]
    result = publisher.request("POST", f"/repos/{REPOSITORY}/check-runs", body)
    verify_check(result, context, app_id)
    return positive(result["id"])


def download_private(destination, username, token):
    require(isinstance(username, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", username), "Reader username unavailable")
    require(isinstance(token, str) and 0 < len(token) <= 4096 and not any(c.isspace() for c in token), "Reader token unavailable")
    authorization = "Basic " + base64.b64encode((username + ":" + token).encode()).decode()
    suffix = pr_dataset.DATASET_MD5[:2] + "/" + pr_dataset.DATASET_MD5[2:]
    for prefix in ("files/md5/", ""):
        request = urllib.request.Request(pr_dataset.REMOTE + "/" + prefix + suffix, headers={"Authorization": authorization})
        try:
            # Auth headers are NEVER forwarded to a redirect host. Unknown layouts fail closed.
            response = opener().open(request, timeout=30)
        except urllib.error.HTTPError as error:
            code = error.code
            error.close()
            if code == 404 and prefix:
                continue
            raise GateError(f"Private dataset retrieval failed (HTTP {code}); redirects/auth failures require operator review") from None
        except urllib.error.URLError:
            raise GateError("Private dataset retrieval unavailable") from None
        with response, Path(destination).open("xb") as stream:
            total = 0
            while block := response.read(min(65536, pr_dataset.SIZE + 1 - total)):
                total += len(block)
                require(total <= pr_dataset.SIZE, "Private dataset size exceeds approved bound")
                stream.write(block)
        return
    raise GateError("Private approved dataset object unavailable")


def validate_private(event, api, context, trusted_sha, check_id, app_id, credentials):
    recheck(event, api, context, trusted_sha)
    require(context["allowed"], "Unauthorized validation inputs; reader credentials must remain unused")
    check = api.request("GET", f"/repos/{REPOSITORY}/check-runs/{positive(check_id)}")
    require(check.get("id") == check_id, "Validation check numeric identity mismatch")
    verify_check(check, context, app_id)
    require(check.get("status") == "in_progress", "Validation check is not pending")
    # Lazy getter: called only AFTER all identity, pin and check-issuer validation.
    username, token = credentials()
    with tempfile.TemporaryDirectory(prefix="signvision-private-validation-") as directory:
        destination = Path(directory) / "main_dataset.db"
        download_private(destination, username, token)
        with redirect_stdout(io.StringIO()):
            result = pr_dataset.validate(destination)
    return {"decision": "PASS", "context_sha256": canonical_sha256(context),
            "dataset_sha256": result["sha256"], "dvc_md5": result["dvc_md5"],
            "bytes": result["bytes"], "validated": result["validated"]}


def finish_check(event, api, publisher, context, trusted_sha, check_id, app_id, job_result, report):
    recheck(event, api, context, trusted_sha)
    check = api.request("GET", f"/repos/{REPOSITORY}/check-runs/{positive(check_id)}")
    require(check.get("id") == check_id, "Publication check numeric identity mismatch")
    verify_check(check, context, app_id)
    if not context["allowed"]:
        require(check.get("status") == "completed" and check.get("conclusion") == "failure",
                "Rejected inputs must retain a failed check")
        return "failure"
    require(check.get("status") == "in_progress" and check.get("conclusion") is None,
            "Completed validation checks cannot be replayed or upgraded")
    expected = {"decision": "PASS", "context_sha256": canonical_sha256(context),
                "dataset_sha256": pr_dataset.SHA256, "dvc_md5": pr_dataset.DATASET_MD5,
                "bytes": pr_dataset.SIZE, "validated": True}
    success = context["allowed"] and job_result == "success" and canonical_sha256(report) == canonical_sha256(expected)
    body = {"status": "completed", "conclusion": "success" if success else "failure",
            "output": {"title": "Actual DVC dataset " + ("validated" if success else "BLOCKED"),
                       "summary": ("Identity, integrity, schema and quality passed using trusted code. "
                           f"PR #{context['pr_number']}, head `{context['head_sha']}`; "
                           f"validator `{trusted_sha}`; dataset MD5 `{pr_dataset.DATASET_MD5}`, "
                           f"SHA-256 `{pr_dataset.SHA256}`. No production model approval.")
                       if success else "Trusted validation failed, was skipped, or evidence did not match. No success issued."}}
    result = publisher.request("PATCH", f"/repos/{REPOSITORY}/check-runs/{check_id}", body)
    verify_check(result, context, app_id)
    require(result.get("status") == "completed" and result.get("conclusion") == body["conclusion"],
            "GitHub did not confirm intended conclusion")
    return body["conclusion"]


def output(name, value):
    if "GITHUB_OUTPUT" in os.environ:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            stream.write(name + "=" + (json.dumps(value, separators=(",", ":")) if not isinstance(value, str) else value) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "recheck", "start", "validate", "finish"))
    args = parser.parse_args()
    try:
        require(os.environ.get("GITHUB_EVENT_NAME") == "workflow_run" and os.environ.get("GITHUB_REPOSITORY") == REPOSITORY,
                "Trusted workflow_run repository context required")
        trusted_sha = sha(os.environ.get("GITHUB_SHA"))
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        require(actual == trusted_sha, "Checkout is not the trusted default-branch event SHA")
        event = read_json(os.environ["GITHUB_EVENT_PATH"])
        api = GitHub(os.environ.get("GITHUB_TOKEN"))
        if args.command == "preflight":
            context = verify_origin(event, api, trusted_sha)
            output("context", context)
            output("allowed", "true" if context["allowed"] else "false")
        else:
            context = json.loads(os.environ["VALIDATION_CONTEXT"])
            recheck(event, api, context, trusted_sha)
            positive(int(os.environ["DATASET_CHECK_APP_ID"]))
            if args.command == "recheck":
                return
            app_id = positive(int(os.environ["DATASET_CHECK_APP_ID"]))
            if args.command == "start":
                output("check_id", str(start_check(context, GitHub(os.environ.get("PUBLISHER_TOKEN")), app_id)))
            elif args.command == "validate":
                report = validate_private(event, api, context, trusted_sha, int(os.environ["DATASET_CHECK_ID"]), app_id,
                    lambda: (os.environ.get("DAGSHUB_DATASET_READER_USERNAME"), os.environ.get("DAGSHUB_DATASET_READER_TOKEN")))
                output("report", report)
            else:
                conclusion = finish_check(event, api, GitHub(os.environ.get("PUBLISHER_TOKEN")), context, trusted_sha,
                    int(os.environ["DATASET_CHECK_ID"]), app_id, os.environ.get("VALIDATION_JOB_RESULT"),
                    json.loads(os.environ.get("VALIDATION_REPORT") or "null"))
                require(conclusion == "success", "Dataset check completed with failure")
        print("Trusted dataset CI stage completed; no dataset contents or credentials logged.")
    except Exception as error:
        # Do not echo server bodies, headers, tokens, artifact contents or arbitrary exception text.
        parser.exit(1, f"Trusted dataset CI blocked ({type(error).__name__}); inspect configuration and trusted metadata.\n")


if __name__ == "__main__":
    main()
