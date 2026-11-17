"""Offline candidate review and approval-gated change plans. Never replaces models.

Local JSON is not authentication: reviewers must verify workflow/evidence origin.
All writes are new files under isolated training outputs, not the serving path.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

from tracking.evaluate_models import read_bundle, read_rows, verify_partitions, SUFFIXES
from tracking.evaluation_gate import (
    TABLES, canonical_sha256, compare_evaluations, digest, read_json, require,
)
from tracking.import_baselines import BACKEND, sha256
from tracking.retraining import write_new


def isolated(path):
    path = Path(path).resolve()
    require(path.is_relative_to((BACKEND / "data/training_runs").resolve()),
            "New evidence must stay under backend/data/training_runs")
    return path


def verify_version(directory, expected_sha256):
    directory = Path(directory).resolve()
    digest(expected_sha256, "reviewed version hash")
    version_path = directory / "version.json"
    require(version_path.resolve().parent == directory, "External version file")
    version = read_json(version_path)
    require(canonical_sha256(version) == expected_sha256, "Version identity mismatch")
    require(type(version.get("schema_version")) is int and version["schema_version"] == 1,
            "Unsupported version schema")
    require(isinstance(version.get("version"), str) and version["version"].strip(), "Missing version identifier")
    files = version.get("files")
    require(isinstance(files, dict) and files, "Missing version files")
    required = {f"{task}_{suffix}" for task in TABLES
                for suffix in (*SUFFIXES.values(), "provenance.json")}
    require(required.issubset(files), "Incomplete two-task bundle")
    for name, expected in files.items():
        require(isinstance(name, str) and Path(name).name == name and name != "version.json",
                "Invalid bundle filename")
        digest(expected, "file hash")
        path = directory / name
        require(path.is_file() and path.resolve().parent == directory, "Missing or external versioned artifact")
        require(sha256(path) == expected, "Versioned artifact changed")
    require({p.name for p in directory.iterdir()} == set(files) | {"version.json"},
            "Unlisted files in versioned bundle")
    return version


def reviewer_fields(reviewer, evidence):
    require(all(isinstance(value, str) and value.strip() and len(value) <= 512
                for value in (reviewer, evidence)), "Reviewer and immutable evidence reference are required")


def archive_current(source, destination, reviewer, evidence):
    """Read-only snapshot of a trusted serving bundle, only with reviewed provenance."""
    reviewer_fields(reviewer, evidence)
    source = Path(source).resolve()
    manifests = {task: read_bundle(source, task) for task in TABLES}
    destination = isolated(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for task in TABLES:
        for suffix in (*SUFFIXES.values(), "provenance.json"):
            name = f"{task}_{suffix}"
            shutil.copyfile(source / name, destination / name)
        # Fail on source/copy mutation, rather than label unverified bytes as archived.
        read_bundle(destination, task, manifests[task][1])
    version = {"schema_version": 1, "version": destination.name,
               "files": {p.name: sha256(p) for p in destination.iterdir()},
               "review": {"reviewer": reviewer, "evidence_reference": evidence},
               "origin": "operator_verified_current_bundle_snapshot", "promotion_approved": False}
    write_new(destination / "version.json", version)
    return canonical_sha256(version)


def review_candidate(source, expected_sha256, destination, reviewer, evidence, database):
    """Explicit human attestation creates a NEW reviewed bundle, never edits capture."""
    reviewer_fields(reviewer, evidence)
    source = Path(source).resolve()
    version = verify_version(source, expected_sha256)
    receipt = read_json(source.parent / "complete.json")
    require(receipt.get("bundle_sha256") == expected_sha256
            and isinstance(receipt.get("mlflow_audit_run_id"), str)
            and bool(receipt["mlflow_audit_run_id"].strip())
            and receipt.get("deployment_performed") is False,
            "Missing completed MLflow bundle receipt")
    require("plan.json" in version["files"], "Frozen plan is not versioned")
    plan = read_json(source / "plan.json")
    require(canonical_sha256(plan) == version["plan_sha256"] and plan.get("seed") == 42,
            "Frozen training plan identity mismatch")
    for task in TABLES:
        require(f"{task}_run_metadata.json" in version["files"], "Training metadata is not versioned")
    records = {}
    for task in TABLES:
        record = read_json(source / f"{task}_provenance.json")
        require(record["provenance"]["status"] == "captured_training_time_unreviewed",
                "Only authentic new training capture can be reviewed here")
        metadata = read_json(source / f"{task}_run_metadata.json")
        require(metadata["training_status"] == "FINISHED" and not metadata["tracking_errors"],
                "Incomplete training or tracking")
        require(metadata["partitions"] == record["partitions"], "Training partition evidence differs")
        require(plan["tasks"][task] == {"dataset": record["dataset"], "partitions": record["partitions"]},
                "Captured data or partitions differ from frozen plan")
        require(metadata["controlled_training"]["plan_sha256"] == version["plan_sha256"]
                == record["provenance"]["training_plan_sha256"], "Training plan identity mismatch")
        require(metadata["mlflow_run_id"] == record["provenance"]["mlflow_run_id"], "MLflow identity mismatch")
        for key, suffix in SUFFIXES.items():
            require(record["artifacts"][key] == sha256(source / f"{task}_{suffix}"), "Captured artifact mismatch")
        rows, database_hash = read_rows(database, task)
        verify_partitions(record, rows, database_hash)
        record["provenance"].update(status="reviewed_training_time_evidence", reviewed_by=reviewer,
                                     evidence_reference=evidence,
                                     reviewed_at=datetime.now(timezone.utc).isoformat())
        records[task] = record
    destination = isolated(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in version["files"]:
        if not name.endswith("_provenance.json"):
            shutil.copyfile(source / name, destination / name)
            require(sha256(destination / name) == version["files"][name],
                    "Copied training evidence changed; review blocked")
    for task, record in records.items():
        write_new(destination / f"{task}_provenance.json", record)
        read_bundle(destination, task)
    reviewed = {**version, "version": destination.name, "parent_version_sha256": expected_sha256,
                "files": {p.name: sha256(p) for p in destination.iterdir()},
                "review": {"reviewer": reviewer, "evidence_reference": evidence},
                "promotion_approved": False}
    write_new(destination / "version.json", reviewed)
    return canonical_sha256(reviewed)


def verify_gate(evaluation_dir, current, target):
    """Recompute both gates and bind records to actual, reviewed bundle artifacts."""
    evaluation_dir = Path(evaluation_dir)
    approval = read_json(evaluation_dir / "approval.json")
    require(approval.get("approved") is True and approval.get("deployment_performed") is False,
            "Missing successful combined evaluation")
    computed = {}
    for task in TABLES:
        production = read_json(evaluation_dir / f"{task}_production.json")
        candidate = read_json(evaluation_dir / f"{task}_candidate.json")
        for directory, record in ((current, production), (target, candidate)):
            manifest, manifest_hash = read_bundle(directory, task)
            require(record["model"]["artifacts"] == manifest["artifacts"], "Gate artifact identity mismatch")
            require(record["held_out"]["manifest_sha256"] == manifest_hash, "Gate provenance identity mismatch")
            require(record["dataset"] == {**manifest["dataset"], "rows": sum(
                len(ids) for ids in manifest["partitions"].values())}, "Gate dataset identity mismatch")
            require(record["held_out"]["test_indices_sha256"] == canonical_sha256(manifest["partitions"]["test"]),
                    "Gate held-out identity mismatch")
        computed[task] = compare_evaluations(production, candidate)
    require(approval.get("tasks") == computed, "Combined approval content differs from recomputed gates")
    return canonical_sha256(approval)


def plan_change(*, action, current, current_sha256, target, target_sha256,
                output, reviewer, evidence, evaluation_dir=None, previous_plan=None,
                previous_plan_sha256=None):
    """Approval records describe a requested change, not the actual deployed state."""
    reviewer_fields(reviewer, evidence)
    current, target = Path(current).resolve(), Path(target).resolve()
    current_version = verify_version(current, current_sha256)
    target_version = verify_version(target, target_sha256)
    require(current_sha256 != target_sha256, "Current and target versions must differ")
    require(action in ("promote", "rollback"), "Unknown change action")
    approval_hash = None
    previous_hash = None
    if action == "promote":
        require(evaluation_dir is not None, "Promotion requires successful evaluation evidence")
        approval_hash = verify_gate(evaluation_dir, current, target)
    else:
        require(previous_plan is not None and previous_plan_sha256 is not None,
                "Rollback requires the pinned previous promotion plan")
        previous = read_json(previous_plan)
        digest(previous_plan_sha256, "previous plan identity")
        require(canonical_sha256(previous) == previous_plan_sha256, "Previous plan changed")
        require(previous.get("action") == "promote" and previous.get("manual_approval") is True
                and previous.get("execution_performed") is False, "Not a reviewed promotion plan")
        require(previous["target"]["sha256"] == current_sha256
                and previous["previous"]["sha256"] == target_sha256, "Rollback must restore recorded previous version")
        require(evaluation_dir is not None, "Rollback requires original gate evidence")
        approval_hash = verify_gate(evaluation_dir, target, current)
        require(approval_hash == previous["gate_approval_sha256"], "Original approval changed")
        previous_hash = previous_plan_sha256
    result = {
        "schema_version": 1, "action": action, "manual_approval": True,
        "reviewer": reviewer, "evidence_reference": evidence,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "previous": {"path": str(current), "version": current_version["version"], "sha256": current_sha256},
        "target": {"path": str(target), "version": target_version["version"], "sha256": target_sha256},
        "gate_approval_sha256": approval_hash, "previous_plan_sha256": previous_hash,
        "execution_performed": False, "deployment_performed": False,
        "status": "approved_plan_only_live_identity_check_and_separate_execution_required",
    }
    write_new(isolated(output), result)
    return canonical_sha256(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    review = sub.add_parser("review")
    review.add_argument("--source", required=True, type=Path)
    review.add_argument("--source-sha256", required=True)
    review.add_argument("--destination", required=True, type=Path)
    review.add_argument("--dataset", type=Path, default=BACKEND / "data/db/main_dataset.db")
    review.add_argument("--attest-training-evidence", action="store_true",
                        help="Explicitly attest that the captured training evidence was independently reviewed")
    archive = sub.add_parser("archive-current")
    archive.add_argument("--source", required=True, type=Path)
    archive.add_argument("--destination", required=True, type=Path)
    for action in ("promote", "rollback"):
        cmd = sub.add_parser(action)
        for name in ("current", "target", "output", "evaluation-dir"):
            cmd.add_argument("--" + name, required=True, type=Path)
        for name in ("current-sha256", "target-sha256"):
            cmd.add_argument("--" + name, required=True)
        cmd.add_argument("--approve-plan", action="store_true",
                         help="Explicit operator approval of a plan only; does not execute it")
        if action == "rollback":
            cmd.add_argument("--previous-plan", required=True, type=Path)
            cmd.add_argument("--previous-plan-sha256", required=True)
    for cmd in (review, archive, *[sub.choices[key] for key in ("promote", "rollback")]):
        cmd.add_argument("--reviewer", required=True)
        cmd.add_argument("--evidence", required=True, help="Trusted immutable review/workflow reference; never a credential")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "archive-current":
        print(archive_current(**args))
    elif command == "review":
        require(args.pop("attest_training_evidence"), "Review requires explicit --attest-training-evidence")
        args["expected_sha256"] = args.pop("source_sha256")
        print(review_candidate(**args))
    else:
        require(args.pop("approve_plan"), "Change plan requires explicit --approve-plan")
        print(plan_change(action=command, **args))


if __name__ == "__main__":
    main()
