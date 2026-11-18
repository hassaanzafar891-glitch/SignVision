"""Required PR sandbox assessment, never a production model approval."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
from unittest.mock import patch

from tracking.evaluation_gate import GateError, canonical_sha256, require


def production_changes(base_sha):
    require(isinstance(base_sha, str) and re.fullmatch(r"[0-9a-f]{40}", base_sha), "Invalid PR base commit")
    result = subprocess.run(["git", "diff", "--name-only", "-z", base_sha, "HEAD", "--", "backend/data/models"],
                            capture_output=True, check=True)
    return [name for name in result.stdout.decode("utf-8").split("\0") if name]


def assess(*, base_sha=None):
    if base_sha is not None:
        require(not production_changes(base_sha),
                "Production model/preprocessing/evidence changes BLOCKED: authentic historical production holdout "
                "provenance is unavailable. Sandbox PASS cannot authorize production changes.")
    from demo_sandbox import foundation, drift, orchestrator, evaluation, evaluation_champion
    # Fresh, temporary runtime. Existing .signvision-demo, models and MLflow remain untouched.
    with tempfile.TemporaryDirectory(prefix="signvision-pr-sandbox-") as temporary, \
         patch.object(foundation, "WORKSPACE", Path(temporary).resolve()):
        foundation.initialize()
        drift.create_reference()
        drift.run("shifted")
        trained = orchestrator.consume(arm=True, job_kind="fitted", timeout_seconds=30, cooldown_seconds=0)
        require(trained["status"] == "FINISHED" and trained["real_model_training"] is True,
                "PR synthetic challenger did not genuinely fit")
        slot = trained["trigger"]["slot"]
        with evaluation._lock():
            evaluation_champion.prepare()
        normal = evaluation.run(job_slot=slot)["receipt"]
        # Normal candidate regression is a job failure, never an expected negative control.
        require(normal["decision"] == "PASS", "PR sandbox candidate BLOCK: " + normal["reason"])
        strict = evaluation.run("stricter-control", job_slot=slot)["receipt"]
        require(strict["decision"] == "BLOCK", "Required stricter negative control did not BLOCK")
        verified = evaluation.inspect(job_slot=slot)
        require(verified["integrity_verified"], "Invalid PR evaluation evidence")
        evaluation.require_receipt(normal["receipt_id"], candidate_version=slot,
            candidate_model_sha256=normal["candidate"]["model_sha256"],
            test_identity_sha256=normal["test_identity_sha256"])
        return {"scope": "synthetic-demo-only", "decision": "PASS", "production_approval": False,
                "production_state": "BLOCKED_missing_historical_holdout_provenance",
                "normal_case": normal, "negative_control": strict,
                "evidence_sha256": canonical_sha256(verified),
                "notice": "Reproduced fitted synthetic models; never production performance evidence or promotion"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", help="PR base commit; production bundle changes fail closed")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = assess(base_sha=args.base_sha)
    except (GateError, OSError, ValueError, subprocess.SubprocessError) as error:
        report = {"scope": "synthetic-demo-only", "decision": "BLOCK", "production_approval": False,
                  "reason": str(error) if isinstance(error, GateError) else type(error).__name__}
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    if report["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
