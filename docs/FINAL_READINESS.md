# Option 2: final MLOps readiness and demo

Prepared 18 November 2026 from the existing project and recorded verification.
Here, **Option 2** means demonstrating the implemented MLOps controls and their
evidence while explicitly retaining unresolved production-evaluation blockers.
It does not mean recovering missing history, collecting a new evaluation set,
approving a candidate or deploying a replacement model.

**Decision: ready to present the implemented workflow with declared limitations;
not cleared for model promotion or a claim of fully verified production operation.**
Original production holdout provenance is **UNAVAILABLE for both tasks**.
Production promotion remains **BLOCKED**. No drift reference has been invented.
The independently collected external-set strategy in [PHASE4_READINESS.md](PHASE4_READINESS.md)
is a future proposal requiring policy approval, not work performed for this demo.

The implemented source revision is
`0529f483871746d5929aed6983bcc579922e0921` on `feature/mlops-phase2`.
This final documentation revision adds the historical-provenance findings and
operator confirmations without changing implementation code. The prior hosted
run applies to its exact commit; check Actions for the published documentation
revision's own result. No training, builds, dependency installation or AWS/DVC/
MLflow network verification was performed by the documentation editor.

## Operator confirmations for submission

On 18 November 2026, the project operator supplied:

- DVC remote verification: `Cache and remote origin are in sync.` This establishes
  the reported cache/remote synchronization; it is not evidence of original
  model splits, a fresh clean-checkout pull or remote MLflow mirroring.
- The live AWS application and monitoring work. This confirms reported live
  functionality, not measured accuracy/drift or an independently audited remote
  Git revision, image digest and six-file model identity.
- Previous GitHub CI passed, consistent with the recorded run linked below.

These confirmations are attributed to the operator; no raw transcript, screenshot
or independent AWS recheck was supplied with this documentation update. The
confirmed functionality is not a reason to rerun training or redeploy.

## Evidence inventory and claim boundaries

"Verified" below describes the named local/recorded check, not an AWS check.
"Operator-verified" attributes a reported successful check to the project operator,
without implying a separate remote check by the documentation editor.
"Blocked" requires missing evidence or a separately
approved policy; a green basic CI run cannot remove that blocker.

| Area | Existing evidence to show | Verified scope / remaining gap |
| --- | --- | --- |
| AWS / HTTPS / Caddy | Operator-verified live site `https://signvision-demo.duckdns.org`; [deployment notes](MONITORING.md#deployment-considerations-not-applied) | Operator confirms the live application works. Exact deployed Git/image/model identity, detailed TLS evidence and Caddy configuration were not supplied or independently checked. No Caddyfile was found in the inspected repository files. U1 describes optional supporting evidence to retain. |
| Docker | [Compose](../docker-compose.yml), [backend Dockerfile](../backend/Dockerfile), [frontend Dockerfile](../frontend/Dockerfile), both `.dockerignore` files | Configuration inspected: Python 3.10, Node 22, native libraries/build tools, loopback bindings for 8000/3000, backend health check and persistent speech/frontend volumes. The live application works according to the operator; exact current image/container-health evidence remains U1. Static files alone are not runtime proof. |
| DVC / dataset provenance | [DVC pointer](../backend/data/db/main_dataset.db.dvc), [.dvc/config](../.dvc/config), operator-supplied synchronization result, [provenance investigation](PHASE4_READINESS.md#historical-production-provenance-investigation-18-november-2026) | Local 6,864,896-byte SQLite file has MD5 `326d65539174dffdad54606bb2271e3c`; its exact historical Git snapshot was recovered. Cache/remote synchronization succeeded according to the operator. Original raw media and original split evidence remain unavailable; U3 does not require repeating the successful sync check. |
| MLflow experiments | Root `mlflow.db` / `mlartifacts/` for the controlled candidate; `backend/mlflow.db` / `backend/mlartifacts/` for November 15 NNs/RF; `backend/mlruns/` for production-artifact imports | Local task/audit IDs, FINISHED statuses, metrics and uploaded bytes were verified. These are separate stores. Remote DagsHub MLflow mirroring/availability is not established; U3 applies if remote tracking is claimed. |
| Model registry | [Registry evidence](../mlflow_models_evidence.json), [run evidence](../mlflow_runs_evidence.json), local backend SQL store | `SignVision-Alphabet` v1 and `SignVision-Word` v1 point to November 15 models, not the production hashes. READY is catalog status; MLflow serving flavor/readiness is unverified. Do not call these the deployed versions. |
| CI/CD | [workflow](../.github/workflows/frontend-ci.yml), [CI guide](CI.md), [successful run 44162305872](https://github.com/hassaanzafar891-glitch/SignVision/actions/runs/44162305872) | Recorded success for commit `0529f48`: backend tests/compilation and frontend ESLint/TypeScript passed. Manual candidate approval was skipped on push. No automatic deployment, training, Docker build or production frontend build runs in basic CI. |
| Evaluation gate | [gate implementation](../backend/tracking/evaluation_gate.py), [evaluator](../backend/tracking/evaluate_models.py), [tests](../backend/tests/test_evaluation_gate.py), [gate policy](EVALUATION_GATE.md) | Validation and synthetic pass/fail scenarios are tested. Both real production bundles fail provenance preflight. There is no real successful candidate-approval result and no permission to promote. |
| Monitoring / drift | [monitor](../backend/monitoring.py), [private router](../backend/routers/monitoring.py), [tests](../backend/tests/test_monitoring.py), [monitoring guide](MONITORING.md), operator confirmation | Bounded per-process counts, latency, errors, model identity and confidence; no added raw-frame/landmark/user logging. Local tests pass and the operator confirms live monitoring works. A legitimate reference remains unavailable; no measured drift or production alert is claimed. U5 covers sanitized demonstration evidence. |
| Controlled retraining | [pipeline](../backend/tracking/retraining.py), [tracking helper](../backend/tracking/training.py), [tests](../backend/tests/test_controlled_retraining.py), [verified candidate](PHASE4_READINESS.md#completed-candidate-verified-18-november-2026) | Existing run completed; all 11 manifest-listed files, receipt, exact split indices and MLflow copies verified read-only. Explicit `--execute` is required for future training; drift never triggers it. No new training is needed for this presentation. Independent review remains U6. |
| Versioning / rollback | [versioning tool](../backend/tracking/versioning.py), [procedure](RETRAINING.md#promotion-and-rollback-procedure-plans-only), candidate `version.json` | Candidate bundle hashes and prior-version/gate bindings are checked. Promotion/rollback commands write plans only. No live executor, production replacement or rollback exercise is implemented or certified. |

Runtime evidence and MLflow artifacts are intentionally local, not required to be
in Git. Keep them intact. Local filesystem links may not work in a GitHub view;
use the version/run identities in [Phase 4 readiness](PHASE4_READINESS.md) to locate
them. Do not commit databases, H5/pickle files, datasets, raw frames or temporary
fixtures just to make the evidence links downloadable.

## Final evaluation checklist

Recorded checks already satisfied:

- [x] Source attribution and MIT license retained; processed data is DVC-tracked,
  while original raw images/videos are private and unavailable.
- [x] Last recorded full backend suite: **101 passed**. Dependency-free run:
  **95 passed, 6 optional tests skipped**. Tests use fixtures, not real training.
- [x] Published implementation commit has successful backend and frontend CI.
- [x] Operator confirms DVC cache/remote synchronization succeeded.
- [x] Operator confirms the live AWS application and monitoring work; this is
  functional confirmation, not measured accuracy/drift or deployment-hash proof.
- [x] Completed candidate verified with the existing read-only command; six
  model/preprocessing files match task and audit uploads.
- [x] Frozen seed-42 partitions reproduce exactly and are duplicate-disjoint:
  alphabet 9,218/3,070/3,075; word 351/88/104 train/validation/test rows.
- [x] Both production provenance preflights stop on missing original manifests;
  November 15 scores/registry entries are distinguished from production artifacts.
- [x] Missing/malformed drift baselines do not produce a fabricated score.
- [x] No automatic retraining, model promotion or deployment is claimed.

Remaining presentation/review checks (not requests to repeat successful checks):

- [ ] U1: retain current AWS/TLS/container/model identity evidence if claiming
  those exact identities; general application functionality is already confirmed.
- [ ] U2: perform the browser-camera and speech/sentence interaction demo.
- [ ] U3: retain the successful DVC sync result; distinguish any remote MLflow
  runs from local runs. No repeated sync or fresh download is needed for this claim.
- [ ] U4: record the final documentation commit and its own CI result when published;
  do not substitute the earlier implementation commit's run.
- [ ] U5: retain sanitized monitoring evidence for presentation; its live
  functionality is already operator-confirmed, but measured drift is not claimed.
- [ ] U6: review the evidence and presentation limitations, and obtain an
  independent candidate-evidence review if claiming reviewed provenance.

These checks cannot establish original production holdout membership. Missing
historical provenance, a legitimate drift reference, live artifact identity and
real promotion/rollback execution remain separate gaps. The final course demo
can show those controls honestly without pretending the blocked steps passed.

## Demo procedure (about 10-12 minutes)

Use existing environments, artifacts and saved evidence. Do not start a new
server/store, rebuild/restart containers, run `--execute`/`--train`, write a
review attestation, dispatch candidate approval or execute a change plan.

| Order | Show / action | Expected evidence and boundary |
| --- | --- | --- |
| 1. Scope, 1 minute | Open this checklist, Git commit and readiness conclusion. | Explain CI, tracking, versioning and safe failure; promotion is blocked. Do not introduce the demo as production-quality model certification. |
| 2. Application, 2 minutes (operator) | If available, open the existing HTTPS sign page. Test alphabet then word mode, sentence controls and speech; stop detection and navigate away. | Show permission handling, changing labels, stop/cleanup and browser audio. This is an interaction smoke test, not a quantitative accuracy evaluation. If unavailable, mark it unverified and show the code/tests instead. |
| 3. DVC, 1 minute | Show the pointer and local dataset hash; optionally show the authorized DagsHub repository/data page. | Exact processed dataset identity, versioning and private raw-media limitation. Do not pull over the working dataset or claim historical model partitions. |
| 4. MLflow, 2 minutes | Open the already running local UI if it is the correct store, or show the saved records. Show candidate task/audit IDs and epoch histories. | Root store has the controlled runs; backend store has the older NN/RF runs and registry. Do not restart/repoint a server to switch stores. Registry evidence JSON can show v1 without changing the UI's store. |
| 5. Candidate, 1 minute | Run the read-only verification command below; inspect `version.json` and `complete.json`. | `capture_verified: true`, with `promotion_approved: false` and `deployment_performed: false`. Explain hashes and frozen splits, not a new training execution. |
| 6. Gate, 1 minute | Show the missing production provenance and run only evidence preflight if desired. | Expected nonzero blocked result before loading models or downloading data. A safe refusal is the correct outcome; do not manufacture manifests for a green screen. |
| 7. Monitoring, 1 minute | Show the private endpoint, bounded counters and drift tests. Use live telemetry only if U5 is already satisfied. | Missing reference gives `baseline_unavailable`. Synthetic drift test fixtures are labeled synthetic; they are never shown as production alerts. |
| 8. CI and versions, 1 minute | Open the successful CI run; show versioning `--help`, recorded previous-version bindings and rollback procedure. | Backend/frontend success, manual approval skipped. Tests cover plans; no live deployment or rollback is demonstrated. |
| 9. Conclusion, 1 minute | Show completed vs pending checks and the original-holdout finding. | Explain the future external-set proposal requires a new reviewed protocol; no collection or policy change occurred. |

Read-only PowerShell commands, from the existing checkout:

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = Join-Path (Get-Location).Path "backend"
$Python = Join-Path (Get-Location).Path ".venv-backend\Scripts\python.exe"
$Candidate = "backend/data/training_runs/candidates/20261117T171728Z-3593468bcb5d4eabb3a01028050a71fb/bundle"
git rev-parse HEAD
git status --short
Get-Content backend/data/db/main_dataset.db.dvc
Get-FileHash backend/data/db/main_dataset.db -Algorithm MD5
& $Python -B -m tracking.versioning verify --source $Candidate --source-sha256 877b90809fd183c4b9f50ffeef88fa85fad49ba6aabeae2d5974e4a18dcc9a7f --dataset backend/data/db/main_dataset.db
if ($LASTEXITCODE -ne 0) { throw "Candidate verification failed; stop the demo and investigate" }
& $Python -B -m tracking.versioning --help
```

Optional **preflight only**, expected to stop because original production
manifests are missing; no evaluation output directory is requested:

```powershell
& $Python -B -m tracking.evaluate_models --check-evidence --production-dir backend/data/models --candidate-dir $Candidate
# Expected: nonzero exit and missing production provenance. Capture that outcome.
# If the outcome differs, stop and inspect the changed evidence; do not claim approval.
```

The existing full unit-test command in [CI.md](CI.md) can be shown as recorded
evidence. Running it for a later rehearsal uses temporary fixtures and the
isolated optional MLflow subprocess, not real model fits. No packages need to be
installed for the dependency-free tests. Do not reinterpret an optional skip as a
failure or as a verified integration test.

## Checks that require your involvement

**U1 — AWS/deployment identity, only through your authorized access.** Check the
current HTTPS certificate/response, actual deployment commit, container health,
image IDs/digests where available, Caddy routing and the actual localhost bindings.
Compare all six running-container artifact hashes with the production identities
in [Phase 4 readiness](PHASE4_READINESS.md#recovered-artifact-and-dataset-lineage).
A local build may have no registry digest; record its image ID without inventing
one. Local Git or Compose configuration does not prove the remote state. This
preparation does not open the site or access EC2.

For your later read-only check, in an already authorized EC2 shell and its actual
deployment directory (no SSH path or key is assumed):

```bash
git rev-parse HEAD
docker compose config --quiet
docker compose ps
docker compose images
docker compose exec -T backend python - <<'PY'
from pathlib import Path
import hashlib, json
names = [f"{task}_{suffix}" for task in ("alphabet", "word")
         for suffix in ("model.h5", "scaler.pkl", "label_encoder.pkl")]
print(json.dumps({name: hashlib.sha256((Path("data/models") / name).read_bytes()).hexdigest()
                  for name in names}, indent=2))
PY
```

Do not use `up`, `build`, `down`, `restart`, `pull`, `git pull`, or file replacement
as part of this evidence-only check. Keep Caddy/environment details private and
redact secrets; do not dump full container environments into presentation logs.

**U2 — Physical browser/camera/audio check.** Allow webcam access yourself; show
known signs in both modes, pending-request mode changes, no-hand behavior,
sentence editing/copy, speech and camera-track cleanup. Avoid recording faces,
raw landmark payloads, identities or token-bearing requests. Use consenting
participants and approved media only. Do not stop a live backend to simulate
errors; demonstrate failure handling with existing tests instead.

**U3 — DagsHub access evidence.** Retain the already successful operator-supplied
cache/remote synchronization result; do not repeat it for documentation. It does
not prove a fresh pull or remote MLflow mirroring. If the course separately
requires a fresh pull, perform it later in a separately authorized
disposable checkout using existing tools, leaving the working dataset/config
intact. Only claim DagsHub MLflow tracking if the relevant remote run IDs and
artifacts can actually be shown; local success does not establish mirroring.
No credential modification, dependency install or download is part of this task.

**U4 — Final presentation/publication evidence.** Publication of these five
documentation files is authorized. Record the resulting commit/event/job
conclusions and note the skipped manual gate. The previous run is historical
implementation evidence; it does not establish the new documentation revision's
CI result. Keep evidence capture dates and commit hashes visible.

**U5 — Monitoring presentation evidence.** Live monitoring functionality is
operator-confirmed. For a demonstration, use the existing operator token
through an approved private channel to check authenticated access and no-store
responses; keep the token off screenshots and command history. Default denial
without a valid token is expected. A 404 alone does not prove that the monitoring
route is deployed. If unavailable during a later demonstration, record that
specific observation rather than silently marking it passed. Do not change live
settings or create a synthetic baseline. Working monitoring is not measured drift.

**U6 — Human review and acceptance of limits.** Review the candidate's original
capture/source hashes and the distinction between production, November 15 registry
artifacts and the November 17 candidate. Obtain independent review before calling
candidate provenance reviewed. Confirm the final submission accurately says
UNAVAILABLE original production holdout, blocked promotion, no operational drift
claim and plan-only rollback. A reviewer name or checkbox cannot recover history.

No final demo check requires retraining, changing production models, collecting
data, installing dependencies or deploying. Establishing a legitimate drift
reference, recovering external historical records, revising the evaluation policy
or executing a production rollout/rollback are **future separately authorized
work**, not actions needed to make the current evidence appear complete.
