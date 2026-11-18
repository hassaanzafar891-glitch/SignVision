# SignVision CI

The existing `.github/workflows/frontend-ci.yml` defines **SignVision CI**
with two preserved basic jobs, an automatic PR model check, and an opt-in manual
production candidate-approval job. Private dataset validation is isolated in
`.github/workflows/dataset-validation.yml`, triggered by trusted `workflow_run`.
The filename is retained to avoid a duplicate workflow.

| Job | Runtime | Checks | Timeout |
| --- | --- | --- | --- |
| Backend | Python 3.10, Ubuntu | Standard-library unittest discovery and Python compilation | 5 minutes |
| Frontend | Node.js 22, Ubuntu | Locked npm install, ESLint, TypeScript | 15 minutes |
| Trusted actual DVC dataset validation | Python 3.10, separate trusted jobs | Private retrieval and actual validation; dedicated App publishes exact-head check | 5/5/10/5 minutes |
| PR model gate (sandbox; production fail-closed) | Python 3.10, Ubuntu | CPU-only fitted synthetic comparison, production artifact guard, full backend tests | 20 minutes |

**Current setup blockers:** no dedicated reader credential, publisher App or
protected environment is claimed to exist. Anonymous object probes returned 401;
the new design keeps data private rather than exposing credentials to PR code.
Local data validation and security tests pass, but a hosted same-repository/fork
smoke run and GitHub enforcement settings are still required. Missing credentials,
stale metadata, changed protected inputs or publication failure cannot issue success.
See [trusted setup and threat boundaries](TRUSTED_DATASET_VALIDATION.md).

## Triggers and permissions

- Pull requests, without branch or path filters.
- Pushes to `main` and `feature/**` branches.
- Manual runs via **Actions > SignVision CI > Run workflow**. GitHub requires a
  workflow-dispatch definition on the default branch for the manual entry point.
- `contents: read` permissions; checkout credentials are not persisted.
- A newer run cancels the previous run for the same workflow/event and PR or
  branch. Push and pull-request runs have separate concurrency groups.

No AWS, DagsHub, authentication, or payment secrets are required for the
basic backend/frontend checks. Those two jobs do not train, deploy, build Docker
images or retrieve DVC data. The separate PR model check fits only tiny synthetic
decision trees in temporary directories; it never retrains production models.
The manual `candidate_approval` job uses DagsHub read credentials and installs
evaluation dependencies only after evidence preflight succeeds. It requires
both task gates before uploading approval evidence, and never deploys. See
[EVALUATION_GATE.md](EVALUATION_GATE.md) for prerequisites, provenance requirements,
GitHub environment configuration and the current blocked production-evidence state.
Application/model files and existing DVC configuration are unchanged.

## Trusted automatic PR dataset validation

`tracking.pr_dataset` uses the actual object referenced by
`backend/data/db/main_dataset.db.dvc`, not validation-test fixtures. It accepts
only the reviewed single-file pointer and checks:

- MD5 `326d65539174dffdad54606bb2271e3c`, size **6,864,896 bytes**, SHA-256
  `e6d6c59c7a0c7a207388d290c7ddae7f05ba1aa1507df8b5d293ac4ec13c33a9`.
- No SQLite WAL/SHM/journal sidecars; both `alphabetDataset` and `wordDataset`.
- Existing `validate_training_database()` SQLite integrity, required `id`, `label`
  and 42 feature columns, finite numerical values, valid labels and sample/class
  adequacy. The file hash is checked again after read-only validation.

Source remains `https://dagshub.com/hassaanzafar891-glitch/SignVision.dvc`, addressed
by the approved MD5. Only the separate trusted job supplies a dedicated reader
credential, after repository/workflow/run/PR/head/base/pin/issuer checks. It uses
reviewed protected-main code and calls the unchanged `tracking.pr_dataset.validate()`.
Authenticated redirects are refused; no credential is forwarded to another host.
No DVC client or local auth configuration, dependencies, PR caches/artifacts, raw
data uploads, public snapshot or PR checkout is used by the trusted workflow.

The original PR workflow no longer downloads private data. A dedicated App publishes
`Trusted actual DVC dataset validation` against the verified PR head SHA. Protected
input proposals are rejected; changed heads/base/run attempts and invalid reports
cannot produce success. App-bound required checks prevent a same-named ordinary
PR job from impersonating this result. Details and mandatory manual setup are in
[TRUSTED_DATASET_VALIDATION.md](TRUSTED_DATASET_VALIDATION.md).

Actual local read-only validation passed: **15,363 alphabet rows / 37 classes**,
**543 word rows / 5 classes**, 42 features each, integrity/schema/quality valid.
This proves local object validity, not runner download availability.

## Automatic PR model gate and its scope

`tracking.pr_gate --base-sha "$PR_BASE_SHA"` runs on **every PR**, independently
of the dataset job so that both assessments remain visible. It:

1. Diffs the PR merge checkout against its base commit for **any** changes under
   `backend/data/models`. Such changes fail closed while historical production
   holdout provenance is unavailable. The guard does not pretend that a sandbox
   PASS authorizes changes to production H5/scaler/encoder/provenance files.
2. Reproduces the existing fitted synthetic sandbox in a fresh temporary root:
   deterministic frozen 24/24/24 train/validation/test rows, actual fixed-configuration
   DecisionTree fitting, train-only baseline, unchanged evaluation protocol.
3. Calls the existing evaluation adapter/shared no-regression rule. A weaker normal
   challenger or invalid training/evaluation evidence fails the job. It also verifies
   the separately labeled stricter negative-control BLOCK; that expected control
   is not a failure of the normal comparison.
4. Installs only `backend/requirements-ci-light.txt`, asserts sklearn/NumPy/SciPy/
   YAML are importable, and runs full backend discovery so genuine sandbox tests
   no longer skip for missing sklearn. Optional unrelated MLflow/FastAPI tests may
   still skip where those packages are intentionally absent.

The retained `pr-sandbox-assessment-*` JSON includes measured metrics, test identities,
artifact/receipt hashes, reasons and `production_approval: false`. It is retained
for seven days even on a blocked assessment if a report exists; it is **not** the
manual production approval artifact. No existing `.signvision-demo`, model,
dataset or MLflow store is written by the PR reproduction.

Local measured evidence: normal champion/challenger accuracy and macro F1 were
both **1.0** on the same 24 synthetic test rows, yielding PASS. The +0.05 control
BLOCKED the measured tie. A separately fitted test challenger with
`min_samples_leaf=20` achieved accuracy **0.5**, macro F1 **0.3333333333333333**,
matrix `[[12,0],[12,0]]`, versus champion `[[12,0],[0,12]]`; the normal no-regression
gate rejected it. Tampered evidence and production-artifact changes also failed
their tests. These are synthetic workflow checks, never production performance.

**A genuine production performance comparison still cannot pass:** authentic
historical holdout provenance remains unavailable. The existing manual/main-only
candidate-approval workflow and its preflight remain unchanged. Recovering that
evidence or approving an independent alternative protocol is separate work.

## Required GitHub settings

After authorized publication and a hosted PR run, configure branch protection or
a ruleset to require these exact status contexts:

- `Backend checks (Python 3.10)`
- `Frontend checks (Node.js 22)`
- `Trusted actual DVC dataset validation` — expected source MUST be the dedicated App
- `PR model gate (sandbox; production fail-closed)`

A workflow failure alone does not prevent merging unless GitHub requires its
status check. No settings were changed here. Fork workflow approval may require
a maintainer action under the repository's Actions policy. Review changes to
workflow/gate/model-path policy; a PR can propose changes to its own CI code.
Protect those changes through required reviews/code ownership where appropriate.
No PR secrets are required or accepted. Store the new reader and publisher secrets
only in their separate main-only environments, not repository/organization-wide.
Existing manual production-approval secrets remain separate. Require up-to-date
branches and protect validation/workflow inputs with owner reviews. Bootstrap the
reviewed default-branch workflow before enabling its required App check; do not
use a generic check-name-only fallback. Merge queues need a separate extension.

Local commands using already installed dependencies:

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = (Join-Path (Get-Location) 'backend')
$python = '.\.venv-backend\Scripts\python.exe'
& $python -B -m tracking.pr_dataset --dataset backend/data/db/main_dataset.db
& $python -B -m tracking.pr_gate --base-sha (git rev-parse HEAD)
& $python -B -m unittest discover -s backend/tests -p test_pr_checks.py -v
```

The first command checks the existing real dataset read-only. It does not prove
private remote access. The old anonymous `--fetch-to` utility remains available
for explicit diagnostics but is no longer a PR workflow dependency. The gate command
fits synthetic models only in a new temporary directory and does not promote them.

Local verification on 19 November 2026: all **14** focused PR-check tests passed;
full sklearn-enabled backend discovery ran **218** cases (**216 passed, two native
symlink skips**). Dependency-free discovery passed **172**, with **46** expected
optional skips. An initial Windows fixture-cleanup failure was corrected by explicitly
closing the new tests' SQLite connections; the rerun passed. YAML parsing and workflow
contract assertions passed, including equality of the original backend/frontend/manual
jobs and triggers against Git HEAD. Compilation, documentation PowerShell parsing and
whitespace checks passed. Existing legacy unclosed-file ResourceWarnings remain.
This is local/static verification, not actionlint or a successful GitHub-hosted run.
Final SHA-256 comparison matched all **330** protected existing dataset/model,
DVC, MLflow and sandbox files; the new checks changed none of those records.

The subsequent trusted two-stage implementation passed **17 focused security
tests** and reran the **14 PR-check tests** successfully. Full dependency-free
discovery passed **188**, with **47** optional skips (235 total). Both workflows
parse and their security/job-boundary assertions pass. Hosted private download,
dedicated-App publication and branch-rule enforcement have not been tested because
the owner setup is still required. No reader token/App installation is assumed.

## Backend checks

Commands run from the repository root:

```powershell
python -m unittest discover -s backend/tests -p "test_*.py" -v
python -m compileall -q -x '(^|[\\/])(data|mlruns|mlartifacts)([\\/]|$)' backend
```

The Python job installs no project dependencies. Camera handling is tested with
mocked FastAPI, OpenCV, MediaPipe, and recognition services. Tracking tests use
fake models and MLflow clients; trainer contract checks inspect syntax without
executing training. Tests use temporary fixtures, not production datasets/models.
The optional real MLflow round-trip test skips when MLflow is absent.

The same discovery includes `test_monitoring.py`: privacy/auth, bounded aggregate
storage, error handling and reference-gated drift tests run without ML packages.
The optional FastAPI ASGI binding test skips if FastAPI is absent. No live webcam,
production drift baseline, external monitoring server or AWS access is needed.
See [MONITORING.md](MONITORING.md) for operational prerequisites and limitations.

`smoke_browser_camera.py` does not match `test_*.py` and is deliberately excluded.
Compilation checks syntax without importing or executing the training modules,
and excludes runtime data and MLflow stores (whose tag filenames may end in `.py`
while containing a checksum rather than Python source).
Tracking is explicitly disabled in the backend job.

## Frontend checks

Commands run from `frontend/`, with Node.js 22 and the lockfile committed:

```powershell
npm ci
npm run lint
npx tsc --noEmit
```

The workflow caches npm's download cache using `frontend/package-lock.json` as
the dependency key. It still runs `npm ci` each time; `node_modules` is not cached.
Native dependencies such as `better-sqlite3` may download a prebuilt binary or
compile using the GitHub-hosted Ubuntu runner's tooling. Registry/network access
is still needed when dependencies are not cached.

The production build is excluded from every trigger, including manual dispatch.
It can be run separately with `npm run build` when explicitly desired; it currently
downloads Google Fonts and needs a writable `backend/data/` directory for SQLite.
The workflow adds no frontend scripts and does not change application behavior.

## Limitations and verification

Discovery also includes `test_controlled_retraining.py`: synthetic frozen plans,
fake trainers/tracking, bundle tampering, evidence-backed promotion and pinned
rollback plans. No production training or activation executor is added to any
workflow; only the separate PR job fits temporary synthetic decision trees.
See [RETRAINING.md](RETRAINING.md). Candidate approval remains manual and blocked
until authentic production holdout evidence is established.

These checks do not establish webcam recognition accuracy, real TensorFlow/
MediaPipe compatibility, real MLflow/DagsHub connectivity, production build success,
or Docker runtime behavior. There is no full frontend browser-test suite. Confidence
parsing/component unit tests are available through `npm test`, but the current
workflow does not run that command; lint and type checking remain its frontend
coverage. See [PREDICTION_CONFIDENCE.md](PREDICTION_CONFIDENCE.md).
ESLint warnings remain warnings; errors fail CI.

The implemented Phase 4 source/tests were published in commit
`0529f483871746d5929aed6983bcc579922e0921`. The recorded
[GitHub run 44162305872](https://github.com/hassaanzafar891-glitch/SignVision/actions/runs/44162305872)
passed backend and frontend checks; manual candidate approval was skipped on the
push event. This is not a successful real-model gate or deployment. Later local
documentation changes are not part of that run. Untracked runtime files are not
uploaded automatically. The final documentation revision needs its own
commit-specific CI result; check the [Actions page](https://github.com/hassaanzafar891-glitch/SignVision/actions)
for that revision instead of reusing the older run's success. See
[FINAL_READINESS.md](FINAL_READINESS.md).

The operator separately confirms that the live AWS application/monitoring work
and DVC reports cache/remote synchronization. Those checks were not performed by
CI and do not validate original production holdouts, candidate promotion or
measured drift. Publishing documentation triggers basic CI only; the workflow
does not deploy or retrain on a push.

Local workflow validation uses an already available YAML parser plus assertions
for events, branch filters, permissions, concurrency, runtimes, timeouts, commands,
and npm cache settings. `git diff --check` checks whitespace. This is static
configuration validation, not a GitHub-hosted execution or actionlint result.

References: [npm caching](https://github.com/actions/setup-node/tree/v6),
[manual workflow runs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).
