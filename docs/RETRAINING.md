# Controlled retraining, candidate versions and rollback plans

Status: implemented locally; no neural training, registry mutation, deployment or
AWS access was performed. The serving API, preprocessing, production models,
dataset, Docker/Caddy settings and GitHub workflow/secrets are unchanged.
**Promotion remains blocked:** authentic production holdout provenance is absent.
New training cannot retrospectively establish it or relax the evaluation policy.

## Explicit lifecycle

1. An operator validates the DVC dataset and freezes a reviewed training plan.
2. A separate command with `--execute` trains both neural tasks on CPU, creates
   fresh MLflow runs and saves isolated candidate artifacts.
3. An independent reviewer checks authentic training evidence. A separate reviewed
   copy is created; the captured original remains unchanged.
4. The existing evaluator scores saved bundles and applies both evidence-backed
   gates. Missing production provenance or any regression blocks approval.
5. An authorized operator may record a promotion **plan** binding both versions
   and the gate results. Rollback plans bind the previous version and original
   promotion evidence. Neither command changes serving files or deploys anything.

Monitoring `drift_detected` is a recommendation for operator investigation only.
There is no monitor-to-training callback, background scheduler, automatic approval
or automatic deployment. A missing monitoring baseline cannot justify retraining.
Ordinary CI runs tests; it never runs the commands below to fit models.

## What is captured

- Reviewed DVC MD5 `326d65539174dffdad54606bb2271e3c`, pointer SHA-256, exact stable
  SQLite SHA-256, table/schema and ordered feature/label row SHA-256 for each task.
- Actual zero-based train/validation/test indices, seed 42 and duplicate-disjoint
  checks. The existing `split_three_way()` and training methodology are reused.
- Source and requirements-file hashes; source snapshots; Git commit/dirty state;
  Python, platform and installed package versions at training time.
- Existing architectures, optimizer/learning rate, epochs/batch sizes, available
  epoch accuracy/loss, validation accuracy/loss and final test accuracy/loss.
- Start/finish/capture times, fresh MLflow run IDs, and SHA-256 of each new model,
  scaler and label encoder. No historical accuracy is invented.

The actual trainer rows and indices must match the frozen plan **before scaler
fitting or model fitting**. Scalers fit training data only; validation and test are
separate. Test data is never supplied to `fit()` as validation. Test metrics are
reported after fitting; do not tune/select successive runs based on this holdout.
Source/dataset changes require a newly reviewed plan; the pipeline checks them
again after training before producing a bundle.

The frozen holdout is isolated from fitting in new runs. It is **not** certified
historically untouched by the deployed models or previous experiments. Repeated
test inspection can bias selection. No signer IDs exist, and near-duplicate or
signer-independent separation is unverified. Authentic production test indices
must ultimately match the candidate's test indices; otherwise the existing gate
rejects comparison. Do not reconstruct historical evidence from today's seed.

TensorFlow operation determinism and random seed 42 are requested in a fresh CLI
process. Cross-platform bitwise weights are not guaranteed. Requirements have
pins/ranges, not a complete transitive lock; captured package versions document
the actual environment. Existing NN architecture/preprocessing is unchanged.

## PowerShell commands

Use the existing Python 3.10 backend environment containing the existing inference
requirements and separate `backend/requirements-tracking.txt`. DVC uses the
existing tools environment. See [HOW_TO_RUN.md](../HOW_TO_RUN.md) for environment,
DVC authentication and MLflow server setup. No new dependency is introduced here.
Do not reinstall tools or modify the existing experiment database unnecessarily.

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = Join-Path (Get-Location).Path "backend"
$Python = Join-Path (Get-Location).Path ".venv-backend\Scripts\python.exe"
$ToolsPython = Join-Path (Get-Location).Path ".venv-tracking\Scripts\python.exe"
$Lifecycle = Join-Path (Get-Location).Path "backend\data\training_runs\lifecycle"
New-Item -ItemType Directory -Force $Lifecycle | Out-Null
# Only if processed data is missing; use the existing authorized DVC configuration.
# & $ToolsPython -m dvc pull backend/data/db/main_dataset.db.dvc
$Plan = Join-Path $Lifecycle "plan.json"
& $Python -m tracking.retraining prepare --output $Plan
if ($LASTEXITCODE -ne 0) { throw "Plan preparation failed" }
$PlanHash = (& $Python -c "import sys; from tracking.evaluation_gate import read_json,canonical_sha256; print(canonical_sha256(read_json(sys.argv[1])))" $Plan).Trim()
if ($LASTEXITCODE -ne 0) { throw "Plan hashing failed" }
```

Preparation validates/splits but never imports TensorFlow or starts MLflow runs.
It writes a new file only; an existing plan is never overwritten. Review its
dataset/sources/indices and pin the canonical hash through your review process.
Do not automatically hash a changed plan and regard that as approval.

**Future operator action only — this command actually trains; not executed here:**

```powershell
# Start the existing compatible MLflow server separately first.
# Use the independently reviewed $PlanHash, not a newly substituted hash.
& $Python -m tracking.retraining train --plan $Plan --plan-sha256 $PlanHash --execute --tracking-uri http://127.0.0.1:5000
if ($LASTEXITCODE -ne 0) { throw "Training incomplete; do not review or promote outputs" }
```

For DagsHub, replace `--tracking-uri` with
`https://dagshub.com/hassaanzafar891-glitch/SignVision.mlflow`; supply existing
`MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` through the operator secret
process, never command arguments or Git. Controlled training requires tracking
and a working new MLflow run. Tracking failure prevents a completed candidate;
ordinary legacy training retains its previous optional/best-effort behavior.

Task outputs remain under `backend/data/training_runs/<task>/<unique-id>/`.
Candidate snapshots live in `backend/data/training_runs/candidates/<unique-id>/`:

```text
plan.json, dataset.dvc, sources/     frozen input and source evidence
bundle/plan.json                    exact plan bound by version manifest
bundle/<task>_model.h5              six new model/preprocessing artifacts
bundle/<task>_scaler.pkl
bundle/<task>_label_encoder.pkl
bundle/<task>_provenance.json        captured_training_time_unreviewed
bundle/<task>_run_metadata.json      parameters, metrics and capture metadata
bundle/version.json                 version ID and raw file hashes
complete.json                       canonical version hash and MLflow audit run ID
```

The separate MLflow audit run uploads the plan, source snapshot and complete
bundle without resuming or editing previous runs. Failed uploads may leave local
files or incomplete remote runs; absent `complete.json` means no completed
pipeline output. Versioning is an immutable-by-hash raw-artifact catalog, not an
MLflow serving flavor, Registry stage transition or filesystem write protection.
Keep trusted backups/access controls; local files can still be tampered with.

## Review and existing evaluation gate

The following are templates for **future** legitimate evidence. Replace the
paths, reviewer and evidence identifiers with independently verified values.
Hash values must come from trusted review records, not arbitrary downloaded JSON.
Review/model deserialization is restricted to trusted operator-owned artifacts.

```powershell
$Candidate = "backend/data/training_runs/candidates/<completed-id>/bundle"
$Reviewed = Join-Path $Lifecycle "reviewed-candidate"
$CandidateHash = "<reviewed-complete.json-bundle_sha256>"
& $Python -m tracking.versioning review --source $Candidate --source-sha256 $CandidateHash --destination $Reviewed --reviewer "<reviewer>" --evidence "<immutable-training-evidence-reference>" --attest-training-evidence
```

This checks training capture, tracking completion, artifacts, dataset and recorded
partitions, then creates a separate reviewed bundle. The printed hash identifies
that reviewed version. The flag is a human attestation, not authentication and
not proof that a claimed source is genuine. Do not attest inferred history.

Production must have authentic reviewed manifests and independently verified
`SIGNVISION_PRODUCTION_ALPHABET_MANIFEST_SHA256` and
`SIGNVISION_PRODUCTION_WORD_MANIFEST_SHA256` pins. Follow
[EVALUATION_GATE.md](EVALUATION_GATE.md) to evaluate both bundles. The existing
manual `candidate_approval` workflow and gate policy are unchanged. Current
production manifests are missing, so that preflight fails closed. No candidate
approval, promotion plan or valid rollback chain is claimed for today's models.

## Promotion and rollback procedure (plans only)

Once genuine production provenance exists, archive a verified current bundle
read-only; the command refuses missing or unreviewed provenance:

```powershell
$Current = Join-Path $Lifecycle "current-reviewed-version"
& $Python -m tracking.versioning archive-current --source backend/data/models --destination $Current --reviewer "<reviewer>" --evidence "<verified-live-bundle-and-training-evidence-reference>"
```

Record its printed canonical version hash as `$CurrentHash`, and the reviewed
candidate hash as `$ReviewedHash`. Verify the archive against actual deployed
bytes through a separate authorized process; a local path does not establish AWS
identity. Evaluate using this exact archived bundle as `--production-dir` and
`$Reviewed` as `--candidate-dir`. Keep the resulting `$Evaluation` directory and
both versioned bundles as immutable trusted evidence.

```powershell
$Promotion = Join-Path $Lifecycle "promotion-plan.json"
& $Python -m tracking.versioning promote --current $Current --current-sha256 $CurrentHash --target $Reviewed --target-sha256 $ReviewedHash --evaluation-dir $Evaluation --output $Promotion --reviewer "<authorized-approver>" --evidence "<immutable-approval-reference>" --approve-plan
```

This recomputes both gates and verifies that their artifact/dataset/holdout hashes
bind the exact versions. It records the previous version, target, gate hash and
review reference, with `execution_performed: false`. Retain its printed canonical
hash as `$PromotionHash`. No command copies files into `backend/data/models/`.

Before any separately designed deployment: verify actual live identity, obtain
independent approval, retain the complete previous bundle, stage all six target
files as one version, reverify hashes, plan a coordinated switch/health check and
record actual execution. This repository adds no deployment executor. GitHub
environment reviewers/access restrictions must be configured by an authorized
operator; a reviewer string or local plan alone is not enforceable authorization.

If that separately authorized promotion was actually executed and the live bundle
is still the recorded target, a new operator approval can request rollback:

```powershell
& $Python -m tracking.versioning rollback --current $Reviewed --current-sha256 $ReviewedHash --target $Current --target-sha256 $CurrentHash --previous-plan $Promotion --previous-plan-sha256 $PromotionHash --evaluation-dir $Evaluation --output (Join-Path $Lifecycle "rollback-plan.json") --reviewer "<authorized-approver>" --evidence "<incident-and-rollback-approval-reference>" --approve-plan
```

Rollback validates the original successful gate, pinned promotion record and exact
previous bundle. It cannot select an arbitrary older model. It writes a new plan
only, without requiring the older model to outperform the new one or silently
changing the promotion gate. The operator must separately verify actual deployed
state and perform any approved coordinated rollback. Do not overwrite an
individual scaler/model while serving requests. Never infer that a plan was
executed: execution/health records are outside this implementation.

## Safe verification

```powershell
& $Python -B -m unittest discover -s backend/tests -p test_controlled_retraining.py -v
& $Python -B -m unittest discover -s backend/tests -p "test_*.py" -v
python -S -B -m unittest discover -s backend/tests -p "test_*.py" -v
git diff --check
```

Tests use temporary synthetic artifacts, mocked MLflow and fake trainers. They
never fit real models, access AWS or use a real tracking store. Existing CI
discovery includes them without heavyweight dependencies. Commands were checked
against the CLI and PowerShell syntax; real training, DagsHub uploads, hosted CI,
promotion/deployment and rollback execution remain untested and unperformed.
