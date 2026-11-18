# Synthetic demonstration sandbox: drift, fitting, evaluation and safe activation

This is a separate local teaching environment, not an approved SignVision model
release. It does not use the unreviewed real candidate, production H5/pickle files,
SQLite training data, live monitoring, AWS, DVC or any MLflow store/registry.
Original production holdout provenance remains **UNAVAILABLE** and production
promotion remains **BLOCKED**.

## What exists

- Source/configuration: `backend/demo_sandbox/`, using existing canonical-JSON/
  SHA-256 helpers. Foundation, drift and legacy simulation use the standard
  library; explicitly selected fitted jobs use the already installed scikit-learn.
- Runtime: the fixed repository-local `.signvision-demo/` directory. It is ignored
  by Git and is outside backend Docker build context, production models, dataset
  directories, training outputs and MLflow stores.
- Two JSON-only, hand-authored threshold-rule fixtures: `demo-champion-v1` and
  `demo-challenger-v1`. These names indicate **fixture roles**, not model approvals
  or a quality ranking. No models were fitted, pretrained artifacts copied, or
  real model deserializers/registries invoked.
- A seed-42 synthetic dataset: 72 rows, 42 integer features, two `demo_class_*`
  labels, and three class-balanced disjoint 24-row partitions. Feature-vector IDs
  are intentionally synthetic. This is not real PSL landmark data.
- An initial `active.json` pointer to the champion fixture, bound to its manifest
  hash. This is initial fixture setup, **not promotion**. That foundation pointer
  remains immutable. Step 5 switches only a separate fitted-model activation
  pointer and never changes the foundation or evaluation pointers. The fitted evaluation
  extension below measures classification quality using a separate initial
  evaluation-champion pointer. Step 2 computes synthetic distribution drift; Step 3 adds an explicitly
  armed synthetic job simulation. The fitted extension below performs actual
  scikit-learn training on synthetic data, never real PSL training.

Immutable foundation layout (eight JSON files):

```text
.signvision-demo/
  config.json
  data/synthetic-v1.json
  versions/demo-champion-v1/model.json
  versions/demo-champion-v1/manifest.json
  versions/demo-challenger-v1/model.json
  versions/demo-challenger-v1/manifest.json
  active.json
  initialized.json
```

Every artifact states `synthetic-demo-only` scope and denies production evidence.
The foundation generates no accuracy, loss, drift score, independent-holdout result
or reviewer approval. Step 2 computes separately labeled synthetic divergence
scores, never production evaluation evidence or real recognition accuracy.
Hand-authored fixture predictions are not a retraining demonstration.

## Isolation and pointer rules

The source configuration is explicit and separate from application `.env` files.
Only the fixed policy is supported: storage `.signvision-demo`, seed 42, disabled
tracking and disabled registry. The CLI exposes no `--root`, model path or URI
override. Production environment variables are ignored, including
`MLFLOW_TRACKING_URI`, `MLFLOW_REGISTRY_URI` and `SIGNVISION_TRACKING_ENABLED`.

Foundation writes occur only during `init`, under the fixed runtime root, using
exclusive creation. Existing foundation storage is never overwritten, repaired or
reset automatically.
The completion receipt is written last. Readers reject incomplete initialization,
altered/missing/unlisted files and changed pointer contents before returning a
fixture. Only the explicitly named, bounded drift and job extension files described
below are additionally permitted. The current format permits only the initial champion pointer; even
manually changing it to the known challenger fails verification.

Both manifests and the initial pointer are checked against the deterministic
fixture contract, not merely against user-editable hashes. Symlinks, Windows
junction/reparse attributes, hard-linked artifact files and path escapes are
rejected. Recomputing a tampered manifest hash does not make it an allowed fixture.
The existing production evaluator also rejects these fixtures: they have no
production provenance or model/scaler/encoder bundle. No MLflow SDK calls exist
in this package, so it cannot upload/register artifacts as part of its commands.

These are application-level guards against misuse and accidental redirection,
not an operating-system security boundary against an administrator, edited source
code or a hostile process concurrently replacing filesystem ancestors. External
tools with production credentials could manually upload any file; the sandbox
does not grant or exercise that authority. Maintain separate OS permissions for
adversarial multi-user environments. No production credential or deployment
configuration is changed by this foundation.

## Commands

Use an existing Python environment; no dependencies need installing. From
PowerShell:

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = Join-Path (Get-Location).Path "backend"
$Python = Join-Path (Get-Location).Path ".venv-backend\Scripts\python.exe"
# Creates only new local synthetic fixtures. Run once, not against an existing root.
& $Python -B -m demo_sandbox init
# Read-only integrity checks after initialization:
& $Python -B -m demo_sandbox verify
& $Python -B -m demo_sandbox status
```

`init` fails if `.signvision-demo/` exists, rather than replacing an active pointer.
If interrupted, leave the partial directory for diagnosis; readers will not treat
it as a valid initialized sandbox. This step has no cleanup/reset command.
No runtime fixture should be committed, copied to `backend/data/models/`, supplied
to production approval workflows or uploaded to the production registry.

Safe tests, using temporary workspace copies and sentinel production files:

```powershell
python -S -B -m unittest discover -s backend/tests -p test_demo_sandbox.py -v
& $Python -B -m unittest discover -s backend/tests -p "test_*.py" -v
```

Tests cover deterministic bytes/predictions, complete disjoint synthetic
partitions, no overwrite, disabled ML imports/tracking, pointer/path manipulation,
file tampering, incomplete initialization, link guards, input validation and
production bundle rejection. Native symlink tests skip when this Windows user
cannot create them; a separate test exercises the Windows reparse-attribute guard.
Passing these tests does not certify production inference or operational drift.
Existing CI discovery will include them after a separately authorized publication;
GitHub Actions is not modified or triggered in this step.

Step 1 verification on 18 November 2026:

- Actual `init` and read-only `verify` succeeded in `E:\LinguaSign\.signvision-demo`.
  Active fixture: `demo-champion-v1`; canonical manifest SHA-256:
  `757cd5c758be75be4e29b529235f507fb90aa5178e215fe212afc580e9d5da1f`.
- All eight runtime files are covered by the deterministic integrity checks;
  Git confirms the runtime directory is ignored.
- Focused suite: 16 passed, two native symlink checks skipped. Full backend suite:
  117 passed, two skipped (119 total). Dependency-free suite: 111 passed, eight
  expected skips (119 total). No test failed.
- Python compilation, documentation PowerShell parsing, local links and whitespace
  checks passed. The native Windows symlink/junction behavior was not exercised
  under this user's privileges; the reparse-attribute guard and native hard-link
  rejection were tested. No administrator permissions were requested.

## Step 2: synthetic drift detection and local alert

`backend/demo_sandbox/drift.py` operates only on the verified sandbox. It imports
the pure `monitoring.js_divergence()` calculation; it never calls the monitoring
endpoint, `get_monitor()`, real model loaders, MLflow or a training function.

An explicit `reference` command computes prediction counts from the **24 validation
rows** and stores `.signvision-demo/drift/reference.json`. Identity includes the
active fixture manifest, synthetic dataset, validation-index hash and sample count.
No labels are substituted for predictions and no test rows are predicted. The
reference is deterministic, synthetic-demo scoped, and never labeled a reviewed
production aggregate. A missing/mismatched reference blocks replay; it is not
silently invented by the run command or adapted to traffic.

The two deterministic replay scenarios are deliberately simple:

- `normal` repeats the validation inputs. With the default 24-row batch, this is
  a control replay of the reference itself, not an independent traffic sample;
  its score is mathematically zero.
- `shifted` cycles **training-partition** inputs whose actual champion-fixture
  prediction is `demo_class_0`. This intentionally changes the predicted class
  mix; repeated toy inputs are not independent production observations.

The detector computes base-2 Jensen-Shannon divergence in [0,1] from the actual
reference and replay counts. Source settings are in
`backend/demo_sandbox/drift_config.json`: default threshold 0.1, minimum 12 samples,
default batch 24, maximum batch 240 and maximum 20 retained events. Only the
threshold is source-configurable; CLI `--threshold` and `--batch-size` allow bounded
demo variations. Thresholds must be finite and in (0,1]. An alert requires
**score strictly greater than threshold**; equality is `no_alert`. This explicitly
differs from production monitoring's existing `>=` rule, which remains unchanged.
Too few current samples produce `insufficient_current_samples`, null score and no
alert. Threshold 0.1 illustrates this synthetic scenario; it is not a calibrated
production guarantee and was not selected using the synthetic test partition.

### Exact demonstration commands

Use the existing initialized sandbox; do **not** rerun `init` or install anything:

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = Join-Path (Get-Location).Path "backend"
$Python = Join-Path (Get-Location).Path ".venv-backend\Scripts\python.exe"
& $Python -B -m demo_sandbox.drift reference
& $Python -B -m demo_sandbox.drift run --scenario normal
& $Python -B -m demo_sandbox.drift run --scenario shifted
& $Python -B -m demo_sandbox.drift inspect
```

The shifted command prints `SYNTHETIC DEMO DRIFT ALERT` only if the calculated
score exceeds its recorded threshold. Both control observations and alerts are
stored in `.signvision-demo/drift/events.json`. `inspect` is read-only: it checks
reference identity, journal/event checksums, sequence numbers and timestamps, and
replays predictions to verify counts, scores and decisions. The foundation's
`verify`/`status` now explicitly report `integrity_scope: immutable_foundation`;
use drift `inspect` for semantic integrity of these dynamic records.

Optional threshold/sample demonstrations, not new production experiments:

```powershell
& $Python -B -m demo_sandbox.drift run --scenario shifted --threshold 1
& $Python -B -m demo_sandbox.drift run --scenario shifted --batch-size 1
```

The first cannot exceed a threshold of 1; the second has too few samples and
must not emit a drift score. Neither launches retraining.

### Bounded audit and isolation

Each event records a real timezone-aware UTC timestamp, deterministic event ID,
sequence, reference/model/dataset hashes, scenario, replay-index hash, sample
count, observed prediction counts, computed score, threshold, comparison rule,
decision and checksum. Every event states `production_evidence: false` and
`training_triggered: false`. No raw feature vectors, webcam frames or user data
are written to the journal.

The journal retains at most **20** events and is capped at **128 KiB**; the reference
is capped at **8 KiB**. Evictions retain an explicit discarded-event count and
sequence continuity. Identical retained observations are deduplicated by event
ID and keep their **first** timestamp; replaying one does not imply a new event
or rewrite the journal. After eviction, a later replay may create a new sequence
with the same deterministic event ID. Scores, distributions and IDs are
deterministic for fixed inputs/settings; wall-clock timestamps are not seeded.

The drift extension permits only `drift/reference.json`, `drift/events.json`, two fixed atomic-write temporary
files (`.reference.tmp`, `.events.tmp`) and `.write.lock` are allowed as extensions.
Writers take an exclusive lock and atomically replace only those two record
paths. A concurrent writer is rejected. Failed replacement preserves the previous
journal and cleans its owned temporary file. A process crash can leave a lock or
temporary file: inspect it manually; there is no automatic force-unlock/reset.
There is no arbitrary file-path/URL option and no active-pointer replacement.

Hashes and replay checks establish local consistency, not cryptographically
authenticated history: a trusted host owner could rewrite timestamps/history and
recompute hashes. The bounded journal is not an immutable remote audit service.
Existing filesystem privilege/race limitations described above still apply.
Reference creation never overwrites a mismatched reference or recreates one
behind existing events. Production baseline rules and all original eight sandbox
artifacts remain unchanged.

### Instructor video (about 60-90 seconds for this step)

1. Keep a visible **SYNTHETIC DEMO ONLY** title and show the reference command's
   validation counts and reference hash.
2. Run normal replay: explain that the control reuses reference validation inputs
   and show zero divergence and `no_alert`.
3. Run shifted replay: show the changed observed counts, calculated score,
   threshold and explicit synthetic alert. Do not describe this as AWS or
   production drift.
4. Run `inspect`; show event timestamps, reference hash, checksum validation and
   `training_triggered: false`. Repeated runs may show a deduplicated event and
   its original timestamp; do not imply that timestamp is the new recording time.

This demonstrates an observable console/file alert, not email/SMS delivery, an
operational production incident, accuracy degradation or automated retraining.

Safe focused tests:

```powershell
python -S -B -m unittest discover -s backend/tests -p test_demo_drift.py -v
```

Step 2 local verification on 18 November 2026:

- All 22 drift tests passed. The full backend suite passed 139 tests with two
  native symlink skips (141 total); the dependency-free suite passed 133 with
  eight expected skips (141 total). No test failed. Compilation, CLI help,
  PowerShell documentation parsing, links and whitespace checks passed.
- The actual CLI reference/run/inspect sequence completed successfully. Reference
  prediction counts were `demo_class_0: 14`, `demo_class_1: 10`; normal replay
  produced score 0 and `no_alert`. Shifted counts were 24/0, yielding calculated
  divergence `0.24835048781787405` against threshold `0.1` and the explicit
  **SYNTHETIC DEMO DRIFT ALERT**. These numbers are toy-fixture replay results,
  not production drift or recognition-accuracy evidence.
- Canonical reference hash:
  `99709d44f3331698383d63ff0455e9649b83f5a55e67642e4b116b229e582b29`.
  Shifted event ID:
  `b53a13f8ec39078f688d8ef5ef419c80617c9f274b2c34b8028c5eb978229956`.
- Read-only inspection verified the two retained observations. Foundation
  verification still reported the original champion manifest identity and
  `training_performed: false`; both events reported `training_triggered: false`.
  No new model, reference approval, job trigger, promotion or registry entry was
  created.

## Step 3: explicitly armed synthetic retraining trigger

`demo_sandbox.orchestrator` consumes verified sandbox drift alerts. It never calls
the real retraining pipeline, MLflow, a registry or a deployment command. The fixed
standard-library child recomputes a toy threshold using only the **24 synthetic
training rows**. This is a **synthetic retraining workflow simulation**, not neural
training or measured PSL performance. It creates no accuracy/loss, evaluation
approval, active-pointer change or production evidence. Validation and test rows
are excluded from its arithmetic operation.

From the repository root, use the existing Python environment:

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = (Join-Path (Get-Location) 'backend')
$python = '.\.venv-backend\Scripts\python.exe'

# Read-only: verifies alerts/history, starts no job, creates no trigger files.
& $python -B -m demo_sandbox.orchestrator consume

# Existing Step 2 alert, or emit the deterministic shifted observation:
& $python -B -m demo_sandbox.drift run --scenario shifted

# Explicitly arm one invocation; consumes at most one previously unprocessed alert.
& $python -B -m demo_sandbox.orchestrator consume --arm --timeout-seconds 10 --cooldown-seconds 30
& $python -B -m demo_sandbox.orchestrator inspect

# Same alert is consumed already: no second child or artifact overwrite.
& $python -B -m demo_sandbox.orchestrator consume --arm
```

Arming is a CLI flag, not a persistent setting. Without `--arm`, every invocation
is read-only even when alerts exist. An optional `--arm --watch-seconds 60` waits
for an unprocessed alert, polls once per second and exits after at most one job or
the watch deadline. Already-pending alerts run immediately. The watch limit is
60 seconds; the child execution timeout is separately bounded to 1–30 seconds.
Only the fixed worker command is accepted; no executable/path/URI overrides exist.
The child receives a minimal environment without MLflow/AWS credential variables.

### Audit, deduplication and failures

The sandbox-only extension is:

```text
.signvision-demo/retraining/
  journal.json
  job-001/request.json
  job-001/output.json
  job-001/model.json
  ...                         # at most ten numbered slots
```

The checksummed journal records trigger ID, source alert ID and full verified
event, source journal hash, real UTC start/completion timestamps, policy, status,
structured job output and SHA-256 artifact hashes. Inspection replays the alert
validation and checks every artifact against the deterministic worker contract.
The raw child streams are capped at **16 KiB combined**; their hashes and captured
byte counts are retained, not arbitrary raw stderr or exception text. Validated
structured stdout is saved in `output.json` and the journal. Every result denies
real training and promotion. Existing producer events remain unchanged: their
`training_triggered: false` describes Step 2, while this separate consumer journal
records the subsequent simulation.

- An exclusive `.write.lock` covers the entire child job. Concurrent consumers
  cannot launch another job. Terminal statuses are `FINISHED`, `FAILED` and
  `TIMED_OUT`; failure/timeout returns a nonzero CLI exit code.
- The alert is durably recorded as `RUNNING` **before** launch. Success, failure
  and timeout all consume its ID; automatic retries are forbidden. The original
  deterministic shifted alert cannot be rerun simply by emitting it again.
- A persisted cooldown (default 30 seconds, configurable 0–3600) follows every
  completed attempt. A later CLI invocation cannot shorten the preceding job's
  cooldown by passing zero.
- History stops after **ten jobs**, with no eviction of consumed alert IDs. The
  journal is capped at 128 KiB; each slot allows only bounded request/output/model
  files (64/16/8 KiB for request/output/model). A failed job has no model artifact. Slot files are exclusively
  created; only the journal is atomically replaced using `.journal.tmp`.
- Invalid alerts, changed artifacts, orphan files, interrupted writes or a stale
  `RUNNING` record block execution. Inspect manually; there is no automatic reset,
  retry, deletion, force-unlock or reuse of a slot. Local checksums establish
  consistency, not authenticated history against a privileged host owner.

For the video, show the unarmed result first, then the Step 2 synthetic alert,
explicit arming, one `FINISHED` job with `jobs_started: 1`, its artifact hashes and
`test_used: false`. Show inspection and a second armed call starting zero jobs.
If this session already consumed the alert, show its original timestamp and saved
receipt; do not imply a new job or reset history to manufacture one.

Safe focused tests (temporary workspaces, including child timeout/failure cases):

```powershell
python -S -B -m unittest discover -s backend/tests -p test_demo_orchestrator.py -v
& .\.venv-backend\Scripts\python.exe -B -m unittest discover -s backend/tests -p 'test_*.py'
```

Step 3 local verification on 18 November 2026:

- All 21 trigger tests passed. Full backend discovery ran 162 tests: 160 passed,
  two native symlink tests skipped, zero failures. Dependency-free discovery ran
  162 tests: 154 passed, eight expected skips, zero failures. Module compilation
  and parsing all seven documented PowerShell blocks passed.
- The actual unarmed command reported one pending alert and started zero jobs.
  Explicit arming completed one simulation; inspection verified its journal and
  artifact contents/hashes. A subsequent armed invocation reported
  `no_unprocessed_alert` and started zero jobs.
- Trigger ID:
  `5c0e7979e1db10ba1129b551b4623163ee874a021a4624e9ffd545d57d5e3003`.
  Its source alert is the Step 2 shifted event recorded above. The worker used
  24 training rows, reported `validation_used: false`, `test_used: false` and
  `quality_metrics_computed: false`; the recomputed toy threshold was `-1.0`.
- Simulation artifact SHA-256:
  `b8f9681dc75665c7de5baf228c7e2d3b4f1fdca989d8eb87554eb4de706827e9`.
  This JSON artifact is not a deployed model, approved challenger or evidence of
  recognition performance.
- Before/after SHA-256 checks matched all 320 snapshotted protected files,
  including production data/models, existing MLflow records/stores, DVC metadata,
  deployment/workflow files, original sandbox foundation and Step 2 records.
  The demonstration added only four runtime JSON files under the ignored
  `.signvision-demo/retraining/` extension. No real training or activation ran.

## Actual automated retraining on synthetic data

The `--job-kind fitted` extension performs a genuine
`sklearn.tree.DecisionTreeClassifier.fit()` in a bounded CPU child process.
The pre-existing simulation mode remains the default for compatibility; **it does
not satisfy actual fitting**. Both modes require `--arm` to launch any job.
Both share the same lock, consumed-alert history, cooldown, timeout and ten-job
limit: changing job mode cannot retry an alert consumed by the simulation.

The frozen seed-42 dataset is unchanged: 72 synthetic rows, 42 numerical features,
two classes and disjoint 24-row train/validation/test partitions. The worker fits
only the training rows, then measures validation accuracy, macro F1 and a confusion
matrix. It never scores test rows. The configuration is fixed before fitting:
Gini decision tree, maximum depth 3, minimum leaf size 1, random state 42. There is
no hyperparameter search, validation-based refit, checkpoint selection or test
selection. This deliberately simple, separable toy dataset can produce perfect
validation results; those are not real sign-recognition performance or independent
production evidence. No dataset change was needed for legitimate fitting.

### Exact demonstration commands

Use the existing `.venv-backend` (verified here with scikit-learn 1.6.1). No package
installation, MLflow server, Docker or network connection is required.

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = (Join-Path (Get-Location) 'backend')
$python = '.\.venv-backend\Scripts\python.exe'

# Different deterministic non-test replay from the earlier 24-input alert.
& $python -B -m demo_sandbox.drift run --scenario shifted --batch-size 25

# Safe default: verifies and reports pending alerts without fitting.
& $python -B -m demo_sandbox.orchestrator consume --job-kind fitted

# Automatically fits one challenger after verifying an unprocessed drift alert.
& $python -B -m demo_sandbox.orchestrator consume --arm --job-kind fitted --timeout-seconds 30 --cooldown-seconds 30
& $python -B -m demo_sandbox.orchestrator inspect

# Human-readable genuine receipt for this verified local demonstration.
Get-Content .signvision-demo/retraining/job-002/training_receipt.json

# Demonstrate deduplication: no further fitting on the same alert.
& $python -B -m demo_sandbox.orchestrator consume --arm --job-kind fitted
```

The batch-25 alert has already been consumed in this local session. Replaying it
now demonstrates deduplication, not another fit. For a separately authorized new
video attempt, use a distinct batch size, such as 26, while remaining within the
240-input replay bound; the next receipt will be in the next available slot.
Do not delete journals or alter timestamps to stage a fresh run. Changing replay
length creates another synthetic observation, not new training data. Training
always uses the same frozen training rows. Optional `--watch-seconds 60` explicitly
arms a bounded watcher; it launches at most one job when an unprocessed alert
arrives and exits. Drift never trains anything unless an operator armed a consumer.

### Genuine artifacts and receipt

Successful fitted slots contain `request.json`, `output.json`, `model.pkl` and
`training_receipt.json`. The fitted model and receipt are bounded to 8 KiB each;
the request is bounded to 64 KiB, output to 16 KiB, shared journal to 128 KiB.
The receipt records dataset and all three split-index hashes, exact train and
validation indices, estimator configuration/full parameters, sklearn version,
real UTC training timestamps, `TRAINED` status, actual measured validation
metrics and fitted model SHA-256. It explicitly states `test_used: false`,
`production_evidence: false`, and `promotion_approved: false`.

The child only returns bounded serialized output; the parent exclusively creates
the four fixed files after validating a successful exit, fitted estimator identity
and recomputed validation metrics. The consumed alert is recorded before launch.
Failure/timeout produces a failed output record and no successful training receipt
or model. A crash during publication leaves a blocked incomplete attempt requiring
manual inspection, never an automatic retry. Inspection does not fit again.

The direct child interpreter uses only its configured dependency directory, with
user-site/.pth startup hooks disabled and native thread counts limited to one.
Production credential variables are not passed. No MLflow calls, registry writes,
real model deserializers, production dataset reads or production training interfaces are used.
The sandbox model uses Python pickle: load only trusted local worker artifacts in
the matching environment, never arbitrary uploaded files. Local hashes and the
application guards are not an OS security boundary against a hostile host owner.

Actual local result, 18 November 2026:

- Alert `c087a3aeb190bb17ff9b3866acf355dac47f2333f69961cf821b20404f088770`
  automatically triggered fitted job `job-002` after explicit arming.
- Trigger `e4c6f75211d9557ce4bb28c525eb083151faf14dea98bb6c18b4b64bc8cf4e20`
  finished successfully. The separate earlier `job-001` remains a simulation.
- Measured validation accuracy **1.0**, macro F1 **1.0**, confusion matrix
  `[[12, 0], [0, 12]]`, from 24 synthetic validation examples. No test metrics exist
  for this fitted challenger.
- Model `.signvision-demo/retraining/job-002/model.pkl` SHA-256:
  `4ebc8d1afcad9f2d5297b1a4abb0dc612af1174d01287b5e0fb53951423e1f6c`.
- Inspection verified both old and new job records; repeating the armed fitted
  command returned `no_unprocessed_alert`, with zero new jobs.
- All eight focused fitted-training tests passed. Full backend discovery ran
  170 tests: 168 passed, two native symlink skips, no failures. Dependency-free
  discovery passed 154 with 16 expected skips, including the eight sklearn tests.
- Before/after SHA-256 verification matched all 322 protected files (production
  data/models, MLflow stores/artifacts, DVC metadata, deployment/workflow files,
  immutable sandbox foundation, drift reference and original job files).
  The only changed prior runtime files were the sandbox drift event journal and
  shared trigger journal; four new files were added under `job-002`. No production
  artifact, registry record, AWS configuration or active pointer was changed.

Focused actual-fit tests:

```powershell
& .\.venv-backend\Scripts\python.exe -B -m unittest discover -s backend/tests -p test_demo_fitted_training.py -v
```

## Step 4: genuine fitted champion–challenger evaluation

`demo_sandbox.evaluation` replaces the unfinished JSON-fixture evaluation draft.
It compares the actual fitted challenger from `retraining/job-002/model.pkl`
against a separately fitted `DecisionTreeClassifier` baseline, using exactly the
same frozen 24 test rows and all 42 raw features. Both use identity preprocessing:
no scaler, encoder fitting, feature selection or tuning occurs during evaluation.
Metrics are calculated with scikit-learn: accuracy, macro F1 and confusion matrices
(rows = actual labels, columns = predicted labels, `demo_class_0`, `demo_class_1`).

The original `active.json` remains the immutable JSON-fixture pointer used by the
earlier drift demonstration. Step 4 explicitly initializes a **separate evaluation
champion role** at `evaluation/champion_pointer.json`, referencing a genuine fitted
model and training receipt. This is initial baseline setup, not promotion or an
application-serving change. Step 5 uses an additional activation pointer; it does
not change this initial evaluation reference.

### Baseline preparation and provenance

`prepare-champion` fits only the 24 training rows with fixed settings declared
before evaluation: Gini decision tree, depth 1, minimum leaf size 1, random state
42. The challenger retains its existing depth-3 configuration and artifact.
Baseline initialization is exclusive, writes its pointer last, never overwrites
files and never scores validation/test rows. Repeating preparation verifies the
existing baseline without fitting again. Partial initialization blocks; the CLI
does not recreate a baseline behind saved evaluation results.

Before test scoring, the adapter verifies:

- Deterministic dataset identity and exact train/validation/test coverage, including
  feature-vector disjointness; the frozen source contract also fixes all rows.
- Original fixture/pointer integrity, the fitted champion pointer, both fitted
  artifact hashes, training-receipt identities, class/feature schema and configuration.
- Challenger drift-event/request/output provenance through `orchestrator.inspect()`:
  recorded training and validation indices must equal the frozen partitions;
  `test_used` must be false and hyperparameters fixed before training. The original
  worker fits only training rows and scores validation only, verified by focused tests.
- Both models' dataset and split hashes against the current frozen data. The
  baseline receipt records train-only fitting, without validation/test scoring.

The fitted artifacts are trusted local pickles; never supply external pickle
files. These local source/receipt/replay checks are controlled sandbox evidence,
not authenticated history against a malicious administrator. Test rows were
excluded from this run's fitting, validation and hyperparameter selection; after
this evaluation they have been observed and must not be described as a new,
unseen holdout in a later development cycle. This is not signer/session-independent
PSL evaluation and does not recover historical production provenance.

### Exact PowerShell demonstration

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = (Join-Path (Get-Location) 'backend')
$python = '.\.venv-backend\Scripts\python.exe'

# One-time fitted baseline; later calls verify without refitting.
& $python -B -m demo_sandbox.evaluation prepare-champion --job-slot job-002

# Actual same-test-set comparison using the existing no-regression rule.
& $python -B -m demo_sandbox.evaluation run --job-slot job-002 --case challenger

# Deliberately stricter, predeclared demo control: requires +0.05 accuracy.
# Expected BLOCK with exit code 1; this is a policy demonstration, not a test failure.
& $python -B -m demo_sandbox.evaluation run --job-slot job-002 --case stricter-control
$LASTEXITCODE

# Read-only receipt/hash checks and recomputation of measured test metrics.
& $python -B -m demo_sandbox.evaluation inspect --job-slot job-002
```

The normal comparison directly calls
`tracking.evaluation_gate.require_no_accuracy_regression()`. Equal accuracy passes;
any regression blocks. Macro F1 and matrices are reported but are not additional
production gate criteria. The control first applies that same rule, then requires
an additional **0.05 absolute accuracy improvement**. Its threshold was declared
in code before test scoring, not chosen by searching test results. With a perfect
champion it is intentionally unattainable; this is clearly labeled a stricter
policy BLOCK, not evidence that the challenger regressed. Production policy is
unchanged and the real production gate still requires unavailable evidence.

### Actual local results (18 November 2026)

| Fitted model | Test accuracy | Macro F1 | Confusion matrix | Test rows |
| --- | --- | --- | --- | --- |
| Depth-1 evaluation champion | 1.0 | 1.0 | `[[12, 0], [0, 12]]` | 24 |
| Existing depth-3 challenger, job-002 | 1.0 | 1.0 | `[[12, 0], [0, 12]]` | Same 24 |

- Normal gate: **PASS**, measured accuracy delta 0.0, required minimum 0.0.
  Receipt ID: `99424711ef0b0f5673c71a4a06663b62c49a408d5999f8619427d6cc901b9e2c`.
- Stricter control: **BLOCK**, measured delta 0.0 is less than required 0.05.
  Receipt ID: `e39990bd8a6c4b5abd5164f89c62a0f5e6636f4b3bfe943b4540ac9205e52fab`.
- Fitted champion model SHA-256:
  `2ad1ce46d212e09da1bcc30f7a8042b58e7fdb4d5f0843e243fa724a30462145`.
- Unchanged fitted challenger SHA-256:
  `4ebc8d1afcad9f2d5297b1a4abb0dc612af1174d01287b5e0fb53951423e1f6c`.
- Test-index SHA-256:
  `ecf1f012663e7ff1d048d2d797e30b70d98a8cf6899facd2f44c4b26e189a81a`.
  Ordered test-row SHA-256:
  `b1531b50c54d80fc3df3679891292390dd2e7d3bf0e3c2138f6fbc860e40c5b4`.

These are actual calculations on intentionally separable synthetic data. They do
not establish independent production accuracy, signer independence or superiority
over the deployed neural networks.

### Bounded audit and future receipt requirement

Runtime additions stay exclusively under `.signvision-demo/evaluation/`:
`champion/model.pkl`, `champion/training_receipt.json`, `champion_pointer.json`
(each at most 8 KiB), and `journal.json` (at most 64 KiB, two fixed-case receipts).
Writers share an exclusive evaluation lock and fixed atomic journal temporary
path. Repeated cases retain the same receipt and original timestamp; no history
eviction or silent overwriting occurs. This journal is pinned to the evaluated
challenger identity; it cannot silently accept a different job under old receipts.

Each receipt binds actual metrics, policy/threshold, reason, dataset/test identities,
artifact and training-receipt hashes, both pointer identities, a real UTC evaluation
timestamp and checksums. Inspection revalidates training provenance and recomputes
scores without fitting. Invalid evidence blocks before issuing a decision; it
does not produce a successful evaluation receipt.

`evaluation.require_receipt()` is the receipt preflight reused by the Step 5 sandbox activator.
It requires a recorded PASS from the normal case, current pointer/data/model
integrity, and exact candidate/test identity. Missing, stale, altered or BLOCK
receipts cannot qualify. Passing this preflight is **not** an operator approval or
production authorization. Step 5 below consumes the normal-case receipt for
explicitly requested sandbox-only activation.

Safe focused tests:

```powershell
& .\.venv-backend\Scripts\python.exe -B -m unittest discover -s backend/tests -p test_demo_evaluation.py -v
```

Step 4 verification on 18 November 2026:

- All 12 focused evaluation tests passed. Full backend discovery ran 182 tests:
  180 passed, two native symlink skips, zero failures. Dependency-free discovery
  ran 182 tests: 154 passed, 28 expected skips (including optional sklearn tests).
- Actual CLI baseline preparation, PASS, stricter BLOCK and inspection completed.
  BLOCK returned the expected exit code 1. Repeating PASS returned the original
  receipt with `deduplicated: true`. No model was refitted by evaluation/inspection.
- Compilation, all 11 documented PowerShell block syntax checks and Git whitespace
  checks passed. The production evaluation-gate source remained unchanged in this
  step; its previously extracted numerical helper is reused directly.
- Before/after SHA-256 checks matched all **329** existing protected files,
  including production data/models, MLflow stores/artifacts, DVC metadata,
  deployment/workflow files, original sandbox pointers/data, drift records and
  both existing retraining jobs. Only four new runtime files were created under
  `evaluation/`: fitted baseline, training receipt, initial evaluation pointer and
  evaluation journal. No production files or prior sandbox records changed.

## Step 5: manual sandbox promotion and rollback

`demo_sandbox.activation` is a local executor, not part of the FastAPI prediction
path, Docker, AWS, GitHub Actions or a production registry. All commands explicitly
report `scope: synthetic-demo-only`. It never trains, copies or overwrites a model.
Only the existing `job-002` fitted challenger and fitted evaluation champion are
supported; there are no model-path, storage-root, registry-URI or version overrides.

Three pointers serve distinct purposes:

- Original `active.json`: immutable JSON fixture for Steps 1–3.
- `evaluation/champion_pointer.json`: immutable fitted baseline reference for Step 4.
- **`activation/active.json`**: the only mutable pointer, used solely by the new
  sandbox executor and its `read_active()` helper. It starts with the fitted
  baseline, can select verified `job-002`, and retains the previous baseline's
  version/model/training-receipt hashes for rollback. No live application reads it.

### Preconditions and transaction safety

Promotion requires an explicit `promote --receipt-id ...` command and a recorded,
revalidated normal-case PASS receipt. The executor reuses `evaluation.require_receipt()`
and verifies the exact shared no-regression policy, zero minimum delta, no additional
control policy, evaluation protocol, reference champion and current dataset/split
identities. Training receipts, fitted artifact hashes and both immutable pointers
are rechecked. A stricter-control BLOCK receipt cannot authorize activation.

An exclusive writer lock covers the operation. The checksummed journal records a
**PENDING transaction before** atomic pointer replacement (`fsync` then `os.replace`
on the same filesystem). After replacement, the executor rereads the exact pointer,
loads its verified fitted model and predicts the **first training sample**, never
a test sample. The label must match the known synthetic label, and class
probabilities must have valid shape/range/sum. A successful operation is committed
only after this post-activation health check. This is a basic load/inference check,
not a latency benchmark or additional accuracy evaluation.

Any caught activation, health or commit failure attempts automatic restoration of
the prior healthy baseline, rechecks the restored pointer/model and records
`RECOVERED`. Interrupts such as Ctrl+C also attempt restoration before exiting.
If restoration fails, the journal records `RECOVERY_FAILED`; active-model reads
remain blocked. Never claim recovery succeeded when storage or fallback integrity
prevents it. Explicit rollback also works when the active challenger file is corrupt,
because it verifies the preserved baseline independently of the broken challenger.
Rollback does not toggle back to an unverified/failed candidate.

A hard process termination cannot run cleanup immediately. A pending journal makes
`status()`/`read_active()` fail closed. The next mutating command (or `recover`)
restores the recorded baseline **instead of executing the requested new action**;
retrying promotion requires another explicit command. A hard kill may also leave a
lock or temporary file. Such files block execution and are never force-unlocked or
deleted automatically: confirm no writer is running and inspect them manually.
This protects against concurrent writers and ambiguous incomplete storage writes;
it is not a background watchdog, power-loss guarantee or privileged-host security
boundary. Local model pickles remain trusted-only inputs.

### Exact PowerShell demonstration

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = (Join-Path (Get-Location) 'backend')
$python = '.\.venv-backend\Scripts\python.exe'
$passReceipt = '99424711ef0b0f5673c71a4a06663b62c49a408d5999f8619427d6cc901b9e2c'
$blockReceipt = 'e39990bd8a6c4b5abd5164f89c62a0f5e6636f4b3bfe943b4540ac9205e52fab'

# Inspect first. Initialize once only if initialized is false.
$state = & $python -B -m demo_sandbox.activation status | ConvertFrom-Json
$state | ConvertTo-Json -Depth 12
if (-not $state.initialized) {
    & $python -B -m demo_sandbox.activation init
}

# Expected rejection, exit 1; the active baseline stays unchanged.
& $python -B -m demo_sandbox.activation promote --receipt-id $blockReceipt
$LASTEXITCODE

# Explicit sandbox promotion and post-activation health check.
& $python -B -m demo_sandbox.activation promote --receipt-id $passReceipt
& $python -B -m demo_sandbox.activation status

# Restore the preserved fitted baseline and verify its inference health.
& $python -B -m demo_sandbox.activation rollback
& $python -B -m demo_sandbox.activation status
& $python -B -m demo_sandbox.activation audit
```

The IDs above bind the recorded local Step 4 results; never substitute invented
receipts. Initialization is one-time: a repeat `init` is rejected and audited,
not a reset. Each explicit operation consumes one audit slot. Repeated `status`
and `audit` are read-only. `REJECTED` and `RECOVERED` return exit 1 (the requested
operation did not succeed normally); successful init/promotion/rollback return 0,
and an interrupted command returns 130 after attempting recovery.

For a detected pending transaction after a confirmed stopped writer:

```powershell
& $python -B -m demo_sandbox.activation audit
& $python -B -m demo_sandbox.activation recover
& $python -B -m demo_sandbox.activation status
```

Do not delete history, unlock a live writer or restore a pointer manually to stage
a video. A recovery command restores the baseline, not a candidate retry. Failed
recovery requires operator inspection; it never authorizes production deployment.

### Bounded audit and retained artifacts

Only `activation/active.json` (8 KiB), `activation/journal.json` (128 KiB), fixed
atomic-write temporary files and one lock are allowed. The journal stops at **20
operations** or the file-size bound, preserving all history without eviction. Capacity exhaustion blocks
new mutations while leaving read-only status/audit available. Recovery of an
already pending transaction updates its reserved slot, even at capacity.

Records include operation ID, action, exact evaluation receipt ID, real UTC start/
completion times, previous/target/result pointers, hashes and dataset/split identities,
status, decision/failure reason and bounded health-check diagnostics. Pointer/history
continuity and checksums are validated on reads. All artifacts and prior step journals
remain unchanged; rollback restores a pointer to the original baseline file rather
than copying a newly generated replacement. Hashes establish local consistency,
not independently authenticated approvals against a malicious host administrator.

Focused fault-injection tests (temporary roots only):

```powershell
& .\.venv-backend\Scripts\python.exe -B -m unittest discover -s backend/tests -p test_demo_activation.py -v
```

Actual local Step 5 demonstration, 18 November 2026:

- Initialized the activation pointer to the healthy fitted baseline.
- The stricter-control receipt was **REJECTED**, with reason `Sandbox evaluation
  receipt does not permit promotion`; the baseline pointer was unchanged.
- Normal PASS receipt promotion **SUCCEEDED**, selecting genuine fitted `job-002`.
  Operation ID: `458ef94a63b310bf25a9d34f113f9baad064c3739b5e02121d5e234b62b4f37a`.
- Explicit rollback **SUCCEEDED**, restoring `fitted-evaluation-champion-v1`.
  Operation ID: `c99356b6b3f247c4e97c709869abd9f18dc666f719ac6b4087c7b7b8c805d0b6`.
- Both post-promotion and post-rollback checks loaded the actual fitted model and
  correctly predicted `demo_class_1` for training sample index 71, with valid class
  probabilities. No test sample was used for these health checks.
- Final read-only audit verified four records (init, rejection, promotion,
  rollback), with `recovery_required: false`. Final active model is the healthy
  fitted baseline. There is no pending transaction.
- Fault-injection tests verified automatic recovery after post-activation health
  failure, pointer replacement failure and Ctrl+C, plus recovery from a simulated
  hard-kill pending transaction. These failures were injected only in temporary
  test roots, not into the recorded demonstration or production deployment.
- All 14 focused activation tests passed. SHA-256 comparison confirmed all **333**
  existing protected files unchanged, including production models/data, MLflow,
  DVC, deployment/workflows and every prior sandbox artifact/receipt/pointer.
  Only two new runtime files were retained: `activation/active.json` and
  `activation/journal.json`; lock and temporary files were cleaned up.
- Full backend discovery ran 196 tests: **194 passed, two native symlink skips,
  zero failures**. Dependency-free discovery ran 196 tests: **154 passed, 42
  expected skips**, including optional sklearn tests. Compilation, AST/whitespace
  checks and all 14 documented PowerShell block syntax checks passed. The printed
  training-output paths in existing tests refer to isolated temporary test data,
  not production neural-model training.

The unreviewed real candidate remains unchanged. Its legitimate evidence review
and the missing historical production holdout are separate from these synthetic
fixtures; neither issue is resolved by initializing this teaching sandbox.
