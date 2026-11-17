# Phase 4 readiness and evidence gaps

This review reuses the existing monitoring, controlled retraining, versioned
bundles, evaluator and plan-only promotion/rollback utilities. No new training,
deployment automation, dependencies or gate policy is introduced.

**Historical production holdout conclusion (18 November 2026): UNAVAILABLE for
both alphabet and word.** Exact local artifact and historical dataset snapshots
were recovered, but no authentic original partition membership was found. This
does not permit promotion against the verified controlled-retraining candidate.

For the final Option 2 checklist, evidence inventory, demo procedure and operator
checks, see [FINAL_READINESS.md](FINAL_READINESS.md). It preserves these blockers
and does not authorize training, policy changes or deployment.

## Final operator confirmations (18 November 2026)

The operator reports successful DVC remote verification with
`Cache and remote origin are in sync.`, and confirms the live AWS application
and monitoring work. These are operator-verified functional/synchronization
results, not independent AWS checks by the documentation editor. No screenshots,
raw transcripts, live artifact hashes or measured-drift results were supplied
with this confirmation. The previous basic CI success remains recorded below
and in [CI.md](CI.md).

Successful synchronization and live functionality do not recover original model
holdouts, validate a production-versus-candidate comparison, create a legitimate
drift reference or authorize promotion. The UNAVAILABLE/BLOCKED conclusions above
are unchanged. No new training, dataset collection or deployment is required to
present the existing evidence accurately.

| Area | Verified locally | Remaining requirement |
| --- | --- | --- |
| Drift | Bounded authenticated aggregates, JS-divergence tests and fail-closed reference checks; live monitoring functionality operator-confirmed | Representative, reviewed production aggregate reference for the exact deployed model/scaler/encoder bundle; no legitimate baseline or measured drift is supplied here. |
| Retraining | Existing 17 November controlled run completed both tasks; local MLflow task/audit records and uploaded artifacts verified read-only on 18 November | Independent provenance review and evidence-backed evaluation; no repeat training is needed for capture verification. |
| Candidate verification | Existing read-only command passed; all 11 listed files, receipt, manifest, metadata and exact recorded splits verified | Capture remains unreviewed; local consistency checks do not constitute independent approval or authenticate remote storage. |
| Evaluation | Exact-dataset, partition, artifact and regression checks remain enforced | Authentic production training-time manifests, independently reviewed manifest pins, and identical held-out test indices. Production manifests are currently missing. |
| Deployment/rollback | Identity-bound, explicit approval plans; prior version and original gate required for rollback | Verified live identity, independent operator approval and a separately reviewed coordinated execution/health-check procedure. No executor or AWS action is added. |

The current processed SQLite database is the pinned DVC version
`326d65539174dffdad54606bb2271e3c`. Reproducible local splits are alphabet
9,218/3,070/3,075 and word 351/88/104 train/validation/test rows. These counts
do **not** establish historical production holdout provenance or signer independence.
Do not use recorded accuracies, reconstructed seed-42 indices or a successful
unit-test run as substitute evaluation evidence.

## Safe local checks

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = Join-Path (Get-Location).Path "backend"
$Python = Join-Path (Get-Location).Path ".venv-backend\Scripts\python.exe"
& $Python -B -m unittest discover -s backend/tests -p "test_*.py" -v
python -S -B -m unittest discover -s backend/tests -p "test_*.py" -v
& $Python -m tracking.versioning verify --help
git diff --check
```

Use the read-only verification command in [RETRAINING.md](RETRAINING.md) only
when a completed candidate and trusted hash exist. It verifies consistency, not
the authenticity of a claimed review or a real MLflow server's state. Synthetic
fixtures remain tests only. At the initial Phase 4 preflight no completed candidate
was found. The subsequent database investigation found a completed candidate;
completion is not provenance review, gate approval or deployment authorization.

## Completed candidate verified (18 November 2026)

Candidate from the 17 November 22:17-22:32 Pakistan-time session:

```text
backend/data/training_runs/candidates/20261117T171728Z-3593468bcb5d4eabb3a01028050a71fb/bundle
```

The existing read-only CLI completed successfully with `capture_verified: true`,
`promotion_approved: false` and `deployment_performed: false`:

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = Join-Path (Get-Location).Path "backend"
& .\.venv-backend\Scripts\python.exe -B -m tracking.versioning verify --source backend/data/training_runs/candidates/20261117T171728Z-3593468bcb5d4eabb3a01028050a71fb/bundle --source-sha256 877b90809fd183c4b9f50ffeef88fa85fad49ba6aabeae2d5974e4a18dcc9a7f --dataset backend/data/db/main_dataset.db
```

Identities checked:

- Canonical candidate manifest SHA-256:
  `877b90809fd183c4b9f50ffeef88fa85fad49ba6aabeae2d5974e4a18dcc9a7f`.
- Canonical frozen-plan SHA-256:
  `6047f6112e617fe7a5f3ca0ba9bd36e7b4585cf5546c25d1039962d0294e2b5d`.
- Raw SQLite SHA-256:
  `e6d6c59c7a0c7a207388d290c7ddae7f05ba1aa1507df8b5d293ac4ec13c33a9`;
  DVC MD5 matches the pinned version above.
- Training Git commit `316a9d8daba4777e7bc8886a46850ee833cbef4f`, with
  `import_checkout_dirty: true`. All captured source snapshots and the captured
  DVC pointer match their frozen-plan hashes; the commit alone does not identify
  the dirty checkout's complete training code.

The manifest pin was read from the existing completion receipt and corroborated
against the saved local MLflow audit bundle. All 11 manifest-listed files, the
manifest itself, the six model/preprocessing files in their task uploads, and
both uploaded task metadata files match the candidate bytes. Local copies were
checked directly; no MLflow client/server mutation or DagsHub request was made.

| Task | MLflow run ID | Status | Epoch records | Train / validation / test |
| --- | --- | --- | --- | --- |
| Alphabet | `82ea15014c4f4e60aefa7718622d68d3` | FINISHED | 25 | 9,218 / 3,070 / 3,075 |
| Word | `cf222a5e454e42a8a2c4aadff8bfe89d` | FINISHED | 50 | 351 / 88 / 104 |
| Bundle audit | `8ae1f489ee764261934dba88f16043c5` | FINISHED | Not a training run | Both tasks |

Run IDs and statuses match the repository-root `mlflow.db`. Every recorded epoch
metric and both final test metrics match the local MLflow SQL records. Both task
metadata files have empty `tracking_errors`. The audit tags match the candidate
version and frozen-plan identity.

| Saved artifact | Raw-file SHA-256 |
| --- | --- |
| `alphabet_model.h5` | `de95048e887e3349db5075558ad48eacfa419ad5fec0d70c6c147e6522cd0838` |
| `alphabet_scaler.pkl` | `975a200f76387cc5fd0adea3949c94f8543f7c8ae5fb9327495d45f3c47a6258` |
| `alphabet_label_encoder.pkl` | `046396fb488f14c943a3aa780de21a072114df97dc65e8e85c325fdab11b2fb2` |
| `word_model.h5` | `8ddd8f8926576d9f49bf6ec99804179f0beeae8a52fcb496e09f9a026baeb857` |
| `word_scaler.pkl` | `182a5d689cf9324ed5f9d581bcd48bcfe22504111ae87f1b5d8786e43b61d294` |
| `word_label_encoder.pkl` | `dea9b1af201b33fc0a05f43762145c866120d65842d8aa2cbe61bf6e3fc90f72` |

Both tasks' recorded partitions completely cover the exact ordered table rows,
keep identical feature vectors within one partition, and include every class in
each partition. Rerunning `split_three_way(..., seed=42)` reproduced
the exact index lists and order. Canonical test-index hashes:

- Alphabet: `3de301e1f6abf917d3f531ded93e715ca90bbd58e21f2fdc982961651b980d05`.
- Word: `3b3bc59318df0fe6d41ba68af014dd30b72a6f92ef77cc285b8ff870cc37e283`.

Existing recorded results, checked against MLflow rather than recomputed:

| Task | Test accuracy (fraction) | Test loss |
| --- | --- | --- |
| Alphabet | 0.9869918823242188 (about 98.70%) | 0.06830180436372757 |
| Word | 1.0 (100%, 104 test rows) | 0.00010311270307283849 |

No model inference, unpickling, training, provenance attestation or gate approval
was performed. These are candidate training records, not new production evaluation
records or proof of comparability with historical production test results. Both
provenance statuses remain `captured_training_time_unreviewed`. Production holdout
evidence, independent review, a legitimate drift reference and live deployment
identity remain separate blockers; promotion/rollback eligibility is not granted.

## Historical production provenance investigation (18 November 2026)

### Recovered artifact and dataset lineage

Git commit `a298fb909fa34020a7bc8121a7cd2874db14ada5` (18 May 2026) contains all
six current `backend/data/models/` model/scaler/encoder files **byte-for-byte**,
and `backend/data/db/main_dataset.db` with the same raw SHA-256 and DVC MD5
recorded above (6,864,896 bytes). Co-commitment establishes a historical snapshot,
not proof that those rows or partitions were actually consumed by training.

Production artifact SHA-256 identities checked:

| Task | Model | Scaler | Label encoder |
| --- | --- | --- | --- |
| Alphabet | `8755d6bc3328edc98c2eeec595817463ea47e6e6d519c2e347c5e9f8806f6371` | `5d7a5f3e22577eea4724e5021d33d2ee379d72e0e36b4942eea0e942ae8789d7` | `046396fb488f14c943a3aa780de21a072114df97dc65e8e85c325fdab11b2fb2` |
| Word | `83c32bc5a01b4e67f27c3077822a2406a8b944ccd356076982aaaefea62fc028` | `b32f4e16d9cb9e9fb856bd028bb311f6de300187407735116e51f3255ed684e2` | `dea9b1af201b33fc0a05f43762145c866120d65842d8aa2cbe61bf6e3fc90f72` |

The matching baseline-import manifests are:

- Alphabet: `backend/mlruns/435313127739737787/9593a88170af42a196d4d9ce13ad1eb1/artifacts/baseline/manifest.json`;
  import run `9593a88170af42a196d4d9ce13ad1eb1`.
- Word: `backend/mlruns/435313127739737787/5e06dd6076544f5b9894285ecb2e0ebf/artifacts/baseline/manifest.json`;
  import run `5e06dd6076544f5b9894285ecb2e0ebf`.

Both bind the exact production model/scaler/encoder hashes, but explicitly state
`historical_training_provenance_verified: false`, no historical metrics, and
`configuration_provenance: current_script_configuration`. They are later imports,
not original training runs. The saved backup under
`E:\SignVision_Backup\phase4-20261114T184933686Z-8dd81a1f\` has the same import
manifests and dataset MD5; it supplies no missing original splits.

The DVC pointer was introduced in available Git history by
`d2931636117e2a6fcb40ec028e3a881ccfc5d855` and
`d10b15499e86fd40fbcbe05f28e4f5a53a42d026` (14/15 November). Both record
`326d65539174dffdad54606bb2271e3c`. No alternate pointer version or DVC training
pipeline/lock binding these production models to a split was found. DVC data
identity alone does not establish model-training provenance.

### Why the other records do not recover the original holdout

- At the May commit, `backend/PSL/alphabet_recognition/train_alphabet_model.py`
  uses `train_test_split(..., test_size=0.2)` without a seed, followed by
  `validation_split=0.20`. No selected indices or RNG state are saved. The saved
  scaler is fitted before that internal validation split. These are source-code
  observations, not verified historical execution claims.
- The co-committed word script uses stratified `train_test_split` with seed 42,
  then `validation_data=(X_test, y_test)` and evaluates on the same rows. If this
  script produced the model, that test was reused for validation, rather than
  untouched. A deterministic reconstruction still depends on unverified execution,
  row order and library versions; it must not be labeled original evidence.
- Read-only HDF5 inspection found Keras 2.13.1 architecture/training configuration,
  weights and optimizer state, but no split/dataset attributes or partition
  records. Static pickle inspection found scaler statistics/sample-count fields,
  not row membership. No models were loaded for inference or pickles executed.
  Weights, scaler moments, sample counts and confusion-matrix images cannot
  authenticate the original held-out rows.
- Current database tables have 44 columns (id, 42 features, label), with no
  split, signer or session fields. The available backend Git object history has
  no saved original index arrays, split manifests or notebooks containing them.
- November 15 neural metadata has dataset fingerprints, seed/settings, counts
  and scores, but no exact partition lists. More decisively, its model hashes
  differ from production. The READY Registry versions point to these newer
  artifacts, not the production files:

| Registry version / task | November 15 training run | Saved model SHA-256 |
| --- | --- | --- |
| `SignVision-Alphabet` v1 | `ea6666a03f814d35bd35588a7edf6a09` | `da24fa5658c4d89bd916fb727dbf89443c6454bfdf63e739e09c910393ba01bc` |
| `SignVision-Word` v1 | `89a7c4b32367407e84f6ed7d85462a76` | `62d3cbe6c05050c411738a48f9bb62997168582adc96b462d7007ba20242c703` |

Their exact metadata files are
`backend/mlartifacts/2/<run-id>/artifacts/training/run_metadata.json`, with matching
local copies under `backend/data/training_runs/<task>/<November-15-run-directory>/`.
The 98.37%/100% documented scores belong to these runs, not verified measurements
of the production hashes above. `mlflow_runs_evidence.json` and
`mlflow_models_evidence.json`, added in commit
`221af54b166779023dc16b4edd39e5c49631f55f`, confirm those run/source links but
contain no original partition fields.

RF `splits.json` records under `backend/data/training_runs/random_forest/` apply
to the RF baselines. The November 17 candidate's actual partition records and
run IDs are documented above and apply to its distinct model hashes. Neither
can retrospectively establish production splits. No production
`alphabet_provenance.json` or `word_provenance.json` exists.

**Comparison decision:** the existing gate cannot legitimately approve a
production-versus-candidate comparison on the current dataset. Same bytes for a
historical dataset, matching labels or a reconstructed seed do not prove a common
unseen holdout. Live EC2 artifact identity also remains unverified; this read-only
local investigation did not contact AWS. Even a later live-hash match would not
recover the missing holdout.

Search scope: all currently available local Git refs/history, production H5 and
preprocessing files, current DVC pointer/history, both local SQL stores, the file
MLflow store, saved training/RF/candidate artifacts, tracked evidence JSON,
documentation and the existing backup. Remote-only/unfetched history, private
original logs/media and unmirrored DagsHub runs were not accessed. Additional
authentic external records could change the conclusion; none were fabricated.

### Scientifically valid alternative (recommendation, not implemented)

Use a **new prospective external evaluation set** collected after freezing both
bundles, with documented provenance and independently checked PSL labels. Include
all supported alphabet/word classes and representative camera, lighting and
session conditions. Retain consent and privacy controls; collect pseudonymous
participant/session identifiers for grouping, not public user identities.
Historical signer IDs are unavailable, so do not claim proven signer independence
from the production training population.

Freeze and hash both six-file bundles before collection. Keep the new evaluation
set entirely out of training, scaler/encoder fitting, validation, threshold tuning
and model selection. Check overlap/exact duplicates against all available old
data, group correlated frames by recording/session, and document the residual
unknown historical overlap. Do not select rows from the current database and call
them newly unseen. Predeclare class coverage, sample-size/precision targets,
metrics and the decision rule; do not collect until a desired score is reached.

Evaluate both unchanged models on the same new examples with their own saved
preprocessing and a reviewed consistent extraction protocol. Report accuracy,
macro F1, per-class recall/confusion matrices and paired uncertainty estimates
using independent sessions/participants as sampling units, rather than treating
video frames as independent. Distinguish landmark classification from end-to-end
browser/MediaPipe performance. No retraining is needed to compare frozen models
on that genuinely new set, and no result is claimed here.

This would be a **new evaluation protocol**, not recovered historical provenance.
It requires independent approval and a versioned gate/schema/policy revision
because today's gate pins the old DVC dataset and original training-time test
indices. Record external-set identity, group assignments, artifact hashes and
protocol version in new evidence. Do not weaken or silently bypass the current
gate, replace production, create reviewed historical manifests, or use the
candidate's existing test set as a substitute. Until authentic original evidence
or an explicitly approved alternative protocol exists, promotion stays blocked.

## Evidence needed before proceeding

1. Through a separately authorized operator process, collect and review a
   representative aggregate traffic window. Bind it to the actual deployed
   artifact bundle and class bins, retain provenance, and calibrate sample/window
   thresholds. Follow [MONITORING.md](MONITORING.md). No raw frames, landmarks or
   identities should be added. Without this evidence, report `baseline_unavailable`.
2. Recover authentic original training-time production partitions and preprocessing
   evidence, then independently pin manifests and verify the deployed bytes.
   If recovery is impossible, an independently reviewed future evaluation-policy
   revision is required; this implementation does not substitute a new reference
   model or dataset. See [EVALUATION_GATE.md](EVALUATION_GATE.md).
3. Do not repeat the completed training for this candidate's verification.
   Independently review its already verified capture, then evaluate both tasks
   only when authentic production evidence permits it and require both gates to
   pass. Any future training still requires explicit authorization; drift never
   starts training or approves anything.
4. Retain the verified previous bundle and original gate evidence. Promotion and
   rollback commands write plans only. Before any separate execution, verify live
   identity, approve the change, stage the entire six-file bundle, and plan health
   checks and a coordinated rollback. Do not hot-swap individual model/scaler files.

Local checks alone do not verify EC2 artifact identity, throughput, live access,
operational alerts, DagsHub availability or production rollback. The operator's
successful live-application/monitoring and DVC sync checks are recorded separately
above; previous hosted CI success is linked in the final checklist. The local
verification work did not access AWS, change the live deployment, restart Docker
or alter credentials. Publishing reviewed source changes is separate from model
approval or deployment. Existing CI discovers the tests without heavy dependencies.

## Resolved MLflow database investigation (17 November 2026)

The earlier root `mlflow.db` checksum changes coincide with separately running
controlled training and its artifact upload, not reproduced test side effects.
Read-only inspection found:

- A local MLflow server launched with `--backend-store-uri sqlite:///mlflow.db`
  on port 5000: launcher PID 9380, Python child 21696, and Uvicorn process chain
  1252/7464 with workers 20864, 23256, 24156 and 23580 at inspection time. PIDs
  are diagnostic observations, not stable configuration. The relative store URI
  allows a repository-root store distinct from `backend/mlflow.db`.
- Root-store alphabet and word runs have 25 and 50 recorded epochs, respectively.
  Their timestamps span 22:17-22:28 Pakistan time, followed by the candidate audit
  upload completing at 22:32. Both task run IDs and the audit run ID exactly match
  the completed candidate receipt/metadata under
  `backend/data/training_runs/candidates/20261117T171728Z-3593468bcb5d4eabb3a01028050a71fb/`.
- The responsible application paths are `TrainingRun.callback()` and
  `record_test()` logging metrics, `TrainingRun.__exit__()` uploading task
  artifacts, and `tracking.retraining.execute()` uploading the candidate audit
  run. Server SQL writes for those client requests are expected MLflow activity;
  an idle heartbeat was not established as the cause. The originating training
  client had exited before inspection; an exact historical worker PID per write
  cannot be recovered from run records alone.
- Five idle observations over 20 seconds showed identical raw hashes, SQLite
  change counters, pages and all 71 logical table contents. The isolated optional
  MLflow test (7 tests) and full backend suite (101 tests) also left **both**
  `mlflow.db` and `backend/mlflow.db` unchanged in raw and logical comparisons.

No additional Python/test fix was warranted: the existing optional test already
selects temporary tracking/registry URIs and working directory before SDK import
in a separate process. That isolation remains in place. No record was deleted,
restored, reset or modified by this investigation; no training or server restart
was executed. The earlier suspicion of a test-induced root-store write is not
supported by these observations. Use an explicit absolute SQLite store URI on
future deliberate server starts to avoid working-directory ambiguity; do not
move, merge or repoint these existing stores without a separately reviewed plan.
