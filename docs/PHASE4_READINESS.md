# Phase 4 readiness and evidence gaps

This review reuses the existing monitoring, controlled retraining, versioned
bundles, evaluator and plan-only promotion/rollback utilities. No new training,
deployment automation, dependencies or gate policy is introduced.

| Area | Verified locally | Remaining requirement |
| --- | --- | --- |
| Drift | Bounded authenticated aggregates, JS-divergence tests, invalid/missing references fail closed | Representative, reviewed production aggregate reference for the exact deployed model/scaler/encoder bundle; no legitimate baseline is supplied here. |
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

Local checks do not verify EC2 identity, throughput, live monitoring access,
operational alerts, hosted CI, DagsHub availability or production rollback. The
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
