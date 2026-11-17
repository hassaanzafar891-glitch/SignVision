# Evidence-backed candidate approval (no deployment)

This change adds a manual evaluation gate to the existing SignVision CI workflow.
It does not train, deploy, restart services, publish a model version, change Docker
ports/Caddy, or contact the AWS application. Basic push/PR CI remains lightweight.

**Current status: blocked for real-model approval, not operationally certified.**
The local processed dataset MD5 matches `326d65539174dffdad54606bb2271e3c`.
Historical NN records have test metrics but no exact training/validation/test
partition evidence linking those partitions to the production artifact bundles.
There are no production provenance manifests. The evaluator fails before loading
models in this situation. It does not treat old accuracy values, RF split files,
or a freshly generated seed-42 split as proof of the production models' holdout.
A successful GitHub workflow run has not been performed.

## Gate policy

`backend/tracking/evaluation_gate.py` uses only the standard library. Both records
must use schema version 1 and protocol `saved-scaler-encoder-softmax-v1`, with:

- Correct `production`/`candidate` role and the same task (`alphabet` or `word`).
- Model task/family identity and SHA-256 hashes of the model, scaler and encoder.
- The reviewed DVC MD5, raw SQLite SHA-256, ordered-row SHA-256, correct table,
  all 42 feature-column names, dataset row count and identical class labels.
- Reviewed held-out provenance; exact test-index and test-row hashes; positive
  partition counts with complete coverage and duplicate-disjoint verification.
- A timezone-aware evaluation timestamp, finite test accuracy in [0,1] and finite,
  non-negative test loss. Missing values, booleans, NaN, Infinity, malformed JSON
  and duplicate JSON keys are rejected.

Candidate accuracy must be **at least** production accuracy, with no regression
tolerance. Equality passes. Loss must be valid but is informational; this version
does not impose loss, latency, per-class recall or statistical-significance limits.
Both alphabet and word must pass before the combined `approval.json` is written.
That report binds canonical evaluation-record hashes, artifact hashes, dataset,
holdout and Git commits. It is candidate evidence, not permission to deploy.
The standalone gate CLI compares records; it does not authenticate arbitrary JSON.
In CI, records must come from the trusted evaluator and reviewed artifact evidence.

## Required training-time provenance

Each trusted model bundle contains the unchanged files:

```text
alphabet_model.h5                 word_model.h5
alphabet_scaler.pkl               word_scaler.pkl
alphabet_label_encoder.pkl        word_label_encoder.pkl
alphabet_provenance.json          word_provenance.json
```

Each task's provenance JSON requires these fields (the following is a schema
description, not substitute historical evidence):

| Field | Required content |
| --- | --- |
| `schema_version` | Integer `1` |
| `task` | `alphabet` or `word` |
| `dataset.dvc_md5` | `326d65539174dffdad54606bb2271e3c` |
| `dataset.sha256` | SHA-256 of the exact stable SQLite file used by training |
| `dataset.ordered_rows_sha256` | Canonical SHA-256 of all selected feature/label rows in original SQL order |
| `dataset.table` | `alphabetDataset` or `wordDataset` |
| `dataset.feature_columns` | Ordered `x1,y1,...,x21,y21` list |
| `artifacts` | `model`, `scaler`, `label_encoder`: raw-file SHA-256 digests |
| `partitions` | Nonempty `train`, `validation`, `test` lists of original zero-based row indices |
| `provenance.status` | `reviewed_training_time_evidence` |
| `provenance.reviewed_by` | Reviewer identity |
| `provenance.evidence_reference` | Immutable training/run evidence reference binding the bundle to its partitions |

Indices must partition the exact dataset, cover every class in every partition,
and keep identical feature vectors in one partition. Candidate training and
validation must exclude every production test example and its exact feature
duplicates. Candidate and production test indices must be identical, including
their order. No scaler or label encoder is refitted during evaluation.

Canonical SHA-256 uses UTF-8 JSON with sorted keys, compact separators,
`ensure_ascii=False`, and no non-finite values, through `canonical_sha256()`.
Rows are the ordered SQL results `SELECT x1,y1,...,x21,y21,label FROM <table>`.
Manifest and evaluation-record hashes are canonical JSON hashes, stable across
Windows/Linux line endings. Model/scaler/encoder and SQLite hashes are raw-file
hashes. The evaluator checks both SQLite content and original SQL row ordering.

The provenance review is a **human trust boundary**, not a proof that can be
manufactured by filling JSON fields or hashing today's data. Reviewers must
establish authentic training-time evidence, including preprocessing fitted only
on training data and validation-driven tuning that did not use the test set.
Never label inferred/reconstructed partitions as verified historical evidence.
If the original production holdout cannot be established, promotion stays blocked.
An independently collected, never-used evaluation set with an appropriate review
policy, or a separately trained fully traceable reference model, needs a future
policy revision; neither is silently substituted by this implementation.

## Reproducible saved-model evaluation

`backend/tracking/evaluate_models.py` checks provenance and bundle hashes before
deserializing anything. It validates the exact DVC dataset without changing it,
rejects pending SQLite WAL/journal sidecars, checks partition coverage and duplicate
leakage, and evaluates **only** the recorded test rows. It loads each saved Keras
model with `compile=False`, applies its saved scaler, and encodes labels with its
saved encoder. It validates softmax probabilities, computes accuracy and categorical
cross entropy with clipping epsilon `1e-7`, and writes new records.

Production and candidate are evaluated under the same protocol on the same
verified holdout. Old training `test_accuracy`/`test_loss` fields are not accepted
as replacements. The metric measures classification on processed landmarks, not
browser capture, MediaPipe extraction, timing or end-to-end deployed performance.
No signer IDs exist, so signer-independent performance remains unverified.

Use only trusted pickles/H5 bundles; pickle deserialization can execute code.
Never download arbitrary artifacts or execute candidate application code with
secrets. Evaluation dependencies are isolated in `backend/requirements-evaluation.txt`.
Direct packages are pinned; a complete transitive lock/image is not yet supplied,
so cross-platform bitwise identity is not promised.

After legitimate evidence exists, reviewers set these non-secret repository or
`candidate-approval` environment variables to **canonical** production manifest hashes:

- `SIGNVISION_PRODUCTION_ALPHABET_MANIFEST_SHA256`
- `SIGNVISION_PRODUCTION_WORD_MANIFEST_SHA256`

Pin them only after an operator verifies the model/scaler/encoder hashes against
the actual deployed bundle. A local directory called production is not proof that
it matches AWS. This implementation has not contacted AWS or established that
identity. Do not generate pins from unverified historical claims.

Example local commands, only once evidence is legitimate (not executed here):

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = Join-Path (Get-Location).Path "backend"
# Set the two reviewed manifest-hash variables before this preflight.
& .\.venv-backend\Scripts\python.exe -m tracking.evaluate_models --check-evidence --production-dir backend/data/models --candidate-dir candidates/reviewed-bundle
# Install requirements-evaluation.txt separately if evaluation dependencies are absent.
$Output = Join-Path $env:TEMP ("signvision-evaluation-" + [guid]::NewGuid().ToString("N"))
& .\.venv-backend\Scripts\python.exe -m tracking.evaluate_models --production-dir backend/data/models --candidate-dir candidates/reviewed-bundle --output-dir $Output
& .\.venv-backend\Scripts\python.exe -m tracking.evaluation_gate --production "$Output/alphabet_production.json" --candidate "$Output/alphabet_candidate.json"
```

Use a fresh output path. No existing evaluation outputs, models or experiments
are overwritten. If the word gate fails after the alphabet passes, records may
remain for diagnosis, but **no combined approval file is issued**.

## GitHub workflow and repository configuration

The existing `.github/workflows/frontend-ci.yml` retains backend/frontend checks.
The third job, `candidate_approval`, runs only for manual dispatch from `main` with
`evaluate_candidate=true`, and only after both basic jobs succeed.

Before using it:

1. Review the workflow/evaluator changes onto `main` through your normal process.
   No branch, commit or push was changed automatically here.
2. Configure the `candidate-approval` GitHub environment with required reviewers,
   prevent self-review where supported, and restrict it to `main`. Environment
   reviewer availability depends on repository visibility/plan. Merely naming an
   environment does not configure these protections.
3. Establish production holdout evidence and reviewed manifest hash variables.
4. Ensure the existing `DAGSHUB_USERNAME` and `DAGSHUB_TOKEN` secrets grant read
   access to the pinned dataset. No AWS or deployment secrets are used.
5. Make trusted candidate bundle files and provenance available on a reviewed
   candidate ref in this same repository. Artifact delivery via DVC/MLflow for
   candidate bundles is a future extension; the current workflow checks them out.
6. Dispatch **SignVision CI** on `main`, select `evaluate_candidate`, and supply
   `candidate_ref` (prefer an immutable commit SHA) and `candidate_bundle` (path
   relative to that checkout). Only model artifacts are used from that ref.

Evidence preflight runs before dataset/ML package downloads. The DVC credentials
exist only in the retrieval step's environment and an ephemeral `.dvc/config.local`.
Command output/arguments are not logged, and that file is removed in `finally`.
DVC fetches only `main_dataset.db.dvc`; the evaluator checks its pinned MD5 and
provenance SHA-256. Heavy TensorFlow dependencies are installed only in this manual
job. No `fit`, training command, SSH, Docker, AWS or deployment command is present.

On success the workflow uploads `gated-candidate-<run-id>-<attempt>` with four
evaluation records and `approval.json`. Failed preflight, downloads, inference or
either gate fail the job and prevent approval upload. A green ordinary CI run or
a skipped manual approval job is **not** candidate approval. No job deploys the
approved candidate; any future deployment must verify the exact artifact/commit
bindings and add a separate reviewed deployment design.

References: [GitHub environment protection](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments),
[artifact action](https://github.com/actions/upload-artifact/tree/v6).

## Local checks and synchronization

```powershell
Set-Location E:\LinguaSign
& .\.venv-backend\Scripts\python.exe -B -m unittest discover -s backend/tests -p "test_*.py" -v
python -S -B -m unittest discover -s backend/tests -p "test_*.py" -v
git diff --check
```

Tests use synthetic evidence, temporary SQLite files and mocked inference. They
exercise metric validation/regression, artifact/dataset/partition mismatch,
leakage, saved-scaler transform-only behavior and all-or-nothing approval. No
real model is trained, no DVC download is performed, and no live server is needed.

The inspected local branch is `feature/browser-camera`; `feature/mlops-phase2`
was not present in the local/remote-tracking branch listing. Branch equivalence
with AWS is not established. Review and synchronize the new evaluation modules,
tests, evaluation requirements and docs into the development branch through a
normal reviewed change. The workflow belongs in GitHub's reviewed `main` branch.
Do not overwrite AWS-only Compose/Caddy changes, alter localhost port bindings,
switch/restart production, or copy model/data/credential directories as part of
this code synchronization. Nothing was merged, force-pushed, deployed or run on AWS.
