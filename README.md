# SignVision: Phase 1 MLOps for Pakistan Sign Language

[![SignVision CI](https://github.com/hassaanzafar891-glitch/SignVision/actions/workflows/frontend-ci.yml/badge.svg)](https://github.com/hassaanzafar891-glitch/SignVision/actions/workflows/frontend-ci.yml)

SignVision is an MLOps course project for alphabet and word recognition from
42 hand-landmark features. Phase 1 focuses on validated data, reproducible
partitioning, tracked experiments, classical baselines and automated verification.

- **GitHub:** [hassaanzafar891-glitch/SignVision](https://github.com/hassaanzafar891-glitch/SignVision)
- **DagsHub:** [hassaanzafar891-glitch/SignVision](https://dagshub.com/hassaanzafar891-glitch/SignVision)
- **Setup and commands:** [HOW_TO_RUN.md](HOW_TO_RUN.md)

## Phase 1 implementation

| Area | Implementation |
| --- | --- |
| Dataset versioning | DVC tracks the processed SQLite database; Git stores its pointer. |
| Validation | Automatic read-only integrity, schema, label, feature and sample-count checks before training data is loaded. |
| Partitioning | Shared `split_three_way()` with seed 42 and duplicate-disjoint train/validation/test partitions. |
| Neural models | Separate alphabet and word classifiers; scalers are fitted only on training data. |
| Classical baseline | Separate Random Forest classifiers with fixed settings and train-only fitting. |
| Experiment tracking | MLflow logs configuration, dataset fingerprints, Git provenance, metrics and artifacts. |
| Model registry | Existing alphabet and word version-1 entries verified; serving readiness remains unverified. |
| Reproduction | One entry point validates by default and requires `--train` to fit models. |
| Verification | Backend tests and lightweight GitHub Actions checks; no training or deployment in CI. |

The application provides browser-webcam inference through FastAPI and a Next.js
frontend, with Urdu text/speech and sentence-building controls. Application
inference uses existing production artifacts and does not require retraining.

## Data and provenance

The training input is `backend/data/db/main_dataset.db`, retrieved using DVC.
It contains `alphabetDataset` (15,363 rows, 37 classes) and `wordDataset`
(543 rows, 5 classes). Inputs are `x1,y1,...,x21,y21`; `label` is the target and
`id` is excluded from model features.

The application and supplied processed data come from the upstream project
credited under [Source attribution and license](#source-attribution-and-license).
Original raw images/videos are private and unavailable in this submission.
Reproduction begins with processed data; the original media extraction and signer
provenance cannot be independently reconstructed from this repository.

Validation checks required columns, finite numerical features, non-empty text
labels, at least 15 samples, two classes and three samples per class. The splitter
separately checks class coverage and whether valid duplicate-disjoint partitions
can be formed. These checks do not prove that labels are semantically correct.

## Models and evaluation

Both neural trainers and the RF baseline use approximately **60% training, 20%
validation and 20% testing**. Identical feature vectors stay in one partition;
actual row ratios vary with duplicate-group sizes. For unchanged ordered input,
splitting is reproducible. Neural weights are not guaranteed bit-identical.
Signer IDs are unavailable, so these are **not signer-independent evaluations**.

| Model | Architecture/settings | Recorded test accuracy |
| --- | --- | ---: |
| Alphabet NN | Dense 120/64/37, dropout 0.3, Adam; 25 epochs, batch size 1 | 98.37% |
| Word NN | Dense 256/128/64/5, dropout 0.3, Adam; 50 epochs, batch size 32 | 100% on 104 test examples |
| Alphabet RF | 100 trees, depth 20, seed 42, two CPU workers | 99.6748% |
| Word RF | Same fixed RF settings | 99.0385% |

These are existing experiment results, not new measurements from this documentation
update. Historical NN partition-index hashes are unavailable, so NN/RF comparisons
are **indicative only**. The word RF validation accuracy is 88.6364%, versus
99.0385% test accuracy, a 10.4021 percentage-point gap. Small sample size and
partition-dependent performance limit generalization claims. RF test macro F1
and full results are in [RANDOM_FOREST_BASELINES.md](docs/RANDOM_FOREST_BASELINES.md).

## Validate and reproduce

After following the environment and DVC setup in [HOW_TO_RUN.md](HOW_TO_RUN.md),
run these safe PowerShell checks:

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = Join-Path (Get-Location).Path "backend"
& .\.venv-backend\Scripts\python.exe -m tracking.train_all
& .\.venv-backend\Scripts\python.exe -B -m unittest discover -s backend/tests -p "test_*.py" -v
```

The first command validates only. To intentionally reproduce training, with a
compatible MLflow server already running:

```powershell
& .\.venv-backend\Scripts\python.exe -m tracking.train_all --train --tracking-uri http://127.0.0.1:5000
```

That command validates both tables, checks server access and runs both NNs and
both RF baselines. New outputs go under `backend/data/training_runs/`; production
models, existing experiments and datasets are not overwritten. No inference model
is automatically replaced. The workflow was verified using mocked training calls;
actual retraining was not performed during Phase 1 verification.

## Tracking, registry and verification limits

Local experiments use `backend/mlflow.db` and `backend/mlartifacts/`; the older
`backend/mlruns/` store remains separate. MLflow client dependencies are in
`backend/requirements-tracking.txt`, separate from inference dependencies.

Phase 1 verification found `SignVision-Alphabet` v1 and `SignVision-Word` v1 both
READY, with existing `.h5` source files. Those sources lack `MLmodel` flavor
manifests, so catalog registration does not establish MLflow serving readiness.
NN logging is best effort; inspect local `tracking_errors` for failed uploads.

The latest Phase 1 backend verification passed **42 tests**. A dependency-free CI
run passed 38 with four expected dependency-related skips. Tests cover invalid
and corrupted data, split separation, train-only scaling, mocked RF fitting,
temporary MLflow storage and workflow orchestration. They do not establish
real-world recognition accuracy or deployment readiness.

## Repository guide

- `backend/tracking/`: validation, partitioning, tracking, baseline and workflow tools.
- `backend/PSL/`: neural training and inference modules.
- `backend/data/models/`: existing production model/scaler/encoder artifacts.
- `backend/tests/`: automated checks using temporary fixtures.
- `frontend/`: Next.js application; Python training uses a separate environment.
- [HOW_TO_RUN.md](HOW_TO_RUN.md): PowerShell setup, DVC, tracking, training and application commands.
- [docs/CI.md](docs/CI.md): CI scope and limitations.
- [docs/BROWSER_CAMERA_DOCKER.md](docs/BROWSER_CAMERA_DOCKER.md): browser-camera and Docker notes.

## Project Attribution and License

SignVision is an MLOps-focused extension of the
[Pakistan Sign Language Detection System](https://github.com/waqaswajla/pakistan-sign-language-detection-system).

This project extends the existing application with dataset versioning
using DVC, automated data validation, reproducible training,
classical ML baselines, MLflow experiment tracking, model registration,
and CI/CD workflows.

The original application and dataset remain attributed to their
respective source. The project retains the original MIT License.
