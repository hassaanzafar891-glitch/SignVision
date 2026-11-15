# Classical Random Forest baselines

`backend/tracking/train_random_forest.py` trains separate alphabet and word
RandomForestClassifier models. It never imports TensorFlow or the NN trainers.
Baseline training has since completed. The MLflow results supplied for the
completed runs are recorded below; no training was repeated for this update.

The loader opens `backend/data/db/main_dataset.db` read-only and selects exactly
`x1,y1,...,x21,y21,label` from `alphabetDataset` or `wordDataset`, excluding `id`.
It retains the existing trainers' natural SQL order. The unchanged
`split_three_way(..., seed=42)` creates duplicate-disjoint partitions targeting
approximately 60/20/20. Trees use the raw 42 numerical features, with no scaler or
label encoder needed. Only the training partition is fitted; validation/test rows
are evaluated separately. There is no tuning or train-plus-validation refit.

Fixed settings: 100 trees, maximum depth 20, `min_samples_leaf=1`,
`max_features="sqrt"`, `random_state=42`, and at most two CPU worker jobs.
No new packages were added: scikit-learn/NumPy are in `backend/requirements.txt`,
and MLflow remains in the separate `backend/requirements-tracking.txt`.
The `.venv-backend` interpreter must have those packages before training.

## Command for a future approved run

First start the existing compatible MLflow server using [HOW_TO_RUN.md](../HOW_TO_RUN.md).
Then, in PowerShell, intentionally run:

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = Join-Path (Get-Location).Path "backend"
& .\.venv-backend\Scripts\python.exe -m tracking.train_random_forest --model both --tracking-uri http://127.0.0.1:5000
```

Use `--model alphabet` or `--model word` for one task. `--help` does not train or
contact MLflow. This explicit baseline command requires tracking; the NN helper's
`SIGNVISION_TRACKING_ENABLED` variable is not used. Tracking defaults to
`http://127.0.0.1:5000`, and the experiment defaults to **SignVision Classical
Baselines**. The script refuses to resume an existing MLflow run. Tracking setup
or start failures stop execution before fitting; later failures propagate so an
incomplete run cannot silently look successful.

Open http://127.0.0.1:5000 and select that experiment. Each task creates a clearly
tagged `classical_random_forest` run with all estimator hyperparameters, dataset
counts/class distribution and ordered-row SHA-256, Git commit/dirty state,
software versions, and model artifacts. Training, validation and test each have
accuracy, macro F1 and a confusion matrix (JSON and CSV, rows actual/columns
predicted). Metrics are fractions in [0,1]. Hyperparameters are fixed in advance.

Outputs go to separate, ignored directories:

```text
backend/data/training_runs/random_forest/<alphabet|word>/<timestamp>-<UUID>/
```

Files include `random_forest.pkl`, `classes.json`, `evaluation.json`, confusion
matrix CSVs, `splits.json` (indices plus SHA-256 fingerprints), `comparison.json`,
source snapshots and `run_metadata.json` with artifact hashes. MLflow stores them
under `random_forest_baseline/`. No existing production model, dataset or
experiment is overwritten. A new invocation creates new outputs and runs.

## Completed Random Forest results

The completed MLflow runs recorded the following metrics (fractions in [0,1]):

| Model | Train accuracy | Validation accuracy | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Alphabet Random Forest | 1.0 | 0.997068 | 0.996748 | 0.996366 |
| Word Random Forest | 1.0 | 0.886364 | 0.990385 | 0.990364 |

The word baseline's validation accuracy is **88.6364%**, compared with
**99.0385%** on the test partition: a **10.4021 percentage-point** difference.
This gap shows that performance varies across these partitions. Given the small
word dataset, the high test score alone should not be treated as evidence of
uniform generalization; these aggregate scores do not establish the cause of
the gap.

## Indicative comparison with previously recorded neural networks

| Task | Random Forest test accuracy | Previously recorded NN test accuracy |
| --- | ---: | ---: |
| Alphabet | 99.6748% | 98.37% |
| Word | 99.0385% | 100% on a small test dataset |

The reference test accuracies are **alphabet 98.37%** and **word 100% on a small
test dataset**. These are historical NN results, not new measurements. NN macro
F1 is unavailable and is not invented. **Historical partition comparability is
not verified, so this comparison is indicative only.** These figures do not
establish that either model family performs better on the same held-out data.

Same seed, split proportions and counts alone do not verify that the exact same
held-out rows were used. `comparison.json` checks ordered dataset SHA-256, all
three partition-index SHA-256 fingerprints, model kind and the historical score.
The existing NN metadata does not contain partition-index hashes, so default
comparison status is **not_verified** and no accuracy-difference claim is made.
Do not add inferred hashes to historical runs to manufacture evidence.

If genuine historical split evidence is available separately, pass its metadata
with `--alphabet-nn-metadata <path>` and/or `--word-nn-metadata <path>`. Expected
fields are the existing NN `model_name`, `dataset.sha256`,
`test_metrics.test_accuracy`, plus `split_indices_sha256` with `train`,
`validation`, and `test` keys. Index fingerprints are SHA-256 of the ordered
index list encoded as compact JSON, using the same algorithm as `splits.json`.
Only matching evidence permits a like-for-like held-out accuracy difference.
Even matching splits do not establish signer-independent performance: signer IDs
are unavailable, and different vectors from the same signer may span partitions.

## Lightweight tests (no baseline fitting)

```powershell
Set-Location E:\LinguaSign
python -B -m unittest discover -s backend/tests -p test_random_forest_baseline.py -v
```

Tests use temporary databases and fake estimators, check train-only fitting and
logging through mocks, and never connect to the live MLflow server. If installed,
scikit-learn is used only to check metric formulas and construct an unfitted
classifier; that test skips in dependency-free CI. No training is done by tests.
