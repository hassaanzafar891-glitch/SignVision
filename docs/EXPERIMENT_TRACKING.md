# Phase 3: MLflow experiment tracking

## Import existing baselines

This step archives the existing alphabet and word bundles as two MLflow experiment
runs. It does not train, evaluate, load TensorFlow, unpickle files, or overwrite
models. It logs raw artifacts, not deployable MLflow model flavors or Model
Registry versions. Every invocation creates two new runs; it does not deduplicate.

Each run contains the existing `.h5`, scaler, label encoder, current training
script, and a `baseline/manifest.json`. The manifest includes SHA-256 hashes and
byte sizes for all four source files, import time, import checkout commit/dirty
state, and configuration extracted from the current script's syntax.
The complete configuration includes layers, dropout, scaler, compile settings,
fit settings, and split options. Fit/compile/split settings and layers also appear
as parameters prefixed `current_script_config.`. Expressions such as `num_classes`
are recorded as expressions, not guessed values. An omitted seed remains omitted.

These are **current script settings, not verified historical training parameters**.
The Git commit describes the checkout at import time, not the original training
revision; a dirty checkout is recorded explicitly. Git absence is recorded as
unknown. Original training dataset, split membership, and historical accuracy/loss
are unknown. No metrics are logged, including placeholder zeros. The existing
confusion-matrix image is not imported because its association with the current
alphabet model is unverified. Datasets are not read or uploaded.

## Setup and local import (PowerShell)

Use a separate tracking environment (Python 3.10–3.12); inference requirements
remain unchanged. The following commands are instructions, not commands executed
during implementation:

```powershell
Set-Location E:\LinguaSign
python -m venv .venv-tracking
.\.venv-tracking\Scripts\python.exe -m pip install -r backend/requirements-tracking.txt
Set-Location backend
..\.venv-tracking\Scripts\python.exe -m tracking.import_baselines
..\.venv-tracking\Scripts\mlflow.exe ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 and select `LinguaSign Baselines`. Inspect the two runs,
their configuration parameters, checksum tags, and `baseline` artifacts; their
metrics should be empty. The default store is `backend/mlruns`, resolved against
the script location, regardless of the process working directory. No server is
needed for the import itself. CLI `--help` also works without MLflow installed.

## Optional DagsHub tracking

Create/select a DagsHub repository first. In PowerShell, set its environment
variables before running the same importer:

```powershell
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/hassaanzafar891-glitch/SignVision.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "<your-dagshub-username>"
$env:MLFLOW_TRACKING_PASSWORD = "<your-dagshub-access-token>"
$env:MLFLOW_EXPERIMENT_NAME = "LinguaSign Baselines"
..\.venv-tracking\Scripts\python.exe -m tracking.import_baselines
```

This intentionally uploads both bundles and the training-script snapshots to the
selected remote. Credentials are read by MLflow from the environment; they are
not recorded as tags, parameters, or artifacts. No DagsHub SDK is required and no
`.env` file is loaded automatically. Clear the URI and credentials to return to
the local default:

```powershell
Remove-Item Env:MLFLOW_TRACKING_URI, Env:MLFLOW_TRACKING_USERNAME, Env:MLFLOW_TRACKING_PASSWORD -ErrorAction SilentlyContinue
```

See [DagsHub's MLflow configuration](https://dagshub.com/docs/integration_guide/mlflow_tracking/)
and [MLflow authentication](https://mlflow.org/docs/latest/self-hosting/security/basic-http-auth/).

## Tests

From the repository root, no third-party packages are needed for unit tests:

```powershell
python -m unittest discover -s backend/tests -p test_tracking_baselines.py -v
python -m compileall -q backend/tracking backend/tests/test_tracking_baselines.py
git diff --check
```

After installing the tracking requirements yourself, run the suite again to
include the real local MLflow round-trip test (otherwise it is skipped):

```powershell
.\.venv-tracking\Scripts\python.exe -m unittest discover -s backend/tests -p test_tracking_baselines.py -v
```

Tests use temporary fake bundles and never load production models or contact
DagsHub. The round-trip test validates persisted runs, empty metrics, artifact
listing, downloaded manifests, and model checksums. Actual local/remote baseline
imports remain untested until tracking dependencies are installed and the importer
is explicitly run. A failed remote import can leave a failed or partially uploaded
run; inspect the run status before treating it as a complete baseline.

## Optional tracking for future training

Both existing trainers now use `tracking/training.py`. MLflow is **disabled by
default**, even if a tracking URI is present. Without tracking enabled, MLflow is
not imported. Enabling it without the package installed, or encountering a remote
logging error, issues a warning and retains local outputs so training can continue.
Tracking errors are recorded separately from training status. An owned MLflow run
is ended as FAILED on training or logging errors; an existing active run is never
resumed or closed by the helper. No autologging or training seeds are enabled.

Use an existing compatible Python 3.10/3.11 training environment containing the
inference/training requirements plus `backend/requirements-tracking.txt` when
tracking is desired. Dependencies remain separate: inference does not import or
require MLflow, and the standalone baseline environment above does not need
TensorFlow. No dependencies were installed and no training was executed while
implementing this integration.

For a **future, intentionally requested training run**, from `backend/`:

```powershell
# Enable local MLflow; leave MLFLOW_TRACKING_URI unset for backend/mlruns.
$env:SIGNVISION_TRACKING_ENABLED = "true"
$env:MLFLOW_EXPERIMENT_NAME = "SignVision Training"
python -m PSL.alphabet_recognition.train_alphabet_model
python -m PSL.word_recognition.train_word_model
```

For DagsHub, set `MLFLOW_TRACKING_URI` to
`https://dagshub.com/hassaanzafar891-glitch/SignVision.mlflow` and the username/token
variables shown above before invoking a trainer. The helper does not automatically
read `.env` or send anything to DagsHub when tracking is disabled. To disable it:

```powershell
$env:SIGNVISION_TRACKING_ENABLED = "false"
```

Every invocation, with or without MLflow, writes to a new directory:

```text
backend/data/training_runs/<alphabet|word>/<UTC timestamp>-<UUID>/
```

The `.h5`, scaler, label encoder, architecture JSON, source snapshots and local
`run_metadata.json` all stay in that directory. Alphabet confusion matrices also
save there. The directory records an MLflow run ID when one is available; the
directory ID remains usable when tracking is off. Production files under
`backend/data/models/` are never overwritten or automatically promoted. Training
outputs are excluded from Git. The database path now resolves from the repository
location and opens read-only, retaining the same queries and row ordering.

For future runs the helper records:

- Model name, architecture/layers, epochs, batch size, actual compiled optimizer
  class, its configured learning rate and optimizer configuration.
- `accuracy`, `val_accuracy`, `loss`, and `val_loss` as available at every epoch,
  with one-based MLflow steps. Missing metrics are omitted, never filled with zero.
- Final `test_accuracy` and `test_loss` from the existing evaluation call. Accuracy
  values are fractions, not percentages.
- Dataset filename, table, row/feature/class counts, class distribution and
  train/test counts. `dataset.sha256` hashes the ordered selected feature/label
  rows before scaling, encoded as compact UTF-8 JSON lines (single-column label
  tuples are unwrapped). It fingerprints the data actually read, including labels,
  rather than SQLite file layout; SQLite WAL contents do not invalidate it.
- Git checkout commit/dirty state, raw model and preprocessing artifacts, source
  snapshots, and artifact sizes/SHA-256 hashes in `run_metadata.json`. This record
  excludes its own checksum. No raw dataset or credentials are uploaded.

MLflow stores the same output directory under `training/` as raw artifacts, not a
serving flavor or Model Registry version. Local metrics and configuration remain
available even with tracking disabled. Remote uploads are best effort: local
metadata records upload/end-run failures, and a partial remote run is not evidence
of a successful complete upload.

Architectures and preprocessing are unchanged. Alphabet retains its unseeded 80/20
train/test split and 20% validation split within training. Word retains its
stratified 80/20 split with `random_state=42` and still uses its test set as
validation data. Its test score is therefore not an independent holdout result;
this task deliberately preserves that methodology.

### Safe tests for the training integration

From the repository root:

```powershell
python -m unittest discover -s backend/tests -p "test_*tracking*.py" -v
python -m compileall -q backend/tracking backend/PSL/alphabet_recognition/train_alphabet_model.py backend/PSL/word_recognition/train_word_model.py backend/tests/test_training_tracking.py
git diff --check
```

The helper tests use fake models, callbacks and MLflow clients. Static trainer
checks verify architecture, scaler behavior, split/evaluation arguments, and safe
output paths without importing or executing either trainer. Real TensorFlow
training and local/DagsHub logging of a training run remain untested here.
