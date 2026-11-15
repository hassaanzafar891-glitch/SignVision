# Run and verify SignVision Phase 1

Use PowerShell. Training, MLflow and application inference are separate processes.
This guide covers the MLOps workflow; source attribution and licensing are in
[README.md](README.md#source-attribution-and-license).

Commands are code-verified against the scripts and CLI options. Validation and
tests were executed during Phase 1 verification; setup/downloads, application
startup and actual retraining were not rerun for this documentation edit.

## 1. Environments

For a fresh checkout only; skip this if `E:\LinguaSign` already exists:

```powershell
Set-Location E:\
git clone https://github.com/hassaanzafar891-glitch/SignVision.git LinguaSign
```

Use Python **3.10** for the pinned ML stack and Node.js **22** for the frontend.
The local Python installations are `E:\Python310\python.exe` and
`E:\Python312\python.exe`; adjust paths on another computer. Explicit interpreter
paths avoid activation-policy and local `py` launcher discovery issues.

```powershell
$ProjectRoot = "E:\LinguaSign"
Set-Location $ProjectRoot
& "E:\Python310\python.exe" -m venv .venv-backend
$BackendPython = Join-Path $ProjectRoot ".venv-backend\Scripts\python.exe"
& $BackendPython -m pip install --timeout 1000 --retries 10 -r backend/requirements.txt
$env:PYTHONPATH = Join-Path $ProjectRoot "backend"
```

Reuse an existing configured environment rather than recreating it. Install
requirements once or when they change; TensorFlow is a large download.
For tracked training, also install the separate MLflow client requirements into
this same interpreter:

```powershell
& $BackendPython -m pip install -r backend/requirements-tracking.txt
```

DVC and the pinned MLflow server can use a separate tools environment:

```powershell
& "E:\Python312\python.exe" -m venv .venv-tracking
$ToolsPython = Join-Path $ProjectRoot ".venv-tracking\Scripts\python.exe"
& $ToolsPython -m pip install -r backend/requirements-tracking.txt "dvc[http]==3.67.1"
```

The tools environment needs no TensorFlow. The tracking requirements pin MLflow
2.22.2; the existing local SQL store uses a separate, newer server installation
as explained in section 4.

## 2. Retrieve processed data

From the repository root, use the already configured DVC remote `origin`:

```powershell
& $ToolsPython -m dvc pull backend/data/db/main_dataset.db.dvc
Test-Path backend/data/db/main_dataset.db
```

The result should be `True`. Do not initialize DVC again. The remote is
`https://dagshub.com/hassaanzafar891-glitch/SignVision.dvc`.
If authentication is required, set credentials in ignored local configuration:

```powershell
& $ToolsPython -m dvc remote modify --local origin auth basic
& $ToolsPython -m dvc remote modify --local origin user "<DagsHub-username>"
& $ToolsPython -m dvc remote modify --local origin password "<DagsHub-token>"
& $ToolsPython -m dvc pull backend/data/db/main_dataset.db.dvc
```

Replace placeholders; never commit `.dvc/config.local` or tokens. DVC authentication
is separate from MLflow authentication. Remote access and data availability were
not rechecked for this edit.

The database contains processed 42-feature vectors and labels for alphabet and
word tasks. Raw images/videos are private and unavailable; reproduction starts
with this database. Signer IDs are absent, and raw-data extraction and label
correctness cannot be independently verified from the processed records alone.

DagsHub project: https://dagshub.com/hassaanzafar891-glitch/SignVision

## 3. Validate and test without training

Repeat the path setup in each new PowerShell terminal:

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = Join-Path (Get-Location).Path "backend"
& .\.venv-backend\Scripts\python.exe -m tracking.train_all
& .\.venv-backend\Scripts\python.exe -B -m unittest discover -s backend/tests -p "test_*.py" -v
```

`train_all` defaults to validation only: no TensorFlow import, MLflow connection or
model fitting. Every trainer also validates automatically before reading training
rows. Checks cover SQLite integrity, required columns, finite numerical features,
non-empty text labels, at least 15 samples, two classes and three samples per class.
The unchanged splitter separately verifies duplicate-disjoint feasibility and
class coverage.

The latest verification passed both real tables and all 42 backend tests. Tests
use temporary data/stores and mocked estimators or training calls. Dependency-free
CI skips four checks requiring NumPy/scikit-learn or MLflow.

The shared seed-42 `split_three_way()` targets about 60/20/20 train/validation/test.
Identical vectors stay together. Audited row counts were 9,218/3,070/3,075 for
alphabet and 351/88/104 for word; both saved RF manifests matched reconstructed
splits. Scalers fit only training features. Signer-independent evaluation is not
established, and bit-identical neural weights are not guaranteed.

## 4. Start MLflow and view existing results

The existing `backend/mlflow.db` and `backend/mlartifacts/` use the global MLflow
**3.18.0** installation. In a separate terminal, start that compatible server:

```powershell
Set-Location E:\LinguaSign\backend
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:///E:/LinguaSign/backend/mlartifacts --no-serve-artifacts --host 127.0.0.1 --port 5000
```

Do not open the existing database with the older 2.22.2 server or downgrade/migrate
it during setup. The older `backend/mlruns/` file store is separate; preserve it.
Direct local artifact paths (`--no-serve-artifacts`) suit clients on this computer.

For a **new store** using the pinned tools environment instead, use different
paths; this will not display the existing SQL experiment history:

```powershell
Set-Location E:\LinguaSign\backend
& ..\.venv-tracking\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///mlflow-course.db --default-artifact-root file:///E:/LinguaSign/backend/mlartifacts-course --no-serve-artifacts --host 127.0.0.1 --port 5000
```

Choose one server and keep its terminal open. At http://127.0.0.1:5000:

- **SignVision Training** contains NN epoch accuracy/loss and final test metrics.
- **SignVision Classical Baselines** contains RF accuracy, macro F1 and confusion matrices.
- **Models** shows `SignVision-Alphabet` v1 and `SignVision-Word` v1, verified READY
  with existing `.h5` sources. No `MLmodel` flavor manifests were found, so MLflow
  loading/serving remains unverified despite catalog registration.

The recorded NN accuracies are alphabet **98.37%** and word **100% on 104 test
examples**. RF test accuracies are **99.6748%** and **99.0385%**, respectively.
Word RF validation accuracy is **88.6364%**, a 10.4021 percentage-point gap from its
test score. Historical NN partition hashes are missing: comparisons are indicative,
not verified same-holdout comparisons. Full metrics are in
[Random Forest results](docs/RANDOM_FOREST_BASELINES.md).

## 5. Intentionally reproduce training

With data, dependencies and the server ready, run the single workflow command
from the repository root:

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = Join-Path (Get-Location).Path "backend"
& .\.venv-backend\Scripts\python.exe -m tracking.train_all --train --tracking-uri http://127.0.0.1:5000
```

It validates both tables, checks server access, then runs alphabet NN, word NN,
alphabet RF and word RF sequentially. Tracking is enabled within the wrapper and
its environment settings are restored afterward. The sequence was tested with
mocked trainers; no actual retraining was performed during verification. Close
alphabet plots to let its command finish.

New outputs go under `backend/data/training_runs/`, with RF outputs in its
`random_forest/` subtree. Production models and existing experiments are never
overwritten or automatically promoted to inference. NN tracking is best effort;
inspect `tracking_errors` in local run metadata for failed uploads.

To run individual models instead, set the training terminal's environment first:

```powershell
$env:SIGNVISION_TRACKING_ENABLED = "true"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_EXPERIMENT_NAME = "SignVision Training"
& .\.venv-backend\Scripts\python.exe -m PSL.alphabet_recognition.train_alphabet_model
& .\.venv-backend\Scripts\python.exe -m PSL.word_recognition.train_word_model
```

The RF command always requires tracking and defaults to the local server:

```powershell
& .\.venv-backend\Scripts\python.exe -m tracking.train_random_forest --model both --tracking-uri http://127.0.0.1:5000
```

### Environment variables and optional DagsHub tracking

| Variable | Purpose |
| --- | --- |
| `PYTHONPATH` | Backend directory, so `PSL` and `tracking` import from the repository root. |
| `SIGNVISION_TRACKING_ENABLED` | NN tracking opt-in; disabled by default for individual NN commands. The combined workflow enables it internally. |
| `MLFLOW_TRACKING_URI` | NN tracking endpoint; unset falls back to `backend/mlruns`. RF/combined commands use their `--tracking-uri` argument. |
| `MLFLOW_EXPERIMENT_NAME` | Optional individual NN experiment name; default is `SignVision Training`. |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | DagsHub username/token; not required by the local server. |
| `NEXT_PUBLIC_API_URL` | Frontend API URL from `frontend/.env.local`. |

For individual NN runs on DagsHub, set:

```powershell
$env:SIGNVISION_TRACKING_ENABLED = "true"
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/hassaanzafar891-glitch/SignVision.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "<DagsHub-username>"
$env:MLFLOW_TRACKING_PASSWORD = "<DagsHub-token>"
```

For combined/RF runs, also pass that URL to `--tracking-uri`. View results in the
DagsHub project's Experiments area. The `.mlflow` endpoint differs from DVC's
`.dvc` endpoint. Trainers do not automatically read `.env` files. `$env:` values
apply only to the current terminal and its child processes; repeat setup in new
terminals. Clear DagsHub credentials when returning to local tracking:

```powershell
Remove-Item Env:MLFLOW_TRACKING_USERNAME, Env:MLFLOW_TRACKING_PASSWORD -ErrorAction SilentlyContinue
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

## 6. Run application inference

Training is optional. Existing artifacts in `backend/data/models/` support
inference without MLflow or the training database. Start both application parts
in separate terminals.

Backend (run from `backend/` because model paths are relative):

```powershell
Set-Location E:\LinguaSign\backend
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
& ..\.venv-backend\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
Set-Location E:\LinguaSign\frontend
npm ci
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
npm run dev
```

Open http://localhost:3000/sign and allow camera access. The browser sends JPEGs
to FastAPI; the backend needs no host-camera device. Use localhost or HTTPS.
Speech requires internet access for Urdu TTS. Configure optional Google sign-in,
Stripe and SMTP settings only if using those integrations. Existing `.env` files
are preserved by the conditional copies.

## Troubleshooting

| Problem | Check |
| --- | --- |
| Cannot import `main` | Start Uvicorn from `E:\LinguaSign\backend`. |
| Cannot import `PSL`/`tracking` | Set `PYTHONPATH` in the current terminal. |
| ML dependency install fails | Use Python 3.10 and the committed requirements; allow time for TensorFlow. |
| Training data missing | Pull the DVC pointer from the repository root. |
| DVC 401/403 or missing objects | Check access, local credentials and remote data availability. |
| MLflow schema/file-store errors | Use the compatible server/store; do not alter existing experiment history. |
| No tracked NN run | Check the endpoint, opt-in flag, installed client and `tracking_errors`. |
| Frontend cannot detect signs | Check backend port 8000 and `NEXT_PUBLIC_API_URL`. |

See [CI](docs/CI.md) and [browser-camera/Docker notes](docs/BROWSER_CAMERA_DOCKER.md)
for checks and runtime limitations.
