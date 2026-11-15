# How to Run SignVision

SignVision is the MLOps course project built around the LinguaSign PSL application.
Use **PowerShell** for all commands below. Application inference, optional model
training, and the MLflow tracking server are separate processes.

Commands were checked against the current scripts, requirement files, DVC pointer
and installed CLI help. Setup, downloads, application/server startup and training
were **not executed** during this documentation update. Existing evaluation metrics
were checked by reading the local MLflow SQLite database without writing to it.

## 1. Project location and Python environment

For a fresh checkout only (do not clone over the existing project):

```powershell
Set-Location E:\
git clone https://github.com/hassaanzafar891-glitch/SignVision.git LinguaSign
```

For the existing checkout, start here. Use Python **3.10**, matching the pinned
TensorFlow/MediaPipe stack, and Node.js **22** for the frontend. The inspected
Python executables are `E:\Python310\python.exe` and `E:\Python312\python.exe`;
adjust those paths on another computer. The local `py` launcher did not discover
these installations, so the examples use explicit executable paths:

```powershell
$ProjectRoot = "E:\LinguaSign"
Set-Location $ProjectRoot
& "E:\Python310\python.exe" -m venv .venv-backend
$BackendPython = Join-Path $ProjectRoot ".venv-backend\Scripts\python.exe"
& $BackendPython -m pip install --timeout 1000 --retries 10 -r backend/requirements.txt
$env:PYTHONPATH = Join-Path $ProjectRoot "backend"
```

Reuse `.venv-backend` if it already contains the required environment. Install
requirements once, or when they change; do not recreate the environment on every
run. Explicit interpreter paths avoid PowerShell activation-policy problems.
TensorFlow is a large download, so setup can take time. A conda Python 3.10
environment is an alternative; use its Python executable consistently.

## 2. Retrieve the DVC training database

Training reads `backend/data/db/main_dataset.db`. Git contains the `.dvc` pointer,
not the database contents. The existing models are sufficient to run inference;
the training database is needed when reproducing training.

Use a separate tools environment for DVC and the pinned MLflow client/server:

```powershell
Set-Location E:\LinguaSign
& "E:\Python312\python.exe" -m venv .venv-tracking
$ToolsPython = Join-Path (Get-Location).Path ".venv-tracking\Scripts\python.exe"
& $ToolsPython -m pip install -r backend/requirements-tracking.txt "dvc[http]==3.67.1"
```

This tools environment does not require TensorFlow. DVC 3.67.1 with HTTP support
is present in the inspected tools environment; the MLflow requirements pin 2.22.2.
For the configured remote `origin`, retrieve only the training database:

```powershell
& $ToolsPython -m dvc pull backend/data/db/main_dataset.db.dvc
Test-Path backend/data/db/main_dataset.db
```

The last command should return `True`. Do not initialize DVC again. The configured
remote is `https://dagshub.com/hassaanzafar891-glitch/SignVision.dvc`.
If the remote requires authentication, configure credentials locally, then retry:

```powershell
& $ToolsPython -m dvc remote modify --local origin auth basic
& $ToolsPython -m dvc remote modify --local origin user "<DagsHub-username>"
& $ToolsPython -m dvc remote modify --local origin password "<DagsHub-token>"
& $ToolsPython -m dvc pull backend/data/db/main_dataset.db.dvc
```

Replace placeholders with your account values. `--local` writes to ignored
`.dvc/config.local`; do not commit credentials. DVC authentication is separate
from MLflow authentication. A 401/403 means access must be granted; missing remote
objects mean the dataset must first be made available by the repository owner.
No remote data availability or authenticated download was verified here.

DagsHub repository: https://dagshub.com/hassaanzafar891-glitch/SignVision

## 3. Run the existing application (inference)

### Terminal 1: FastAPI backend

```powershell
Set-Location E:\LinguaSign\backend
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
& ..\.venv-backend\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Keep this terminal open. The backend uses existing model/scaler/encoder files in
`backend/data/models/`; it does not train models. Run from `backend/` because the
inference loaders use relative model paths. MLflow is not needed for inference.
The `.env` example provides localhost CORS defaults; configure SMTP only if needed.

### Terminal 2: Next.js frontend

```powershell
Set-Location E:\LinguaSign\frontend
npm ci
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
npm run dev
```

Use `npm ci` once after checkout or lockfile changes. Open
http://localhost:3000/sign and allow camera access. Keep both terminals running.
The browser captures the webcam and sends JPEG frames to FastAPI; the backend
needs no direct host-camera device. Use localhost or HTTPS for browser camera
access. Speech uses Urdu Edge TTS with an internet connection.

`NEXT_PUBLIC_API_URL` in `frontend/.env.local` defaults to
`http://127.0.0.1:8000`. Google sign-in needs the `AUTH_*` settings; donations need
`STRIPE_SECRET_KEY`. These optional services are not prerequisites for training.
Existing `.env` files are preserved by the conditional copy commands above.

## 4. Optional model training

Run training only when intentionally reproducing an experiment. It is not a
prerequisite for starting the application. In a separate PowerShell terminal:

```powershell
$ProjectRoot = "E:\LinguaSign"
$BackendPython = Join-Path $ProjectRoot ".venv-backend\Scripts\python.exe"
Set-Location $ProjectRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "backend"
$env:SIGNVISION_TRACKING_ENABLED = "false"
```

Train the alphabet model (25 epochs, batch size 1):

```powershell
& $BackendPython -m PSL.alphabet_recognition.train_alphabet_model
```

Train the word model (50 epochs, batch size 32):

```powershell
& $BackendPython -m PSL.word_recognition.train_word_model
```

Both commands read the DVC-retrieved database. `split_three_way()` uses a
reproducible, seed-42 group split targeting approximately **60% training, 20%
validation, 20% testing**. Identical 42-feature vectors remain in the same
partition; actual row proportions can vary with duplicate-group sizes. Every
partition must contain every class. `StandardScaler` is fitted only on training
features; validation and test features are transformed separately.

Signer IDs are unavailable, so this is **not signer-independent splitting**.
The same signer may contribute different feature vectors to multiple partitions.
Reproducible partitioning does not promise bit-for-bit deterministic training.
Alphabet training displays plots; close them to let the command finish.

Every invocation saves new model/preprocessing artifacts and metadata under:

```text
backend/data/training_runs/<alphabet|word>/<UTC timestamp>-<UUID>/
```

Existing production `.h5`, scaler and label encoder files are not overwritten.
Training new artifacts does not automatically replace the models used by inference.

## 5. Optional MLflow tracking and local server

To track training, install the separate tracking requirements into the **training
interpreter**, not just the tools environment (from `E:\LinguaSign`):

```powershell
& $BackendPython -m pip install -r backend/requirements-tracking.txt
```

### View the existing course experiments

This workspace already has `backend/mlflow.db` and `backend/mlartifacts/`, used by
the global **MLflow 3.18.0** installation. In a separate PowerShell terminal, use
that compatible server installation to view the existing store:

```powershell
Set-Location E:\LinguaSign\backend
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:///E:/LinguaSign/backend/mlartifacts --no-serve-artifacts --host 127.0.0.1 --port 5000
```

Do not open this existing database with the older pinned 2.22.2 server, downgrade
its schema, or migrate/delete experiments as part of setup. `--no-serve-artifacts`
uses direct local artifact paths, suitable for clients on this same computer.
The older `backend/mlruns/` file store is separate from `mlflow.db`; keep it intact.

### Fresh course store (alternative)

If the existing compatible server installation is unavailable, use the pinned
server in `.venv-tracking` with a **different, new** store and artifact directory.
This does not show or alter the existing SQL experiment history:

```powershell
Set-Location E:\LinguaSign\backend
& ..\.venv-tracking\Scripts\python.exe -m mlflow server --backend-store-uri sqlite:///mlflow-course.db --default-artifact-root file:///E:/LinguaSign/backend/mlartifacts-course --no-serve-artifacts --host 127.0.0.1 --port 5000
```

Choose one server; both use port 5000. Keep its terminal open. In the **training
terminal**, set the endpoint before running either training command from section 4:

```powershell
$env:SIGNVISION_TRACKING_ENABLED = "true"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_EXPERIMENT_NAME = "SignVision Training"
```

Open http://127.0.0.1:5000 and select **SignVision Training**. Compare runs, inspect
`accuracy`, `val_accuracy`, `loss`, `val_loss` curves, and read `test_accuracy` and
`test_loss`. Under **Artifacts > training**, inspect the model bundle, architecture,
dataset fingerprint, Git provenance and `run_metadata.json`.

If tracking fails, training can continue and retain local outputs. Inspect
`tracking_errors` in the run's local metadata before assuming remote logging
succeeded. `SIGNVISION_TRACKING_ENABLED=false` disables tracking. If tracking is
enabled but `MLFLOW_TRACKING_URI` is unset, the helper defaults to the older local
`backend/mlruns/` file store; setting the URI explicitly avoids this ambiguity.

### Optional DagsHub tracking instead of the local server

```powershell
$env:SIGNVISION_TRACKING_ENABLED = "true"
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/hassaanzafar891-glitch/SignVision.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "<DagsHub-username>"
$env:MLFLOW_TRACKING_PASSWORD = "<DagsHub-token>"
$env:MLFLOW_EXPERIMENT_NAME = "SignVision Training"
```

Then intentionally run either training command from section 4. View results in
[the DagsHub repository's Experiments area](https://dagshub.com/hassaanzafar891-glitch/SignVision).
The `.mlflow` endpoint is for experiments, while `.dvc` is for dataset storage.
No token is stored in source code, and the trainers do not load `.env` automatically.

### Environment variables at a glance

| Variable | Purpose |
| --- | --- |
| `PYTHONPATH` | Set to `E:\LinguaSign\backend` so `PSL` and `tracking` import from the repository root. |
| `SIGNVISION_TRACKING_ENABLED` | Explicit opt-in (`true`); default is disabled. |
| `MLFLOW_TRACKING_URI` | Local server URL or DagsHub `.mlflow` URL; unset defaults to `backend/mlruns`. |
| `MLFLOW_EXPERIMENT_NAME` | Optional; training defaults to `SignVision Training`. |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | DagsHub account and token; not required by the local server. |
| `NEXT_PUBLIC_API_URL` | Frontend's FastAPI endpoint; configured in `frontend/.env.local`. |

PowerShell `$env:` assignments apply to the current terminal and its child
processes, not other open terminals. Repeat the setup in a new terminal. When
switching back to the local server, remove DagsHub credentials from that terminal:

```powershell
Remove-Item Env:MLFLOW_TRACKING_USERNAME, Env:MLFLOW_TRACKING_PASSWORD -ErrorAction SilentlyContinue
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

## 6. Recorded evaluation results

The existing local SQL-backed **SignVision Training** experiment records:

| Model | Test accuracy | Test examples |
| --- | --- | --- |
| Alphabet | **98.37%** | 3,075 |
| Word | **100% on a small test dataset** | 104 |

Accuracy is stored as a fraction in MLflow (approximately `0.98374` and `1.0`).
These observations are not guarantees for new signers or real-world webcam input.
They were read from existing experiments; no training was rerun for this update.

## 7. Phase 1 validation and one-command workflow

### Dataset provenance and automatic checks

Original raw images/videos are private and unavailable in this submission.
Reproduction starts with the DVC-tracked processed SQLite database, not original
media extraction. The database has no signer IDs, so signer-independent evaluation
cannot be established. Keep the existing database, DVC pointers and credentials
unchanged when performing verification.

Alphabet, word and RF entry points now validate the database before loading rows.
`tracking/data_validation.py` checks SQLite integrity, required tables and columns,
42 finite numerical features (not numeric strings), non-empty text labels,
at least 15 samples, at least two classes, and at least three samples per class.
The split helper then checks class coverage and duplicate-disjoint feasibility.
Validation cannot prove label correctness or reconstruct the unavailable raw data.

### Safe one-command verification (no training)

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = Join-Path (Get-Location).Path "backend"
& .\.venv-backend\Scripts\python.exe -m tracking.train_all
```

This command was executed successfully in validation-only mode: alphabet has
15,363 samples/37 classes and word has 543 samples/5 classes. It does not import
TensorFlow, contact MLflow or create training runs. The split audit also confirmed
current train/validation/test counts of 9,218/3,070/3,075 for alphabet and
351/88/104 for word, with full index coverage and no duplicate-vector overlap.
Both existing RF split manifests match the reconstructed partition fingerprints.
Historical NN partition-index hashes remain unavailable, so NN/RF comparison is
still indicative, not a proven same-holdout comparison.

Run the backend suite without training:

```powershell
& .\.venv-backend\Scripts\python.exe -B -m unittest discover -s backend/tests -p "test_*.py" -v
```

Tests use temporary databases, fake estimators and mocked neural training calls.
The optional installed-MLflow test writes only to a temporary test store. Existing
experiments, production models and DVC data are not changed. Dependency-free CI
skips tests that require absent NumPy/scikit-learn or MLflow packages.

### Future training reproduction (not executed during verification)

With the dependencies installed and the compatible MLflow server from section 5
already running, this is the single training command:

```powershell
& .\.venv-backend\Scripts\python.exe -m tracking.train_all --train --tracking-uri http://127.0.0.1:5000
```

It validates both tables, checks server access, then invokes alphabet NN, word NN,
alphabet RF and word RF sequentially. NN runs use **SignVision Training**; RF runs
use **SignVision Classical Baselines**. The wrapper temporarily enables tracking
and restores its environment settings afterward. No existing MLflow run is resumed.
The sequence was verified with mocked trainers; no model was retrained to test it.
Seeded duplicate-disjoint splitting is repeatable for unchanged ordered data;
TensorFlow weights and floating-point results are not guaranteed bit-identical.
Alphabet plots must be closed for its command to finish. NN tracking remains
best effort: inspect local `tracking_errors` before assuming every upload succeeded.

### Verified registry entries and remaining limits

Read-only inspection of the existing local SQL store found:

| Registered name | Version | Status | Source |
| --- | --- | --- | --- |
| `SignVision-Alphabet` | 1 | READY | Existing alphabet `.h5` artifact |
| `SignVision-Word` | 1 | READY | Existing word `.h5` artifact |

Both source files exist. No versions were created, overwritten, promoted or deleted.
The registered sources are raw `.h5` files without an `MLmodel` flavor manifest;
`mlflow.pyfunc` loading and serving these versions have not been verified and may
require separate future packaging. Registry status does not prove deployment
readiness. The local MLflow `/health` endpoint returned HTTP 200 during inspection.

In the UI at http://127.0.0.1:5000, use **Models** to inspect those versions and
**Experiments** to inspect the existing NN and RF results. Completed RF results
and the word validation/test gap are documented in
[RANDOM_FOREST_BASELINES.md](docs/RANDOM_FOREST_BASELINES.md).

## Troubleshooting

| Problem | Check |
| --- | --- |
| `Could not import module "main"` | Start Uvicorn from `E:\LinguaSign\backend`. |
| `No module named tracking` or `PSL` | Set `PYTHONPATH` to the backend directory in the training terminal. |
| ML dependencies fail to install | Use Python 3.10 and the committed requirements; allow time for the TensorFlow download. |
| `unable to open database file` during training | Run DVC pull from the repository root and verify the database exists. |
| DVC 401/403 or missing objects | Check repository access, local credentials, and remote data availability. |
| MLflow file-store/schema errors | Use the matching server/store; do not downgrade or modify existing experiments to fix setup. |
| No new MLflow run appears | Verify the URI, `SIGNVISION_TRACKING_ENABLED`, client installation and local `tracking_errors`. |
| Frontend loads but detection fails | Keep FastAPI running on port 8000 and check `NEXT_PUBLIC_API_URL`. |
| No camera or speech | Allow browser permissions, use localhost/HTTPS, and check internet access for TTS. |

For more detail see [CI](docs/CI.md) and [browser-camera/Docker notes](docs/BROWSER_CAMERA_DOCKER.md).
