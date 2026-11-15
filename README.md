# SignVision - Pakistan Sign Language Detection System

MLOps course project based on the LinguaSign application. Original attribution is retained below.

[![SignVision CI](https://github.com/hassaanzafar891-glitch/SignVision/actions/workflows/frontend-ci.yml/badge.svg)](https://github.com/hassaanzafar891-glitch/SignVision/actions/workflows/frontend-ci.yml)

A real-time Pakistani Sign Language (PSL) recognition system that translates hand gestures into Urdu text and speech, with a full learning platform built on top.

![LinguaSign Project Poster](docs/linguasign-poster.png)

---

## Application Overview

### Landing Page

![LinguaSign Home](frontend/public/images/screenshots/home.png)

The landing page introduces LinguaSign to new visitors with a bold hero section — a full-bleed animated background with a live hand-skeleton preview demonstrating PSL detection in real time. The headline communicates the core value proposition immediately: *Sign it. See it translated, instantly.* Below the fold, trust indicators display the system's reach (users, detections, accuracy), followed by a feature highlights section and navigation links to the core platform modules. The top navigation provides direct access to all major sections — Detect, Learn, Contact, and Feedback — with a persistent theme toggle and a prominent **Try the Detector** call-to-action. The overall aesthetic uses a dark base with a green accent system, reflecting the PSL branding throughout.

---

### Sign Detector

![Sign Detector](frontend/public/images/screenshots/sign.png)

The Sign Detector uses the browser's `getUserMedia()` webcam access. The browser sends JPEG frames to FastAPI at `/api/predict-frame`; MediaPipe extracts landmarks and the existing alphabet or word model returns a prediction. The backend does not need direct access to the host webcam. Users can:

- **Accept a letter** to append it to the current word
- **Add a word to the sentence** and remove individual words
- **Speak the sentence** using Urdu Edge TTS audio played by the browser
- Switch to **Word Mode** for direct whole-word detection
- Start and stop the camera; frame requests are sent sequentially

The layout is split into a control panel on the left and the live camera feed on the right, keeping detection context and controls in the same view without context switching.

---

### How It Works

![How It Works](frontend/public/images/screenshots/how-it-works.png)

An interactive, animated breakdown of the six-stage detection pipeline — from camera capture through MediaPipe landmark extraction, feature scaling, and neural network classification, to building words and speaking them aloud.

---

### Dictionary

![PSL Dictionary](frontend/public/images/screenshots/dictionary.png)

A searchable reference covering all 37 alphabet signs and the 5 supported words, for learners who want to browse outside of live detection.

---

## Features

### Detection
- **Real-time recognition** — detects 37 PSL alphabet letters and 5 common words via webcam
- **Sentence builder** — accept letters → build words → compose full sentences
- **Urdu speech output** - Urdu Edge TTS audio played in the browser (internet required)
- **Word mode** — switch between letter-by-letter and full-word detection
- **Browser camera** - captures JPEGs without a backend webcam device

### Learning Platform
- **Google sign-in** — account creation and login via NextAuth, backed by a local SQLite user store
- **Dashboard** — day streak tracker, weekly activity chart, signs-detected counter, recent history
- **Learn page** — browse all 37 PSL letters with hover-to-reveal sign images; quick quiz on PSL words
- **Live Learn** — real-time webcam practice with collapsible letter sidebar
- **Live Quiz** — timed quiz mode using the live camera
- **Dictionary** — searchable reference for all letters and words
- **Donate** — Stripe-powered donation page supporting the project
- **Contact & Feedback** — forms that email submissions via Gmail SMTP and log them locally

### UI / UX
- **Light & dark mode** — persisted theme toggle across all pages
- **Collapsible sidebar** — fold to icon-only view; hover the hand logo to reveal the expand button
- **Fully responsive** — mobile-optimised layouts at 860 px and 560 px breakpoints
- **Design system** — custom `ls-*` CSS classes, green accent, display font, Urdu font support

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16.2.4 (App Router), TypeScript |
| Styling | Custom CSS design system (`globals.css`) |
| Backend | FastAPI, Python 3.10 for the pinned ML stack |
| ML / CV | MediaPipe, TensorFlow / Keras, scikit-learn, OpenCV |
| Speech | Urdu Edge TTS audio with browser playback |

---

## Project Structure

```text
backend/
  main.py                         # FastAPI application
  routers/browser_camera.py       # Browser JPEG inference endpoint
  services/                       # Recognition and speech services
  PSL/alphabet_recognition/       # Alphabet inference and training
  PSL/word_recognition/           # Word inference and training
  tracking/data_splits.py         # Duplicate-disjoint three-way split
  tracking/training.py            # Optional MLflow tracking
  data/db/main_dataset.db.dvc     # DVC pointer to the training database
  data/models/                   # Existing production .h5 and .pkl files
  data/training_runs/            # New, isolated training outputs (ignored)
  mlruns/                       # Older local file-store experiments (ignored)
  mlflow.db                     # Existing local SQL experiment store (ignored)
  mlartifacts/                  # Existing local experiment artifacts (ignored)
  requirements.txt
  requirements-tracking.txt
frontend/app/                    # Next.js pages, UI and API integrations
.github/workflows/frontend-ci.yml # Lightweight backend/frontend CI
```

---

## Getting Started

> Looking for a more detailed, step-by-step walkthrough (including common errors)? See [HOW_TO_RUN.md](HOW_TO_RUN.md).

### Prerequisites

- Python **3.10** for the pinned TensorFlow/MediaPipe training and inference environment (the local executable is `E:\Python310\python.exe`; adjust this path if installed elsewhere)
- Node.js **22**, Git, and a webcam
- DVC with HTTP support for retrieving training data; MLflow is optional for inference

For a fresh checkout in PowerShell (skip cloning if `E:\LinguaSign` already exists):

```powershell
Set-Location E:\
git clone https://github.com/hassaanzafar891-glitch/SignVision.git LinguaSign
Set-Location E:\LinguaSign
& "E:\Python310\python.exe" -m venv .venv-backend
& .\.venv-backend\Scripts\python.exe -m pip install --timeout 1000 --retries 10 -r backend/requirements.txt
```

Start the backend in one PowerShell terminal:

```powershell
Set-Location E:\LinguaSign\backend
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
& ..\.venv-backend\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal:

```powershell
Set-Location E:\LinguaSign\frontend
npm ci
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
npm run dev
```

Open [http://localhost:3000/sign](http://localhost:3000/sign) and allow camera access. Google sign-in, Stripe and SMTP need their own optional settings from the environment examples. The two application terminals must remain running.

Existing models under `backend/data/models/` are used for **inference**. Running the app does not retrain them and does not require MLflow. For DVC retrieval, optional training, tracking and result viewing, follow the PowerShell sequence in [HOW_TO_RUN.md](HOW_TO_RUN.md).

### MLOps workflow

The DagsHub project is [hassaanzafar891-glitch/SignVision](https://dagshub.com/hassaanzafar891-glitch/SignVision). DVC remote `origin` points to its `.dvc` endpoint; MLflow uses the separate `.mlflow` endpoint.

After setting up the training environment and retrieving DVC data as described in the run guide, these **optional training commands** run from the repository root:

```powershell
Set-Location E:\LinguaSign
$env:PYTHONPATH = (Join-Path (Get-Location).Path "backend")
$env:SIGNVISION_TRACKING_ENABLED = "false"
& .\.venv-backend\Scripts\python.exe -m PSL.alphabet_recognition.train_alphabet_model
& .\.venv-backend\Scripts\python.exe -m PSL.word_recognition.train_word_model
```

For tracked training, install the separate `backend/requirements-tracking.txt` into the training environment, start the MLflow server from the run guide, and set `SIGNVISION_TRACKING_ENABLED=true` and `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`. Tracking is disabled by default. `PYTHONPATH` makes `tracking` and `PSL` importable when launching modules from the repository root.

New training runs write their `.h5`, scaler, label encoder and metadata to `backend/data/training_runs/<model>/<timestamp>-<UUID>/`. They do **not** overwrite the existing production artifacts. Repeating training is optional and does not guarantee bit-for-bit identical weights.

---

## How to Use

1. Open the app and navigate to **Sign Detector** from the sidebar
2. Click **Start Detection** and allow camera access
3. Make a PSL hand sign in front of your webcam
4. The detected letter appears in the **Detected Letter** panel
5. Click **Accept Letter** to add it to your current word
6. Click **Add Word to Sentence** when a word is complete
7. Use **Speak Sentence** to hear the output in Urdu
8. Click **Stop Detection** when finished to release the camera

To learn PSL signs, visit the **Learn** page — hover any letter card to see its hand sign image, or take the quick quiz on common words.

### Tips

- Use good lighting for best accuracy
- Keep your hand centered in the camera frame
- Enable **Word Mode** to detect full words directly
- Toggle **Enable Speech** to speak new live predictions

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/start-capture` | Legacy server-camera capture; not used by the browser sign page |
| `POST` | `/api/stop-capture` | Release webcam |
| `POST` | `/api/match` | Run detection on current frame, return label |
| `GET` | `/api/stream` | MJPEG live video stream |
| `POST` | `/api/predict-frame?mode=0` | Infer an uploaded JPEG; mode 0 = alphabet, 1 = word |
| `GET` | `/api/speech-audio/{label}` | Generate/serve Urdu speech for browser playback |

---

## Model Details

Two separate models are trained and served — one for alphabet letters, one for words. Both share the same input pipeline.

### Input pipeline

MediaPipe Holistic extracts 21 hand landmarks per frame, each carrying `[x, y, confidence]` (63 values total). Confidence scores are discarded, leaving **42 features per frame** (21 landmarks × XY pixel coordinates). These are standardised with a `StandardScaler` before being passed to the model.

### Alphabet model — 37 classes

| Layer | Units | Activation |
|-------|-------|-----------|
| Dense | 120 | ReLU |
| Dropout | 0.3 | — |
| Dense | 64 | ReLU |
| Dropout | 0.3 | — |
| Dense (output) | 37 | Softmax |

- Optimizer: Adam · Loss: Categorical cross-entropy
- Train/validation/test targets: approximately 60 / 20 / 20; epochs: 25; batch size: 1

### Word model — 5 classes

| Layer | Units | Activation |
|-------|-------|-----------|
| Dense | 256 | ReLU |
| Dropout | 0.3 | — |
| Dense | 128 | ReLU |
| Dropout | 0.3 | — |
| Dense | 64 | ReLU |
| Dense (output) | 5 | Softmax |

- Optimizer: Adam · Loss: Categorical cross-entropy
- Train/validation/test targets: approximately 60 / 20 / 20; epochs: 50; batch size: 32

### Training data

Custom-collected PSL keypoint dataset captured via MediaPipe, stored in SQLite. Alphabet dataset uses `alphabetDataset` table; word dataset uses `wordDataset`. Confusion-matrix analysis was used iteratively to identify and correct misclassified letter pairs.

The extracted keypoint dataset is versioned by DVC at `backend/data/db/main_dataset.db`; Git stores its `.dvc` pointer. Retrieve the data before training. Existing production models remain in Git. Raw source images/video frames are private and are not included.

### Splitting and evaluation

Both trainers call `split_three_way()` from `backend/tracking/data_splits.py`. It uses deterministic group shuffles with base seed 42 and keeps identical 42-feature vectors in the same partition. The targets are approximately **60% training, 20% validation and 20% testing**; row counts can vary because splitting operates on duplicate groups. Every accepted partition contains every class. The scaler is fitted only on training data, validation data is used during fitting, and final evaluation uses the separate test data.

This is **not signer-independent evaluation**: signer IDs are unavailable. Different feature vectors from the same person or closely related examples can still appear in different partitions. Reproducible splitting requires unchanged input rows and ordering; it does not guarantee deterministic neural-network training.

Recorded results from the existing local `SignVision Training` MLflow experiment:

| Model | Test accuracy | Test examples |
| --- | --- | --- |
| Alphabet | **98.37%** | 3,075 |
| Word | **100% on a small test dataset** | 104 |

These are observed experiment results, not guaranteed accuracy on unseen signers or real-world webcam input. No training was repeated to update this documentation. See [HOW_TO_RUN.md](HOW_TO_RUN.md) for viewing the existing SQL-backed experiment and for creating new runs without changing production models.

---

## Deployment

- **Backend** — deploy as a standard FastAPI app (e.g. Render, Railway, Fly.io). Set the environment variables from `backend/.env.example`, including `ALLOWED_ORIGINS` pointed at your deployed frontend's URL.
- **Frontend** — deploy to Vercel (or any Node host) with the variables from `frontend/.env.example`, pointing `NEXT_PUBLIC_API_URL` at your deployed backend.
- The live camera stream and detection rely on the browser's webcam access, so both frontend and backend should be served over HTTPS in production.

---

## Project Docs

- [HOW_TO_RUN.md](HOW_TO_RUN.md) - PowerShell application, DVC, training and MLflow instructions
- [docs/CI.md](docs/CI.md) - GitHub Actions checks and limitations
- [docs/BROWSER_CAMERA_DOCKER.md](docs/BROWSER_CAMERA_DOCKER.md) - browser camera and Docker notes

- [`docs/final-presentation.pptx`](docs/final-presentation.pptx) — final FYP defense presentation
- [`docs/fyp-logics.docx`](docs/fyp-logics.docx) — design/logic notes written during development
- An earlier, simpler iteration of the UI (before the LinguaSign redesign) is preserved at [linguasign-sign-to-speech-converter](https://github.com/waqaswajla/linguasign-sign-to-speech-converter)

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Waqas Ahmed

---

## Author

**Waqas Ahmed** — [waqaswajla](https://github.com/waqaswajla)

For any queries, reach out at wow.992du@gmail.com or +92 300 0700908.
