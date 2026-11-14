# Browser camera in Docker (Windows)

Use Docker Desktop with Linux containers and open PowerShell in `E:\LinguaSign`.
The backend image uses Python 3.10 on amd64; the frontend uses Node 22 on Debian
with Python/make/g++ available to compile better-sqlite3. No webcam or sound
hardware is passed to Docker. Existing server-camera routes remain available.

The sign page uses getUserMedia, previews a mirrored video, and sends unmirrored
JPEGs at up to 640x480 to `/api/predict-frame?mode=0` (alphabet) or `mode=1` (word).
Requests run sequentially with a one-second pause after each response. The backend
serializes MediaPipe access and passes a temporary request-specific keypoint file
to the unchanged model preprocessing. It never selects a browser frame by mtime.
Speech audio is generated with the existing Urdu Edge TTS voice and played by the
browser through `/api/speech-audio/{text}`. TTS requires an internet connection.

## Build and test

Start Docker Desktop first, then run exactly:

```powershell
Set-Location E:\LinguaSign
docker compose config --quiet
docker compose build
docker compose up -d
docker compose ps
curl.exe --fail http://localhost:8000/
docker compose exec backend python -m pip check
docker compose exec backend python -m compileall -q .
docker compose exec backend python -m unittest discover -s tests -p test_*.py -v
docker compose exec backend python tests/smoke_browser_camera.py
docker compose exec frontend npx tsc --noEmit --incremental false
docker compose exec frontend npm run lint
```

The HTTP smoke test uses blank JPEGs, real MediaPipe and the loaded model
pipelines; it expects `no confidence`. It does not test a hand gesture or inference
accuracy. The dependency-isolated unit tests check frame isolation and cleanup.

Open `http://localhost:3000/sign`, allow the camera, and test manually:

1. Start detection; confirm preview and sequential `predict-frame` requests in
   DevTools. Show known alphabet signs, then switch Word Mode and show known words.
2. Confirm live labels correspond to the current mode, including when toggling
   while a request is pending. Remove hands and confirm the old label clears.
3. Accept letters, delete a letter, add words, remove words, clear and copy a
   sentence. Enable Speech and use Speak Sentence; confirm sound in the browser.
4. Stop; confirm requests stop and the webcam indicator turns off. Restart, then
   navigate away and confirm camera tracks stop. Deny permission and disconnect
   the camera to verify visible errors. Stop the backend to verify request errors.
5. Check narrow/mobile layouts and both themes retain their existing controls.

```powershell
docker compose logs --tail=100 backend frontend
docker compose down
```

Use localhost (or HTTPS); a plain HTTP LAN address cannot use getUserMedia.
`NEXT_PUBLIC_API_URL` is baked into the frontend build. If changing host/port,
update the Compose build argument and backend ALLOWED_ORIGINS, then rebuild.
Optional Google authentication and Stripe settings are passed from shell/root
.env variables listed in Compose; use a persistent AUTH_SECRET for sign-in.
Existing local frontend/.env.local and backend/.env are excluded from images.
The frontend SQLite database and speech cache persist in separate named volumes.
Models remain in the backend image; no data volume hides them.

## Validation in the implementation environment

Docker Compose configuration passed. The Docker Desktop Linux engine was not
running, so image builds and container smoke tests could not be executed.
Python compilation and the dependency-isolated unit tests passed. Real webcam
recognition, audio playback, model inference and Docker runtime behavior remain
untested here. TypeScript passed and ESLint passed with two existing warnings: an img element
in app/components/ls/Components.tsx and a missing words.length effect dependency
in app/page.tsx. The sign page has no ESLint warnings. The frontend production build also passed
after allowing network access to download its existing Google Fonts. Docker
builds likewise need access to Google Fonts, npm and PyPI.
