# Private inference monitoring and prediction-distribution drift

This feature adds aggregate, per-process monitoring to the existing FastAPI
application. It does not retrain, change classification/preprocessing, replace
models, alter candidate approval, deploy, restart Docker, or access AWS.
There are no new dependencies or background jobs.

**Operational status:** implementation and local tests are available, but no
legitimate production reference baseline was found or created. Drift therefore
reports `baseline_unavailable`, not a score or an assurance of stability. AWS
runtime behavior and overhead have not been verified. Candidate model approval
remains blocked independently by the missing original holdout provenance.

## What is measured

- Completed `/api/*` request counts, 2xx/3xx/4xx/5xx counts, HTTP error rates,
  429 counts and fixed latency histograms. Routes have only three categories:
  `predict_frame`, `match`, and `other`; arbitrary paths are never retained.
- Prediction-service calls and latency by `alphabet`, `word`, or `unknown` task,
  accepted predictions, no-confidence/no-match results and inference exceptions.
- A model identity derived from SHA-256 hashes of the loaded `.h5`, scaler and
  label encoder. The bundle is hashed once at service startup, not per request.
- Maximum softmax probability when the existing neural helper already produces
  it. No second prediction or scaler fit is performed. Confidence is aggregated
  as a mean and five fixed bins. It is not calibrated probability of correctness.

API latency includes parsing, JPEG/MediaPipe processing and response handling.
Service latency measures recognition/preprocessing and model invocation, excluding
the earlier MediaPipe stage. Legacy `predict()` -> `predict_file()` calls are
counted once. Existing swallowed inference exceptions still
return `no match`; they now increment the **inference** error counter even if
the API returns HTTP 200. HTTP and inference error rates must be interpreted
separately. Long-lived streams count when they finish, not at connection opening.

The middleware does not read request bodies, query strings, headers or identities.
It preserves responses/exceptions. The recognition service no longer prints
per-frame labels, temporary keypoint paths or exception payloads. Monitoring
failures are isolated from inference, and invalid confidence values are discarded.

## Private operator endpoint

`GET /api/monitoring` is hidden from OpenAPI and requires an exact Bearer token.
It returns 404 when monitoring is disabled, no sufficiently long token is
configured, or authorization fails. Successful responses use `Cache-Control:
no-store`. No metrics are included in the public health endpoint or frontend.

Configuration fields are in `backend/core/config.py` and the **example** environment
file. No real token or credential was generated, modified or committed.

| Setting | Default | Purpose |
| --- | --- | --- |
| `MONITORING_ENABLED` | `true` | Collect in-memory aggregates; `false` bypasses collection. |
| `MONITORING_TOKEN` | empty | Operator secret; endpoint inaccessible until set. Use a cryptographically random 32-512 character token with no whitespace. |
| `MONITORING_WINDOW_SIZE` | `1000` | Rolling outcome window per task; 20-10,000 events. |
| `MONITORING_MIN_SAMPLES` | `100` | Current observations required for drift and distribution disclosure; 1 to window size. |
| `MONITORING_MIN_REFERENCE_SAMPLES` | `100` | Minimum reviewed reference observations. |
| `MONITORING_DRIFT_THRESHOLD` | `0.1` | Jensen-Shannon divergence alert threshold, in (0,1]. |
| `MONITORING_BASELINE_PATH` | empty | Optional reviewed aggregate JSON file, loaded once. |

Keep the token server-side. Never place it in `NEXT_PUBLIC_*`, frontend code, a URL,
command-line query parameter, or an example committed to Git. Manage it using the
existing operator secret process. After a future reviewed local server setup,
an operator can query loopback without printing the token:

```powershell
$headers = @{ Authorization = "Bearer $env:MONITORING_TOKEN" }
Invoke-RestMethod http://127.0.0.1:8000/api/monitoring -Headers $headers
```

Counters are process-local and reset on process restart; task metrics reset when
the loaded model identity is re-registered. Multiple workers/containers do not
share them. Use one worker for the current deployment or explicitly aggregate
scrapes per worker in a future design. No exporter, dashboard, retention service,
alert delivery or persistent metric store is installed by this change.

## Bounded storage and privacy

There is no disk write per request or prediction. Each task retains only a bounded
deque of outcome-bin names plus counters, not images, raw landmarks, timestamps,
prediction text sequences, probabilities arrays, users, IPs, session IDs or headers.
Class bins are anonymous encoder positions (`class_0`, ...), not label strings in
the exported snapshot. Known label mappings remain internal and bounded to 64
classes. Current distributions are withheld until the minimum sample count.
Histograms, route categories, task keys and error counters have fixed sizes;
lifetime counters saturate at signed 64-bit maximum. References are capped at
256 KiB and 68 bins per task. No unbounded label/URL dictionary is created.

The endpoint excludes its own traffic to avoid polluting inference metrics. Its
snapshot contains model hashes and aggregate operational data, so protect the
token even though it contains no raw input. Existing Caddy/Uvicorn access-log
policies and legacy capture behavior are separate: audit them independently and
do not enable logging of bodies, authorization headers or user-specific URLs.
The browser route's existing temporary keypoint file remains part of inference;
monitoring does not copy or retain it.

## Drift policy and reference requirements

Drift is prediction/outcome distribution change measured with base-2
**Jensen-Shannon divergence** in [0,1]. It includes accepted class bins and the
fixed `no_confidence`, `no_match`, `error`, `unknown` bins. A score at least the
configured threshold yields `drift_detected`; below it yields `no_alert`.
Zero counts are handled mathematically without smoothing or fabricated counts.
The score is computed on an authenticated snapshot request, not on every frame.

A reference must be an operator-reviewed, representative production **aggregate**
window for the exact same model/scaler/encoder bundle and class bins. Training
label frequencies, synthetic test fixtures, the first requests after startup,
or a hand-entered balanced distribution are not legitimate production references.
There is no automatic baseline creation or online reference adaptation.

Reference JSON schema (field descriptions, not production evidence):

| Field | Requirement |
| --- | --- |
| `schema_version` | Integer `1`. |
| `provenance.status` | `reviewed_production_aggregate`. |
| `provenance.reviewed_by` / `reference_id` | Nonempty operator review identifiers, maximum 256 characters. |
| `provenance.window_start` / `window_end` | ISO timestamps with explicit timezone offsets; start before end. |
| `tasks` | One or both of `alphabet`, `word`. |
| `tasks.<task>.model_identity` | Exact lowercase SHA-256 identity from the monitored loaded bundle. |
| `tasks.<task>.sample_count` | Positive integer equal to the sum of all bin counts. |
| `tasks.<task>.counts` | Every current `class_N` bin and all four fixed outcome bins, including zeros; nonnegative integer counts. |

To establish a reference later: securely collect aggregate counts for an approved
representative window, verify bundle identity and collection provenance, review
sample adequacy and traffic mix, then mount that reviewed file read-only. Do not
invent a source, reviewer, counts or dates to enable the detector. Review is a
human trust boundary; schema checks cannot establish that claimed traffic really
occurred. Keep reference preparation separate from monitoring ordinary users.

Missing, unreadable, malformed, unreviewed, model-mismatched or bin-mismatched
references produce `baseline_unavailable` with a bounded reason code. Too few
reference/current samples produce the corresponding `insufficient_*_samples`
status. All these states have `score: null`, never `0` or `no_alert`.

An alert is **not** proof of concept drift or degraded accuracy. Usage patterns,
repeated video frames, lighting, camera placement and the small word vocabulary
can change distributions. Conversely, input drift can occur with no label-frequency
change. There are no ground-truth labels, signer-independent checks, significance
test or confidence calibration here. Threshold 0.1 is a configurable starting
point, not a calibrated production guarantee. Calibrate thresholds/window sizes
using legitimate reviewed traffic and account for correlated successive frames.
No alert automatically retrains, approves or deploys a model.

## Deployment considerations (not applied)

The existing Compose ports remain bound to `127.0.0.1`, and Docker/Caddy files are
unchanged. The existing Compose file does not automatically forward newly added
host environment variables. During a future reviewed rollout, explicitly supply
monitoring settings to the backend container, for example through a separate
operator-managed override:

```yaml
services:
  backend:
    environment:
      MONITORING_ENABLED: "true"
      MONITORING_TOKEN: ${MONITORING_TOKEN:?Set an operator-managed token}
      MONITORING_WINDOW_SIZE: "1000"
      MONITORING_MIN_SAMPLES: "100"
      MONITORING_DRIFT_THRESHOLD: "0.1"
```

Do not set `MONITORING_BASELINE_PATH` or add a bind mount until legitimate reference
data exists. When it does, mount its actual host path read-only and pass its
container path explicitly. The file is loaded once; replacing it does not silently
change the active reference. No reload/upload/reset HTTP endpoints are exposed.
Invalid numeric range settings disable telemetry; normal settings type-validation
errors still need operator correction at startup.

Caddy may proxy `/api/*`, so the bearer check remains required even with loopback
Docker ports. Prefer an operator-only path/network policy as an additional layer;
do not expose telemetry anonymously. Poll sparingly (for example every 30 seconds).
Review changes on `feature/mlops-phase2` before any rollout. No live deployment,
SSH access, service restart, credential change or AWS request was performed here.

## Tests and local verification

```powershell
Set-Location E:\LinguaSign
& .\.venv-backend\Scripts\python.exe -B -m unittest discover -s backend/tests -p test_monitoring.py -v
& .\.venv-backend\Scripts\python.exe -B -m unittest discover -s backend/tests -p "test_*.py" -v
python -S -B -m unittest discover -s backend/tests -p test_monitoring.py -v
git diff --check
```

The existing backend CI discovery automatically includes `test_monitoring.py`.
Tests check bounded memory, invalid data/configuration, missing/invalid references,
drift/non-drift synthetic distributions, auth/default denial, no-store responses,
error handling, nested/thread-isolated confidence, unchanged prediction/scaler
calls, and middleware behavior without reading bodies. The optional real FastAPI
ASGI test needs only FastAPI and never loads models; it skips in dependency-free
CI. Test references are synthetic fixtures, not deployment baselines.

Real webcam traffic, live CPU overhead, a valid production reference, external
scraping/alerts and AWS runtime integration remain unverified until an operator
performs a separately reviewed rollout. Passing unit tests is not an operational
drift-detection claim.
