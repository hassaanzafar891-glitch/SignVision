# SignVision CI

The existing `.github/workflows/frontend-ci.yml` defines **SignVision CI**
with two independent basic jobs plus an opt-in manual candidate-approval job.
The filename is retained to avoid a duplicate workflow.

| Job | Runtime | Checks | Timeout |
| --- | --- | --- | --- |
| Backend | Python 3.10, Ubuntu | Standard-library unittest discovery and Python compilation | 5 minutes |
| Frontend | Node.js 22, Ubuntu | Locked npm install, ESLint, TypeScript | 15 minutes |

## Triggers and permissions

- Pull requests, without branch or path filters.
- Pushes to `main` and `feature/**` branches.
- Manual runs via **Actions > SignVision CI > Run workflow**. GitHub requires a
  workflow-dispatch definition on the default branch for the manual entry point.
- `contents: read` permissions; checkout credentials are not persisted.
- A newer run cancels the previous run for the same workflow/event and PR or
  branch. Push and pull-request runs have separate concurrency groups.

No AWS, DagsHub, authentication, or payment secrets are required for the basic
checks. They do not train, deploy, build Docker images or retrieve DVC data.
The manual `candidate_approval` job uses DagsHub read credentials and installs
evaluation dependencies only after evidence preflight succeeds. It requires
both task gates before uploading approval evidence, and never deploys. See
[EVALUATION_GATE.md](EVALUATION_GATE.md) for prerequisites, provenance requirements,
GitHub environment configuration and the current blocked production-evidence state.
Application/model files and existing DVC configuration are unchanged.

## Backend checks

Commands run from the repository root:

```powershell
python -m unittest discover -s backend/tests -p "test_*.py" -v
python -m compileall -q -x '(^|[\\/])(data|mlruns|mlartifacts)([\\/]|$)' backend
```

The Python job installs no project dependencies. Camera handling is tested with
mocked FastAPI, OpenCV, MediaPipe, and recognition services. Tracking tests use
fake models and MLflow clients; trainer contract checks inspect syntax without
executing training. Tests use temporary fixtures, not production datasets/models.
The optional real MLflow round-trip test skips when MLflow is absent.

The same discovery includes `test_monitoring.py`: privacy/auth, bounded aggregate
storage, error handling and reference-gated drift tests run without ML packages.
The optional FastAPI ASGI binding test skips if FastAPI is absent. No live webcam,
production drift baseline, external monitoring server or AWS access is needed.
See [MONITORING.md](MONITORING.md) for operational prerequisites and limitations.

`smoke_browser_camera.py` does not match `test_*.py` and is deliberately excluded.
Compilation checks syntax without importing or executing the training modules,
and excludes runtime data and MLflow stores (whose tag filenames may end in `.py`
while containing a checksum rather than Python source).
Tracking is explicitly disabled in the backend job.

## Frontend checks

Commands run from `frontend/`, with Node.js 22 and the lockfile committed:

```powershell
npm ci
npm run lint
npx tsc --noEmit
```

The workflow caches npm's download cache using `frontend/package-lock.json` as
the dependency key. It still runs `npm ci` each time; `node_modules` is not cached.
Native dependencies such as `better-sqlite3` may download a prebuilt binary or
compile using the GitHub-hosted Ubuntu runner's tooling. Registry/network access
is still needed when dependencies are not cached.

The production build is excluded from every trigger, including manual dispatch.
It can be run separately with `npm run build` when explicitly desired; it currently
downloads Google Fonts and needs a writable `backend/data/` directory for SQLite.
The workflow adds no frontend scripts and does not change application behavior.

## Limitations and verification

Discovery also includes `test_controlled_retraining.py`: synthetic frozen plans,
fake trainers/tracking, bundle tampering, evidence-backed promotion and pinned
rollback plans. No training or lifecycle executor is added to any workflow.
See [RETRAINING.md](RETRAINING.md). Candidate approval remains manual and blocked
until authentic production holdout evidence is established.

These checks do not establish webcam recognition accuracy, real TensorFlow/
MediaPipe compatibility, real MLflow/DagsHub connectivity, production build success,
or Docker runtime behavior. There is no frontend browser/unit-test suite configured
in `package.json`; lint and type checking provide the frontend coverage here.
ESLint warnings remain warnings; errors fail CI.

All required tests and tracking helpers must be committed in a future reviewed
commit before GitHub can execute them; untracked local files are not uploaded by
the workflow. No commit or push is part of this implementation.

Local workflow validation uses an already available YAML parser plus assertions
for events, branch filters, permissions, concurrency, runtimes, timeouts, commands,
and npm cache settings. `git diff --check` checks whitespace. This is static
configuration validation, not a GitHub-hosted execution or actionlint result.

References: [npm caching](https://github.com/actions/setup-node/tree/v6),
[manual workflow runs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).
