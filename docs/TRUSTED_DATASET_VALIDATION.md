# Trusted private dataset validation for PRs

Implemented locally, 19 November 2026. No reader credential, publisher App or
hosted success is claimed. Nothing is published, made public, deployed or approved
by these files. Production model approval remains blocked independently.

## Trust boundary

`.github/workflows/frontend-ci.yml` runs PR code without dataset or publisher
secrets. Its backend/frontend/model-gate jobs are preserved; anonymous dataset
retrieval is removed. Completion triggers `dataset-validation.yml` from the default
branch, including when ordinary PR tests fail (their required checks remain failed).

Every trusted job checks out the explicit base repository at `${{ github.sha }}`:
the **workflow_run default-branch commit**, never the originating run's head SHA.
The helper verifies that checkout, the live `main` commit and branch protection.
Only reviewed protected-main code executes. It runs Python 3.10 with `-S -B`, no
package installation, startup site hooks, shared caches, submodules, LFS or executable
PR artifacts. Referenced actions are pinned to full commit SHAs.

Four fresh hosted jobs isolate authority:

| Job | Authority and checks |
| --- | --- |
| `preflight` | Read-only GitHub token. Validates repository name/ID, originating workflow name/path/ID, run event/status/attempt, event/API source-fork identity, unique open PR/current head, base/main commit, complete changed-file list and approved pin. No reader/App secrets. |
| `start_check` | Repeats preflight before App-key access; creates an exact-head check using a dedicated publisher App, restricted to Checks write on SignVision. |
| `validate` | Runs only when inputs are allowed. Rechecks metadata and pending check identity/issuer before reading dedicated DagsHub credentials. Downloads actual private bytes and calls `tracking.pr_dataset.validate()`. No publisher key. |
| `finish_check` | Separate publisher environment; no DagsHub credentials. Rechecks current head/base/run/policy and exact check ID/App issuer. Success requires the validation job's success and an exact bound report. Failed/skipped/missing evidence fails; completed checks cannot be upgraded/replayed. |

GitHub can return an empty check `pull_requests` array for fork commits. The helper
does not trust that array or PR-uploaded metadata: it resolves the open PR through
the base repository API, verifies the fork ID and branch, and binds the immutable
head SHA. If the originating run points at a merge commit, both ordered parents
must match the current trusted base and PR head. Supplied historical associations
must agree. Missing, ambiguous, stale, closed or unreadable metadata blocks.

The check is **`Trusted actual DVC dataset validation`**, created through the
Checks API with explicit `head_sha`, and bound to run ID/attempt and validator
commit by `external_id`. A separate GitHub App issuer is necessary: merely using
the GitHub Actions issuer would let a PR propose a job with the same check name.
Required-check settings MUST select the dedicated App as expected source.

## Approved inputs and actual data

The PR dataset pin must match the protected-main file byte for byte. Changes or
renames involving `.github/**`, `.dvc/**`, the pin/database, trusted validation
modules and their local imports, tracking packages, top-level Python modules or
stdlib-shadowing packages/startup hooks are rejected. The current conservative policy rejects all such proposals before
reader use; it does not automatically accept a GitHub review as a policy update.
Security/pin updates require a separately reviewed protected-main rollout.

Actual data remains private. The reader requests only the fixed reviewed DagsHub
object: MD5 `326d65539174dffdad54606bb2271e3c`, SHA-256
`e6d6c59c7a0c7a207388d290c7ddae7f05ba1aa1507df8b5d293ac4ec13c33a9`,
6,864,896 bytes. It uses the modern object layout, with legacy fallback only on
404, disables ambient proxies and refuses authenticated redirects. A redirect or
auth failure requires operator review; credentials are never sent to another host.
The bounded download goes to a fresh temporary directory, then the existing
validator checks identity, SQLite integrity, both required schemas, finite features,
labels and data quality. Local validation already established 15,363 alphabet rows
and 543 word rows; the remote authenticated path still needs an integration run.

No private database, raw rows, cookies, headers, server bodies or credentials are
uploaded/logged. Only non-sensitive identities and validation status cross job
boundaries. Database files are removed by temporary-directory cleanup. Neither the
private dataset nor its read token is ever passed to PR-controlled code.

## Owner setup — not performed

1. **Protect `main`.** Require reviews for `.github/**`, `.dvc/**` and the trusted
   validation code/dependencies. Ensure the branch API reports `protected: true`.
   The implementation currently requires the repository's default branch to be
   `main`; a different default branch blocks until explicitly reviewed.
2. **Provision a dedicated DagsHub reader.** Grant only read access to this repository
   and create its token. Verify permissions in DagsHub; do not test with a write or
   reuse the general owner token. This code cannot prove token permissions from its
   string. If read-only credentials cannot be provisioned under the account plan,
   setup remains blocked; do not silently substitute a write-capable token.
3. **Create/install a dedicated GitHub App** on SignVision only, with repository
   **Checks: read and write**, plus automatic metadata read. No Contents write,
   Actions write, deployments or organization permissions are needed. Disable
   webhooks if unused. Generate a private key and retain it privately.
4. Set repository variables `SIGNVISION_DATASET_APP_ID` (numeric App ID) and
   `SIGNVISION_DATASET_APP_CLIENT_ID` (App client ID). Neither is a secret.
5. Create environment **`dataset-validation`** with secrets
   `DAGSHUB_DATASET_READER_USERNAME` and `DAGSHUB_DATASET_READER_TOKEN`.
   Create environment **`dataset-check-publisher`** with secret
   `SIGNVISION_DATASET_APP_PRIVATE_KEY` (PEM). Store these secrets ONLY in those
   environments, not repository/organization-wide. Restrict both environments to
   the selected branch **`main` only**; PR refs/feature branches must be denied.
   Human environment approvals are optional but would make validation approval-gated
   rather than fully automatic. Leave fork-secret/write-token forwarding disabled.
6. Publish the reviewed workflow/helper to protected `main` through the owner's
   existing authorized review/bootstrap process. They must exist there before
   `workflow_run` can use them. This initial workflow/policy change cannot approve
   itself through the new protected-input guard. No automatic bypass is provided.
7. After an authorized same-repository and fork smoke PR, require these checks:
   `Backend checks (Python 3.10)`, `Frontend checks (Node.js 22)`,
   `PR model gate (sandbox; production fail-closed)`, and
   **`Trusted actual DVC dataset validation` from the dedicated App**. Remove the
   superseded anonymous `PR actual DVC dataset validation` context if required.
   Require branches to be up to date before merge, so old head results cannot be
   reused against a changed base/policy. Merge queues are not supported by this
   PR-only protocol; they need a separately reviewed `merge_group` extension.

Do not configure a generic success/check-name substitute when App or fork check
creation fails. A missing/pending required check blocks merging. The old manual
production candidate workflow/secrets remain separate and unchanged.

## Verification and operational limits

Focused tests use mock GitHub metadata and dummy credentials, never a live token:

```powershell
Set-Location E:\LinguaSign
& .\.venv-backend\Scripts\python.exe -B -m unittest discover -s backend/tests -p test_trusted_dataset_ci.py -v
python -S -B -m unittest discover -s backend/tests -p test_trusted_dataset_ci.py
```

Tests cover forks/empty associations, exact merge parents, wrong workflow/repository,
unprotected/advanced default branch, changed heads/attempts, protected renames/pins,
truncated metadata, lazy reader access, redirects, incorrect check issuer/ID/head,
report mismatch and completed-check replay. YAML tests assert job isolation,
default-branch-only checkouts, action SHA pins and absence of caches/PR artifacts.

Required owner integration evidence: verify success on the exact current fork PR
SHA; push a new head during validation and confirm it cannot inherit success; verify
protected-input PRs fail without reader execution; check missing/invalid credentials
fail without leaking them. Default-branch movement, API unavailability or source run
reruns with mismatched attempts block rather than reusing evidence. Rebase/update
and rerun source CI when needed. Cancellation after check creation can leave a
pending check; rerun the full source CI, not just a privileged finishing job.

There is no fundamental exact-SHA API design blocker: GitHub supports App check
creation on explicit commit SHAs, including fork commits with empty association
arrays. Actual installation permissions, protected environments, anonymous/private
storage behavior and branch enforcement remain unverified until the owner runs it.
If check creation is rejected, no success is issued. No operational-compliance
claim is made before that hosted evidence exists.

Local verification on 19 November: **17 focused security tests passed**, and all
**14 existing PR-check tests passed** with available sklearn/YAML dependencies.
Full dependency-free backend discovery ran 235 cases: **188 passed, 47 optional
skips**, no failures. YAML/job-boundary assertions, preserved original jobs/triggers,
Python compilation, whitespace and documented PowerShell syntax checks passed.
The earlier draft fork-identity test failure was corrected by binding the event's
fork ID/name and branch to the API result. Existing legacy unclosed-file warnings
remain unrelated. All six production H5/preprocessing artifacts match Git HEAD;
the actual SQLite SHA-256 still matches the approved pin. No live credential use,
status publication, production training, deployment, commit or push occurred.

References: [GitHub Checks API](https://docs.github.com/en/rest/checks/runs),
[privileged workflow security](https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target),
[expected status-check issuer](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
