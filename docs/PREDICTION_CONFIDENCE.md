# Live model confidence

The browser-camera API adds `confidence` to its existing `label` and `mode`
response. It is the **predicted class's softmax probability from the same inference
call**, or JSON `null` when unavailable. Labels, preprocessing, saved artifacts,
hand-quality thresholds and existing HTTP error handling are unchanged. Legacy
`predict()` and `predict_file()` callers still receive strings.

Both inspected production-path H5 artifacts support this: the alphabet model has
a 37-class softmax output and the word model a 5-class softmax output. Their existing
Keras `model.predict()` produces the probability vector; scikit-learn's
`predict_proba()` is not needed. The new reporting code reads the probability at
the existing `argmax` class index. It does not run another prediction, rescale
scores, refit preprocessing or change the selected label.

Only finite, bounded, normalized single-row class probabilities are exposed.
Invalid shape/range/sum, missing outputs, missing hands, `no confidence` and
`no match` yield `confidence: null`. Detector/landmark confidence is never used as
classification confidence. The score capture is request-local, thread-isolated
and independent of monitoring being enabled. Errors cannot leak a previous frame's
score into a later response.

The sign page shows **Model confidence (uncalibrated)** as a percentage, or
**Unavailable**. This is not calibrated probability of correctness, measured
accuracy, an explanation of feature attribution or a guarantee that a gesture is
correct. Scores refresh even when the predicted label repeats; speech remains
deduplicated. Stop/restart, mode changes, no-hand responses and request errors clear
the displayed score. Older backends without the field remain usable and show
Unavailable. Old API clients can ignore the additive field.

## Local verification

No new dependencies are required. From PowerShell:

```powershell
Set-Location E:\LinguaSign
python -S -B -m unittest discover -s backend/tests -p test_prediction_confidence.py -v
python -S -B -m unittest discover -s backend/tests -p test_browser_camera.py -v
Set-Location frontend
npm.cmd test
npx.cmd --no-install tsc --noEmit --incremental false
npm.cmd run lint
```

Backend tests execute the actual alphabet/word inference function bodies with
controlled model outputs, checking unchanged labels/preprocessing and one predict
call, malformed scores, request/thread isolation and errors. Frontend tests run the
actual parsing/formatting functions and render the confidence component using the
installed React/TypeScript packages and Node's built-in test runner. They do not
require a browser or new test dependencies. The existing workflow does not yet
invoke `npm test`; its lint/TypeScript checks remain unchanged.

Both real saved H5 models were additionally loaded read-only with `compile=False`
and called on a synthetic zero vector (42 features). They returned finite softmax
outputs of shapes `(1, 37)` and `(1, 5)`; the captured confidence exactly matched
the selected output entry. This is an artifact/output smoke check, not a gesture
accuracy evaluation. Both file hashes remained unchanged.

Verification results: 34 focused backend cases and five frontend cases passed.
Full dependency-free backend discovery passed 162 cases with 42 optional skips
(204 total). TypeScript, compilation and ESLint passed; ESLint retained two existing
warnings outside the sign page. Exercising the legacy inference readers also
surfaced their existing unclosed-file ResourceWarnings. SHA-256 comparison matched
all 334 protected model/data/MLflow/DVC/deployment/sandbox files. No dependency
installation or inference dependency change was needed.

Physical webcam, mobile layout and AWS runtime verification were not performed.
Before claiming deployment success, an operator must verify both recognition
modes in the browser, repeated-label score updates, no-hand/error behavior and
stop/mode-switch resets. No models were retrained or replaced, and nothing was
deployed by this change.
