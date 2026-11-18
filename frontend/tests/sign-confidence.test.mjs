/* Uses installed TypeScript/React and Node's built-in runner; no new dependencies. */
import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

const directory = path.dirname(fileURLToPath(import.meta.url));

// Transpile the actual small UI modules in memory; do not emit files or boot Next.js.
function load(relative, dependencies = {}) {
  const filename = path.resolve(directory, "../app/sign", relative);
  const source = ts.transpileModule(readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const realRequire = createRequire(filename);
  const localRequire = (name) => dependencies[name] || realRequire(name);
  const loaded = { exports: {} };
  new Function("require", "module", "exports", source)(localRequire, loaded, loaded.exports);
  return loaded.exports;
}
const helpers = load("prediction-confidence.ts");
const ModelConfidence = load("ModelConfidence.tsx", { "./prediction-confidence": helpers }).default;

test("actual component displays model confidence with the uncalibrated label", () => {
  const html = renderToStaticMarkup(React.createElement(ModelConfidence, { value: 0.873 }));
  assert.match(html, /Model confidence/);
  assert.match(html, /uncalibrated/);
  assert.match(html, /87\.3%/);
  assert.match(html, /not a calibrated probability of correctness/);
  assert.match(html, /role="status"/);
});
test("missing, malformed and detector-only confidence stays unavailable", () => {
  for (const value of [undefined, null, NaN, Infinity, -0.1, 1.1, "0.9", true]) {
    const prediction = helpers.readFramePrediction({ label: "word", confidence: value });
    assert.equal(prediction.confidence, null);
    assert.match(renderToStaticMarkup(React.createElement(ModelConfidence, { value: prediction.confidence })), /Unavailable/);
  }
  assert.equal(helpers.readFramePrediction({ label: "word", detection_confidence: 0.9 }).confidence, null);
});
test("missing hands and errors suppress scores while preserving server labels", () => {
  for (const label of ["no match", "no confidence", ""]) {
    assert.deepEqual(helpers.readFramePrediction({ label, confidence: 0.9 }), { label, confidence: null });
  }
  for (const response of [null, {}, { label: 2 }, "bad"]) {
    assert.throws(() => helpers.readFramePrediction(response), /Invalid prediction response/);
  }
});
test("legacy API responses remain valid and zero is not treated as missing", () => {
  assert.deepEqual(helpers.readFramePrediction({ label: "alphabet", mode: 0 }), { label: "alphabet", confidence: null });
  assert.equal(helpers.formatModelConfidence(0), "0.0%");
  assert.equal(helpers.formatModelConfidence(1), "100.0%");
});
test("repeated labels can carry independently updated per-frame scores", () => {
  const values = [0.6, 0.9, null].map(confidence => helpers.readFramePrediction({ label: "word", confidence }));
  assert.deepEqual(values.map(value => value.confidence), [0.6, 0.9, null]);
  assert.deepEqual(values.map(value => value.label), ["word", "word", "word"]);
});
