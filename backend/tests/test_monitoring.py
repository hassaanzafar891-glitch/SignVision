"""Monitoring tests use synthetic aggregates, fake inference and raw ASGI calls."""

import asyncio
import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitoring


TOKEN = "synthetic-test-token-" + "x" * 32
IDENTITY = "a" * 64


def reference(count_a=50, count_b=50, identity=IDENTITY):
    return {
        "schema_version": 1,
        "provenance": {"status": "reviewed_production_aggregate", "reviewed_by": "test-fixture-only",
                       "reference_id": "synthetic-not-production", "window_start": "2026-01-01T00:00:00+00:00",
                       "window_end": "2026-01-01T01:00:00+00:00"},
        "tasks": {"alphabet": {"model_identity": identity, "sample_count": count_a + count_b,
                               "counts": {"class_0": count_a, "class_1": count_b,
                                          **{key: 0 for key in monitoring.OUTCOMES}}}},
    }


class MonitoringTests(unittest.TestCase):
    def make_monitor(self, **kwargs):
        monitor = monitoring.Monitor(token=TOKEN, window_size=20, min_samples=10,
                                     min_reference_samples=10, **kwargs)
        monitor.register_task("alphabet", ["a", "b"], IDENTITY)
        monitor.register_task("word", ["a", "b"], "b" * 64)
        return monitor

    def with_reference(self, data):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "reference.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return self.make_monitor(baseline_path=path)

    def fill(self, monitor, values):
        for value in values:
            monitor.record_prediction("alphabet", value, 0.1, 0.8)

    def test_counts_latencies_http_errors_and_confidence(self):
        monitor = self.make_monitor()
        for status in (200, 400, 429, 500):
            monitor.record_request("predict_frame", status, 0.25)
        monitor.record_prediction("alphabet", "a", 0.1, 0.9)
        monitor.record_prediction("alphabet", "b", 0.3, 0.7)
        monitor.record_prediction("alphabet", "no confidence", 0.2)
        monitor.record_prediction("alphabet", "no match", 0.4, failed=True)
        snapshot = monitor.snapshot()
        api = snapshot["api"]["predict_frame"]
        self.assertEqual(api["completed_requests"], 4)
        self.assertEqual(api["http_error_rate"], 0.75)
        self.assertEqual(api["server_error_rate"], 0.25)
        self.assertEqual(api["rate_limited"], 1)
        self.assertEqual(sum(api["latency"]["bucket_counts"]), 4)
        task = snapshot["tasks"]["alphabet"]
        self.assertEqual(task["calls"], 4)
        self.assertEqual(task["outcomes"]["prediction"], 2)
        self.assertEqual(task["inference_error_rate"], 0.25)
        self.assertAlmostEqual(task["confidence"]["mean"], 0.8)
        self.assertEqual(task["confidence"]["count"], 2)
        self.assertIsNone(task["window"]["counts"])  # Suppress undersampled distributions.
        json.dumps(snapshot, allow_nan=False)

    def test_missing_reference_is_not_a_drift_score_or_automatic_baseline(self):
        monitor = self.make_monitor()
        self.fill(monitor, ["a"] * 30)
        drift = monitor.snapshot()["tasks"]["alphabet"]["drift"]
        self.assertEqual(drift["status"], "baseline_unavailable")
        self.assertEqual(drift["reason"], "not_configured")
        self.assertIsNone(drift["score"])

    def test_representative_and_shifted_distributions(self):
        same = self.with_reference(reference())
        self.fill(same, ["a", "b"] * 10)
        result = same.snapshot()["tasks"]["alphabet"]["drift"]
        self.assertEqual(result["status"], "no_alert")
        self.assertAlmostEqual(result["score"], 0.0)
        shifted = self.with_reference(reference(100, 0))
        self.fill(shifted, ["b"] * 20)
        result = shifted.snapshot()["tasks"]["alphabet"]["drift"]
        self.assertEqual(result["status"], "drift_detected")
        self.assertAlmostEqual(result["score"], 1.0)

    def test_model_mismatch_and_bins_mismatch_never_produce_scores(self):
        for mode in ("identity", "bins"):
            data = reference()
            if mode == "identity":
                data["tasks"]["alphabet"]["model_identity"] = "f" * 64
            else:
                data["tasks"]["alphabet"]["counts"]["class_2"] = 0
            monitor = self.with_reference(data)
            self.fill(monitor, ["a"] * 20)
            result = monitor.snapshot()["tasks"]["alphabet"]["drift"]
            self.assertEqual(result["status"], "baseline_unavailable")
            self.assertIsNone(result["score"])

    def test_minimum_reference_and_current_samples(self):
        low_reference = self.with_reference(reference(3, 3))
        self.fill(low_reference, ["a"] * 20)
        self.assertEqual(low_reference.snapshot()["tasks"]["alphabet"]["drift"]["status"], "insufficient_reference_samples")
        low_current = self.with_reference(reference())
        self.fill(low_current, ["a"] * 5)
        self.assertEqual(low_current.snapshot()["tasks"]["alphabet"]["drift"]["status"], "insufficient_current_samples")

    def test_invalid_or_unreviewed_reference_is_unavailable(self):
        for change in ("schema", "provenance", "negative", "boolean", "sum", "time"):
            data = reference()
            if change == "schema":
                data["schema_version"] = True
            elif change == "provenance":
                data["provenance"]["status"] = "invented"
            elif change == "negative":
                data["tasks"]["alphabet"]["counts"]["class_0"] = -1
            elif change == "boolean":
                data["tasks"]["alphabet"]["counts"]["class_0"] = True
            elif change == "sum":
                data["tasks"]["alphabet"]["sample_count"] = 123
            else:
                data["provenance"]["window_end"] = "not a date"
            with self.subTest(change=change):
                monitor = self.with_reference(data)
                self.assertEqual(monitor.snapshot()["tasks"]["alphabet"]["drift"]["status"], "baseline_unavailable")
        monitor = self.make_monitor(baseline_path="definitely-absent-reference.json")
        self.assertEqual(monitor.snapshot()["tasks"]["alphabet"]["drift"]["status"], "baseline_unavailable")

    def test_bounded_windows_no_arbitrary_labels_routes_or_sensitive_values(self):
        monitor = self.make_monitor()
        marker = "private-frame-landmarks-user-identity"
        for index in range(1000):
            monitor.record_prediction("alphabet", f"{marker}-{index}", 0.001)
            monitor.record_request(f"/api/{marker}-{index}", 404, 0.001)
        snapshot = monitor.snapshot()
        task = snapshot["tasks"]["alphabet"]
        self.assertEqual(task["window"]["samples"], 20)
        self.assertEqual(sum(task["window"]["counts"].values()), 20)
        self.assertEqual(len(task["window"]["counts"]), 6)
        self.assertEqual(len(snapshot["api"]), 3)
        self.assertNotIn(marker, json.dumps(snapshot))
        self.assertNotIn(TOKEN, json.dumps(snapshot))

    def test_invalid_data_and_config(self):
        for kwargs in ({"window_size": 0}, {"window_size": 10001}, {"min_samples": 1001},
                       {"threshold": float("nan")}, {"threshold": 0}, {"min_reference_samples": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                monitoring.Monitor(**kwargs)
        monitor = self.make_monitor()
        for value in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(ValueError):
                monitor.record_prediction("alphabet", "a", value)
            monitor.record_prediction("alphabet", "a", 0.1, confidence=value)
        for status in (0, 600, True):
            with self.assertRaises(ValueError):
                monitor.record_request("match", status, 0.1)
        self.assertEqual(monitor.snapshot()["tasks"]["alphabet"]["confidence"]["invalid_values"], 4)
        self.assertEqual(monitoring.js_divergence({"a": 1, "b": 0}, {"a": 0, "b": 1}), 1.0)
        with self.assertRaises(ValueError):
            monitoring.js_divergence({"a": 0}, {"a": 1})

    def test_saturated_lifetime_counters_do_not_grow_but_windows_continue(self):
        monitor = self.make_monitor()
        with patch.object(monitoring, "MAX_COUNT", 2):
            for _ in range(5):
                monitor.record_prediction("alphabet", "a", 0.1, 0.9)
                monitor.record_request("match", 500, 0.1)
            snapshot = monitor.snapshot()
        self.assertEqual(snapshot["tasks"]["alphabet"]["calls"], 2)
        self.assertTrue(snapshot["tasks"]["alphabet"]["counters_saturated"])
        self.assertEqual(snapshot["tasks"]["alphabet"]["window"]["samples"], 5)
        self.assertEqual(snapshot["api"]["match"]["completed_requests"], 2)

    def test_runtime_caches_config_and_fails_disabled_on_invalid_config(self):
        config = SimpleNamespace(
            MONITORING_ENABLED=True, MONITORING_TOKEN=SimpleNamespace(get_secret_value=lambda: TOKEN),
            MONITORING_WINDOW_SIZE=20, MONITORING_MIN_SAMPLES=10, MONITORING_MIN_REFERENCE_SAMPLES=10,
            MONITORING_DRIFT_THRESHOLD=0.1, MONITORING_BASELINE_PATH="",
        )
        with patch.dict(sys.modules, {"core.config": SimpleNamespace(settings=config)}), \
             patch.object(monitoring, "_runtime", None):
            first = monitoring.get_monitor()
            self.assertTrue(first.authorized("Bearer " + TOKEN))
            self.assertIs(first, monitoring.get_monitor())
        config.MONITORING_WINDOW_SIZE = 0
        with patch.dict(sys.modules, {"core.config": SimpleNamespace(settings=config)}), \
             patch.object(monitoring, "_runtime", None):
            self.assertFalse(monitoring.get_monitor().enabled)

    def test_token_authorization_default_deny_and_disable(self):
        monitor = self.make_monitor()
        self.assertTrue(monitor.authorized("Bearer " + TOKEN))
        for header in (None, TOKEN, "Bearer wrong", "Basic " + TOKEN, "Bearer " + TOKEN + " "):
            self.assertFalse(monitor.authorized(header))
        self.assertFalse(monitoring.Monitor().authorized("Bearer " + TOKEN))
        self.assertFalse(monitoring.Monitor(token="too-short").authorized("Bearer too-short"))
        disabled = self.make_monitor(enabled=False)
        disabled.record_request("match", 200, 0.1)
        self.assertFalse(disabled.authorized("Bearer " + TOKEN))
        self.assertEqual(disabled.snapshot()["api"]["match"]["completed_requests"], 0)

    def test_nested_inference_confidence_reset_and_error_reporting(self):
        monitor = self.make_monitor()

        @monitoring.observe_prediction(mode_index=0)
        def inner(mode):
            monitoring.note_confidence(0.9)
            return "a"

        @monitoring.observe_prediction(mode_index=0)
        def outer(mode):
            return inner(mode)

        @monitoring.observe_prediction(mode_index=0)
        def no_landmarks(mode):
            return "no confidence"

        @monitoring.observe_prediction(mode_index=0)
        def swallowed_error(mode):
            monitoring.note_prediction_error()
            return "no match"

        with patch.object(monitoring, "get_monitor", return_value=monitor):
            self.assertEqual(outer(0), "a")
            self.assertEqual(no_landmarks(0), "no confidence")
            self.assertEqual(swallowed_error(1), "no match")
        alpha, word = monitor.snapshot()["tasks"]["alphabet"], monitor.snapshot()["tasks"]["word"]
        self.assertEqual(alpha["calls"], 2)  # No double-counting nested service calls.
        self.assertEqual(alpha["confidence"]["count"], 1)
        self.assertEqual(word["outcomes"]["error"], 1)
        self.assertEqual(word["confidence"]["count"], 0)

    def test_thread_contexts_do_not_mix_tasks_or_probabilities(self):
        monitor = self.make_monitor()

        @monitoring.observe_prediction(mode_index=0)
        def fake(mode):
            monitoring.note_confidence(0.2 if mode == 0 else 0.9)
            return "a"

        with patch.object(monitoring, "get_monitor", return_value=monitor), ThreadPoolExecutor(max_workers=4) as executor:
            self.assertEqual(list(executor.map(fake, [0, 1] * 30)), ["a"] * 60)
        tasks = monitor.snapshot()["tasks"]
        self.assertEqual(tasks["alphabet"]["calls"], 30)
        self.assertEqual(tasks["word"]["calls"], 30)
        self.assertAlmostEqual(tasks["alphabet"]["confidence"]["mean"], 0.2)
        self.assertAlmostEqual(tasks["word"]["confidence"]["mean"], 0.9)

    def test_monitor_failure_preserves_results_and_original_exceptions(self):
        monitor = self.make_monitor()
        error = RuntimeError("original-inference-error")

        @monitoring.observe_prediction(mode_index=0)
        def fake(mode):
            if mode:
                raise error
            return "unchanged-prediction"

        with patch.object(monitoring, "get_monitor", return_value=monitor), \
             patch.object(monitor, "record_prediction", side_effect=RuntimeError("monitor unavailable")):
            self.assertEqual(fake(0), "unchanged-prediction")
            with self.assertRaises(RuntimeError) as raised:
                fake(1)
            self.assertIs(raised.exception, error)
        with patch.object(monitoring, "get_monitor", side_effect=RuntimeError("monitor init failed")):
            self.assertEqual(fake(0), "unchanged-prediction")

    def test_model_bundle_identity_includes_saved_preprocessing_and_resets_counts(self):
        monitor = self.make_monitor()
        with tempfile.TemporaryDirectory() as directory:
            paths = {key: Path(directory) / key for key in ("model", "scaler", "label_encoder")}
            for key, path in paths.items():
                path.write_bytes(key.encode())
            with patch.object(monitoring, "get_monitor", return_value=monitor):
                monitoring.register_loaded_model("alphabet", ["a", "b"], paths)
                first = monitor.snapshot()["tasks"]["alphabet"]["model_identity"]
                self.fill(monitor, ["a"] * 20)
                paths["scaler"].write_bytes(b"changed-synthetic-scaler")
                monitoring.register_loaded_model("alphabet", ["a", "b"], paths)
            task = monitor.snapshot()["tasks"]["alphabet"]
            self.assertNotEqual(first, task["model_identity"])
            self.assertEqual(task["calls"], 0)

    def test_probability_observation_is_read_only_and_optional(self):
        values = SimpleNamespace(max=lambda: 0.85)
        monitor = self.make_monitor()

        @monitoring.observe_prediction(mode_index=0)
        def fake(mode):
            monitoring.note_model_confidence(values)
            return "a"

        with patch.object(monitoring, "get_monitor", return_value=monitor):
            fake(0)
        self.assertAlmostEqual(monitor.snapshot()["tasks"]["alphabet"]["confidence"]["mean"], 0.85)
        monitoring.note_model_confidence(object())  # Unscoped calls are harmless.


class MiddlewareTests(unittest.TestCase):
    def test_asgi_success_validation_error_and_server_error_without_body_reads(self):
        monitor = monitoring.Monitor()
        for status in (200, 422):
            messages = []
            async def app(scope, receive, send):
                await send({"type": "http.response.start", "status": status, "headers": []})
                await send({"type": "http.response.body", "body": b"unchanged"})
            async def receive():
                raise AssertionError("Monitoring must not read request bodies")
            async def send(message):
                messages.append(message)
            asyncio.run(monitoring.MonitoringMiddleware(app, monitor)(
                {"type": "http", "path": "/api/predict-frame", "query_string": b"private-data"}, receive, send))
            self.assertEqual(messages[-1]["body"], b"unchanged")
        async def failing(scope, receive, send):
            raise RuntimeError("sensitive-exception-payload")
        with self.assertRaises(RuntimeError):
            asyncio.run(monitoring.MonitoringMiddleware(failing, monitor)(
                {"type": "http", "path": "/api/match"}, receive, send))
        snapshot = monitor.snapshot()
        self.assertEqual(snapshot["api"]["predict_frame"]["completed_requests"], 2)
        self.assertEqual(snapshot["api"]["match"]["status_counts"]["5xx"], 1)
        self.assertNotIn("private-data", json.dumps(snapshot))
        self.assertNotIn("sensitive-exception-payload", json.dumps(snapshot))

    def test_monitor_endpoint_excluded_and_monitor_errors_do_not_change_responses(self):
        monitor = monitoring.Monitor()
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 404, "headers": []})
        async def receive():
            return {}
        messages = []
        async def send(message):
            messages.append(message)
        asyncio.run(monitoring.MonitoringMiddleware(app, monitor)(
            {"type": "http", "path": "/api/monitoring"}, receive, send))
        self.assertEqual(sum(value["completed_requests"] for value in monitor.snapshot()["api"].values()), 0)
        with patch.object(monitor, "record_request", side_effect=RuntimeError("telemetry failed")):
            asyncio.run(monitoring.MonitoringMiddleware(app, monitor)(
                {"type": "http", "path": "/api/match"}, receive, send))
        self.assertEqual(messages[-1]["status"], 404)


class OperatorEndpointTests(unittest.TestCase):
    def test_endpoint_auth_hidden_route_and_no_store_headers_without_fastapi_dependency(self):
        fastapi = ModuleType("fastapi")
        route_options = []
        def get(*args, **kwargs):
            route_options.append(kwargs)
            return lambda function: function
        class HTTPError(Exception):
            def __init__(self, status_code, detail):
                self.status_code, self.detail = status_code, detail
        fastapi.APIRouter = lambda: SimpleNamespace(get=get)
        fastapi.Header = lambda default=None: default
        fastapi.HTTPException = HTTPError
        fastapi.Response = object
        spec = importlib.util.spec_from_file_location("operator_route_fixture", Path(__file__).parents[1] / "routers/monitoring.py")
        route = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"fastapi": fastapi}):
            spec.loader.exec_module(route)
        monitor = monitoring.Monitor(token=TOKEN)
        response = SimpleNamespace(headers={})
        with patch.object(route, "get_monitor", return_value=monitor):
            for header in (None, "Bearer invalid"):
                with self.assertRaises(HTTPError) as denied:
                    route.monitoring_snapshot(response, header)
                self.assertEqual(denied.exception.status_code, 404)
                self.assertEqual(denied.exception.detail, "Not found")
            snapshot = route.monitoring_snapshot(response, "Bearer " + TOKEN)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertFalse(route_options[0]["include_in_schema"])
        self.assertNotIn(TOKEN, json.dumps(snapshot))

    @unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI is not installed")
    def test_real_fastapi_binding_and_asgi_middleware_without_models(self):
        from fastapi import FastAPI
        from routers import monitoring as route

        monitor = monitoring.Monitor(token=TOKEN)
        app = FastAPI()
        app.include_router(route.router, prefix="/api")
        app.add_middleware(monitoring.MonitoringMiddleware, monitor=monitor)
        async def request(path, authorization=None):
            messages = []
            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}
            async def send(message):
                messages.append(message)
            headers = [(b"authorization", authorization.encode())] if authorization else []
            await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                       "method": "GET", "scheme": "http", "path": path, "raw_path": path.encode(),
                       "root_path": "", "query_string": b"", "headers": headers,
                       "server": ("test", 80), "client": ("127.0.0.1", 12345)}, receive, send)
            status = next(message for message in messages if message["type"] == "http.response.start")
            body = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
            return status, body

        with patch.object(route, "get_monitor", return_value=monitor):
            self.assertEqual(asyncio.run(request("/api/monitoring"))[0]["status"], 404)
            self.assertEqual(asyncio.run(request("/api/monitoring", "Bearer wrong"))[0]["status"], 404)
            response, body = asyncio.run(request("/api/monitoring", "Bearer " + TOKEN))
            self.assertEqual(response["status"], 200)
            self.assertEqual(dict(response["headers"])[b"cache-control"], b"no-store")
            self.assertNotIn(TOKEN, body.decode())
            self.assertNotIn("/api/monitoring", json.loads(asyncio.run(request("/openapi.json"))[1])["paths"])


class PredictionContractTests(unittest.TestCase):
    def test_probability_hook_adds_no_predict_or_preprocessing_calls(self):
        for kind, center_input in (("alphabet", "handPoints"), ("word", "result")):
            path = Path(__file__).parents[1] / f"PSL/{kind}_recognition/{kind}_recognition.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            predictions = [node for node in calls if ast.unparse(node.func) == "model.predict"]
            self.assertEqual(len(predictions), 1)
            self.assertEqual(ast.unparse(predictions[0].args[0]), "scaler.transform(np.array([handRightResults]))")
            centers = [node for node in calls if ast.unparse(node.func) == "move.centerPoints"]
            self.assertEqual([ast.unparse(arg) for arg in centers[0].args], [center_input])
            argmax = [node for node in calls if ast.unparse(node.func) == "np.argmax"]
            self.assertEqual(len(argmax), 1)
            self.assertEqual(ast.unparse(argmax[0].args[0]), "y_pred")
            self.assertEqual(len([node for node in calls if ast.unparse(node.func) == "note_model_confidence"]), 1)


if __name__ == "__main__":
    unittest.main()
