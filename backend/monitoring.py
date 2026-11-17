"""Bounded per-process aggregate telemetry; no frames, landmarks or identities.

Only standard-library dependencies. Monitoring failures must never affect inference.
"""

from bisect import bisect_right
from collections import deque
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import threading
import time


TASKS = ("alphabet", "word", "unknown")
OUTCOMES = ("no_confidence", "no_match", "error", "unknown")
MAX_COUNT = 2 ** 63 - 1
LATENCY_EDGES = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
_observation = ContextVar("signvision_prediction_telemetry", default=None)
_runtime = None
_runtime_lock = threading.Lock()


def js_divergence(reference, current):
    """Base-2 Jensen-Shannon divergence in [0,1], without smoothing."""
    if set(reference) != set(current) or not reference:
        raise ValueError("Distribution bins differ")
    for distribution in (reference, current):
        if any(type(value) is not int or value < 0 for value in distribution.values()) or not sum(distribution.values()):
            raise ValueError("Counts must be nonnegative integers with a positive total")
    p_total, q_total = sum(reference.values()), sum(current.values())
    result = 0.0
    for key in reference:
        p, q = reference[key] / p_total, current[key] / q_total
        mean = (p + q) / 2
        if p:
            result += p * math.log2(p / mean) / 2
        if q:
            result += q * math.log2(q / mean) / 2
    return min(1.0, max(0.0, result))


def _duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 604800:
        raise ValueError("Duration must be finite and between zero and seven days")


class _Latency:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0
        self.buckets = [0] * (len(LATENCY_EDGES) + 1)

    def add(self, seconds):
        if self.count >= MAX_COUNT:
            return
        self.count += 1
        self.total += seconds
        self.maximum = max(self.maximum, seconds)
        self.buckets[bisect_right(LATENCY_EDGES, seconds)] += 1

    def snapshot(self):
        return {"count": self.count, "mean_seconds": self.total / self.count if self.count else None,
                "max_seconds": self.maximum if self.count else None,
                "bucket_edges_seconds": list(LATENCY_EDGES), "bucket_counts": list(self.buckets)}


class _Task:
    def __init__(self, window_size, labels=(), identity=None):
        self.identity = identity
        self.label_bins = {label: f"class_{index}" for index, label in enumerate(labels)}
        self.bins = {key: 0 for key in (*self.label_bins.values(), *OUTCOMES)}
        self.window = deque(maxlen=window_size)
        self.calls = 0
        self.outcomes = {key: 0 for key in ("prediction", *OUTCOMES)}
        self.latency = _Latency()
        self.confidence_count = 0
        self.confidence_sum = 0.0
        self.confidence_buckets = [0] * 5
        self.invalid_confidence = 0

    def add(self, label, confidence, failed, seconds):
        if failed:
            category = bucket = "error"
        elif label == "no confidence":
            category = bucket = "no_confidence"
        elif label == "no match":
            category = bucket = "no_match"
        else:
            bucket = self.label_bins.get(label, "unknown") if isinstance(label, str) else "unknown"
            category = "prediction" if bucket != "unknown" else "unknown"
        if len(self.window) == self.window.maxlen:
            self.bins[self.window.popleft()] -= 1
        self.window.append(bucket)
        self.bins[bucket] += 1
        if self.calls >= MAX_COUNT:
            return
        self.calls += 1
        self.outcomes[category] += 1
        self.latency.add(seconds)
        if confidence is not None and category == "prediction":
            if type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1:
                self.confidence_count += 1
                self.confidence_sum += confidence
                self.confidence_buckets[bisect_right((0.2, 0.4, 0.6, 0.8), confidence)] += 1
            else:
                self.invalid_confidence += 1


class Monitor:
    def __init__(self, *, enabled=True, token="", window_size=1000, min_samples=100,
                 min_reference_samples=100, threshold=0.1, baseline_path=""):
        if type(window_size) is not int or not 20 <= window_size <= 10000:
            raise ValueError("Window size must be between 20 and 10000")
        if type(min_samples) is not int or not 1 <= min_samples <= window_size:
            raise ValueError("Invalid minimum current sample count")
        if type(min_reference_samples) is not int or not 1 <= min_reference_samples <= 10_000_000:
            raise ValueError("Invalid minimum reference sample count")
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("Jensen-Shannon threshold must be in (0,1]")
        self.enabled = bool(enabled)
        self.window_size, self.min_samples = window_size, min_samples
        self.min_reference_samples, self.threshold = min_reference_samples, threshold
        # Retain only a digest, never include the token in snapshots or logs.
        valid_token = isinstance(token, str) and 32 <= len(token) <= 512 and not any(c.isspace() for c in token)
        self._token_hash = hashlib.sha256(token.encode()).digest() if valid_token else None
        self._lock = threading.Lock()
        self._tasks = {name: _Task(window_size) for name in TASKS}
        self._routes = {name: {"requests": 0, "statuses": {s: 0 for s in ("2xx", "3xx", "4xx", "5xx", "other")},
                               "rate_limited": 0, "latency": _Latency()}
                        for name in ("predict_frame", "match", "other")}
        self._reference = {}
        self._reference_reason = "not_configured"
        if baseline_path:
            self._load_reference(baseline_path)

    def authorized(self, authorization):
        if not self.enabled or self._token_hash is None or not isinstance(authorization, str) or len(authorization) > 1024:
            return False
        scheme, separator, value = authorization.partition(" ")
        return bool(separator and scheme.lower() == "bearer"
                    and secrets.compare_digest(self._token_hash, hashlib.sha256(value.encode()).digest()))

    def register_task(self, task, labels, identity):
        labels = list(labels)
        if task not in TASKS[:2] or not 2 <= len(labels) <= 64 or len(set(labels)) != len(labels):
            raise ValueError("Invalid model labels or task")
        if not all(isinstance(label, str) and 0 < len(label) <= 128 for label in labels):
            raise ValueError("Invalid class label")
        if identity is not None and not re.fullmatch(r"[0-9a-f]{64}", identity):
            raise ValueError("Invalid model identity")
        with self._lock:
            # Never combine distributions from different models/encoders.
            self._tasks[task] = _Task(self.window_size, labels, identity)

    def _load_reference(self, path):
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("Duplicate reference key")
                value[key] = item
            return value
        try:
            path = Path(path)
            with path.open("rb") as stream:
                raw = stream.read(262145)
            if len(raw) > 262144:
                raise ValueError("Reference too large")
            data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique)
            if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
                raise ValueError("Invalid schema")
            provenance = data["provenance"]
            if provenance["status"] != "reviewed_production_aggregate":
                raise ValueError("Reference has not been reviewed")
            if not all(isinstance(provenance.get(k), str) and 0 < len(provenance[k]) <= 256
                       for k in ("reviewed_by", "reference_id")):
                raise ValueError("Missing reference provenance")
            start, end = (datetime.fromisoformat(provenance[key]) for key in ("window_start", "window_end"))
            if start.utcoffset() is None or end.utcoffset() is None or start >= end:
                raise ValueError("Invalid collection window")
            tasks = data["tasks"]
            if not isinstance(tasks, dict) or not tasks or not set(tasks).issubset(TASKS[:2]):
                raise ValueError("Invalid reference tasks")
            for reference in tasks.values():
                if not re.fullmatch(r"[0-9a-f]{64}", reference["model_identity"]):
                    raise ValueError("Invalid reference model")
                counts = reference["counts"]
                if not isinstance(counts, dict) or not 1 <= len(counts) <= 68:
                    raise ValueError("Invalid reference bins")
                for key, count in counts.items():
                    if key not in OUTCOMES and not re.fullmatch(r"class_([0-9]|[1-5][0-9]|6[0-3])", key):
                        raise ValueError("Unknown reference bin")
                    if type(count) is not int or not 0 <= count <= MAX_COUNT:
                        raise ValueError("Invalid reference count")
                if type(reference["sample_count"]) is not int or not 0 < reference["sample_count"] <= MAX_COUNT:
                    raise ValueError("Invalid reference sample count")
                if sum(counts.values()) != reference["sample_count"]:
                    raise ValueError("Reference counts do not sum to sample count")
            self._reference, self._reference_reason = tasks, None
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
            self._reference, self._reference_reason = {}, "invalid_or_unreadable_reference"

    def record_request(self, route, status, seconds):
        if not self.enabled:
            return
        _duration(seconds)
        if type(status) is not int or not 100 <= status <= 599:
            raise ValueError("Invalid HTTP status")
        with self._lock:
            stats = self._routes.get(route, self._routes["other"])
            if stats["requests"] >= MAX_COUNT:
                return
            stats["requests"] += 1
            stats["statuses"][f"{status // 100}xx" if status >= 200 else "other"] += 1
            stats["rate_limited"] += int(status == 429)
            stats["latency"].add(seconds)

    def record_prediction(self, task, label, seconds, confidence=None, failed=False):
        if not self.enabled:
            return
        _duration(seconds)
        with self._lock:
            self._tasks.get(task, self._tasks["unknown"]).add(label, confidence, failed, seconds)

    def _drift(self, name, stats):
        reference = self._reference.get(name)
        result = {"method": "jensen_shannon_divergence_base2", "threshold": self.threshold,
                  "current_samples": len(stats.window), "score": None}
        if reference is None:
            return {**result, "status": "baseline_unavailable", "reason": self._reference_reason or "task_not_in_reference"}
        if not stats.identity or stats.identity != reference["model_identity"]:
            return {**result, "status": "baseline_unavailable", "reason": "model_identity_mismatch"}
        if set(stats.bins) != set(reference["counts"]):
            return {**result, "status": "baseline_unavailable", "reason": "reference_bins_mismatch"}
        if reference["sample_count"] < self.min_reference_samples:
            return {**result, "status": "insufficient_reference_samples"}
        if len(stats.window) < self.min_samples:
            return {**result, "status": "insufficient_current_samples"}
        score = js_divergence(reference["counts"], stats.bins)
        return {**result, "status": "drift_detected" if score >= self.threshold else "no_alert", "score": score}

    def snapshot(self):
        with self._lock:
            routes, tasks = {}, {}
            for name, stats in self._routes.items():
                count = stats["requests"]
                routes[name] = {"completed_requests": count, "status_counts": dict(stats["statuses"]),
                                "http_error_rate": (stats["statuses"]["4xx"] + stats["statuses"]["5xx"]) / count if count else None,
                                "server_error_rate": stats["statuses"]["5xx"] / count if count else None,
                                "counters_saturated": count == MAX_COUNT,
                                "rate_limited": stats["rate_limited"], "latency": stats["latency"].snapshot()}
            for name, stats in self._tasks.items():
                tasks[name] = {
                    "model_identity": stats.identity, "calls": stats.calls, "outcomes": dict(stats.outcomes),
                    "inference_error_rate": stats.outcomes["error"] / stats.calls if stats.calls else None,
                    "latency": stats.latency.snapshot(),
                    "confidence": {"count": stats.confidence_count,
                                   "mean": stats.confidence_sum / stats.confidence_count if stats.confidence_count else None,
                                   "bucket_edges": [0.2, 0.4, 0.6, 0.8],
                                   "bucket_counts": list(stats.confidence_buckets), "invalid_values": stats.invalid_confidence},
                    "window": {"samples": len(stats.window), "capacity": self.window_size,
                               "counts": dict(stats.bins) if len(stats.window) >= self.min_samples else None},
                    "drift": self._drift(name, stats),
                    "counters_saturated": stats.calls == MAX_COUNT,
                }
            return {"schema_version": 1, "scope": "single_process_since_start_or_model_registration",
                    "enabled": self.enabled, "api": routes, "tasks": tasks}


def get_monitor():
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                try:
                    from core.config import settings
                    _runtime = Monitor(
                        enabled=settings.MONITORING_ENABLED, token=settings.MONITORING_TOKEN.get_secret_value(),
                        window_size=settings.MONITORING_WINDOW_SIZE, min_samples=settings.MONITORING_MIN_SAMPLES,
                        min_reference_samples=settings.MONITORING_MIN_REFERENCE_SAMPLES,
                        threshold=settings.MONITORING_DRIFT_THRESHOLD, baseline_path=settings.MONITORING_BASELINE_PATH,
                    )
                except Exception:
                    _runtime = Monitor(enabled=False)
    return _runtime


def register_loaded_model(task, labels, paths):
    """Hash the loaded bundle once at service startup, never once per frame."""
    try:
        hashes = {}
        for key, path in paths.items():
            digest = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[key] = digest.hexdigest()
        identity = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        get_monitor().register_task(task, labels, identity)
    except Exception:
        # A monitoring/hash problem must not stop the already-loaded model.
        pass


def note_confidence(value):
    observation = _observation.get()
    if observation is not None:
        observation["confidence"] = value


def note_model_confidence(probabilities):
    if _observation.get() is not None:
        try:
            note_confidence(float(probabilities.max()))
        except Exception:
            pass


def note_prediction_error():
    observation = _observation.get()
    if observation is not None:
        observation["failed"] = True


def observe_prediction(mode_index):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            try:
                monitor = get_monitor()
            except Exception:
                return function(*args, **kwargs)
            if not monitor.enabled or _observation.get() is not None:
                return function(*args, **kwargs)
            mode = kwargs.get("mode", args[mode_index] if len(args) > mode_index else None)
            task = "alphabet" if mode == 0 else "word" if mode == 1 else "unknown"
            observation = {"confidence": None, "failed": False}
            token = _observation.set(observation)
            started, result = time.perf_counter(), None
            try:
                result = function(*args, **kwargs)
                return result
            except BaseException:
                observation["failed"] = True
                raise
            finally:
                _observation.reset(token)
                try:
                    monitor.record_prediction(task, result, time.perf_counter() - started,
                                              observation["confidence"], observation["failed"])
                except Exception:
                    pass
        return wrapped
    return decorate


class MonitoringMiddleware:
    """Pure ASGI wrapper: never reads bodies, query strings, identities or headers."""
    def __init__(self, app, monitor):
        self.app, self.monitor = app, monitor

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not self.monitor.enabled or not path.startswith("/api/") or path == "/api/monitoring":
            return await self.app(scope, receive, send)
        route = {"/api/predict-frame": "predict_frame", "/api/match": "match"}.get(path, "other")
        started, status = time.perf_counter(), 500

        async def observed_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            return await self.app(scope, receive, observed_send)
        except BaseException:
            status = 500
            raise
        finally:
            try:
                self.monitor.record_request(route, status, time.perf_counter() - started)
            except Exception:
                pass
