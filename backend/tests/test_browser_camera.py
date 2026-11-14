"""Dependency-isolated regressions; these do not validate real ML inference."""
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import MagicMock, patch


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code


def load_route():
    fastapi = ModuleType("fastapi")
    router = SimpleNamespace(post=lambda *args, **kwargs: lambda function: function)
    fastapi.APIRouter = lambda: router
    fastapi.File = lambda *args: None
    fastapi.HTTPException = HTTPException
    fastapi.UploadFile = object
    cv = MagicMock()
    cv.imdecode.return_value = SimpleNamespace(shape=(480, 640, 3))
    helper = MagicMock()
    helper._get_pose_keypoints.return_value = [0.0] * 57
    helper._get_hand_keypoints.return_value = [1.0] * 63
    service = MagicMock()
    stubs = {
        "fastapi": fastapi, "cv2": cv, "numpy": MagicMock(),
        "mediapipe": MagicMock(),
        "PSL.helper": SimpleNamespace(mediapipe_helper=helper),
        "services": SimpleNamespace(recognition_service=service),
    }
    path = Path(__file__).parents[1] / "routers" / "browser_camera.py"
    spec = importlib.util.spec_from_file_location("tested_browser_camera", path)
    route = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(route)
    return route, cv, helper, service


class BrowserFrameTests(unittest.TestCase):
    def setUp(self):
        self.route, self.cv, self.helper, self.service = load_route()

    def upload(self, data=b"\xff\xd8test"):
        return SimpleNamespace(file=io.BytesIO(data))

    def assert_status(self, status, upload=None, mode=0):
        with self.assertRaises(HTTPException) as raised:
            self.route.predict_frame(upload or self.upload(), mode)
        self.assertEqual(raised.exception.status_code, status)

    def test_invalid_mode_and_busy(self):
        self.assert_status(400, mode=2)
        self.route._processing_lock.acquire()
        try:
            self.assert_status(429)
        finally:
            self.route._processing_lock.release()

    def test_invalid_uploads_release_lock(self):
        self.assert_status(400, self.upload(b""))
        self.assert_status(413, self.upload(b"x" * 2_000_001))
        self.cv.imdecode.return_value = None
        self.assert_status(400)
        self.assertFalse(self.route._processing_lock.locked())

    def test_current_frame_and_mode_are_used_and_files_removed(self):
        paths = []
        def predict(path, mode):
            paths.append(Path(path))
            payload = json.loads(Path(path).read_text())
            self.assertEqual(payload["people"][0]["hand_right_keypoints_2d"],
                             self.helper._get_hand_keypoints.return_value)
            return "alphabet" if mode == 0 else "word"
        self.service.predict_file.side_effect = predict
        self.assertEqual(self.route.predict_frame(self.upload(), 0)["label"], "alphabet")
        self.helper._get_hand_keypoints.return_value = [2.0] * 63
        self.assertEqual(self.route.predict_frame(self.upload(), 1)["label"], "word")
        self.assertNotEqual(paths[0], paths[1])
        self.assertTrue(all(not path.exists() for path in paths))
        self.service.predict.assert_not_called()

    def test_inference_failure_releases_lock_and_removes_file(self):
        paths = []
        def fail(path, mode):
            paths.append(Path(path))
            raise RuntimeError("failed")
        self.service.predict_file.side_effect = fail
        with self.assertRaises(RuntimeError):
            self.route.predict_frame(self.upload(), 0)
        self.assertFalse(self.route._processing_lock.locked())
        self.assertFalse(paths[0].exists())


if __name__ == "__main__":
    unittest.main()
