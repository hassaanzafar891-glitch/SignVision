"""Run inside the backend container after starting Compose.

Exercises HTTP decoding, MediaPipe and loading both real model pipelines on a
blank JPEG. It does NOT establish recognition accuracy or webcam operation.
"""
import json
import urllib.error
import urllib.request

import cv2
import numpy as np


def request_frame(mode):
    ok, image = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
    assert ok
    body = (b"--psl-test\r\n"
            b'Content-Disposition: form-data; name="frame"; filename="blank.jpg"\r\n'
            b"Content-Type: image/jpeg\r\n\r\n" + image.tobytes() +
            b"\r\n--psl-test--\r\n")
    request = urllib.request.Request(
        f"http://localhost:8000/api/predict-frame?mode={mode}", data=body,
        headers={"Content-Type": "multipart/form-data; boundary=psl-test"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    assert result == {"label": "no confidence", "mode": mode}, result
    print(f"Mode {mode}: blank-frame HTTP/MediaPipe smoke test passed")


if __name__ == "__main__":
    request_frame(0)
    request_frame(1)
