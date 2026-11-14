import json
import tempfile
from pathlib import Path
import threading

import cv2
import mediapipe as mp
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from PSL.helper import mediapipe_helper
from services import recognition_service

router = APIRouter()
_processing_lock = threading.Lock()
_holistic = None

@router.post("/predict-frame")
def predict_frame(frame: UploadFile = File(...), mode: int = 0):
    global _holistic
    if mode not in (0, 1):
        raise HTTPException(status_code=400, detail="Invalid mode")

    if not _processing_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Frame processing busy")

    try:
        data = frame.file.read(2_000_001)

        if len(data) > 2_000_000:
            raise HTTPException(status_code=413, detail="Frame too large")

        if not data.startswith(b"\xff\xd8"):
            raise HTTPException(status_code=400, detail="Expected a JPEG frame")

        image = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        height, width = image.shape[:2]
        if width > 1920 or height > 1080:
            raise HTTPException(status_code=413, detail="Frame dimensions too large")
        if _holistic is None:
            _holistic = mp.solutions.holistic.Holistic(
                static_image_mode=True, min_detection_confidence=0.5
            )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = _holistic.process(rgb)

        pose = mediapipe_helper._get_pose_keypoints(
            results.pose_landmarks, width, height
        )
        right = mediapipe_helper._get_hand_keypoints(
            results.right_hand_landmarks, width, height
        )
        left = mediapipe_helper._get_hand_keypoints(
            results.left_hand_landmarks, width, height
        )

        payload = {
            "version": 1.3,
            "people": [{
                "person_id": [-1],
                "pose_keypoints_2d": pose,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": left,
                "hand_right_keypoints_2d": right,
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }]
        }

        # Keep this request outside the legacy capture directory. Never select
        # a frame by modification time; the existing models read this exact file.
        with tempfile.TemporaryDirectory(prefix="psl-frame-") as directory:
            path = Path(directory) / "keypoints.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            label = recognition_service.predict_file(str(path), mode)
        return {"label": label, "mode": mode}

    finally:
        _processing_lock.release()
