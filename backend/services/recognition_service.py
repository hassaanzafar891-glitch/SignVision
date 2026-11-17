
import os
import json
from typing import Optional
from core.config import settings
import PSL.alphabet_recognition.alphabet_recognition as alphabet
import PSL.word_recognition.word_recognition as word
import PSL.helper.helperFunc as helper
from monitoring import note_prediction_error, observe_prediction, register_loaded_model


for _task, _module in (("alphabet", alphabet), ("word", word)):
    register_loaded_model(_task, list(_module.le.classes_), {
        key: os.path.join("data", "models", f"{_task}_{suffix}")
        for key, suffix in (("model", "model.h5"), ("scaler", "scaler.pkl"),
                            ("label_encoder", "label_encoder.pkl"))
    })


def get_latest_keypoint_file() -> Optional[str]:
    if not os.path.exists(settings.KEYPOINTS_DIR):
        return None
    entries = [
        e for e in os.scandir(settings.KEYPOINTS_DIR)
        if e.is_file() and e.name.endswith(".json")
    ]
    if not entries:
        return None
    return max(entries, key=lambda e: e.stat().st_mtime).name


@observe_prediction(mode_index=0)
def predict(mode: int) -> str:
    filename = get_latest_keypoint_file()
    if not filename:
        return "no match"

    filepath = os.path.join(settings.KEYPOINTS_DIR, filename)

    return predict_file(filepath, mode)


@observe_prediction(mode_index=1)
def predict_file(filepath: str, mode: int) -> str:
    """Run existing preprocessing and models against an explicitly chosen frame."""
    if mode not in (0, 1):
        raise ValueError("Invalid recognition mode")
    try:
        js = json.loads(open(filepath).read())
        for items in js['people']:
            hr = items["hand_right_keypoints_2d"]
            hl = items["hand_left_keypoints_2d"]
        r_conf = sum(hr[i] for i in range(2, len(hr), 3))
        l_conf = sum(hl[i] for i in range(2, len(hl), 3))

        if mode == 0:
            result = alphabet.match_ann(filepath)
        else:
            result = word.match_ann(filepath)
        return result
    except Exception:
        # Preserve the response while reporting an aggregate inference error.
        # Do not log per-frame labels, keypoint paths or exception payloads.
        note_prediction_error()
        return "no match"
