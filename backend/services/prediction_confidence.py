"""Request-local classification scores; independent of optional monitoring."""

from contextlib import contextmanager
from contextvars import ContextVar
import math


_capture = ContextVar("signvision_classification_confidence", default=None)


@contextmanager
def capture_confidence():
    observation = {"confidence": None}
    token = _capture.set(observation)
    try:
        yield observation
    finally:
        _capture.reset(token)


def record_model_probability(probabilities, predicted_index, class_count):
    """Read the selected class from one genuine softmax row; never normalize scores.

    Invalid/unavailable outputs suppress confidence without changing inference.
    No detector/landmark confidence enters this function.
    """
    observation = _capture.get()
    if observation is None:
        return
    observation["confidence"] = None
    try:
        values = probabilities.tolist()
        if len(values) != 1 or len(values[0]) != class_count or class_count < 2:
            return
        row = values[0]
        if not all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in row):
            return
        if abs(sum(row) - 1) > 1e-5 or not 0 <= predicted_index < class_count:
            return
        if predicted_index != max(range(class_count), key=row.__getitem__):
            return
        observation["confidence"] = float(row[predicted_index])
    except Exception:
        # Optional score reporting must not change the label or raise new inference errors.
        pass
