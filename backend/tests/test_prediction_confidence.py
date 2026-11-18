"""Execute real inference functions with fake model outputs; never load/train PSL models."""

import ast
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.prediction_confidence import capture_confidence, record_model_probability


class Probabilities:
    def __init__(self, values):
        self.values = values
    def tolist(self):
        return self.values


def function_from_source(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class PredictionConfidenceTests(unittest.TestCase):
    def test_selected_probability_is_read_without_normalizing_or_modifying_output(self):
        probabilities = Probabilities([[0.2, 0.8]])
        with capture_confidence() as result:
            record_model_probability(probabilities, 1, 2)
        self.assertEqual(result["confidence"], 0.8)
        self.assertEqual(probabilities.values, [[0.2, 0.8]])

    def test_invalid_outputs_are_unavailable_not_fabricated(self):
        invalid = [[], [[]], [[.5, .5], [.5, .5]], [[2, -1]], [[.2, .2]],
                   [[float("nan"), .5]], [[float("inf"), 0]], [["0.2", .8]], [[True, 0]]]
        for values in invalid:
            with self.subTest(values=values), capture_confidence() as result:
                record_model_probability(Probabilities(values), 1, 2)
            self.assertIsNone(result["confidence"])
        for index in (-1, 2, 0):
            with capture_confidence() as result:
                record_model_probability(Probabilities([[.2, .8]]), index, 2)
            self.assertIsNone(result["confidence"])

    def test_missing_nested_and_threaded_captures_do_not_reuse_other_frame_scores(self):
        record_model_probability(object(), 0, 2)  # Legacy calls need no capture.
        with capture_confidence() as outer:
            record_model_probability(Probabilities([[.3, .7]]), 1, 2)
            with capture_confidence() as inner:
                self.assertIsNone(inner["confidence"])
            self.assertEqual(outer["confidence"], .7)
        with capture_confidence() as fresh:
            self.assertIsNone(fresh["confidence"])
        def scoped(value):
            with capture_confidence() as result:
                record_model_probability(Probabilities([[1-value, value]]), 1, 2)
                return result["confidence"]
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.assertEqual(list(executor.map(scoped, [.6, .7, .8, .9] * 8)), [.6, .7, .8, .9] * 8)

    def inference(self, task, confidence=21):
        points = list(range(1, 43))
        scaled = [value * 2 for value in points]
        scaler = SimpleNamespace(transform=MagicMock(return_value="saved-scaler-output"))
        model = SimpleNamespace(predict=MagicMock(return_value=Probabilities([[.15, .85]])))
        move = SimpleNamespace(centerPoints=MagicMock(side_effect=lambda values: (values, None)))
        namespace = {"json": json, "math": math, "record_model_probability": record_model_probability,
            "note_model_confidence": MagicMock(), "model": model, "scaler": scaler,
            "np": SimpleNamespace(array=lambda values: values, argmax=lambda probabilities: 1),
            "le": SimpleNamespace(classes_=["first", "second"], inverse_transform=lambda indices: ["second"]),
            "helper": SimpleNamespace(confidencePoints=lambda hand: hand, confidence=lambda hand: confidence,
                                      removePoints=lambda hand: points),
            "scale": SimpleNamespace(scalePoints=lambda hand, distance: (scaled, None)), "move": move}
        path = Path(__file__).parents[1] / "PSL" / (task + "_recognition") / (task + "_recognition.py")
        match = function_from_source(path, "match_ann", namespace)
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.json"
            frame.write_text(json.dumps({"people": [{"hand_right_keypoints_2d": [0] * 63}]}))
            with capture_confidence() as result:
                label = match(str(frame))
        return label, result["confidence"], model, scaler, move, points, scaled

    def test_both_real_inference_functions_keep_labels_preprocessing_and_one_predict_call(self):
        for task in ("alphabet", "word"):
            with self.subTest(task=task):
                label, score, model, scaler, move, original, scaled = self.inference(task)
                self.assertEqual((label, score), ("second", .85))
                model.predict.assert_called_once_with("saved-scaler-output", verbose=0)
                expected = original if task == "alphabet" else scaled
                move.centerPoints.assert_called_once_with(expected)
                scaler.transform.assert_called_once_with([expected])

    def test_missing_hands_do_not_predict_or_reuse_detection_confidence(self):
        for task in ("alphabet", "word"):
            label, score, model, *_ = self.inference(task, confidence=0)
            self.assertEqual(label, "no confidence")
            self.assertIsNone(score)
            model.predict.assert_not_called()

    def wrapper(self, predict):
        return function_from_source(Path(__file__).parents[1] / "services/recognition_service.py",
            "predict_file_with_confidence", {"capture_confidence": capture_confidence, "predict_file": predict})

    def test_service_wrapper_preserves_label_and_discards_confidence_on_no_match(self):
        for label in ("word", "no match", "no confidence"):
            def predict(path, mode):
                record_model_probability(Probabilities([[.1, .9]]), 1, 2)
                return label
            wrapped_predict = MagicMock(side_effect=predict)
            self.assertEqual(self.wrapper(wrapped_predict)("exact-frame.json", 1),
                             (label, .9 if label == "word" else None))
            wrapped_predict.assert_called_once_with("exact-frame.json", 1)

    def test_errors_cleanup_capture_and_legacy_missing_scores_remain_null(self):
        def fail(path, mode):
            record_model_probability(Probabilities([[.1, .9]]), 1, 2)
            raise RuntimeError("inference failed")
        with self.assertRaises(RuntimeError):
            self.wrapper(fail)("frame", 0)
        self.assertEqual(self.wrapper(lambda path, mode: "same label")("frame", 0), ("same label", None))


if __name__ == "__main__":
    unittest.main()
