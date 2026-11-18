"""Temporary synthetic fixtures only; no training, inference stack or registry."""

import builtins
from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_sandbox import foundation as demo
from tracking.evaluation_gate import GateError
from tracking.evaluate_models import read_bundle


class DemoSandboxTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.addCleanup(patch.stopall)
        patch.object(demo, "WORKSPACE", self.workspace).start()
        self.root = self.workspace / ".signvision-demo"
        self.protected = {}
        for name in ("backend/data/models/alphabet_model.h5", "backend/data/models/word_scaler.pkl",
                     "backend/data/db/main_dataset.db", "mlflow.db", "backend/mlflow.db",
                     "docker-compose.yml", "Caddyfile"):
            path = self.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("protected-sentinel:" + name).encode())
            self.protected[path] = path.read_bytes()

    def assert_protected(self):
        self.assertEqual(self.protected, {p: p.read_bytes() for p in self.protected})

    def snapshot(self, root=None):
        root = self.root if root is None else root
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    def make_link(self, link, target, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation is not available for this test user")

    def test_initialize_has_two_fixtures_and_initial_demo_only_pointer(self):
        result = demo.initialize()
        self.assertEqual(result["active_version"], demo.CHAMPION)
        self.assertEqual(result["available_fixtures"], list(demo.VERSIONS))
        self.assertEqual(result["registry"], "disabled")
        for key in ("training_performed", "metrics_computed", "promotion_performed", "production_evidence"):
            self.assertIs(result[key], False)
        self.assertEqual(demo.read_active()["artifact_type"], "hand_authored_threshold_fixture")
        self.assertEqual(len(self.snapshot()), 8)
        self.assertFalse(list(self.root.rglob("*.h5")))
        self.assertFalse(list(self.root.rglob("*.pkl")))
        self.assert_protected()

    def test_fixtures_are_byte_identical_in_another_workspace(self):
        demo.initialize()
        before = self.snapshot()
        with tempfile.TemporaryDirectory() as other:
            with patch.object(demo, "WORKSPACE", Path(other).resolve()):
                demo.initialize()
                after = self.snapshot(Path(other) / ".signvision-demo")
        self.assertEqual(before, after)

    def test_synthetic_partitions_are_complete_disjoint_and_class_complete(self):
        demo.initialize()
        dataset = demo.read_dataset()
        self.assertFalse(dataset["production_evidence"])
        self.assertIn("not PSL", dataset["notice"])
        rows = dataset["rows"]
        self.assertEqual(len(rows), 72)
        self.assertEqual(len({tuple(r["features"]) for r in rows}), 72)
        self.assertTrue(all(len(r["features"]) == 42 for r in rows))
        ids = [i for group in dataset["partitions"].values() for i in group]
        self.assertEqual(sorted(ids), list(range(72)))
        for group in dataset["partitions"].values():
            self.assertEqual(len(group), 24)
            self.assertEqual({rows[i]["label"] for i in group}, set(demo.POLICY["class_labels"]))

    def test_initialization_refuses_overwrite_without_changing_files(self):
        demo.initialize()
        before = self.snapshot()
        with self.assertRaises(FileExistsError):
            demo.initialize()
        self.assertEqual(before, self.snapshot())
        self.assert_protected()

    def test_source_configuration_cannot_redirect_paths_or_enable_tracking(self):
        path = self.workspace / "invalid-config.json"
        for field, value in (("runtime_directory", "backend/data/models"), ("registry", "production"),
                             ("tracking", "enabled"), ("schema_version", True)):
            config = deepcopy(demo.POLICY)
            config[field] = value
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.subTest(field=field), patch.object(demo, "CONFIG_PATH", path), \
                 self.assertRaisesRegex(GateError, "configuration"):
                demo.initialize()
            self.assertFalse(self.root.exists())
        self.assert_protected()

    def test_production_environment_and_ml_libraries_are_unused(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("mlflow", "tensorflow", "keras", "sklearn", "mediapipe"):
                raise AssertionError("Sandbox tried to import an ML/registry library")
            return original(name, *args, **kwargs)
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "https://not-a-demo.invalid",
                                   "MLFLOW_REGISTRY_URI": "sqlite:///mlflow.db",
                                   "SIGNVISION_TRACKING_ENABLED": "true",
                                   "SIGNVISION_DEMO_ROOT": str(self.workspace / "backend/data/models")}), \
             patch("builtins.__import__", side_effect=guarded):
            demo.initialize()
            demo.read_active()
            demo.predict_fixture(demo.CHAMPION, [0] * 42)
        self.assert_protected()

    def test_pointer_cannot_reference_external_paths_or_switch_to_challenger(self):
        demo.initialize()
        path = self.root / "active.json"
        original = path.read_bytes()
        edits = [
            {"active_version": "../backend/data/models"},
            {"active_version": str(self.workspace / "backend/data/models")},
            {"active_version": "https://models.invalid/model.h5"},
            {"scope": "production"}, {"model_path": "backend/data/models/alphabet_model.h5"},
            {"manifest_sha256": "0" * 64},
            {"active_version": demo.CHALLENGER, "manifest_sha256": demo.canonical_sha256(
                json.loads((self.root / f"versions/{demo.CHALLENGER}/manifest.json").read_text()))},
        ]
        for change in edits:
            value = json.loads(original)
            value.update(change)
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.subTest(change=change), self.assertRaises(GateError):
                demo.read_active()
            self.assert_protected()
            path.write_bytes(original)
        self.assertEqual(demo.status()["active_version"], demo.CHAMPION)

    def test_tampered_or_missing_fixture_is_rejected(self):
        demo.initialize()
        path = self.root / f"versions/{demo.CHAMPION}/model.json"
        path.write_bytes(b'{"scope":"synthetic-demo-only"}')
        with self.assertRaisesRegex(GateError, "identity"):
            demo.status()
        path.unlink()  # Own temporary fixture only.
        with self.assertRaisesRegex(GateError, "incomplete"):
            demo.status()
        self.assert_protected()

    def test_unlisted_artifacts_and_runtime_config_tampering_are_rejected(self):
        demo.initialize()
        path = self.root / "unexpected.h5"
        path.write_bytes(b"not-a-model")
        with self.assertRaisesRegex(GateError, "Unexpected"):
            demo.status()
        path.unlink()
        (self.root / "config.json").write_text('{"registry":"production"}')
        with self.assertRaisesRegex(GateError, "identity"):
            demo.status()

    def test_missing_completion_receipt_never_yields_an_active_model(self):
        demo.initialize()
        (self.root / "initialized.json").unlink()
        with self.assertRaisesRegex(GateError, "incomplete"):
            demo.read_active()

    def test_interrupted_initialization_does_not_publish_valid_state(self):
        original = Path.open
        def interrupted(path, *args, **kwargs):
            if path.name == "active.json" and args and args[0] == "xb":
                raise OSError("synthetic interruption")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", interrupted), self.assertRaises(OSError):
            demo.initialize()
        self.assertFalse((self.root / "initialized.json").exists())
        with self.assertRaises(GateError):
            demo.read_active()
        self.assert_protected()

    def test_root_symlink_cannot_redirect_initialization_to_production(self):
        self.make_link(self.root, self.workspace / "backend/data/models", directory=True)
        with self.assertRaisesRegex(GateError, "symlinks"):
            demo.initialize()
        self.assert_protected()

    def test_nested_symlink_and_hardlinked_pointer_are_rejected(self):
        demo.initialize()
        path = self.root / "active.json"
        path.unlink()
        try:
            os.link(self.workspace / "mlflow.db", path)
        except (OSError, NotImplementedError):
            self.skipTest("Hard links are unavailable")
        with self.assertRaisesRegex(GateError, "hard-linked"):
            demo.read_active()
        self.assert_protected()

    def test_nested_directory_symlink_cannot_escape_sandbox(self):
        demo.initialize()
        versions = self.root / "versions"
        versions.rename(self.workspace / "original-versions")
        self.make_link(versions, self.workspace / "backend/data/models", directory=True)
        with self.assertRaisesRegex(GateError, "symlinks"):
            demo.status()
        self.assert_protected()

    def test_windows_reparse_attribute_blocks_initialization(self):
        self.root.mkdir()
        original = Path.lstat
        def reparse(path, *args, **kwargs):
            info = original(path, *args, **kwargs)
            if path == self.root:
                return SimpleNamespace(st_mode=info.st_mode, st_nlink=info.st_nlink,
                                       st_file_attributes=0x400)
            return info
        with patch.object(Path, "lstat", reparse), self.assertRaisesRegex(GateError, "reparse"):
            demo.initialize()
        self.assertEqual(list(self.root.iterdir()), [])
        self.assert_protected()

    def test_production_evaluator_rejects_the_synthetic_fixture(self):
        demo.initialize()
        with self.assertRaisesRegex(GateError, "provenance"):
            read_bundle(self.root / "versions" / demo.CHAMPION, "alphabet")
        from tracking.versioning import verify_version
        manifest = json.loads((self.root / f"versions/{demo.CHAMPION}/manifest.json").read_text())
        # It is not even shaped as a production two-task bundle. The production
        # versioning API cannot accept this as a complete neural model release.
        path = self.root / "versions" / demo.CHAMPION
        with self.assertRaises(GateError):
            verify_version(path, demo.canonical_sha256(manifest))

    def test_fixture_predictions_are_deterministic_and_input_checked(self):
        demo.initialize()
        features = [60] + [0] * 41
        self.assertEqual(demo.predict_fixture(demo.CHAMPION, features), "demo_class_0")
        self.assertEqual(demo.predict_fixture(demo.CHALLENGER, features), "demo_class_1")
        for bad in ([], [0] * 41, [True] * 42, [float("nan")] * 42,
                    [float("inf")] * 42, [10 ** 1000] * 42, ["0"] * 42):
            with self.subTest(type=type(bad[0]).__name__ if bad else "empty"), self.assertRaises(GateError):
                demo.predict_fixture(demo.CHAMPION, bad)
        with self.assertRaises(GateError):
            demo.predict_fixture("../alphabet_model.h5", features)

    def test_cli_has_no_activation_training_or_root_override(self):
        for args in (["promote"], ["train"], ["set-active"], ["init", "--root", "backend/data/models"]):
            with self.subTest(args=args), patch.object(sys, "argv", ["demo_sandbox", *args]), \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                demo.main()
        self.assertFalse(self.root.exists())
        self.assert_protected()


if __name__ == "__main__":
    unittest.main()
