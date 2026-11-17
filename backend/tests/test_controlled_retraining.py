"""Synthetic plans/bundles and mocked trainers only; never fit or touch real runs."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import builtins
import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking import retraining as retrain, versioning, training
from tracking import evaluation_gate as gate
from tracking.evaluate_models import read_bundle, SUFFIXES


class ControlledRetrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.backend = Path(temporary.name)
        self.outputs = self.backend / "data/training_runs"
        self.outputs.mkdir(parents=True)
        patch.object(versioning, "BACKEND", self.backend).start()
        self.addCleanup(patch.stopall)
        self.rows = [[*[float(i)] * 42, "a" if i % 2 == 0 else "b"] for i in range(6)]
        self.parts = {"train": [0, 1], "validation": [2, 3], "test": [4, 5]}
        self.datasets = {task: {"dvc_md5": gate.DATASET_MD5, "sha256": "a" * 64,
                         "ordered_rows_sha256": gate.canonical_sha256(self.rows),
                         "table": table, "feature_columns": gate.FEATURES}
                         for task, table in gate.TABLES.items()}
        self.plan = {"schema_version": 1, "seed": 42, "sources": {},
                     "tasks": {task: {"dataset": dataset, "partitions": self.parts}
                               for task, dataset in self.datasets.items()}}

    def make_bundle(self, name, *, reviewed=True):
        root = self.outputs / name
        root.mkdir()
        directory = root / "bundle"
        directory.mkdir()
        records = {}
        for task in gate.TABLES:
            artifacts = {}
            for key, suffix in SUFFIXES.items():
                path = directory / f"{task}_{suffix}"
                path.write_bytes(f"synthetic-{name}-{task}-{key}".encode())
                artifacts[key] = versioning.sha256(path)
            manifest = {"schema_version": 1, "task": task, "dataset": self.datasets[task],
                        "partitions": self.parts, "artifacts": artifacts,
                        "provenance": {"status": "reviewed_training_time_evidence" if reviewed
                                       else "captured_training_time_unreviewed",
                                       "reviewed_by": "test-only", "evidence_reference": "synthetic-only",
                                       "training_plan_sha256": gate.canonical_sha256(self.plan),
                                       "mlflow_run_id": "fixture-run"}}
            retrain.write_new(directory / f"{task}_provenance.json", manifest)
            retrain.write_new(directory / f"{task}_run_metadata.json", {
                "training_status": "FINISHED", "tracking_errors": [], "partitions": self.parts,
                "controlled_training": {"plan_sha256": gate.canonical_sha256(self.plan)},
                "mlflow_run_id": "fixture-run"})
            records[task] = manifest
        retrain.write_new(directory / "plan.json", self.plan)
        version = {"schema_version": 1, "version": name,
                   "plan_sha256": gate.canonical_sha256(self.plan),
                   "files": {p.name: versioning.sha256(p) for p in directory.iterdir()}}
        retrain.write_new(directory / "version.json", version)
        retrain.write_new(root / "complete.json", {"bundle_sha256": gate.canonical_sha256(version),
            "mlflow_audit_run_id": "fixture-audit", "deployment_performed": False})
        return directory, gate.canonical_sha256(version), records

    def evaluations(self, current, target):
        directory = self.outputs / "evaluation"
        directory.mkdir()
        approvals = {}
        for task in gate.TABLES:
            records = {}
            for role, bundle in (("production", current), ("candidate", target)):
                manifest, manifest_hash = read_bundle(bundle, task)
                record = {"schema_version": 1, "role": role, "task": task, "protocol": gate.PROTOCOL,
                          "evaluated_at": "2026-11-17T00:00:00+00:00",
                          "dataset": {**self.datasets[task], "rows": 6},
                          "model": {"task": task, "family": "keras_dense", "artifacts": manifest["artifacts"]},
                          "classes": ["a", "b"],
                          "held_out": {"provenance_status": "reviewed_training_time_evidence",
                                       "duplicate_disjoint": True, "manifest_sha256": manifest_hash,
                                       "test_indices_sha256": gate.canonical_sha256(self.parts["test"]),
                                       "test_rows_sha256": "b" * 64, "sample_count": 2,
                                       "partition_counts": {key: 2 for key in self.parts}},
                          "metrics": {"test_accuracy": 0.8, "test_loss": 0.3}}
                retrain.write_new(directory / f"{task}_{role}.json", record)
                records[role] = record
            approvals[task] = gate.compare_evaluations(records["production"], records["candidate"])
        retrain.write_new(directory / "approval.json", {
            "approved": True, "tasks": approvals, "deployment_performed": False})
        return directory

    def test_no_training_without_explicit_execute(self):
        with patch.object(sys, "argv", ["retraining", "train", "--plan", "unused", "--plan-sha256", "0" * 64]), \
             patch.object(retrain, "execute") as execute:
            with self.assertRaisesRegex(gate.GateError, "explicit"):
                retrain.main()
        execute.assert_not_called()

    def test_plan_or_dataset_change_blocks_before_mlflow_or_training(self):
        with self.assertRaisesRegex(gate.GateError, "hash"):
            retrain.execute(self.plan, "0" * 64, "unused", self.outputs, "unused")
        with patch.object(retrain, "prepare_plan", return_value={}), \
             self.assertRaisesRegex(gate.GateError, "changed"):
            retrain.execute(self.plan, gate.canonical_sha256(self.plan), "unused", self.outputs, "unused")

    def test_candidate_output_cannot_target_production(self):
        with patch.object(retrain, "prepare_plan", return_value=self.plan), \
             self.assertRaisesRegex(gate.GateError, "Candidate root"):
            retrain.execute(self.plan, gate.canonical_sha256(self.plan), "unused", self.backend / "data/models", "unused")

    def test_actual_rows_and_partitions_checked_before_fitting(self):
        output = self.outputs / "capture"
        output.mkdir()
        run = SimpleNamespace(name="alphabet", output_dir=output)
        session = retrain.Session(self.plan, self.outputs)
        features, labels = [r[:-1] for r in self.rows], [(r[-1],) for r in self.rows]
        session.partitions(run, features, labels, self.parts)
        self.assertTrue((output / "partition_plan.json").exists())
        bad = deepcopy(self.parts)
        bad["train"], bad["test"] = bad["test"], bad["train"]
        with self.assertRaisesRegex(gate.GateError, "holdout"):
            session.partitions(run, features, labels, bad)
        with self.assertRaisesRegex(gate.GateError, "rows"):
            session.partitions(run, list(reversed(features)), labels, self.parts)

    def test_finish_captures_unreviewed_not_gate_ready_evidence(self):
        directory, _, _ = self.make_bundle("finish", reviewed=False)
        (directory / "alphabet_provenance.json").unlink()  # temporary fixture only
        retrain.write_new(directory / "partition_plan.json", self.plan["tasks"]["alphabet"])
        run = SimpleNamespace(name="alphabet", output_dir=directory,
                              metadata={"mlflow_run_id": "fixture-run", "git": {"commit": "b" * 40},
                                        "test_metrics": {"test_accuracy": 0.8, "test_loss": 0.3}})
        retrain.Session(self.plan, self.outputs).finished(run)
        with self.assertRaisesRegex(gate.GateError, "Unverified"):
            read_bundle(directory, "alphabet")

    def test_disabled_tracking_is_rejected_in_controlled_context(self):
        script = self.backend / "fixture.py"
        script.write_text("# not executed")
        token = training.CONTROLLED_SESSION.set(MagicMock())
        self.addCleanup(training.CONTROLLED_SESSION.reset, token)
        with self.assertRaisesRegex(RuntimeError, "working MLflow"):
            with training.TrainingRun("alphabet", script, backend=self.backend, environ={}):
                self.fail("Must not fit")

    def test_failed_capture_closes_run_as_failed(self):
        script = self.backend / "fixture.py"
        script.write_text("# fixture")
        client = MagicMock()
        client.active_run.return_value = None
        client.start_run.return_value.info.run_id = "fixture-run"
        session = MagicMock()
        session.finished.side_effect = ValueError("invalid capture")
        token = training.CONTROLLED_SESSION.set(session)
        self.addCleanup(training.CONTROLLED_SESSION.reset, token)
        with patch.object(training.importlib, "import_module", return_value=client), \
             self.assertRaisesRegex(ValueError, "invalid capture"):
            with training.TrainingRun("alphabet", script, backend=self.backend,
                                      environ={"SIGNVISION_TRACKING_ENABLED": "true"}) as run:
                pass
        client.end_run.assert_called_once_with(status="FAILED")
        self.assertEqual(run.metadata["training_status"], "FAILED")

    def test_candidate_verification_is_read_only_and_does_not_approve(self):
        source, identity, _ = self.make_bundle("verify-only", reviewed=False)
        before = {str(p): p.read_bytes() for p in source.parent.rglob("*") if p.is_file()}
        output = io.StringIO()
        with patch.object(versioning, "read_rows", return_value=(self.rows, "a" * 64)), \
             patch.object(sys, "argv", ["versioning", "verify", "--source", str(source),
                                       "--source-sha256", identity, "--dataset", "unused"]), \
             redirect_stdout(output):
            versioning.main()
        report = json.loads(output.getvalue())
        self.assertTrue(report["capture_verified"])
        self.assertFalse(report["promotion_approved"])
        self.assertFalse(report["deployment_performed"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in source.parent.rglob("*") if p.is_file()})
        self.assertEqual(gate.read_json(source / "alphabet_provenance.json")["provenance"]["status"],
                         "captured_training_time_unreviewed")
        with self.assertRaises(gate.GateError):
            versioning.verify_candidate(source, "0" * 64, "unused")

    def test_gate_rejects_partition_counts_not_bound_to_manifest(self):
        current, _, _ = self.make_bundle("partition-current")
        target, _, _ = self.make_bundle("partition-target")
        evaluation = self.evaluations(current, target)
        record_path = evaluation / "alphabet_candidate.json"
        record = gate.read_json(record_path)
        record["held_out"]["partition_counts"].update(train=1, validation=3)
        record_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(gate.GateError, "partition counts"):
            versioning.verify_gate(evaluation, current, target)

    def test_review_creates_separate_bundle_leaving_capture_unmodified(self):
        source, identity, records = self.make_bundle("unreviewed", reviewed=False)
        before = {p.name: p.read_bytes() for p in source.iterdir()}
        with patch.object(versioning, "read_rows", return_value=(self.rows, "a" * 64)):
            new_hash = versioning.review_candidate(source, identity, self.outputs / "reviewed",
                                                    "fixture-reviewer", "synthetic-only", "unused")
        versioning.verify_version(self.outputs / "reviewed", new_hash)
        read_bundle(self.outputs / "reviewed", "alphabet")
        self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})

    def test_version_tampering_missing_files_and_wrong_pin_are_blocked(self):
        directory, identity, _ = self.make_bundle("tamper")
        with self.assertRaisesRegex(gate.GateError, "identity"):
            versioning.verify_version(directory, "0" * 64)
        path = directory / "word_scaler.pkl"
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(gate.GateError, "changed"):
            versioning.verify_version(directory, identity)
        path.unlink()
        with self.assertRaisesRegex(gate.GateError, "Missing"):
            versioning.verify_version(directory, identity)

    def test_manual_promotion_and_rollback_are_plans_only(self):
        current, current_hash, _ = self.make_bundle("current")
        target, target_hash, _ = self.make_bundle("target")
        evaluation = self.evaluations(current, target)
        before = {str(p): p.read_bytes() for d in (current, target) for p in d.iterdir()}
        promotion = self.outputs / "promotion.json"
        pin = versioning.plan_change(action="promote", current=current, current_sha256=current_hash,
            target=target, target_sha256=target_hash, output=promotion, reviewer="synthetic-reviewer",
            evidence="synthetic-only", evaluation_dir=evaluation)
        versioning.plan_change(action="rollback", current=target, current_sha256=target_hash,
            target=current, target_sha256=current_hash, output=self.outputs / "rollback.json",
            reviewer="synthetic-reviewer", evidence="synthetic-only", evaluation_dir=evaluation,
            previous_plan=promotion, previous_plan_sha256=pin)
        self.assertFalse(gate.read_json(promotion)["execution_performed"])
        self.assertEqual(before, {str(p): p.read_bytes() for d in (current, target) for p in d.iterdir()})
        with self.assertRaises(FileExistsError):
            versioning.plan_change(action="promote", current=current, current_sha256=current_hash,
                target=target, target_sha256=target_hash, output=promotion, reviewer="r", evidence="e",
                evaluation_dir=evaluation)

    def test_missing_historical_provenance_blocks_archive(self):
        source = self.backend / "production"
        source.mkdir()
        with self.assertRaisesRegex(gate.GateError, "Missing"):
            versioning.archive_current(source, self.outputs / "archive", "r", "e")
        self.assertFalse((self.outputs / "archive").exists())

    def test_archive_hashes_both_tasks_without_changing_source(self):
        current, _, _ = self.make_bundle("original")
        identity = versioning.archive_current(current, self.outputs / "archive", "r", "e")
        versioning.verify_version(self.outputs / "archive", identity)
        self.assertEqual((current / "alphabet_model.h5").read_bytes(),
                         (self.outputs / "archive/alphabet_model.h5").read_bytes())

    def test_gate_regression_stale_approval_and_wrong_bundle_fail(self):
        current, _, _ = self.make_bundle("current")
        target, _, _ = self.make_bundle("target")
        wrong, _, _ = self.make_bundle("wrong")
        evaluation = self.evaluations(current, target)
        with self.assertRaisesRegex(gate.GateError, "artifact identity"):
            versioning.verify_gate(evaluation, current, wrong)
        candidate_file = evaluation / "alphabet_candidate.json"
        candidate = gate.read_json(candidate_file)
        candidate["metrics"]["test_accuracy"] = 0.7
        candidate_file.write_text(json.dumps(candidate))
        with self.assertRaisesRegex(gate.GateError, "regression"):
            versioning.verify_gate(evaluation, current, target)
        candidate["metrics"]["test_accuracy"] = 0.9
        candidate_file.write_text(json.dumps(candidate))
        with self.assertRaisesRegex(gate.GateError, "Combined"):
            versioning.verify_gate(evaluation, current, target)

    def test_rollback_requires_pinned_previous_promotion_and_original_gate(self):
        current, current_hash, _ = self.make_bundle("current")
        target, target_hash, _ = self.make_bundle("target")
        with self.assertRaisesRegex(gate.GateError, "pinned previous"):
            versioning.plan_change(action="rollback", current=current, current_sha256=current_hash,
                target=target, target_sha256=target_hash, output=self.outputs / "rollback.json",
                reviewer="r", evidence="e")

    def test_writes_restricted_and_required_review_not_inferred(self):
        with self.assertRaises(gate.GateError):
            versioning.isolated(self.backend / "data/models/new.json")
        with self.assertRaises(gate.GateError):
            versioning.reviewer_fields("", "evidence")
        path = self.outputs / "immutable.json"
        retrain.write_new(path, {"original": True})
        with self.assertRaises(FileExistsError):
            retrain.write_new(path, {"replacement": True})

    def test_invalid_final_metrics_cannot_create_provenance(self):
        directory = self.outputs / "invalid-metrics"
        directory.mkdir()
        retrain.write_new(directory / "partition_plan.json", self.plan["tasks"]["alphabet"])
        for metrics in ({}, {"test_accuracy": True, "test_loss": 0.2},
                        {"test_accuracy": 1.01, "test_loss": 0.2},
                        {"test_accuracy": 0.8, "test_loss": float("nan")},
                        {"test_accuracy": 0.8, "test_loss": -1}):
            with self.subTest(metrics=metrics), self.assertRaisesRegex(gate.GateError, "metrics"):
                retrain.Session(self.plan, self.outputs).finished(SimpleNamespace(
                    name="alphabet", output_dir=directory, metadata={"test_metrics": metrics}))
        self.assertFalse((directory / "alphabet_provenance.json").exists())

    def test_extra_unversioned_files_and_metadata_are_rejected(self):
        directory, identity, _ = self.make_bundle("extra")
        (directory / "unreviewed.txt").write_text("not versioned")
        with self.assertRaisesRegex(gate.GateError, "Unlisted"):
            versioning.verify_version(directory, identity)
        source, _, _ = self.make_bundle("missing-metadata", reviewed=False)
        metadata_file = source / "alphabet_run_metadata.json"
        version = gate.read_json(source / "version.json")
        del version["files"][metadata_file.name]
        metadata_file.unlink()  # temporary fixture only
        (source / "version.json").write_text(json.dumps(version))
        (source.parent / "complete.json").write_text(json.dumps({
            "bundle_sha256": gate.canonical_sha256(version), "mlflow_audit_run_id": "fixture-audit",
            "deployment_performed": False}))
        with self.assertRaisesRegex(gate.GateError, "metadata is not versioned"):
            versioning.review_candidate(source, gate.canonical_sha256(version),
                self.outputs / "should-not-exist", "r", "e", "unused")

    def test_cli_requires_explicit_review_and_change_approval(self):
        commands = [
            ["review", "--source", "unused", "--source-sha256", "a" * 64,
             "--destination", "unused", "--reviewer", "r", "--evidence", "e"],
            ["promote", "--current", "unused", "--current-sha256", "a" * 64,
             "--target", "unused", "--target-sha256", "b" * 64, "--output", "unused",
             "--evaluation-dir", "unused", "--reviewer", "r", "--evidence", "e"],
        ]
        for command in commands:
            with self.subTest(command=command[0]), patch.object(sys, "argv", ["versioning", *command]), \
                 self.assertRaisesRegex(gate.GateError, "explicit"):
                versioning.main()

    def test_review_rejects_missing_completion_and_inconsistent_frozen_plan(self):
        source, identity, _ = self.make_bundle("incomplete", reviewed=False)
        receipt = source.parent / "complete.json"
        original = receipt.read_bytes()
        receipt.unlink()  # temporary fixture only
        with self.assertRaises(gate.GateError):
            versioning.review_candidate(source, identity, self.outputs / "blocked", "r", "e", "unused")
        receipt.write_bytes(original)
        plan = gate.read_json(source / "plan.json")
        plan["seed"] = 123
        (source / "plan.json").write_text(json.dumps(plan))
        version = gate.read_json(source / "version.json")
        version["files"]["plan.json"] = versioning.sha256(source / "plan.json")
        (source / "version.json").write_text(json.dumps(version))
        new_identity = gate.canonical_sha256(version)
        receipt.write_text(json.dumps({"bundle_sha256": new_identity,
            "mlflow_audit_run_id": "fixture-audit", "deployment_performed": False}))
        with self.assertRaisesRegex(gate.GateError, "plan identity"):
            versioning.review_candidate(source, new_identity, self.outputs / "blocked", "r", "e", "unused")
        self.assertFalse((self.outputs / "blocked").exists())

    def test_end_to_end_bundle_with_fake_trainers_and_tracking_only(self):
        """Exercise lifecycle wiring, never import TensorFlow or fit a model."""
        database = self.backend / "fixture.db"
        database.with_suffix(".db.dvc").write_text("synthetic-only")
        client = MagicMock()
        client.active_run.return_value = None
        client.start_run.return_value.info.run_id = "fixture-run"
        client.start_run.return_value.__enter__.return_value.info.run_id = "fixture-audit"
        tf = SimpleNamespace(config=SimpleNamespace(experimental=SimpleNamespace(
            enable_op_determinism=MagicMock())), keras=SimpleNamespace(utils=SimpleNamespace(
            set_random_seed=MagicMock())))

        def trainer(task):
            script = self.backend / f"{task}.py"
            script.write_text("# synthetic source; never executed")
            def run_fake():
                with training.TrainingRun(task, script, backend=self.backend,
                    environ={"SIGNVISION_TRACKING_ENABLED": "true"}) as run:
                    run.record_partitions([r[:-1] for r in self.rows], [r[-1] for r in self.rows],
                        self.parts["train"], self.parts["validation"], self.parts["test"])
                    for suffix in SUFFIXES.values():
                        (run.output_dir / f"{task}_{suffix}").write_bytes(b"synthetic-only")
                    run.record_test(loss=0.3, accuracy=0.8)
            return run_fake

        imports = {"mlflow": client, "tensorflow": tf,
            "PSL.alphabet_recognition.train_alphabet_model": SimpleNamespace(train_alphabets=trainer("alphabet")),
            "PSL.word_recognition.train_word_model": SimpleNamespace(train_words=trainer("word"))}
        original_import = builtins.__import__
        def fake_import(name, *args, **kwargs):
            return imports[name] if name in imports else original_import(name, *args, **kwargs)
        with patch.object(retrain, "BACKEND", self.backend), \
             patch.object(retrain, "prepare_plan", return_value=self.plan), \
             patch.object(retrain.metadata, "distributions", return_value=[]), \
             patch.object(training, "importlib", SimpleNamespace(import_module=lambda name: client)), \
             patch.object(training, "git_metadata", return_value={"import_commit": "b" * 40, "import_checkout_dirty": False}), \
             patch("builtins.__import__", side_effect=fake_import), \
             patch.dict(sys.modules, {k: v for k, v in sys.modules.items() if k != "tensorflow"}, clear=True):
            root = retrain.execute(self.plan, gate.canonical_sha256(self.plan), database,
                                   self.outputs / "candidates", "http://127.0.0.1:5000")
            # Fail the final audit upload, after both fake task runs completed.
            # There must be no completion receipt for that failed attempt.
            client.log_artifacts.side_effect = [None, None, ConnectionError("synthetic outage")]
            with self.assertRaises(ConnectionError):
                retrain.execute(self.plan, gate.canonical_sha256(self.plan), database,
                                self.outputs / "failed-candidates", "http://127.0.0.1:5000")
            self.assertFalse(list((self.outputs / "failed-candidates").rglob("complete.json")))
        receipt = gate.read_json(root / "complete.json")
        versioning.verify_version(root / "bundle", receipt["bundle_sha256"])
        self.assertEqual(gate.read_json(root / "bundle/plan.json"), self.plan)
        self.assertFalse(receipt["deployment_performed"])
        self.assertIsNone(training.CONTROLLED_SESSION.get())
        self.assertEqual(tf.keras.utils.set_random_seed.call_count, 4)
        with self.assertRaisesRegex(gate.GateError, "Unverified"):
            read_bundle(root / "bundle", "alphabet")


if __name__ == "__main__":
    unittest.main()
