"""Read-only validation and entry-point contracts; never trains models."""

import ast
from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking.data_validation import FEATURE_COLUMNS, validate_training_database
from tracking.import_baselines import BACKEND, script_path


class ValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def database(self, *, missing=None, samples=18, classes=2):
        path = self.root / "fixture.db"
        connection = sqlite3.connect(path)
        columns = ["id", *FEATURE_COLUMNS, "label"]
        columns = [name for name in columns if name != missing]
        # No type affinity: invalid numeric strings stay strings for validation.
        for table in ("alphabetDataset", "wordDataset"):
            connection.execute(f"CREATE TABLE {table} ({','.join(columns)})")
            for index in range(samples):
                row = {"id": index, "label": f"class-{index % classes}",
                       **{name: float(index) for name in FEATURE_COLUMNS}}
                connection.execute(f"INSERT INTO {table} VALUES ({','.join('?' for _ in columns)})",
                                   [row[name] for name in columns])
        connection.commit()
        connection.close()
        return path

    def test_both_valid_tables_and_file_unchanged(self):
        path = self.database()
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        for table in ("alphabetDataset", "wordDataset"):
            with patch("builtins.print"):
                result = validate_training_database(path, table)
            self.assertEqual(result["samples"], 18)
            self.assertEqual(result["classes"], 2)
            self.assertEqual(result["features"], 42)
            self.assertTrue(result["quality_valid"])
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_corrupt_and_missing_database(self):
        path = self.root / "corrupt.db"
        path.write_bytes(b"this is not a SQLite database")
        with self.assertRaisesRegex(ValueError, "Invalid SQLite"):
            validate_training_database(path, "alphabetDataset")
        with self.assertRaises(FileNotFoundError):
            validate_training_database(self.root / "absent.db", "alphabetDataset")
        self.assertFalse((self.root / "absent.db").exists())

    def test_missing_columns(self):
        for name in ("id", "label", "x1", "y21"):
            with self.subTest(column=name), tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory)
                path = self.database(missing=name)
                with self.assertRaisesRegex(ValueError, "missing required columns"):
                    validate_training_database(path, "alphabetDataset")

    def test_missing_and_unsupported_tables(self):
        path = self.database()
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("DROP TABLE wordDataset")
        with self.assertRaisesRegex(ValueError, "Required table is missing"):
            validate_training_database(path, "wordDataset")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            validate_training_database(path, "invalid")

    def test_invalid_labels(self):
        path = self.database()
        for label in (None, "", "  ", 12, b"binary"):
            with self.subTest(label=label):
                with closing(sqlite3.connect(path)) as connection, connection:
                    connection.execute("UPDATE alphabetDataset SET label=? WHERE id=0", (label,))
                with self.assertRaisesRegex(ValueError, "invalid label"):
                    validate_training_database(path, "alphabetDataset")

    def test_invalid_features(self):
        path = self.database()
        for feature in (None, "text", "1.23", b"binary", float("inf"), -float("inf"), float("nan")):
            with self.subTest(feature=feature):
                with closing(sqlite3.connect(path)) as connection, connection:
                    connection.execute("UPDATE alphabetDataset SET x1=? WHERE id=0", (feature,))
                with self.assertRaisesRegex(ValueError, "feature"):
                    validate_training_database(path, "alphabetDataset")

    def test_too_few_samples_or_classes(self):
        for samples, classes in ((0, 2), (14, 2), (18, 1), (18, 7)):
            with self.subTest(samples=samples, classes=classes), tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory)
                path = self.database(samples=samples, classes=classes)
                with self.assertRaises(ValueError):
                    validate_training_database(path, "alphabetDataset")

    def test_neural_wrappers_validate_before_creating_runs(self):
        for kind, function in (("alphabet", "train_alphabets"), ("word", "train_words")):
            tree = ast.parse(script_path(BACKEND, kind).read_text(encoding="utf-8"))
            wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
            validation = wrapper.body[0].value
            self.assertEqual(ast.unparse(validation.func), "validate_training_database")
            self.assertEqual(ast.literal_eval(validation.args[1]), f"{kind}Dataset")
            self.assertIsInstance(wrapper.body[1], ast.With)

    def test_invalid_rf_dataset_stops_before_fitting_or_logging(self):
        from tracking import train_random_forest as baseline
        path = self.root / "bad.db"
        path.write_bytes(b"corrupted")
        with patch.object(baseline, "build_classifier") as classifier, patch.object(baseline, "partition_indices") as split:
            with self.assertRaises(ValueError):
                baseline.load_dataset(path, "alphabet")
            classifier.assert_not_called()
            split.assert_not_called()


if __name__ == "__main__":
    unittest.main()
