"""Pre-training validation for SignVision SQLite datasets."""

import math
from collections import Counter
from numbers import Real
import sqlite3
from pathlib import Path

TABLES = ("alphabetDataset", "wordDataset")
FEATURE_COLUMNS = [
    name
    for i in range(1, 22)
    for name in (f"x{i}", f"y{i}")
]


def validate_training_database(database_path, table):
    if table not in TABLES:
        raise ValueError(f"Unsupported dataset table: {table}")

    path = Path(database_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training database not found: {path}")

    connection = sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro", uri=True
    )

    try:
        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise ValueError(f"SQLite integrity check failed: {integrity}")

        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

        if table not in tables:
            raise ValueError(f"Required table is missing: {table}")

        columns = {
            row[1]
            for row in connection.execute(f'PRAGMA table_info("{table}")')
        }

        required = {"id", "label", *FEATURE_COLUMNS}
        missing = required - columns

        if missing:
            raise ValueError(
                f"{table}: missing required columns: {sorted(missing)}"
            )

        selected = ", ".join(
            f'"{column}"' for column in FEATURE_COLUMNS
        )

        cursor = connection.execute(
            f'SELECT {selected}, "label" FROM "{table}"'
        )

        samples = 0
        classes = Counter()

        for row_number, row in enumerate(cursor, start=1):
            features = row[:-1]
            label = row[-1]

            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"{table}: invalid label at row {row_number}; expected non-empty text"
                )

            for index, value in enumerate(features):
                if value is None:
                    raise ValueError(
                        f"{table}: missing feature at row {row_number}, "
                        f"column {FEATURE_COLUMNS[index]}"
                    )

                if not isinstance(value, Real):
                    raise ValueError(
                        f"{table}: non-numeric feature at row {row_number}, "
                        f"column {FEATURE_COLUMNS[index]}"
                    )

                if not math.isfinite(value):
                    raise ValueError(
                        f"{table}: non-finite feature at row {row_number}, "
                        f"column {FEATURE_COLUMNS[index]}"
                    )

            samples += 1
            classes[label] += 1

        if samples < 15:
            raise ValueError(
                f"{table}: insufficient samples ({samples}); minimum is 15"
            )

        if len(classes) < 2:
            raise ValueError(
                f"{table}: at least two classes are required"
            )

        if min(classes.values()) < 3:
            raise ValueError(f"{table}: at least three samples per class are required")

        result = {
            "table": table,
            "samples": samples,
            "classes": len(classes),
            "features": len(FEATURE_COLUMNS),
            "sqlite_integrity": integrity,
            "schema_valid": True,
            "quality_valid": True,
        }

        print(f"VALIDATION PASSED: {result}")
        return result

    except sqlite3.DatabaseError as error:
        raise ValueError(f"Invalid SQLite training database: {path}: {error}") from error
    finally:
        connection.close()
