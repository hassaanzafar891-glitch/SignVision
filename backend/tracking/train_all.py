"""Validate by default; --train explicitly runs both NNs and both RF baselines."""

import argparse
import json
import os
from contextlib import contextmanager

from tracking.import_baselines import BACKEND
from tracking.data_validation import TABLES, validate_training_database


@contextmanager
def tracking_environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def train_neural_models():
    # Lazy imports keep validation-only execution free of TensorFlow.
    from PSL.alphabet_recognition.train_alphabet_model import train_alphabets
    from PSL.word_recognition.train_word_model import train_words

    train_alphabets()
    train_words()


def run_workflow(*, train=False, tracking_uri="http://127.0.0.1:5000"):
    database = BACKEND / "data/db/main_dataset.db"
    validation = [validate_training_database(database, table) for table in TABLES]
    if not train:
        return {"mode": "validation_only", "datasets": validation}

    import mlflow
    from tracking.train_random_forest import train_task

    if mlflow.active_run() is not None or os.environ.get("MLFLOW_RUN_ID"):
        raise RuntimeError("Refusing to resume an existing MLflow run")
    mlflow.set_tracking_uri(tracking_uri)
    # Check server access before importing/fitting neural networks.
    mlflow.tracking.MlflowClient(tracking_uri=tracking_uri).search_experiments(max_results=1)
    with tracking_environment({
        "SIGNVISION_TRACKING_ENABLED": "true", "MLFLOW_TRACKING_URI": tracking_uri,
        "MLFLOW_EXPERIMENT_NAME": "SignVision Training",
    }):
        train_neural_models()
        mlflow.set_experiment("SignVision Classical Baselines")
        for kind in ("alphabet", "word"):
            train_task(kind, database, mlflow)
    return {"mode": "training_requested", "datasets": validation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Explicitly authorize model fitting")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    print(json.dumps(run_workflow(train=args.train, tracking_uri=args.tracking_uri), indent=2))


if __name__ == "__main__":
    main()
