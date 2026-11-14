"""Archive existing model bundles as MLflow runs without importing ML code.

Run from backend/: python -m tracking.import_baselines
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


BACKEND = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("alphabet", "word")
DISCLAIMER = (
    "Configuration extracted from the current training script, not verified "
    "historical training parameters of this artifact. Historical metrics and "
    "the original training commit/dataset/splits are unknown."
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def script_path(backend, name):
    return backend / "PSL" / f"{name}_recognition" / f"train_{name}_model.py"


def current_script_config(path):
    """Read syntax only: never import/execute training or unpickle model files."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))

    def value(node):
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            return {"expression": ast.unparse(node)}

    def describe(call):
        return {
            "args": [value(arg) for arg in call.args],
            "kwargs": {kw.arg or "**": value(kw.value) for kw in call.keywords},
        }

    config = {"layers": []}
    calls = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for call in calls:
        if isinstance(call.func, ast.Name):
            name = call.func.id
            if name in ("Dense", "Dropout"):
                config["layers"].append({"type": name, **describe(call)})
            elif name == "train_test_split":
                config["split"] = describe(call)["kwargs"]
            elif name == "StandardScaler":
                config["scaler"] = {"type": name, **describe(call)}
        elif (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "model"
            and call.func.attr in ("compile", "fit")
        ):
            config[call.func.attr] = describe(call)["kwargs"]
    return config


def git_metadata(root):
    """The checkout at import time is not claimed to be the training revision."""
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()

    try:
        return {"import_commit": git("rev-parse", "HEAD"),
                "import_checkout_dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"import_commit": None, "import_checkout_dirty": None}


def tracking_uri(backend, environ):
    return environ.get("MLFLOW_TRACKING_URI", "").strip() or (backend / "mlruns").as_uri()


def import_baselines(mlflow, backend=BACKEND, environ=None):
    backend = Path(backend).resolve()
    environ = os.environ if environ is None else environ
    # Validate BOTH bundles before creating any runs or contacting a server.
    sources = {}
    for name in MODEL_NAMES:
        sources[name] = [
            backend / "data" / "models" / f"{name}_{suffix}"
            for suffix in ("model.h5", "scaler.pkl", "label_encoder.pkl")
        ] + [script_path(backend, name)]
    for paths in sources.values():
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Required baseline artifact missing: {path}")

    # Snapshot first so checksums describe the bytes actually uploaded.
    with tempfile.TemporaryDirectory(prefix="linguasign-baselines-") as temporary:
        snapshot = Path(temporary)
        manifests = {}
        git = git_metadata(backend.parent)
        for name, paths in sources.items():
            bundle = snapshot / name
            bundle.mkdir()
            artifacts = []
            for source in paths:
                copied = bundle / source.name
                shutil.copyfile(source, copied)
                artifacts.append({
                    "source": source.relative_to(backend).as_posix(),
                    "artifact": source.name,
                    "size_bytes": copied.stat().st_size,
                    "sha256": sha256(copied),
                })
            manifests[name] = {
                "schema_version": 1,
                "model_kind": name,
                "imported_at_utc": datetime.now(timezone.utc).isoformat(),
                "provenance": git,
                "historical_metrics_available": False,
                "historical_training_provenance_verified": False,
                "configuration_provenance": "current_script_configuration",
                "note": DISCLAIMER,
                "artifacts": artifacts,
                "current_script_config": current_script_config(bundle / paths[-1].name),
            }
            (bundle / "manifest.json").write_text(
                json.dumps(manifests[name], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        mlflow.set_tracking_uri(tracking_uri(backend, environ))
        mlflow.set_experiment(environ.get("MLFLOW_EXPERIMENT_NAME") or "LinguaSign Baselines")
        run_ids = {}
        for name in MODEL_NAMES:
            manifest = manifests[name]
            tags = {
                "baseline.kind": "existing_artifact_import",
                "model.kind": name,
                "training.performed": "false",
                "historical.metrics_available": "false",
                "configuration.provenance": "current_script_configuration",
                "provenance.note": DISCLAIMER,
                "import.git.commit": git["import_commit"] or "unavailable",
                "import.git.dirty": str(git["import_checkout_dirty"]).lower(),
            }
            with mlflow.start_run(run_name=f"{name}-existing-baseline", tags=tags) as run:
                params = {
                    f"current_script_config.{section}.{key}": json.dumps(value)
                    for section in ("fit", "compile", "split")
                    for key, value in manifest["current_script_config"].get(section, {}).items()
                }
                params["current_script_config.layers"] = json.dumps(
                    manifest["current_script_config"]["layers"]
                )
                mlflow.log_params(params)
                for artifact in manifest["artifacts"]:
                    mlflow.set_tag(f"sha256.{artifact['artifact']}", artifact["sha256"])
                mlflow.log_artifacts(str(snapshot / name), artifact_path="baseline")
                run_ids[name] = run.info.run_id
        return run_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        import mlflow
    except ImportError:
        parser.exit(1, "MLflow is not installed. Use a separate environment with "
                    "backend/requirements-tracking.txt.\n")
    runs = import_baselines(mlflow)
    for name, run_id in runs.items():
        print(f"{name}: run_id={run_id}")


if __name__ == "__main__":
    main()
