"""Deterministic JSON fixtures and a read-checked, local-only initial pointer.

No fit, activation, promotion, deserialization of ML models, registry or network
operations. Production and the unreviewed real candidate are never inputs.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import stat

from tracking.evaluation_gate import GateError, canonical_sha256, digest, read_json, require
from tracking.import_baselines import sha256


WORKSPACE = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("config.json")
SCOPE = "synthetic-demo-only"
CHAMPION = "demo-champion-v1"
CHALLENGER = "demo-challenger-v1"
VERSIONS = (CHAMPION, CHALLENGER)
# Optional, bounded Step 2 records. Their semantic integrity is checked by
# demo_sandbox.drift; the foundation continues to verify its immutable core.
EXTENSION_LIMITS = {
    "drift/reference.json": 8192, "drift/events.json": 131072,
    "drift/.reference.tmp": 8192, "drift/.events.tmp": 131072,
    "drift/.write.lock": 512,
    "retraining/journal.json": 131072, "retraining/.journal.tmp": 131072,
    "retraining/.write.lock": 512,
    "evaluation/journal.json": 65536, "evaluation/.journal.tmp": 65536,
    "evaluation/.write.lock": 512,
    "evaluation/champion/model.pkl": 8192, "evaluation/champion/training_receipt.json": 8192,
    "evaluation/champion_pointer.json": 8192,
    "activation/active.json": 8192, "activation/.active.tmp": 8192,
    "activation/journal.json": 131072, "activation/.journal.tmp": 131072,
    "activation/.write.lock": 512,
}
for _slot in range(1, 11):
    for _name, _limit in (("request.json", 65536), ("output.json", 16384), ("model.json", 8192),
                          ("model.pkl", 8192), ("training_receipt.json", 8192)):
        EXTENSION_LIMITS[f"retraining/job-{_slot:03d}/{_name}"] = _limit
POLICY = {
    "schema_version": 1, "scope": SCOPE, "runtime_directory": ".signvision-demo",
    "seed": 42, "feature_count": 42, "rows_per_partition": 24,
    "class_labels": ["demo_class_0", "demo_class_1"],
    "tracking": "disabled", "registry": "disabled",
}


def _encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _config():
    config = read_json(CONFIG_PATH)
    # Exact, type-sensitive policy: editing configuration cannot enable a registry
    # or redirect storage. Changes to the fixture contract require code review.
    require(canonical_sha256(config) == canonical_sha256(POLICY),
            "Unsupported sandbox configuration; external paths/tracking are forbidden")
    return config


def _plain(path, *, directory):
    info = path.lstat()
    redirected = stat.S_ISLNK(info.st_mode) or (
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )
    require(not redirected, "Sandbox symlinks, junctions and reparse points are forbidden")
    require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
            "Unexpected sandbox filesystem object")
    if not directory:
        require(info.st_nlink == 1, "Sandbox hard-linked artifacts are forbidden")


def _root():
    workspace = WORKSPACE.absolute()
    for path in reversed((workspace, *workspace.parents)):
        _plain(path, directory=True)
    root = workspace / POLICY["runtime_directory"]
    if os.path.lexists(root):
        _plain(root, directory=True)
    require(root.resolve() == workspace / POLICY["runtime_directory"], "Sandbox root redirected")
    return root


def _path(relative):
    # No path from a model manifest, pointer, environment variable or CLI argument
    # is used to open an arbitrary artifact.
    require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative,
            "Invalid sandbox path")
    parts = relative.split("/")
    require(all(part not in ("", ".", "..") for part in parts), "Invalid sandbox path")
    root = _root()
    current = root
    for index, part in enumerate(parts):
        current = current / part
        if os.path.lexists(current):
            _plain(current, directory=index < len(parts) - 1)
    require(current.resolve().is_relative_to(root), "Sandbox path escapes storage")
    return current


def _dataset(config):
    rng = random.Random(config["seed"])
    rows, classes = [], config["class_labels"]
    per_class = config["rows_per_partition"] // 2
    for index in range(config["rows_per_partition"] * 3):
        label = index % 2
        values = [rng.randrange(-1000, 1001) for _ in range(config["feature_count"])]
        values[0] = rng.randrange(50, 201) * (1 if label else -1)
        values[1] = index  # Deliberately synthetic unique vector identifier.
        rows.append({"id": f"synthetic-{index:03d}", "features": values, "label": classes[label]})
    by_class = [[i for i in range(len(rows)) if i % 2 == label] for label in range(2)]
    for indices in by_class:
        rng.shuffle(indices)
    partitions = {}
    for number, name in enumerate(("train", "validation", "test")):
        ids = [i for indices in by_class for i in indices[number * per_class:(number + 1) * per_class]]
        rng.shuffle(ids)
        partitions[name] = ids
    return {"schema_version": 1, "scope": SCOPE, "kind": "generated_integer_fixture",
            "production_evidence": False, "seed": config["seed"],
            "feature_count": config["feature_count"], "class_labels": classes,
            "rows": rows, "partitions": partitions,
            "notice": "Synthetic workflow data, not PSL landmarks or an independent production holdout"}


def _payloads():
    config = _config()
    dataset = _dataset(config)
    files = {"config.json": _encode(config), "data/synthetic-v1.json": _encode(dataset)}
    manifests = {}
    for version, threshold, role in ((CHAMPION, 70, "champion-fixture"),
                                     (CHALLENGER, 0, "challenger-fixture")):
        model = {"schema_version": 1, "scope": SCOPE, "version": version,
                 "artifact_type": "hand_authored_threshold_fixture", "fixture_role": role,
                 "feature_count": config["feature_count"], "class_labels": config["class_labels"],
                 "rule": {"feature_index": 0, "threshold": threshold},
                 "training_performed": False, "production_evidence": False}
        model_bytes = _encode(model)
        manifest = {"schema_version": 1, "scope": SCOPE, "version": version,
                    "artifact_type": model["artifact_type"], "fixture_role": role,
                    "files": {"model.json": hashlib.sha256(model_bytes).hexdigest()},
                    "synthetic_dataset_sha256": canonical_sha256(dataset),
                    "training_performed": False, "production_evidence": False,
                    "registry": "disabled"}
        manifests[version] = manifest
        files[f"versions/{version}/model.json"] = model_bytes
        files[f"versions/{version}/manifest.json"] = _encode(manifest)
    files["active.json"] = _encode({
        "schema_version": 1, "scope": SCOPE, "active_version": CHAMPION,
        "manifest_sha256": canonical_sha256(manifests[CHAMPION]),
        "state": "initial-fixture-only", "promotion_performed": False,
        "production_evidence": False, "registry": "disabled",
    })
    # Published last. Readers refuse partial initialization, including a partial
    # active.json write. This is an initialization receipt, never an approval.
    files["initialized.json"] = _encode({
        "schema_version": 1, "scope": SCOPE, "files": {
            name: hashlib.sha256(value).hexdigest() for name, value in sorted(files.items())},
        "training_performed": False, "promotion_performed": False,
        "production_evidence": False, "registry": "disabled",
    })
    return files


def _inventory(files):
    root = _root()
    require(root.is_dir(), "Sandbox is not initialized")
    directories = {str(parent) for name in files for parent in PurePosixPath(name).parents
                   if str(parent) != "."}
    directories.update(str(parent) for name in EXTENSION_LIMITS
                       for parent in PurePosixPath(name).parents if str(parent) != ".")
    found = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        _plain(directory, directory=True)
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                name = path.relative_to(root).as_posix()
                if name in directories:
                    _plain(path, directory=True)
                    pending.append(path)
                else:
                    require(name in files or name in EXTENSION_LIMITS, "Unexpected sandbox artifact")
                    _plain(path, directory=False)
                    if name in EXTENSION_LIMITS:
                        require(path.stat().st_size <= EXTENSION_LIMITS[name], "Sandbox drift file exceeds size limit")
                    else:
                        found.add(name)
    require(found == set(files), "Sandbox initialization is incomplete")


def _verify():
    files = _payloads()
    _inventory(files)
    for name, expected in files.items():
        require(sha256(_path(name)) == hashlib.sha256(expected).hexdigest(),
                "Sandbox artifact identity mismatch")
    pointer = read_json(_path("active.json"))
    require(pointer["active_version"] == CHAMPION and pointer["scope"] == SCOPE,
            "Only the initial demo champion pointer is supported")
    digest(pointer["manifest_sha256"], "demo manifest identity")
    manifest = read_json(_path(f"versions/{CHAMPION}/manifest.json"))
    require(canonical_sha256(manifest) == pointer["manifest_sha256"], "Demo pointer identity mismatch")
    return pointer


def initialize():
    """Create one new synthetic sandbox. Never overwrite or repair an existing one."""
    files = _payloads()  # Validate source configuration before creating anything.
    root = _root()
    root.mkdir(exist_ok=False)
    for name, payload in files.items():
        path = _path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Check ancestors again after mkdir. Exclusive creation prevents ordinary
        # stale-file/symlink overwrite; there is no replacement/reset operation.
        path = _path(name)
        with path.open("xb") as stream:
            stream.write(payload)
    return status()


def status():
    pointer = _verify()
    return {"scope": SCOPE, "integrity_verified": True, "integrity_scope": "immutable_foundation",
            "active_version": pointer["active_version"],
            "active_manifest_sha256": pointer["manifest_sha256"],
            "available_fixtures": list(VERSIONS), "seed": POLICY["seed"],
            "rows_per_partition": POLICY["rows_per_partition"],
            "training_performed": False, "metrics_computed": False,
            "promotion_performed": False, "production_evidence": False,
            "tracking": "disabled", "registry": "disabled"}


def read_dataset():
    _verify()
    return read_json(_path("data/synthetic-v1.json"))


def read_active():
    pointer = _verify()
    return read_json(_path(f"versions/{pointer['active_version']}/model.json"))


def _predict(model, features):
    """Pure rule calculation for an already integrity-checked fixture context."""
    require(isinstance(features, (list, tuple)) and len(features) == POLICY["feature_count"]
            and all(type(value) in (int, float) and abs(value) <= 1_000_000
                    and math.isfinite(value) for value in features),
            "Expected 42 finite, bounded synthetic numeric features")
    label = int(features[model["rule"]["feature_index"]] > model["rule"]["threshold"])
    return model["class_labels"][label]


def predict_fixture(version, features):
    """Evaluate a hand-authored toy rule, not a trained recognition model."""
    _verify()
    require(version in VERSIONS, "Unknown demo fixture")
    return _predict(read_json(_path(f"versions/{version}/model.json")), features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "verify", "status"))
    args = parser.parse_args()
    try:
        print(json.dumps(initialize() if args.command == "init" else status(), indent=2))
    except (GateError, OSError) as error:
        parser.exit(1, f"Sandbox blocked: {error}\n")


if __name__ == "__main__":
    main()
