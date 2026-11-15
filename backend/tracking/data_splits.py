"""Reproducible, exact-duplicate-disjoint train/validation/test splitting.

The source database does not contain signer IDs, so signer-disjoint splits
cannot be verified. Identical feature vectors are kept in one partition.
"""
import hashlib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

SEED = 42


def split_three_way(features, labels, seed=SEED):
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels).reshape(-1)
    if x.ndim != 2 or x.shape[1] != 42 or not np.isfinite(x).all():
        raise ValueError("Expected finite 42-dimensional landmark features")
    if len(x) != len(y) or len(x) < 15 or any(str(z).strip() == "" for z in y):
        raise ValueError("Missing labels or insufficient samples")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or min(counts) < 3:
        raise ValueError("At least 3 examples per class and 2 classes required")
    groups = np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in x])
    all_classes = set(classes.tolist())
    for trial in range(200):
        a, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed + trial).split(x, y, groups))
        train, val = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed + trial + 10000).split(x[a], y[a], groups[a]))
        train, val = a[train], a[val]
        if all(set(y[z].tolist()) == all_classes for z in (train, val, test)):
            assert not (set(groups[train]) & set(groups[val]))
            assert not (set(groups[train]) & set(groups[test]))
            assert not (set(groups[val]) & set(groups[test]))
            return train, val, test
    raise ValueError("Cannot create class-complete duplicate-disjoint splits; collect more data or signer IDs")
