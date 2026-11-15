import unittest
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HAS_SPLIT_DEPENDENCIES = all(importlib.util.find_spec(name) is not None
                             for name in ("numpy", "sklearn"))
if HAS_SPLIT_DEPENDENCIES:
    import numpy as np
    from tracking.data_splits import split_three_way

@unittest.skipUnless(HAS_SPLIT_DEPENDENCIES, "NumPy/scikit-learn are not installed")
class DataSplitTests(unittest.TestCase):
    def test_deterministic_and_duplicate_disjoint(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(300,42))
        x[50] = x[0]
        x[51] = x[0]
        y = np.array(['a'] * 150 + ['b'] * 150)
        first = split_three_way(x,y)
        second = split_three_way(x,y)
        for a,b in zip(first,second):
            np.testing.assert_array_equal(a,b)
        ids = [set(map(tuple,x[i])) for i in first]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    def test_bad_input_fails(self):
        with self.assertRaises(ValueError):
            split_three_way(np.full((30,42),np.nan), ['a']*15+['b']*15)
