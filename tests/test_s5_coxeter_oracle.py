from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "s5_coxeter_oracle", ROOT / "map" / "s5_coxeter_oracle.py"
)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


def test_simple_root_representation_is_exact_and_local():
    n = 5
    oracle.verify_simple_root_representation(n)
    for i in range(n - 1):
        M = oracle.simple_root_matrix(n, i)
        changed_rows = [r for r in range(n - 1) if any(M[r, c] != (1 if r == c else 0) for c in range(n - 1))]
        assert changed_rows == [i]
        support = [c for c in range(n - 1) if M[i, c] != 0]
        assert all(abs(c - i) <= 1 for c in support)


def test_full_s5_exact_resource_floor():
    stats = oracle.compile_stats(5)
    full = stats["subsets"]["full"]
    assert stats["group_size"] == 120
    assert stats["behavioral_state_channels"] == 4
    assert full["sequential_mean"] == 5.0
    assert full["sequential_max"] == 10
    assert abs(full["parallel_mean"] - (403.0 / 120.0)) < 1e-12
    assert full["parallel_max"] == 5
    assert stats["full_parallel_depth_histogram"] == {
        "0": 1,
        "1": 7,
        "2": 16,
        "3": 35,
        "4": 46,
        "5": 15,
    }


def test_exact_long_horizon_full_s5_tracking():
    result = oracle.long_horizon_check(5, length=4096, seed=123)
    assert result["exact"] is True
