from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "local_transformation_monoid_oracle"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "local_transformation_monoid_oracle.py"
)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = oracle
SPEC.loader.exec_module(oracle)


def test_one_local_pinch_plus_adjacent_swaps_generates_full_small_transformation_monoid():
    for n in (2, 3, 4):
        row = oracle.measure(n)
        expected = n**n
        assert row.full_transformation_monoid_size == expected
        assert row.path_swaps_plus_one_pinch["size"] == expected
        assert row.parallel_path_one_pinch["size"] == expected


def test_more_local_irreversible_ports_reduce_exact_word_depth_small_cases():
    r3 = oracle.measure(3)
    assert r3.path_swaps_plus_one_pinch["max_depth"] == 6
    assert r3.path_swaps_plus_all_local_pinches["max_depth"] == 3
    assert r3.parallel_path_one_pinch["max_depth"] == 6
    assert r3.parallel_path_all_pinches["max_depth"] == 3

    r4 = oracle.measure(4)
    assert r4.path_swaps_plus_one_pinch["max_depth"] == 11
    assert r4.path_swaps_plus_all_local_pinches["max_depth"] == 6
    assert r4.parallel_path_one_pinch["max_depth"] == 9
    assert r4.parallel_path_all_pinches["max_depth"] == 4


def test_rank_one_step_cannot_be_built_from_swaps_alone():
    n = 4
    swaps = [oracle.swap(n, i, i + 1) for i in range(n - 1)]
    distances = oracle.bfs_word_lengths(n, swaps)
    assert len(distances) == 24  # exactly S4, not T4
    pinch = oracle.merge(n, 0, 1)
    assert pinch not in distances
