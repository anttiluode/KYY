from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "pinch_pareto_oracle"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "pinch_pareto_oracle.py"
)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = oracle
SPEC.loader.exec_module(oracle)


def test_every_nonempty_write_edge_set_generates_full_small_transformation_monoid():
    for n in (3, 4):
        for k in range(1, n):
            for row in oracle.all_placements(n, k):
                assert row.sequential["size"] == n**n
                assert row.parallel["size"] == n**n


def test_n4_exact_write_site_pareto():
    rows = oracle.pareto_rows(4)
    expected = {
        1: ((1,), 7, 6),
        2: ((0, 2), 6, 5),
        3: ((0, 1, 2), 6, 4),
    }
    for item in rows:
        k = item["write_site_count"]
        seq = item["best_sequential"]
        par = item["best_parallel"]
        edges, seq_max, par_max = expected[k]
        assert tuple(seq["write_edges"]) == edges
        assert tuple(par["write_edges"]) == edges
        assert seq["sequential"]["max_depth"] == seq_max
        assert par["parallel"]["max_depth"] == par_max


def test_n6_key_resource_points():
    one = oracle.best(oracle.all_placements(6, 1), "parallel")
    two = oracle.best(oracle.all_placements(6, 2), "parallel")
    three = oracle.best(oracle.all_placements(6, 3), "parallel")

    assert one.write_edges == (2,)
    assert one.sequential["max_depth"] == 16
    assert one.parallel["max_depth"] == 11

    assert two.write_edges == (1, 3)
    assert two.sequential["max_depth"] == 15
    assert two.parallel["max_depth"] == 8

    assert three.write_edges == (0, 2, 4)
    assert three.sequential["max_depth"] == 15
    assert three.parallel["max_depth"] == 7
