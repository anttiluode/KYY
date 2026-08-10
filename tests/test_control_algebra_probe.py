from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "control_algebra_probe", ROOT / "map" / "control_algebra_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_path_one_local_control_reaches_so_n_small_cases():
    for n in range(3, 9):
        row = probe.measure(n)
        assert row.path_drift_plus_one_local == row.so_n


def test_even_ring_one_control_has_restricted_algebra():
    expected = {4: 4, 6: 9, 8: 16, 10: 25, 12: 36}
    for n, dim in expected.items():
        row = probe.measure(n)
        assert row.ring_drift_plus_one_local == dim
        assert dim < row.so_n


def test_second_neighbor_control_restores_even_ring_small_cases():
    for n in (4, 6, 8, 10, 12):
        row = probe.measure(n)
        assert row.ring_drift_plus_two_local == row.so_n


def test_odd_ring_one_control_reaches_so_n_small_cases():
    for n in (3, 5, 7, 9, 11):
        row = probe.measure(n)
        assert row.ring_drift_plus_one_local == row.so_n
