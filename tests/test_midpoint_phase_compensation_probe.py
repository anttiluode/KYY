from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "midpoint_phase_compensation_probe_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "midpoint_phase_compensation_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_midpoint_compensation_halves_linear_endpoint_error() -> None:
    n = 101
    learned = np.array([0.20, 0.70], dtype=np.float64)
    projected = learned + np.array([0.002, -0.004], dtype=np.float64)
    phi = probe.midpoint_phase_compensation(n, learned, projected)
    before_max, _ = probe.phase_mismatch_stats(n, learned, projected, np.zeros_like(phi))
    after_max, _ = probe.phase_mismatch_stats(n, learned, projected, phi)
    assert np.isclose(after_max, before_max / 2.0, atol=1e-12)


def test_no_snap_change_needs_no_compensation() -> None:
    angles = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    phi = probe.midpoint_phase_compensation(31, angles, angles.copy())
    assert np.allclose(phi, 0.0)
