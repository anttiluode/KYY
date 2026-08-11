from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "cyclic_reset_orbit_oracle"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "cyclic_reset_orbit_oracle.py"
)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = oracle
SPEC.loader.exec_module(oracle)


def test_cyclic_reset_stays_two_dimensional():
    for n in (3, 5, 7, 16, 101):
        row = oracle.probe(n)
        assert row.recurrent_real_dimension == 2
        assert row.exact_cycle_state_error < 1e-12
        assert row.exact_reset_state_error == 0.0


def test_exact_margin_formulas():
    for n in (3, 4, 8, 32):
        row = oracle.probe(n)
        assert math.isclose(row.minimum_pair_distance, 2.0 * math.sin(math.pi / n), rel_tol=1e-12)
        assert math.isclose(row.nearest_prototype_noise_radius, math.sin(math.pi / n), rel_tol=1e-12)
        assert math.isclose(row.unit_logit_margin, 1.0 - math.cos(2.0 * math.pi / n), rel_tol=1e-12)


def test_noise_radius_shrinks_with_behavioral_state_count():
    radii = [oracle.probe(n).nearest_prototype_noise_radius for n in (4, 8, 16, 32, 64)]
    assert all(a > b for a, b in zip(radii, radii[1:]))
