from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "state_packing_floor"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "state_packing_floor.py"
)
assert SPEC is not None and SPEC.loader is not None
packing = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = packing
SPEC.loader.exec_module(packing)


def test_scalar_counter_saturates_volume_bound_in_one_dimension():
    for n in (3, 4, 10, 100):
        eps = packing.scalar_counter_radius(n, radius=0.5)
        bound = packing.max_robust_radius_volume_bound(n, dimension=1, radius=0.5)
        assert math.isclose(eps, bound, rel_tol=1e-12, abs_tol=1e-12)


def test_volume_bound_contains_standard_examples():
    for n in (3, 5, 10, 100):
        for ex in packing.standard_examples(n):
            assert ex.robust_radius <= ex.volume_bound_radius * (1.0 + 1e-12)
            assert ex.capacity_bits_upper_bound + 1e-12 >= ex.state_information_bits


def test_dimension_bound_inverts_capacity_inequality():
    cases = [
        (16, 0.5, 1.0),
        (1000, 0.1, 1.0),
        (1000000, 0.01, 1.0),
    ]
    for n, eps, radius in cases:
        d = packing.minimum_dimension_volume_bound(n, eps, radius)
        assert packing.packing_capacity_bits(d, eps, radius) + 1e-12 >= math.log2(n)
        if d > 1:
            assert packing.packing_capacity_bits(d - 1, eps, radius) < math.log2(n) + 1e-12


def test_cyclic_orbit_precision_shrinks_as_inverse_n():
    for n in (100, 1000, 10000):
        eps = packing.cyclic_orbit_radius(n)
        asymptotic = math.pi / n
        assert abs(eps / asymptotic - 1.0) < 1e-3
