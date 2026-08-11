import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "metacircuit_cyclic_backend_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "metacircuit_cyclic_backend.py")
assert SPEC is not None and SPEC.loader is not None
backend = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = backend
SPEC.loader.exec_module(backend)


def test_companion_is_similar_to_exact_rotation_and_has_finite_order():
    n, f = 101, 25
    row = backend.lower_mode(n, f)
    assert row.relation_defect < 1e-10
    assert row.stable_interior
    theta = 2 * math.pi * f / n
    a = backend.companion_block(theta)
    t = backend.phase_from_companion(theta)
    assert np.linalg.norm(t @ a @ np.linalg.inv(t) - backend.rotation(theta)) < 1e-10


def test_faithful_character_margin_is_frequency_independent_for_coprime_frequencies():
    n = 101
    margins = [backend.prototype_margin(n, f) for f in [1, 4, 25, 49, 50]]
    assert max(margins) - min(margins) < 1e-12


def test_physical_conditioning_is_not_frequency_independent():
    slow = backend.lower_mode(101, 1)
    quarter = backend.lower_mode(101, 25)
    near_nyquist = backend.lower_mode(101, 50)
    assert quarter.phase_map_condition < slow.phase_map_condition
    assert quarter.phase_map_condition < near_nyquist.phase_map_condition


def test_near_nyquist_character_has_tiny_positive_component_tolerance_headroom():
    quarter = backend.lower_mode(101, 25)
    near_nyquist = backend.lower_mode(101, 50)
    assert quarter.positive_relative_stability_headroom > 1.0
    assert near_nyquist.positive_relative_stability_headroom < 3e-4
    assert near_nyquist.relative_phase_sensitivity_to_ratio > 50.0


def test_nyquist_character_is_degenerate_in_central_difference_companion_coordinates():
    row = backend.lower_mode(100, 50)
    assert row.degenerate
    assert not row.stable_interior
    assert row.relation_defect > 1.0


def test_port_transport_preserves_logits_exactly():
    theta = 2 * math.pi * 7 / 31
    t = backend.phase_from_companion(theta)
    h = np.array([0.37, -0.22])
    p = t @ h
    w_phase = np.array([[1.2, -0.3], [-0.4, 0.7], [0.1, 0.9]])
    w_comp = backend.transport_phase_port_to_companion(w_phase, theta)
    assert np.allclose(w_phase @ p, w_comp @ h, atol=1e-12)
