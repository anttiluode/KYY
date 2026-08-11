from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "cyclic_relation_defect"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "cyclic_relation_defect.py")
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_exact_characters_satisfy_cyclic_relation_to_roundoff():
    n = 31
    angles = probe.exact_character_angles(n, [1, 3, 7, 11, 17])
    assert probe.operator_relation_defect(n, angles) < 2e-13
    assert probe.state_wrap_defect(n, angles) < 2e-13
    assert probe.operator_relation_defect(n, angles, wraps=13) < 3e-12


def test_uniform_angle_error_has_closed_form_one_wrap_defect():
    n = 31
    eta = 1e-4
    angles = probe.exact_character_angles(n, [1, 3, 7, 11]) + eta
    expected = 2.0 * abs(math.sin(0.5 * n * eta))
    assert math.isclose(probe.operator_relation_defect(n, angles), expected, rel_tol=1e-11, abs_tol=1e-12)
    assert math.isclose(probe.state_wrap_defect(n, angles), expected, rel_tol=1e-11, abs_tol=1e-12)


def test_small_defect_slope_predicts_early_wrap_growth():
    n = 101
    eta = 2e-6
    angles = probe.exact_character_angles(n, [1, 7, 19, 37, 61]) + eta
    slope = probe.small_defect_wrap_slope(n, angles)
    for q in (1, 4, 12):
        actual = probe.state_wrap_defect(n, angles, wraps=q)
        approx = q * slope
        assert math.isclose(actual, approx, rel_tol=2e-5, abs_tol=1e-12)


def test_state_wrap_formula_matches_direct_phase_state_distance():
    n = 17
    angles = probe.exact_character_angles(n, [1, 4, 6]) + np.asarray([1e-3, -5e-4, 2e-4])
    k = len(angles)
    S = 23
    q = 3
    z0 = np.exp(1j * S * angles) / math.sqrt(k)
    z1 = np.exp(1j * (S + q * n) * angles) / math.sqrt(k)
    direct = float(np.linalg.norm(z1 - z0))
    assert math.isclose(direct, probe.state_wrap_defect(n, angles, wraps=q), rel_tol=1e-12, abs_tol=1e-12)
