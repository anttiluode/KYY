from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "transition_lowering_audit_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "transition_lowering_audit.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_c4_cycle_lowers_linearly() -> None:
    z = probe.cyclic_code(4)
    y = probe.transition_targets(z, [1, 2, 3, 0])
    r = probe.linear_lowering(z, y)
    assert r.exact
    A = np.asarray(r.operator)
    assert np.linalg.norm(A @ z - y) < 1e-9
    assert r.continuous_rank == 2


def test_c4_partial_merge_lowers_to_exact_singular_pinch() -> None:
    z = probe.cyclic_code(4)
    y = probe.transition_targets(z, [0, 0, 2, 2])
    r = probe.linear_lowering(z, y)
    assert r.exact
    A = np.asarray(r.operator)
    expected = np.asarray([[1.0, 1.0], [0.0, 0.0]])
    assert np.linalg.norm(A - expected) < 1e-9
    assert r.continuous_rank == 1


def test_total_reset_is_not_linear_on_centered_square_but_is_affine() -> None:
    z = probe.cyclic_code(4)
    y = probe.transition_targets(z, [0, 0, 0, 0])
    linear = probe.linear_lowering(z, y)
    affine = probe.affine_lowering(z, y)
    assert not linear.exact
    assert linear.dependency_violation > 1e-3
    assert affine.exact
    A = np.asarray(affine.operator)
    b = np.asarray(affine.bias)
    assert np.linalg.norm(A) < 1e-9
    assert np.linalg.norm(b - z[:, 0]) < 1e-9


def test_dependency_criterion_matches_residual_in_demo_cases() -> None:
    for row in probe.demo().values():
        for kind in ("linear", "affine"):
            r = row[kind]
            assert bool(r["exact"]) == (
                float(r["dependency_violation"]) <= 1e-9
                and float(r["residual"]) <= 1e-9
            )
