import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "hybrid_fiber_lowering_audit_for_tests"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "hybrid_fiber_lowering_audit.py")
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = audit
SPEC.loader.exec_module(audit)


def rails():
    centers = np.array(
        [[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]]
    )
    v = np.array([[0.0], [0.0], [1.0]])
    return centers, [v.copy() for _ in range(4)]


def test_partial_merge_preserving_shared_analog_fiber_is_linear():
    centers, tangents = rails()
    result = audit.audit_linear_fiber_lowering(
        centers, tangents, [0, 0, 2, 2], [np.eye(1) for _ in range(4)]
    )
    assert result.realizable
    assert result.residual < 1e-10
    assert result.dependency_residual < 1e-10


def test_shared_tangent_forbids_source_mode_dependent_analog_action():
    centers, tangents = rails()
    maps = [np.array([[1.0]]), np.array([[0.5]]), np.array([[1.0]]), np.array([[1.0]])]
    result = audit.audit_linear_fiber_lowering(centers, tangents, [0, 0, 2, 2], maps)
    assert not result.realizable
    assert result.dependency_residual > 1e-3


def test_separating_tangent_copies_spends_dimension_and_restores_realizability():
    # Four independent fiber tangents let one global operator act differently
    # on the analog coordinate in each source mode, at the cost of dimension.
    centers = np.zeros((6, 4))
    centers[0, 0] = 1.0
    centers[1, 1] = 1.0
    centers[0, 2] = -1.0
    centers[1, 3] = -1.0
    tangents = []
    for q in range(4):
        v = np.zeros((6, 1))
        v[2 + q, 0] = 1.0
        tangents.append(v)
    # Keep digital states fixed so target tangent bases stay mode-specific.
    maps = [np.array([[1.0]]), np.array([[0.5]]), np.array([[0.8]]), np.array([[1.2]])]
    result = audit.audit_linear_fiber_lowering(centers, tangents, [0, 1, 2, 3], maps)
    assert result.realizable
    assert result.residual < 1e-10
