from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "dihedral_coset_recenter_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "dihedral_coset_recenter_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_coset_midpoint_reduces_symmetric_snap_alignment_error() -> None:
    n = 31
    modes = 4
    frequencies = np.array([1, 5, 9, 13])
    projected = 2.0 * math.pi * frequencies / n
    delta = np.array([0.03, -0.02, 0.01, -0.025])
    learned = projected - delta
    h0 = probe.base.default_h0(modes, dtype=probe.torch.float64)
    z_learned = probe.base.orbit_prototypes(n, learned, h0)
    z_projected = probe.base.orbit_prototypes(n, projected, h0)
    z_coset = probe.coset_midpoint_recenter(n, z_projected, learned, projected)
    raw = float(probe.torch.linalg.matrix_norm(z_projected - z_learned).item())
    corrected = float(probe.torch.linalg.matrix_norm(z_coset - z_learned).item())
    assert corrected < raw


def test_coset_midpoint_uses_opposite_phase_on_reflected_branch() -> None:
    n = 7
    projected = np.array([2.0 * math.pi / n])
    learned = projected - np.array([0.02])
    phi = probe.coset_midpoint_phase_vector(n, learned, projected)
    h0 = probe.base.default_h0(1, dtype=probe.torch.float64)
    z = probe.base.orbit_prototypes(n, projected, h0)
    out = probe.coset_midpoint_recenter(n, z, learned, projected)
    rot_expected = probe.apply_per_mode_phase(z[:n], phi)
    ref_expected = probe.apply_per_mode_phase(z[n:], -phi)
    assert probe.torch.allclose(out[:n], rot_expected)
    assert probe.torch.allclose(out[n:], ref_expected)
