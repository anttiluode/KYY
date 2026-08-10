from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "dihedral_legalization_probe_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "dihedral_legalization_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_symbolic_updates_match_matrix_action() -> None:
    n = 7
    angles = np.array([2.0 * math.pi / n, 4.0 * math.pi / n], dtype=np.float64)
    model = probe.DihedralHarmonicTracker(n, angles, learn_angles=False).double()
    # Word: r^2, s, r^3. Symbolically: r^2 -> s r^2 -> s r^(2-3)=s r^6.
    tokens = torch.tensor([[2, n, 3]], dtype=torch.long)
    h = model.h0.unsqueeze(0)
    for t in range(tokens.shape[1]):
        h = model.step(h, tokens[:, t])
    orbit = probe.orbit_prototypes(n, angles, model.h0)
    expected = orbit[n + 6]
    assert torch.allclose(h.reshape(-1), expected, atol=1e-10, rtol=1e-10)


def test_reflection_relations_hold_for_arbitrary_angles() -> None:
    angles = np.array([0.31, -1.17, 2.2], dtype=np.float64)
    for theta in angles:
        c, s = math.cos(theta), math.sin(theta)
        r = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
        f = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
        eye = torch.eye(2, dtype=torch.float64)
        assert torch.allclose(f @ f, eye, atol=1e-12)
        assert torch.allclose(f @ r @ f, r.T, atol=1e-12)


def test_projection_makes_finite_rotation_relation_exact() -> None:
    n = 31
    angles = np.array([0.23, -1.7, 2.91], dtype=np.float64)
    projected, _ = probe.project_angles_to_dn_characters(n, angles)
    assert probe.rotation_relation_defect(n, projected) < 1e-12


def test_orthogonal_procrustes_recovers_known_port_rotation() -> None:
    g = torch.Generator().manual_seed(3)
    x = torch.randn(50, 6, generator=g, dtype=torch.float64)
    q0, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
    y = x @ q0
    q = probe.orthogonal_procrustes_port(x, y)
    assert torch.allclose(x @ q, y, atol=1e-10, rtol=1e-10)


def test_complete_orbit_has_two_n_rows() -> None:
    n = 11
    angles = np.array([2.0 * math.pi / n, 6.0 * math.pi / n])
    h0 = probe.default_h0(2, dtype=torch.float64)
    z = probe.orbit_prototypes(n, angles, h0)
    assert z.shape == (2 * n, 4)
    assert torch.unique(torch.round(z * 1e10) / 1e10, dim=0).shape[0] == 2 * n
