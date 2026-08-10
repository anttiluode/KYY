from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "dihedral_equivariant_port_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "dihedral_equivariant_port_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def reflection(beta: float) -> np.ndarray:
    c, s = math.cos(2.0 * beta), math.sin(2.0 * beta)
    return np.asarray([[c, s], [s, -c]], dtype=np.float64)


def test_dihedral_projection_fixes_exact_equivariant_decoder() -> None:
    n = 7
    angles = np.asarray([2.0 * math.pi / n, 4.0 * math.pi / n])
    refs = np.stack([reflection(0.2), reflection(-0.35)])
    w0 = torch.tensor([0.7, -0.2, 0.4, 0.9], dtype=torch.float64)
    rows = []
    for branch in (0, 1):
        for k in range(n):
            rows.append(probe.representation_matrix(angles, refs, k, branch) @ w0)
    W = torch.stack(rows, dim=0)
    b = torch.full((2*n,), 0.23, dtype=torch.float64)
    Wp, bp, recovered = probe.project_dihedral_equivariant_decoder(n, angles, refs, W, b)
    assert torch.allclose(Wp, W, atol=1e-10, rtol=1e-10)
    assert torch.allclose(bp, b, atol=1e-12, rtol=1e-12)
    assert torch.allclose(recovered, w0, atol=1e-10, rtol=1e-10)
    assert probe.equivariance_defect(n, angles, refs, Wp, bp) < 1e-10


def test_positive_orbit_kernel_is_exact_when_stabilizer_trivial() -> None:
    n = 11
    frequencies = np.asarray([1, 3], dtype=np.int64)
    angles = 2.0 * math.pi * frequencies / n
    refs = np.stack([reflection(0.0), reflection(0.17)])
    h0 = torch.tensor(
        [[math.cos(math.pi/(2*n)), math.sin(math.pi/(2*n))], [0.6, 0.8]],
        dtype=torch.float64,
    )
    z = probe.joint.canonical_orbit(n, angles, refs, h0)
    # Feed an already positive-kernel decoder through the projector.
    alpha = [1.2, 0.7]
    u = h0.reshape(-1, 2)
    base = torch.stack([a * ui for a, ui in zip(alpha, u)], dim=0).reshape(-1)
    rows = []
    for branch in (0, 1):
        for k in range(n):
            rows.append(probe.representation_matrix(angles, refs, k, branch) @ base)
    W = torch.stack(rows, dim=0)
    b = torch.zeros(2*n, dtype=torch.float64)
    Wp, bp, learned_alpha, _ = probe.project_positive_orbit_kernel(
        n, angles, refs, h0, W, b
    )
    cert = probe.certmod.dihedral_stabilizer_certificate(n, frequencies, refs, h0)
    assert cert.trivial_stabilizer_certified
    assert np.all(learned_alpha > 0.0)
    acc, margin, mistakes = probe.joint.metrics(z, Wp, bp)
    assert acc == 1.0
    assert mistakes == 0
    assert margin > 0.0
