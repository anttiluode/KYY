from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "cyclic_equivariant_port_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "cyclic_equivariant_port_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_equivariant_projection_fixes_exact_equivariant_decoder() -> None:
    n = 11
    angles = 2.0 * math.pi * np.array([1, 3], dtype=np.float64) / n
    d = 4
    w0 = torch.tensor([0.7, -0.2, 0.4, 0.9], dtype=torch.float64)
    rows = []
    for j in range(n):
        rows.append(probe.block_rotation(angles, j) @ w0)
    W = torch.stack(rows, dim=0)
    b = torch.full((n,), 0.37, dtype=torch.float64)
    Wp, bp, recovered = probe.project_cyclic_equivariant_decoder(n, angles, W, b)
    assert torch.allclose(Wp, W, atol=1e-11, rtol=1e-11)
    assert torch.allclose(bp, b, atol=1e-12, rtol=1e-12)
    assert torch.allclose(recovered, w0, atol=1e-11, rtol=1e-11)
    assert W.shape[1] == d


def test_projected_decoder_has_circulant_logits() -> None:
    n = 13
    angles = 2.0 * math.pi * np.array([2, 5], dtype=np.float64) / n
    h0 = torch.tensor([[1.0, 0.0], [0.6, 0.8]], dtype=torch.float64)
    z = probe.exact_orbit(n, angles, h0)
    g = torch.Generator().manual_seed(3)
    W = torch.randn(n, 4, generator=g, dtype=torch.float64)
    b = torch.randn(n, generator=g, dtype=torch.float64)
    Wp, bp, _ = probe.project_cyclic_equivariant_decoder(n, angles, W, b)
    assert probe.circulant_logit_defect(z, Wp, bp) < 1e-10


def test_prototype_decoder_is_exact_for_distinct_equal_norm_orbit() -> None:
    n = 17
    angles = 2.0 * math.pi * np.array([1, 4, 7], dtype=np.float64) / n
    h0 = torch.zeros(3, 2, dtype=torch.float64)
    h0[:, 0] = 1.0 / math.sqrt(3.0)
    z = probe.exact_orbit(n, angles, h0)
    W, b = probe.prototype_decoder(z)
    acc, margin, mistakes = probe.readout_metrics(z, W, b)
    assert acc == 1.0
    assert mistakes == 0
    assert margin > 0.0
