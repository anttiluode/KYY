from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "cyclic_positive_kernel_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "cyclic_positive_kernel_port_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_positive_kernel_certificate_accepts_faithful_active_characters() -> None:
    ok, g = probe.positive_kernel_certificate(
        12,
        np.asarray([2, 5]),
        np.asarray([0.7, 0.2]),
    )
    assert ok
    assert g == 1


def test_positive_kernel_certificate_ignores_zero_weight_mode() -> None:
    ok, g = probe.positive_kernel_certificate(
        15,
        np.asarray([5, 2]),
        np.asarray([1.0, 0.0]),
    )
    assert not ok
    assert g == 5


def test_positive_kernel_certificate_rejects_negative_weight() -> None:
    ok, _ = probe.positive_kernel_certificate(
        11,
        np.asarray([1, 3]),
        np.asarray([0.5, -0.1]),
    )
    assert not ok


def test_positive_kernel_decoder_is_exact_for_faithful_orbit() -> None:
    n = 13
    frequencies = np.asarray([2, 5], dtype=np.int64)
    angles = 2.0 * math.pi * frequencies / n
    h0 = torch.tensor([[0.6, 0.0], [0.8, 0.0]], dtype=torch.float64)
    z = probe.base.exact_orbit(n, angles, h0)
    alpha = np.asarray([1.7, 0.4], dtype=np.float64)
    u = z[0].reshape(-1, 2)
    w0 = torch.stack(
        [float(a) * ui for a, ui in zip(alpha.tolist(), u)], dim=0
    ).reshape(-1)
    W = torch.stack(
        [probe.base.block_rotation(angles, j) @ w0 for j in range(n)], dim=0
    )
    b = torch.zeros(n, dtype=torch.float64)
    accuracy, min_margin, mistakes = probe.base.readout_metrics(z, W, b)
    assert accuracy == 1.0
    assert mistakes == 0
    assert min_margin > 0.0
