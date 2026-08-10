from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "phase_readout_legalization_probe"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "phase_readout_legalization_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_zero_phase_matches_integer_character_orbit():
    n = 7
    angles = 2.0 * math.pi * np.asarray([1, -2, 3], dtype=np.float64) / n
    z = probe.phase_shifted_prototypes(n, angles, 0.0)
    assert z.shape == (n, 6)
    assert torch.allclose(torch.linalg.vector_norm(z, dim=-1), torch.ones(n, dtype=torch.float64))


def test_fractional_time_shift_advances_each_harmonic_by_its_generator_angle():
    n = 11
    angles = 2.0 * math.pi * np.asarray([1, 3], dtype=np.float64) / n
    tau = 0.25
    z = probe.phase_shifted_prototypes(n, angles, tau)
    scale = 1.0 / math.sqrt(2)
    expected = torch.tensor(
        [
            scale * math.cos(tau * angles[0]),
            scale * math.sin(tau * angles[0]),
            scale * math.cos(tau * angles[1]),
            scale * math.sin(tau * angles[1]),
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(z[0], expected, atol=1e-12, rtol=1e-12)


def test_phase_search_keeps_zero_when_zero_has_strictly_best_readout():
    n = 5
    angles = 2.0 * math.pi * np.asarray([1, 2], dtype=np.float64) / n
    z0 = probe.phase_shifted_prototypes(n, angles, 0.0)
    # Prototype classifier: each class weight points exactly at its legal state.
    W = z0.clone()
    b = torch.zeros(n, dtype=torch.float64)
    result = probe.search_phase_offset(n, angles, W, b, half_span=0.4, grid=401)
    assert result.baseline_accuracy == 1.0
    assert result.best_accuracy == 1.0
    assert abs(result.best_tau) < 1e-12
    assert result.best_min_margin >= result.baseline_min_margin - 1e-12
