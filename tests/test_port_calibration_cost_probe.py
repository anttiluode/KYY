from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "port_calibration_cost_probe_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "port_calibration_cost_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_mode_phase_contains_common_time_shift() -> None:
    n = 31
    freqs = np.array([1, 3, 7, 11], dtype=np.float64)
    angles = 2.0 * math.pi * freqs / n
    tau = 0.237
    z = probe.phase.phase_shifted_prototypes(n, angles, 0.0)
    shifted = probe.mode_phase_transform(
        z, torch.tensor(tau * angles, dtype=torch.float64)
    )
    direct = probe.phase.phase_shifted_prototypes(n, angles, tau)
    assert torch.allclose(shifted, direct, atol=1e-10, rtol=1e-10)


def test_zero_low_rank_adapter_is_identity() -> None:
    z = torch.randn(9, 6, dtype=torch.float64)
    u = torch.zeros(6, 1, dtype=torch.float64)
    v = torch.randn(6, 1, dtype=torch.float64)
    out = probe.low_rank_hidden_transform(z, u, v)
    assert torch.equal(out, z)


def test_zero_full_hidden_adapter_is_identity() -> None:
    z = torch.randn(9, 6, dtype=torch.float64)
    delta = torch.zeros(6, 6, dtype=torch.float64)
    out = probe.full_hidden_transform(z, delta)
    assert torch.equal(out, z)


def test_full_calibration_states_are_complete() -> None:
    states = probe.make_calibration_states(17, 17, seed=123)
    assert torch.equal(states, torch.arange(17))


def test_calibration_subset_has_unique_states() -> None:
    states = probe.make_calibration_states(101, 16, seed=123)
    assert states.numel() == 16
    assert torch.unique(states).numel() == 16
    assert bool(torch.all(states[1:] > states[:-1]))
