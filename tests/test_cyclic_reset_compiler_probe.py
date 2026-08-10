from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "cyclic_reset_compiler_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "cyclic_reset_compiler_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def exact_positive_port(n: int, frequencies: np.ndarray) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    angles = 2.0 * math.pi * frequencies.astype(np.float64) / n
    modes = len(frequencies)
    h0 = torch.zeros(modes, 2, dtype=torch.float64)
    h0[:, 0] = 1.0 / math.sqrt(modes)
    orbit = probe.eq.exact_orbit(n, angles, h0)
    W, b = probe.eq.prototype_decoder(orbit)
    return angles, h0, W, b


def test_compiled_cycle_reset_runtime_is_exact() -> None:
    n = 11
    angles, h0, W, b = exact_positive_port(n, np.asarray([1, 3], dtype=np.int64))
    # increments: +4,+2, reset, +3,+8, reset,+5 => states 4,6,0,3,0,0,5
    tokens = torch.tensor([[4, 2, n, 3, 8, n, 5]], dtype=torch.long)
    expected = torch.tensor([[4, 6, 0, 3, 0, 0, 5]], dtype=torch.long)
    pred = probe.compiled_runtime(tokens, n, angles, h0, W, b).argmax(dim=-1)
    assert torch.equal(pred, expected)


def test_compiled_relations_vanish_and_orbit_is_faithful() -> None:
    n = 13
    angles, h0, _, _ = exact_positive_port(n, np.asarray([2, 5], dtype=np.int64))
    rel = probe.reset_relation_defects(n, angles, h0)
    assert rel["cycle_operator_relation_defect"] < 1e-10
    assert rel["cycle_state_relation_defect"] < 1e-10
    assert rel["reset_idempotence_defect"] == 0.0
    assert rel["reset_after_cycle_defect"] == 0.0
    assert rel["minimum_nonidentity_orbit_distance"] > 0.0


def test_positive_port_certificate_for_compiled_reset_orbit() -> None:
    n = 15
    frequencies = np.asarray([2, 5], dtype=np.int64)
    angles, h0, _, _ = exact_positive_port(n, frequencies)
    orbit = probe.eq.exact_orbit(n, angles, h0)
    # Construct a learned-looking equivariant port with positive per-mode rays.
    alpha = [1.3, 0.4]
    u = orbit[0].reshape(-1, 2)
    w0 = torch.stack([a * ui for a, ui in zip(alpha, u)], dim=0).reshape(-1)
    W = torch.stack([probe.eq.block_rotation(angles, j) @ w0 for j in range(n)], dim=0)
    b = torch.zeros(n, dtype=torch.float64)
    Wp, bp, learned_alpha, _ = probe.pos.positive_kernel_projection(n, angles, orbit[0], W, b)
    acc, margin, mistakes = probe.eq.readout_metrics(orbit, Wp, bp)
    certified, g = probe.pos.positive_kernel_certificate(n, frequencies, learned_alpha)
    assert acc == 1.0
    assert mistakes == 0
    assert margin > 0.0
    assert certified
    assert g == 1
