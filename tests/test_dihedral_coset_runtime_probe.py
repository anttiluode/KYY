from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "dihedral_coset_runtime_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "dihedral_coset_runtime_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_branch_sidecar_toggles_only_on_reflection() -> None:
    n = 7
    angles = np.array([2.0 * math.pi / n], dtype=np.float64)
    model = probe.base.DihedralHarmonicTracker(n, angles, learn_angles=False).double()
    phi = np.array([0.2], dtype=np.float64)
    tokens = torch.tensor([[1, 2, n, 3, n, 0]], dtype=torch.long)
    branch = torch.zeros(1, dtype=torch.long)
    expected = [0, 0, 1, 1, 0, 0]
    h = model.h0.unsqueeze(0)
    for t, e in enumerate(expected):
        tok = tokens[:, t]
        h = model.step(h, tok)
        branch = torch.where(tok == n, 1 - branch, branch)
        assert int(branch.item()) == e
        out = probe.apply_branch_conditioned_phase_batch(h, branch, phi)
        assert out.shape == h.shape


def test_batch_branch_phase_matches_static_coset_formula() -> None:
    n = 11
    projected = np.array([2.0 * math.pi / n, 6.0 * math.pi / n])
    learned = projected - np.array([0.01, -0.015])
    phi = probe.coset.coset_midpoint_phase_vector(n, learned, projected)
    h0 = probe.base.default_h0(2, dtype=torch.float64)
    z = probe.base.orbit_prototypes(n, projected, h0).reshape(2 * n, 2, 2)
    branch = torch.cat((torch.zeros(n), torch.ones(n))).long()
    live = probe.apply_branch_conditioned_phase_batch(z, branch, phi).reshape(2 * n, -1)
    static = probe.coset.coset_midpoint_recenter(
        n, z.reshape(2 * n, -1), learned, projected
    )
    assert torch.allclose(live, static, atol=1e-12, rtol=1e-12)
