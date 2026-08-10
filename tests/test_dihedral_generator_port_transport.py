from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "dihedral_generator_transport_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "dihedral_generator_port_transport.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def explicit_cross(n: int, learned: float, projected: float, v: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(2, 2, dtype=torch.float64)
    rl = probe.joint.rotation_matrix(learned, dtype=torch.float64)
    rp = probe.joint.rotation_matrix(projected, dtype=torch.float64)
    xl = v.reshape(2,1)
    xp = v.reshape(2,1)
    for _ in range(n):
        out += xp @ xl.T
        xp = rp @ xp
        xl = rl @ xl
    return out


def test_matrix_power_sum_matches_explicit_sum() -> None:
    k = torch.tensor([[0.8, -0.2], [0.1, 0.9]], dtype=torch.float64)
    p, s = probe.matrix_power_sum(k, 13)
    pe = torch.linalg.matrix_power(k, 13)
    se = torch.zeros_like(k)
    cur = torch.eye(2, dtype=torch.float64)
    for _ in range(13):
        se += cur
        cur = cur @ k
    assert torch.allclose(p, pe, atol=1e-12, rtol=1e-12)
    assert torch.allclose(s, se, atol=1e-12, rtol=1e-12)


def test_generator_cross_matches_orbit_enumeration() -> None:
    n = 31
    learned = 0.713
    projected = 2.0 * math.pi * 4 / n
    v = torch.tensor([0.37, 0.91], dtype=torch.float64)
    cg = probe.rotation_cross_covariance_from_generators(n, learned, projected, v)
    ce = explicit_cross(n, learned, projected, v)
    assert torch.allclose(cg, ce, atol=1e-10, rtol=1e-10)


def test_generator_quotient_ports_match_orbit_procrustes() -> None:
    n = 17
    learned_a = np.array([0.51, -1.13])
    projected_a = np.array([2.0*math.pi*1/n, 2.0*math.pi*(-3)/n])
    learned_s = np.array([
        [[0.98,0.18],[0.07,-0.91]],
        [[0.73,0.55],[0.51,-0.76]],
    ], dtype=np.float64)
    projected_s = probe.joint.project_reflections(learned_s)
    h0 = probe.base.default_h0(2, dtype=torch.float64)
    qg0,qg1 = probe.generator_quotient_block_port(
        n, learned_a, projected_a, learned_s, projected_s, h0
    )
    qo0,qo1 = probe.orbit_block_port_for_check(
        n, learned_a, projected_a, learned_s, projected_s, h0
    )
    assert torch.allclose(qg0, qo0, atol=1e-10, rtol=1e-10)
    assert torch.allclose(qg1, qo1, atol=1e-10, rtol=1e-10)
