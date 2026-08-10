from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "dihedral_stabilizer_certificate_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "dihedral_stabilizer_certificate.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def reflection(beta: float) -> np.ndarray:
    c, s = math.cos(2.0 * beta), math.sin(2.0 * beta)
    return np.asarray([[c, s], [s, -c]], dtype=np.float64)


def seed(gammas: list[float]) -> torch.Tensor:
    return torch.tensor(
        [[math.cos(g), math.sin(g)] for g in gammas], dtype=torch.float64
    )


def test_reflection_fixed_seed_is_not_trivial() -> None:
    n = 11
    cert = probe.dihedral_stabilizer_certificate(
        n,
        [1],
        np.stack([reflection(0.0)]),
        seed([0.0]),
    )
    assert cert.rotation_kernel_size == 1
    assert cert.reflection_stabilizer_exists
    assert cert.reflection_candidate_k == 0
    assert not cert.trivial_stabilizer_certified


def test_off_grid_seed_has_trivial_stabilizer_and_exact_prototype_decode() -> None:
    n = 11
    frequencies = np.asarray([1], dtype=np.int64)
    angle = 2.0 * math.pi * frequencies / n
    gamma = math.pi / (2.0 * n)
    h0 = seed([gamma])
    refs = np.stack([reflection(0.0)])
    cert = probe.dihedral_stabilizer_certificate(n, frequencies, refs, h0)
    assert cert.rotation_kernel_size == 1
    assert not cert.reflection_stabilizer_exists
    assert cert.trivial_stabilizer_certified

    z = probe.joint.canonical_orbit(n, angle, refs, h0)
    W, b = probe.prototype_decoder(z)
    acc, margin, mistakes = probe.joint.metrics(z, W, b)
    assert acc == 1.0
    assert mistakes == 0
    assert margin > 0.0


def test_incompatible_reflection_congruences_remove_reflected_stabilizer() -> None:
    n = 11
    # beta=0 gives target t=-n*gamma/pi. These two modes demand k=1 and k=2.
    h0 = seed([-math.pi / n, -2.0 * math.pi / n])
    refs = np.stack([reflection(0.0), reflection(0.0)])
    cert = probe.dihedral_stabilizer_certificate(n, [1, 1], refs, h0)
    assert cert.rotation_kernel_size == 1
    assert not cert.reflection_stabilizer_exists
    assert cert.trivial_stabilizer_certified


def test_linear_congruence_solver_handles_composite_modulus() -> None:
    # 4 k = 8 mod 12 -> k = 2 mod 3.
    assert probe.solve_linear_congruence(4, 8, 12) == (2, 3)
    # 4 k = 6 mod 12 has no solution.
    assert probe.solve_linear_congruence(4, 6, 12) is None
    # Intersect k=2 mod 3 with k=5 mod 6 -> k=5 mod 6.
    assert probe._crt_pair(2, 3, 5, 6) == (5, 6)
