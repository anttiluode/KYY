from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "abelian_harmonic_state_oracle"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "abelian_harmonic_state_oracle.py"
)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = oracle
SPEC.loader.exec_module(oracle)


def test_group_action_is_exact_up_to_roundoff_for_small_products():
    cases = [
        ((5,), [(1,), (2,)]),
        ((2, 3), [(1, 0), (0, 1), (1, 2)]),
        ((3, 3), [(1, 0), (0, 1), (1, 1), (1, 2)]),
    ]
    for moduli, freqs in cases:
        assert oracle.verify_group_action(moduli, freqs) < 2e-13
        states, proto = oracle.prototypes(moduli, freqs)
        assert len(states) == math.prod(moduli)
        assert np.allclose(np.linalg.norm(proto, axis=1), 1.0, atol=1e-13)


def test_cyclic_special_case_matches_character_phase_formula():
    n = 17
    freqs = [(1,), (3,), (7,), (11,)]
    moduli = (n,)
    _, proto = oracle.prototypes(moduli, freqs)
    token = (1,)
    A = oracle.token_operator(moduli, freqs, token)
    assert np.allclose(A.T @ A, np.eye(A.shape[0]), atol=2e-13)
    assert np.allclose((A @ proto.T).T, np.roll(proto, -1, axis=0), atol=2e-13)


def test_random_search_finds_constant_margin_for_small_noncyclic_group():
    moduli = (4, 5)
    freqs, metrics = oracle.random_search(moduli, k=8, trials=500, seed=0)
    assert len(freqs) == 8
    assert metrics["group_order"] == 20
    assert metrics["real_dimension"] == 16
    assert float(metrics["nearest_prototype_noise_radius"]) > 0.45


def test_existence_bound_depends_on_group_order_not_factorization():
    alpha = 0.5
    assert oracle.existence_k(64, alpha) == oracle.existence_k(8 * 8, alpha)
    k = oracle.existence_k(64, alpha)
    assert oracle.hoeffding_failure_bound(64, k, alpha) < 1.0
