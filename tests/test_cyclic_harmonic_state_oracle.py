from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "cyclic_harmonic_state_oracle"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "cyclic_harmonic_state_oracle.py"
)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = oracle
SPEC.loader.exec_module(oracle)


def test_cycle_operator_moves_every_harmonic_prototype_exactly_up_to_roundoff():
    cases = [
        (7, [1]),
        (31, [1, 3, 7, 11]),
        (101, [0, 4, 17, 29, 43, 61, 77, 95]),
    ]
    for n, freqs in cases:
        assert oracle.verify_cycle(n, freqs) < 2e-13
        states = oracle.harmonic_prototypes(n, freqs)
        assert np.allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-13)
        A = oracle.cycle_operator(n, freqs)
        assert np.allclose(A.T @ A, np.eye(A.shape[0]), atol=2e-13)


def test_geometry_formula_matches_bruteforce_pair_distances():
    n = 31
    freqs = [1, 3, 5, 9, 12, 17, 24, 28]
    states = oracle.harmonic_prototypes(n, freqs)
    brute = min(
        float(np.linalg.norm(states[i] - states[j]))
        for i in range(n)
        for j in range(i)
    )
    metrics = oracle.geometric_metrics(n, freqs)
    assert math.isclose(brute, metrics["minimum_pair_distance"], rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(
        0.5 * brute,
        metrics["nearest_prototype_noise_radius"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_elementary_existence_bound_is_logarithmic_for_fixed_alpha():
    alpha = 0.5
    ks = [oracle.existence_k(n, alpha) for n in (10, 100, 1000, 10000)]
    for n, k in zip((10, 100, 1000, 10000), ks):
        assert oracle.hoeffding_one_sided_failure_bound(n, k, alpha) < 1.0
        assert k <= math.ceil(8.0 * math.log(max(2, n - 1))) + 2
    # Increasing n by three orders of magnitude should not increase k linearly.
    assert ks[-1] < 5 * ks[0]


def test_random_harmonic_code_substantially_improves_margin_over_single_phase():
    n = 101
    k = 16
    _, metrics = oracle.random_search(n, k, trials=500, seed=0)
    harmonic_radius = float(metrics["nearest_prototype_noise_radius"])
    single_phase_radius = math.sin(math.pi / n)
    assert harmonic_radius > 0.55
    assert harmonic_radius > 10.0 * single_phase_radius
