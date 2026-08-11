from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "sigma_affine_compiler"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "sigma_affine_compiler.py"
)
assert SPEC is not None and SPEC.loader is not None
compiler = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = compiler
SPEC.loader.exec_module(compiler)


def test_length_threshold_front_advances_one_neighbor_per_symbol():
    h = 6
    chain = compiler.length_threshold_sigma_chain(h, alphabet_size=2)
    hist = compiler.threshold_front(chain, h + 1)
    assert hist[0] == (0, 0, 0, 0, 0, 0)
    for t in range(1, h + 1):
        assert hist[t] == tuple([1] * t + [0] * (h - t))
    assert hist[h + 1] == tuple([1] * h)


def test_length_threshold_language_semantics():
    for h in range(1, 7):
        chain = compiler.length_threshold_sigma_chain(h, alphabet_size=3)
        for length in range(0, h + 3):
            state = chain.run([2] * length)[-1]
            assert (state[-1] == 1) == (length >= h)


def test_exact_affine_lowering_threshold_and_mixed_chain():
    threshold = compiler.length_threshold_sigma_chain(5, alphabet_size=2)
    result = compiler.verify_exact_affine(threshold, max_length=6)
    assert result["exact"]
    assert result["max_state_error"] == 0.0

    mixed = compiler.mixed_demo_sigma_chain()
    result = compiler.verify_exact_affine(mixed, max_length=5)
    assert result["exact"]
    assert result["max_state_error"] == 0.0


def test_sigma_description_grows_linearly_while_explicit_prefix_table_grows_exponentially():
    alphabet_size = 2
    costs = []
    for h in range(1, 9):
        cost = compiler.length_threshold_sigma_chain(h, alphabet_size).cost()
        costs.append(cost)
        assert cost.sigma_transition_entries == 2 * alphabet_size + (h - 1) * 4 * alphabet_size
        assert cost.explicit_cascade_transition_entries_same_components == (
            2 * alphabet_size * (2**h - 1)
        )
        assert cost.sigma_intercomponent_state_edges == max(0, h - 1)
        assert cost.all_prefix_intercomponent_state_edges == h * (h - 1) // 2

    assert costs[-1].sigma_transition_entries < costs[-1].explicit_cascade_transition_entries_same_components


def test_reset_lowering_is_singular_and_permutation_lowering_is_orthogonal():
    import numpy as np

    P, b = compiler.affine_lowering((1, 2, 0))
    assert np.allclose(P.T @ P, np.eye(3))
    assert np.allclose(b, 0.0)

    A, b = compiler.affine_lowering((2, 2, 2))
    assert np.linalg.matrix_rank(A) == 0
    assert np.allclose(b, np.array([0.0, 0.0, 1.0]))
