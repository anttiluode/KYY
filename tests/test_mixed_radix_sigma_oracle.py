from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "mixed_radix_sigma_oracle"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "mixed_radix_sigma_oracle.py"
)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = oracle
SPEC.loader.exec_module(oracle)


def test_increment_visits_all_mixed_radix_states_before_return():
    for p, q in ((3, 5), (5, 7), (4, 9)):
        assert oracle.verify_increment_cycle(p, q)
        state = (0, 0)
        values = []
        for _ in range(p * q):
            values.append(oracle.symbolic_value(state, p, q))
            state = oracle.symbolic_step(state, oracle.INC_TOKEN, p, q)
        assert values == list(range(p * q))
        assert state == (0, 0)


def test_reset_erases_both_digits():
    p, q = 7, 5
    for low in range(p):
        for high in range(q):
            assert oracle.symbolic_step((low, high), oracle.RESET_TOKEN, p, q) == (0, 0)


def test_harmonic_factor_equal_norm_radius_accounts_for_two_unit_cells():
    resources, detail = oracle.compare_resources(11, 7, modes_per_factor=6, trials=200, seed=0)
    by_name = {r.name: r for r in resources}
    factor = by_name["sigma_factor_harmonic"]
    assert detail["increment_cycle_verified"]
    assert math.isclose(
        factor.nearest_prototype_radius_equal_total_norm,
        factor.nearest_prototype_radius_native_scale / math.sqrt(2.0),
        rel_tol=1e-12,
    )
    assert factor.intercell_state_edges == 1
    assert factor.sufficient_structured_carry_bits == 1


def test_same_harmonic_mode_budget_does_not_automatically_make_factorization_win():
    # Deterministic search point used as a guard against accidentally comparing
    # unequal state norm or unequal mode budgets.  This is not a universal theorem.
    resources, _ = oracle.compare_resources(31, 29, modes_per_factor=8, trials=500, seed=0)
    by_name = {r.name: r for r in resources}
    mono = by_name["monolithic_harmonic"]
    fact = by_name["sigma_factor_harmonic"]
    assert mono.real_dimension == fact.real_dimension == 32
    assert mono.total_state_norm == 1.0
    assert math.isclose(fact.total_state_norm, math.sqrt(2.0), rel_tol=1e-12)
    # At equal total norm, this particular monolithic code is better packed.
    assert mono.nearest_prototype_radius_equal_total_norm > fact.nearest_prototype_radius_equal_total_norm
