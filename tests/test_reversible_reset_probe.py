from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reversible_reset_probe", ROOT / "map" / "reversible_reset_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_permutations_preserve_full_rank():
    for n in range(2, 8):
        monoid = probe.monoid_closure(n, [probe.cycle(n)])
        assert len(monoid) == n
        assert probe.rank_histogram(monoid) == {n: n}


def test_cycle_plus_reset_generates_rotations_and_constants():
    for n in range(2, 8):
        monoid = probe.monoid_closure(n, [probe.cycle(n), probe.reset(n, 0)])
        assert len(monoid) == 2 * n
        assert probe.rank_histogram(monoid) == {1: n, n: n}
        assert probe.rank_monotonicity_holds(monoid)


def test_visible_reset_can_hide_old_state_in_reversible_ancilla():
    for n in range(2, 8):
        swap = probe.reversible_swap_with_ancilla(n)
        assert probe.is_permutation(swap)
        for q in range(n):
            src = q * n
            dst = swap[src]
            visible, ancilla = divmod(dst, n)
            assert visible == 0
            assert ancilla == q


def test_probe_summary_n3():
    result = probe.probe(3)
    assert result.permutation_only_size == 3
    assert result.permutation_only_rank_histogram == {3: 3}
    assert result.cycle_plus_reset_size == 6
    assert result.cycle_plus_reset_rank_histogram == {1: 3, 3: 3}
    assert result.ancilla_swap_is_permutation
    assert result.visible_reset_with_blank_ancilla
    assert result.old_state_retained_in_ancilla
