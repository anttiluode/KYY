from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "task_transition_kernel_audit_for_tests"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "task_transition_kernel_audit.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = probe
SPEC.loader.exec_module(probe)


def test_permreset3_cycle_and_reset_get_different_primitives() -> None:
    rows = probe.audit_task("permreset3")
    assert [x.primitive_hint for x in rows] == ["permutation", "permutation", "constant_reset"]
    assert rows[1].rank == 3
    assert rows[1].kernel_block_sizes == [1, 1, 1]
    assert rows[2].rank == 1
    assert rows[2].kernel_blocks == [[0, 1, 2]]
    assert rows[2].is_idempotent


def test_flipflop_has_two_permutations_and_two_constant_writes() -> None:
    rows = probe.audit_task("flipflop")
    assert [x.primitive_hint for x in rows] == [
        "permutation",
        "constant_reset",
        "constant_reset",
        "permutation",
    ]
    assert rows[1].image == [1]
    assert rows[2].image == [0]


def test_group_tasks_have_discrete_kernels() -> None:
    for task in ("parity", "mod3", "perm3"):
        for row in probe.audit_task(task):
            assert row.is_bijective
            assert row.rank == row.n_states
            assert all(size == 1 for size in row.kernel_block_sizes)
            assert row.irreversible_merge_count == 0


def test_partial_merge_classification_on_synthetic_column() -> None:
    blocks = probe.kernel_partition([0, 0, 2, 2, 4])
    assert blocks == [[0, 1], [2, 3], [4]]
