from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Import the exact task module rather than duplicating perm3's group table.
TASKS_PATH = ROOT / "kyy" / "tasks.py"
MODULE_NAME = "kyy_tasks_for_kernel_audit"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, TASKS_PATH)
assert SPEC is not None and SPEC.loader is not None
tasks = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = tasks
SPEC.loader.exec_module(tasks)


@dataclass(frozen=True)
class TransitionKernel:
    task: str
    token: int
    n_states: int
    image: list[int]
    rank: int
    kernel_blocks: list[list[int]]
    kernel_block_sizes: list[int]
    is_bijective: bool
    is_constant: bool
    is_idempotent: bool
    irreversible_merge_count: int
    primitive_hint: str


def exact_transition_table(task: str) -> list[list[int]]:
    """Return state x token -> next-state table for KYY's exact toy tasks."""
    if task == "parity":
        return [[0, 1], [1, 0]]
    if task == "mod3":
        return [[0, 1], [1, 2], [2, 0]]
    if task == "flipflop":
        # tokens: no-op, set-1, reset-0, toggle
        return [[0, 1, 0, 1], [1, 1, 0, 0]]
    if task == "perm3":
        return tasks._PERM_TABLE.tolist()
    if task == "permreset3":
        return [[0, 1, 0], [1, 2, 0], [2, 0, 0]]
    raise KeyError(task)


def kernel_partition(column: list[int]) -> list[list[int]]:
    fibers: dict[int, list[int]] = {}
    for state, target in enumerate(column):
        fibers.setdefault(int(target), []).append(int(state))
    return sorted((sorted(block) for block in fibers.values()), key=lambda b: (b[0], len(b)))


def analyze_transition(task: str, token: int) -> TransitionKernel:
    table = exact_transition_table(task)
    n = len(table)
    if not table or not 0 <= token < len(table[0]):
        raise ValueError("invalid token")
    column = [int(row[token]) for row in table]
    image = sorted(set(column))
    blocks = kernel_partition(column)
    rank = len(image)
    bijective = rank == n
    constant = rank == 1
    # Idempotence of this single state transformation: f(f(q))=f(q).
    idempotent = all(column[column[q]] == column[q] for q in range(n))
    merges = sum(len(block) - 1 for block in blocks)
    if bijective:
        hint = "permutation"
    elif constant:
        hint = "constant_reset"
    else:
        hint = "partial_merge"
    return TransitionKernel(
        task=task,
        token=int(token),
        n_states=n,
        image=image,
        rank=rank,
        kernel_blocks=blocks,
        kernel_block_sizes=[len(block) for block in blocks],
        is_bijective=bijective,
        is_constant=constant,
        is_idempotent=idempotent,
        irreversible_merge_count=int(merges),
        primitive_hint=hint,
    )


def audit_task(task: str) -> list[TransitionKernel]:
    table = exact_transition_table(task)
    return [analyze_transition(task, token) for token in range(len(table[0]))]


def audit_all() -> dict[str, list[TransitionKernel]]:
    return {name: audit_task(name) for name in tasks.TASKS}


def main() -> None:
    p = argparse.ArgumentParser(description="Classify exact task transitions by image/rank/kernel before choosing recurrent primitives")
    p.add_argument("--tasks", nargs="+", default=list(tasks.TASKS))
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    payload = {name: [asdict(x) for x in audit_task(name)] for name in args.tasks}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for name, rows in payload.items():
        print(name)
        for row in rows:
            print(
                f"  token {row['token']}: rank={row['rank']}/{row['n_states']} "
                f"kernel={row['kernel_block_sizes']} primitive={row['primitive_hint']} "
                f"idempotent={row['is_idempotent']}"
            )


if __name__ == "__main__":
    main()
