from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class TaskSpec:
    name: str
    vocab_size: int
    n_classes: int


TASKS: Dict[str, TaskSpec] = {
    "parity": TaskSpec("parity", vocab_size=2, n_classes=2),
    "mod3": TaskSpec("mod3", vocab_size=2, n_classes=3),
    "flipflop": TaskSpec("flipflop", vocab_size=4, n_classes=2),
    "perm3": TaskSpec("perm3", vocab_size=3, n_classes=6),
}


# Six permutations of (0, 1, 2). A state stores the current permutation.
_PERMS = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]
_PERM_TO_ID = {p: i for i, p in enumerate(_PERMS)}
_GENERATORS = [
    (0, 1, 2),  # identity / no-op
    (1, 0, 2),  # swap 0 <-> 1
    (1, 2, 0),  # cycle 0 -> 1 -> 2 -> 0
]


def _compose(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return left o right, both represented as image tuples."""
    return tuple(left[right[i]] for i in range(3))


_PERM_TABLE = torch.tensor(
    [[_PERM_TO_ID[_compose(_GENERATORS[g], p)] for g in range(3)] for p in _PERMS],
    dtype=torch.long,
)


def generate_batch(
    task: str,
    batch_size: int,
    length: int,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate integer tokens and the exact running state target at every step."""
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; choose from {sorted(TASKS)}")
    spec = TASKS[task]
    x = torch.randint(spec.vocab_size, (batch_size, length), device=device)
    y = torch.empty((batch_size, length), dtype=torch.long, device=device)

    if task == "parity":
        y.copy_(torch.cumsum(x, dim=1).remainder(2))
        return x, y

    if task == "mod3":
        y.copy_(torch.cumsum(x, dim=1).remainder(3))
        return x, y

    if task == "flipflop":
        # 0=no-op, 1=set, 2=reset, 3=toggle
        state = torch.zeros(batch_size, dtype=torch.long, device=device)
        for t in range(length):
            tok = x[:, t]
            state = torch.where(tok == 1, torch.ones_like(state), state)
            state = torch.where(tok == 2, torch.zeros_like(state), state)
            state = torch.where(tok == 3, 1 - state, state)
            y[:, t] = state
        return x, y

    if task == "perm3":
        # Non-commutative composition in S3. State 0 is identity.
        table = _PERM_TABLE.to(device)
        state = torch.zeros(batch_size, dtype=torch.long, device=device)
        for t in range(length):
            state = table[state, x[:, t]]
            y[:, t] = state
        return x, y

    raise AssertionError("unreachable")
