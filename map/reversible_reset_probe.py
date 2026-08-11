from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import asdict, dataclass


Transformation = tuple[int, ...]


def identity(n: int) -> Transformation:
    return tuple(range(n))


def compose(left: Transformation, right: Transformation) -> Transformation:
    """Return left o right for finite transformations represented by image tuples."""
    if len(left) != len(right):
        raise ValueError("transformations must act on the same state set")
    return tuple(left[right[i]] for i in range(len(right)))


def transformation_rank(t: Transformation) -> int:
    return len(set(t))


def is_permutation(t: Transformation) -> bool:
    return transformation_rank(t) == len(t)


def cycle(n: int) -> Transformation:
    if n < 2:
        raise ValueError("cycle needs n >= 2")
    return tuple((i + 1) % n for i in range(n))


def reset(n: int, target: int = 0) -> Transformation:
    if not 0 <= target < n:
        raise ValueError("reset target out of range")
    return tuple(target for _ in range(n))


def monoid_closure(n: int, generators: list[Transformation]) -> set[Transformation]:
    """Exact finite closure under composition, including identity."""
    for g in generators:
        if len(g) != n:
            raise ValueError("generator size mismatch")
    seen = {identity(n)}
    queue: deque[Transformation] = deque([identity(n)])
    while queue:
        a = queue.popleft()
        for g in generators:
            for candidate in (compose(g, a), compose(a, g)):
                if candidate not in seen:
                    seen.add(candidate)
                    queue.append(candidate)
    return seen


def rank_histogram(monoid: set[Transformation]) -> dict[int, int]:
    counts = Counter(transformation_rank(t) for t in monoid)
    return dict(sorted(counts.items()))


def rank_monotonicity_holds(monoid: set[Transformation]) -> bool:
    for a in monoid:
        for b in monoid:
            if transformation_rank(compose(a, b)) > min(
                transformation_rank(a), transformation_rank(b)
            ):
                return False
    return True


def reversible_swap_with_ancilla(n: int) -> Transformation:
    """Permutation on Q x A implementing (q,a) -> (a,q).

    If the ancilla starts at 0, then (q,0) -> (0,q): the visible coordinate
    looks reset to zero, but the old q is retained exactly in the ancilla.
    """
    size = n * n
    out = [0] * size
    for q in range(n):
        for a in range(n):
            src = q * n + a
            dst = a * n + q
            out[src] = dst
    return tuple(out)


@dataclass(frozen=True)
class ProbeResult:
    n: int
    permutation_only_size: int
    permutation_only_rank_histogram: dict[int, int]
    cycle_plus_reset_size: int
    cycle_plus_reset_rank_histogram: dict[int, int]
    mixed_rank_monotonicity_holds: bool
    ancilla_swap_is_permutation: bool
    visible_reset_with_blank_ancilla: bool
    old_state_retained_in_ancilla: bool


def probe(n: int = 3) -> ProbeResult:
    c = cycle(n)
    r = reset(n, 0)

    permutation_only = monoid_closure(n, [c])
    mixed = monoid_closure(n, [c, r])

    swap = reversible_swap_with_ancilla(n)
    ancilla_ok = is_permutation(swap)
    visible_reset = True
    retained = True
    for q in range(n):
        src = q * n + 0
        dst = swap[src]
        visible, ancilla = divmod(dst, n)
        visible_reset &= visible == 0
        retained &= ancilla == q

    return ProbeResult(
        n=n,
        permutation_only_size=len(permutation_only),
        permutation_only_rank_histogram=rank_histogram(permutation_only),
        cycle_plus_reset_size=len(mixed),
        cycle_plus_reset_rank_histogram=rank_histogram(mixed),
        mixed_rank_monotonicity_holds=rank_monotonicity_holds(mixed),
        ancilla_swap_is_permutation=ancilla_ok,
        visible_reset_with_blank_ancilla=visible_reset,
        old_state_retained_in_ancilla=retained,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Exact finite probe: reversible permutations versus rank-reducing reset"
    )
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    result = probe(args.n)
    payload = asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"n={result.n}")
    print(
        "permutation-only closure: "
        f"size={result.permutation_only_size}, "
        f"rank_hist={result.permutation_only_rank_histogram}"
    )
    print(
        "cycle + reset closure: "
        f"size={result.cycle_plus_reset_size}, "
        f"rank_hist={result.cycle_plus_reset_rank_histogram}"
    )
    print(f"rank monotonicity under composition: {result.mixed_rank_monotonicity_holds}")
    print(
        "reversible ancilla embedding: "
        f"permutation={result.ancilla_swap_is_permutation}, "
        f"visible_reset={result.visible_reset_with_blank_ancilla}, "
        f"old_state_retained={result.old_state_retained_in_ancilla}"
    )
    print()
    print(
        "Interpretation: a visible reset can be embedded in reversible dynamics only "
        "by moving the discarded distinction into hidden/ancillary state."
    )


if __name__ == "__main__":
    main()
