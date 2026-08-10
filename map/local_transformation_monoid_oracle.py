from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import asdict, dataclass


Transformation = tuple[int, ...]


def identity(n: int) -> Transformation:
    return tuple(range(n))


def compose(left: Transformation, right: Transformation) -> Transformation:
    """Return left o right for transformations represented by image tuples."""
    return tuple(left[right[i]] for i in range(len(right)))


def swap(n: int, i: int, j: int) -> Transformation:
    out = list(range(n))
    out[i], out[j] = out[j], out[i]
    return tuple(out)


def merge(n: int, source: int, target: int) -> Transformation:
    """Rank-(n-1) idempotent: source -> target, all other states fixed."""
    out = list(range(n))
    out[source] = target
    return tuple(out)


def cycle(n: int) -> Transformation:
    return tuple((i + 1) % n for i in range(n))


def bfs_word_lengths(n: int, generators: list[Transformation]) -> dict[Transformation, int]:
    """Exact right-Cayley BFS from identity in the generated transformation monoid."""
    start = identity(n)
    distance = {start: 0}
    queue: deque[Transformation] = deque([start])
    while queue:
        current = queue.popleft()
        depth = distance[current] + 1
        for generator in generators:
            nxt = compose(generator, current)
            if nxt not in distance:
                distance[nxt] = depth
                queue.append(nxt)
    return distance


def all_local_parallel_layers(n: int, all_pinches: bool) -> list[Transformation]:
    """All non-identity layers of disjoint path-local operations.

    Every selected edge may perform a swap.  If all_pinches=True, it may instead
    perform either directed merge.  If False, the only irreversible primitive is
    the single fixed pinch 0->1.
    """
    layers: set[Transformation] = set()

    def rec(edge: int, current: Transformation) -> None:
        if edge >= n - 1:
            layers.add(current)
            return

        # Leave this edge unused.
        rec(edge + 1, current)

        # Use a swap; skip the neighboring edge to keep the layer disjoint.
        rec(edge + 2, compose(swap(n, edge, edge + 1), current))

        if all_pinches:
            rec(edge + 2, compose(merge(n, edge, edge + 1), current))
            rec(edge + 2, compose(merge(n, edge + 1, edge), current))
        elif edge == 0:
            rec(edge + 2, compose(merge(n, 0, 1), current))

    rec(0, identity(n))
    layers.discard(identity(n))
    return sorted(layers)


def summary(distances: dict[Transformation, int]) -> dict[str, object]:
    hist = Counter(distances.values())
    return {
        "size": len(distances),
        "max_depth": max(distances.values()),
        "mean_depth": sum(distances.values()) / len(distances),
        "depth_histogram": dict(sorted(hist.items())),
    }


@dataclass(frozen=True)
class Row:
    n: int
    full_transformation_monoid_size: int
    global_three_generators: dict[str, object]
    path_swaps_plus_one_pinch: dict[str, object]
    path_swaps_plus_all_local_pinches: dict[str, object]
    parallel_path_one_pinch: dict[str, object]
    parallel_path_all_pinches: dict[str, object]


def measure(n: int) -> Row:
    if n < 2:
        raise ValueError("n must be >= 2")

    # Classical rank-3 style generating set: two generators of S_n plus one
    # defect-1 map.  The n-cycle is not a local hardware primitive; this row is
    # the algebraic generator-count reference.
    global_three = [cycle(n), swap(n, 0, 1), merge(n, 0, 1)]

    # Strictly local path generators.  Adjacent swaps generate S_n; conjugating
    # the one defect-1 map by those permutations is enough to recover arbitrary
    # singular behavior, but may require long words.
    adjacent_swaps = [swap(n, i, i + 1) for i in range(n - 1)]
    one_pinch = adjacent_swaps + [merge(n, 0, 1)]
    all_pinches = adjacent_swaps.copy()
    for i in range(n - 1):
        all_pinches.append(merge(n, i, i + 1))
        all_pinches.append(merge(n, i + 1, i))

    libraries = {
        "global_three": global_three,
        "one_pinch": one_pinch,
        "all_pinches": all_pinches,
        "parallel_one": all_local_parallel_layers(n, all_pinches=False),
        "parallel_all": all_local_parallel_layers(n, all_pinches=True),
    }
    distances = {name: bfs_word_lengths(n, gens) for name, gens in libraries.items()}
    expected = n**n
    for name, d in distances.items():
        if len(d) != expected:
            raise AssertionError(
                f"{name} generated {len(d)} transformations, expected full T_{n} size {expected}"
            )

    return Row(
        n=n,
        full_transformation_monoid_size=expected,
        global_three_generators=summary(distances["global_three"]),
        path_swaps_plus_one_pinch=summary(distances["one_pinch"]),
        path_swaps_plus_all_local_pinches=summary(distances["all_pinches"]),
        parallel_path_one_pinch=summary(distances["parallel_one"]),
        parallel_path_all_pinches=summary(distances["parallel_all"]),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Exact small-n resource floor for local reversible transport + rank-lowering pinch"
    )
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument(
        "--n-max",
        type=int,
        default=5,
        help="BFS scales as n^n; values above 6 become expensive quickly",
    )
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = [measure(n) for n in range(args.n_min, args.n_max + 1)]
    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2, sort_keys=True))
        return

    print(
        " n | |T_n| | global3 max | path+1 pinch max | path+all pinch max "
        "| parallel+1 max | parallel+all max"
    )
    print("---+-------+-------------+------------------+--------------------+----------------+-----------------")
    for r in rows:
        print(
            f"{r.n:2d} | {r.full_transformation_monoid_size:5d} | "
            f"{r.global_three_generators['max_depth']:11d} | "
            f"{r.path_swaps_plus_one_pinch['max_depth']:16d} | "
            f"{r.path_swaps_plus_all_local_pinches['max_depth']:18d} | "
            f"{r.parallel_path_one_pinch['max_depth']:14d} | "
            f"{r.parallel_path_all_pinches['max_depth']:15d}"
        )


if __name__ == "__main__":
    main()
