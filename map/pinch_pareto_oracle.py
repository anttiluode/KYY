from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from itertools import combinations


Transformation = tuple[int, ...]


def identity(n: int) -> Transformation:
    return tuple(range(n))


def compose(left: Transformation, right: Transformation) -> Transformation:
    return tuple(left[right[i]] for i in range(len(right)))


def swap(n: int, i: int, j: int) -> Transformation:
    out = list(range(n))
    out[i], out[j] = out[j], out[i]
    return tuple(out)


def merge(n: int, source: int, target: int) -> Transformation:
    out = list(range(n))
    out[source] = target
    return tuple(out)


def bfs(n: int, generators: list[Transformation]) -> dict[Transformation, int]:
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


def sequential_library(n: int, write_edges: tuple[int, ...]) -> list[Transformation]:
    """Adjacent swaps everywhere; bidirectional defect-1 merges only at write edges."""
    generators = [swap(n, i, i + 1) for i in range(n - 1)]
    for edge in write_edges:
        generators.append(merge(n, edge, edge + 1))
        generators.append(merge(n, edge + 1, edge))
    return generators


def parallel_layers(n: int, write_edges: tuple[int, ...]) -> list[Transformation]:
    """All nonempty layers of disjoint local swaps and allowed bidirectional pinches."""
    write = set(write_edges)
    layers: set[Transformation] = set()

    def rec(edge: int, current: Transformation) -> None:
        if edge >= n - 1:
            layers.add(current)
            return

        # Edge unused.
        rec(edge + 1, current)

        # Reversible local swap.
        rec(edge + 2, compose(swap(n, edge, edge + 1), current))

        # One physical write site can be driven in either merge direction.
        if edge in write:
            rec(edge + 2, compose(merge(n, edge, edge + 1), current))
            rec(edge + 2, compose(merge(n, edge + 1, edge), current))

    rec(0, identity(n))
    layers.discard(identity(n))
    return list(layers)


def summarize(distance: dict[Transformation, int]) -> dict[str, float | int]:
    values = list(distance.values())
    return {
        "size": len(values),
        "max_depth": max(values),
        "mean_depth": sum(values) / len(values),
    }


@dataclass(frozen=True)
class Placement:
    n: int
    write_edges: tuple[int, ...]
    sequential: dict[str, float | int]
    parallel: dict[str, float | int]


def measure(n: int, write_edges: tuple[int, ...]) -> Placement:
    if not write_edges:
        raise ValueError("at least one write edge is required to generate singular maps")
    if len(set(write_edges)) != len(write_edges):
        raise ValueError("write_edges must be unique")
    if min(write_edges) < 0 or max(write_edges) >= n - 1:
        raise ValueError("write edge out of path range")

    seq = bfs(n, sequential_library(n, write_edges))
    par = bfs(n, parallel_layers(n, write_edges))
    expected = n**n
    if len(seq) != expected or len(par) != expected:
        raise AssertionError(
            f"write_edges={write_edges} generated seq={len(seq)}, par={len(par)}, expected={expected}"
        )
    return Placement(
        n=n,
        write_edges=write_edges,
        sequential=summarize(seq),
        parallel=summarize(par),
    )


def all_placements(n: int, k: int) -> list[Placement]:
    if not 1 <= k <= n - 1:
        raise ValueError("k must satisfy 1 <= k <= n-1")
    return [measure(n, edges) for edges in combinations(range(n - 1), k)]


def best(rows: list[Placement], mode: str) -> Placement:
    if mode not in {"sequential", "parallel"}:
        raise ValueError("mode must be sequential or parallel")
    return min(
        rows,
        key=lambda row: (
            int(getattr(row, mode)["max_depth"]),
            float(getattr(row, mode)["mean_depth"]),
            row.write_edges,
        ),
    )


def pareto_rows(n: int) -> list[dict[str, object]]:
    out = []
    for k in range(1, n):
        rows = all_placements(n, k)
        seq = best(rows, "sequential")
        par = best(rows, "parallel")
        out.append(
            {
                "write_site_count": k,
                "best_sequential": asdict(seq),
                "best_parallel": asdict(par),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Exact small-n Pareto curve: local write-site count/placement versus transformation depth"
    )
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = pareto_rows(args.n)
    if args.json:
        print(json.dumps({"n": args.n, "rows": rows}, indent=2, sort_keys=True))
        return

    print(f"n={args.n}, |T_n|={args.n ** args.n}")
    print(" sites | best seq edges  max  mean | best par edges  max  mean")
    print("-------+---------------------------+--------------------------")
    for item in rows:
        seq = item["best_sequential"]
        par = item["best_parallel"]
        print(
            f" {item['write_site_count']:5d} | "
            f"{str(tuple(seq['write_edges'])):14s} {seq['sequential']['max_depth']:3d} "
            f"{seq['sequential']['mean_depth']:6.3f} | "
            f"{str(tuple(par['write_edges'])):14s} {par['parallel']['max_depth']:3d} "
            f"{par['parallel']['mean_depth']:6.3f}"
        )


if __name__ == "__main__":
    main()
