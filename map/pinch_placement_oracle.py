from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass


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
        next_depth = distance[current] + 1
        for generator in generators:
            nxt = compose(generator, current)
            if nxt not in distance:
                distance[nxt] = next_depth
                queue.append(nxt)
    return distance


def parallel_layers(n: int, pinch: tuple[int, int]) -> list[Transformation]:
    """All nonempty disjoint path-local layers with one fixed directed pinch site."""
    source, target = pinch
    if abs(source - target) != 1:
        raise ValueError("pinch must lie on one path edge")
    pinch_edge = min(source, target)
    layers: set[Transformation] = set()

    def rec(edge: int, current: Transformation) -> None:
        if edge >= n - 1:
            layers.add(current)
            return

        # edge unused
        rec(edge + 1, current)

        # local reversible swap; consuming this edge prevents use of neighbor
        rec(edge + 2, compose(swap(n, edge, edge + 1), current))

        # the one physical irreversible site
        if edge == pinch_edge:
            rec(edge + 2, compose(merge(n, source, target), current))

    rec(0, identity(n))
    layers.discard(identity(n))
    return list(layers)


def path_sequential_library(n: int, pinch: tuple[int, int]) -> list[Transformation]:
    source, target = pinch
    return [swap(n, i, i + 1) for i in range(n - 1)] + [merge(n, source, target)]


def summarize(distance: dict[Transformation, int]) -> dict[str, float | int]:
    values = list(distance.values())
    return {
        "size": len(values),
        "max_depth": max(values),
        "mean_depth": sum(values) / len(values),
    }


@dataclass(frozen=True)
class PlacementRow:
    n: int
    source: int
    target: int
    edge: int
    sequential: dict[str, float | int]
    parallel: dict[str, float | int]


def measure_placement(n: int, source: int, target: int) -> PlacementRow:
    expected = n**n
    seq = bfs(n, path_sequential_library(n, (source, target)))
    par = bfs(n, parallel_layers(n, (source, target)))
    if len(seq) != expected or len(par) != expected:
        raise AssertionError(
            f"pinch {source}->{target} did not generate T_{n}: seq={len(seq)}, par={len(par)}, expected={expected}"
        )
    return PlacementRow(
        n=n,
        source=source,
        target=target,
        edge=min(source, target),
        sequential=summarize(seq),
        parallel=summarize(par),
    )


def all_one_pinch_placements(n: int) -> list[PlacementRow]:
    rows = []
    for edge in range(n - 1):
        rows.append(measure_placement(n, edge, edge + 1))
        rows.append(measure_placement(n, edge + 1, edge))
    return rows


def best_rows(rows: list[PlacementRow], mode: str) -> list[PlacementRow]:
    if mode not in {"sequential", "parallel"}:
        raise ValueError("mode must be sequential or parallel")
    best = min(int(getattr(row, mode)["max_depth"]) for row in rows)
    return [row for row in rows if int(getattr(row, mode)["max_depth"]) == best]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Exact small-n placement cost for one directed irreversible pinch on a path"
    )
    p.add_argument("--n-min", type=int, default=3)
    p.add_argument("--n-max", type=int, default=6)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    payload = []
    for n in range(args.n_min, args.n_max + 1):
        rows = all_one_pinch_placements(n)
        payload.append(
            {
                "n": n,
                "full_transformation_monoid_size": n**n,
                "placements": [asdict(row) for row in rows],
                "best_sequential": [asdict(row) for row in best_rows(rows, "sequential")],
                "best_parallel": [asdict(row) for row in best_rows(rows, "parallel")],
            }
        )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    for item in payload:
        n = item["n"]
        print(f"\nn={n}, |T_n|={item['full_transformation_monoid_size']}")
        print(" pinch | seq max mean | par max mean")
        print("-------+--------------+-------------")
        for row in item["placements"]:
            s = row["sequential"]
            q = row["parallel"]
            print(
                f" {row['source']}->{row['target']}   | "
                f"{s['max_depth']:3d} {s['mean_depth']:6.3f} | "
                f"{q['max_depth']:3d} {q['mean_depth']:6.3f}"
            )
        print(
            "best sequential:",
            [(r["source"], r["target"], r["sequential"]["max_depth"]) for r in item["best_sequential"]],
        )
        print(
            "best parallel:",
            [(r["source"], r["target"], r["parallel"]["max_depth"]) for r in item["best_parallel"]],
        )


if __name__ == "__main__":
    main()
