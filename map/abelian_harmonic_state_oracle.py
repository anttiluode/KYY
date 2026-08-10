from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass

import numpy as np


GroupElement = tuple[int, ...]
Frequency = tuple[int, ...]


def group_order(moduli: tuple[int, ...]) -> int:
    if not moduli or any(n < 2 for n in moduli):
        raise ValueError("moduli must be a nonempty tuple with every n>=2")
    return math.prod(moduli)


def elements(moduli: tuple[int, ...]) -> list[GroupElement]:
    group_order(moduli)
    return list(itertools.product(*(range(n) for n in moduli)))


def normalize_element(x: GroupElement, moduli: tuple[int, ...]) -> GroupElement:
    if len(x) != len(moduli):
        raise ValueError("element length mismatch")
    return tuple(int(a) % n for a, n in zip(x, moduli))


def add(a: GroupElement, b: GroupElement, moduli: tuple[int, ...]) -> GroupElement:
    a = normalize_element(a, moduli)
    b = normalize_element(b, moduli)
    return tuple((x + y) % n for x, y, n in zip(a, b, moduli))


def character_phase(f: Frequency, g: GroupElement, moduli: tuple[int, ...]) -> float:
    if len(f) != len(moduli) or len(g) != len(moduli):
        raise ValueError("frequency/element length mismatch")
    return 2.0 * math.pi * sum((fi % n) * (gi % n) / n for fi, gi, n in zip(f, g, moduli))


def random_frequencies(
    moduli: tuple[int, ...],
    k: int,
    rng: np.random.Generator,
) -> list[Frequency]:
    if k < 1:
        raise ValueError("k must be >=1")
    return [tuple(int(rng.integers(0, n)) for n in moduli) for _ in range(k)]


def prototypes(
    moduli: tuple[int, ...],
    frequencies: list[Frequency],
) -> tuple[list[GroupElement], np.ndarray]:
    if not frequencies:
        raise ValueError("need at least one frequency")
    states = elements(moduli)
    k = len(frequencies)
    rows = []
    for g in states:
        blocks = []
        for f in frequencies:
            phase = character_phase(f, g, moduli)
            blocks.extend((math.cos(phase), math.sin(phase)))
        rows.append(np.asarray(blocks, dtype=np.float64) / math.sqrt(k))
    return states, np.stack(rows, axis=0)


def token_operator(
    moduli: tuple[int, ...],
    frequencies: list[Frequency],
    token: GroupElement,
) -> np.ndarray:
    token = normalize_element(token, moduli)
    k = len(frequencies)
    A = np.zeros((2 * k, 2 * k), dtype=np.float64)
    for i, f in enumerate(frequencies):
        theta = character_phase(f, token, moduli)
        c, s = math.cos(theta), math.sin(theta)
        A[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = np.array(
            [[c, -s], [s, c]], dtype=np.float64
        )
    return A


def verify_group_action(
    moduli: tuple[int, ...],
    frequencies: list[Frequency],
) -> float:
    states, proto = prototypes(moduli, frequencies)
    index = {g: i for i, g in enumerate(states)}
    max_error = 0.0
    for token in states:
        A = token_operator(moduli, frequencies, token)
        for i, g in enumerate(states):
            target = add(g, token, moduli)
            err = float(np.max(np.abs(A @ proto[i] - proto[index[target]])))
            max_error = max(max_error, err)
    return max_error


def max_nontrivial_inner_product(
    moduli: tuple[int, ...],
    frequencies: list[Frequency],
) -> float:
    identity = tuple(0 for _ in moduli)
    vals = []
    for delta in elements(moduli):
        if delta == identity:
            continue
        vals.append(
            sum(math.cos(character_phase(f, delta, moduli)) for f in frequencies)
            / len(frequencies)
        )
    return max(vals)


def metrics(moduli: tuple[int, ...], frequencies: list[Frequency]) -> dict[str, float | int]:
    mu = max_nontrivial_inner_product(moduli, frequencies)
    min_dist = math.sqrt(max(0.0, 2.0 * (1.0 - mu)))
    return {
        "group_order": group_order(moduli),
        "complex_modes": len(frequencies),
        "real_dimension": 2 * len(frequencies),
        "max_nontrivial_inner_product": mu,
        "minimum_pair_distance": min_dist,
        "nearest_prototype_noise_radius": 0.5 * min_dist,
    }


def random_search(
    moduli: tuple[int, ...],
    k: int,
    trials: int,
    seed: int,
) -> tuple[list[Frequency], dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    best_f: list[Frequency] | None = None
    best_metrics: dict[str, float | int] | None = None
    best_mu = float("inf")
    for _ in range(trials):
        f = random_frequencies(moduli, k, rng)
        m = metrics(moduli, f)
        mu = float(m["max_nontrivial_inner_product"])
        if mu < best_mu:
            best_mu = mu
            best_f = f
            best_metrics = m
    assert best_f is not None and best_metrics is not None
    return best_f, best_metrics


def hoeffding_failure_bound(order: int, k: int, alpha: float) -> float:
    if order < 2 or k < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid arguments")
    return min(1.0, (order - 1) * math.exp(-0.5 * k * alpha * alpha))


def existence_k(order: int, alpha: float) -> int:
    k = max(1, math.floor(2.0 * math.log(order - 1) / (alpha * alpha)) + 1)
    while hoeffding_failure_bound(order, k, alpha) >= 1.0:
        k += 1
    return k


@dataclass(frozen=True)
class GroupRow:
    moduli: tuple[int, ...]
    order: int
    complex_modes: int
    real_dimension: int
    max_nontrivial_inner_product: float
    noise_radius: float
    action_error: float
    frequencies: list[Frequency]


def probe(
    moduli: tuple[int, ...],
    k: int,
    trials: int,
    seed: int,
) -> GroupRow:
    f, m = random_search(moduli, k, trials, seed)
    return GroupRow(
        moduli=moduli,
        order=group_order(moduli),
        complex_modes=k,
        real_dimension=2 * k,
        max_nontrivial_inner_product=float(m["max_nontrivial_inner_product"]),
        noise_radius=float(m["nearest_prototype_noise_radius"]),
        action_error=verify_group_action(moduli, f),
        frequencies=f,
    )


def parse_group(spec: str) -> tuple[int, ...]:
    parts = tuple(int(x) for x in spec.lower().replace("c", "").split("x") if x)
    group_order(parts)
    return parts


def main() -> None:
    p = argparse.ArgumentParser(
        description="Finite-abelian harmonic-frame recurrent state oracle"
    )
    p.add_argument(
        "--groups",
        nargs="+",
        default=["C31", "C4xC5", "C3xC3xC3"],
        help="products such as C31, C4xC5, C3xC3xC3",
    )
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--trials", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = []
    for i, spec in enumerate(args.groups):
        moduli = parse_group(spec)
        row = probe(moduli, args.k, args.trials, args.seed + i)
        rows.append(
            {
                "result": asdict(row),
                "existence_bound": {
                    "alpha": args.alpha,
                    "complex_modes_k": existence_k(row.order, args.alpha),
                    "real_dimension": 2 * existence_k(row.order, args.alpha),
                },
            }
        )

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    for item in rows:
        r = item["result"]
        e = item["existence_bound"]
        print(
            f"G={tuple(r['moduli'])}, |G|={r['order']:4d}, "
            f"search d={r['real_dimension']:3d}, radius={r['noise_radius']:.4f}, "
            f"action_err={r['action_error']:.2e}, "
            f"existence d(alpha={e['alpha']})={e['real_dimension']}"
        )


if __name__ == "__main__":
    main()
