from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import numpy as np


def harmonic_prototypes(n: int, frequencies: np.ndarray | list[int]) -> np.ndarray:
    """Return n unit states in R^(2k) from k characters of C_n.

    State j is

        1/sqrt(k) [cos(2pi f_1 j/n), sin(...), ..., cos(2pi f_k j/n), sin(...)].

    Frequencies may repeat; this matches the iid character-sampling existence proof.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    f = np.asarray(frequencies, dtype=np.int64).reshape(-1)
    if f.size < 1:
        raise ValueError("at least one frequency is required")
    f = np.mod(f, n)
    j = np.arange(n, dtype=np.float64)[:, None]
    phase = 2.0 * math.pi * j * f[None, :] / n
    z = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
    return (z / math.sqrt(f.size)).reshape(n, 2 * f.size)


def cycle_operator(n: int, frequencies: np.ndarray | list[int], increment: int = 1) -> np.ndarray:
    """Exact-real-arithmetic block rotation implementing j -> j+increment mod n."""
    f = np.asarray(frequencies, dtype=np.int64).reshape(-1)
    k = f.size
    A = np.zeros((2 * k, 2 * k), dtype=np.float64)
    for idx, freq in enumerate(f):
        theta = 2.0 * math.pi * int(freq) * int(increment) / n
        c, s = math.cos(theta), math.sin(theta)
        A[2 * idx : 2 * idx + 2, 2 * idx : 2 * idx + 2] = np.array(
            [[c, -s], [s, c]], dtype=np.float64
        )
    return A


def shift_correlations(n: int, frequencies: np.ndarray | list[int]) -> np.ndarray:
    """Real inner product <v_0,v_delta> for every nonzero cyclic shift delta."""
    f = np.asarray(frequencies, dtype=np.int64).reshape(-1)
    delta = np.arange(1, n, dtype=np.float64)[:, None]
    phase = 2.0 * math.pi * delta * f[None, :] / n
    return np.mean(np.cos(phase), axis=1)


def geometric_metrics(n: int, frequencies: np.ndarray | list[int]) -> dict[str, float | int]:
    corr = shift_correlations(n, frequencies)
    max_corr = float(np.max(corr))
    min_dist = math.sqrt(max(0.0, 2.0 * (1.0 - max_corr)))
    return {
        "n_states": n,
        "complex_modes": int(np.asarray(frequencies).size),
        "real_dimension": int(2 * np.asarray(frequencies).size),
        "max_nontrivial_inner_product": max_corr,
        "minimum_pair_distance": min_dist,
        "nearest_prototype_noise_radius": 0.5 * min_dist,
    }


def verify_cycle(n: int, frequencies: np.ndarray | list[int]) -> float:
    v = harmonic_prototypes(n, frequencies)
    A = cycle_operator(n, frequencies, increment=1)
    pred = (A @ v.T).T
    target = np.roll(v, -1, axis=0)
    return float(np.max(np.abs(pred - target)))


def hoeffding_one_sided_failure_bound(n: int, k: int, alpha: float) -> float:
    """Union bound for max_delta <v_0,v_delta> >= alpha under iid uniform frequencies.

    For every nonzero delta, X_l=cos(2pi f_l delta/n) has E[X_l]=0 and lies in
    [-1,1]. Hoeffding gives P(mean X_l >= alpha) <= exp(-k alpha^2/2).
    Union over n-1 shifts.
    """
    if n < 2 or k < 1:
        raise ValueError("need n>=2 and k>=1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    return min(1.0, (n - 1) * math.exp(-0.5 * k * alpha * alpha))


def existence_k(n: int, alpha: float) -> int:
    """Smallest integer k for which the elementary union bound is strictly < 1.

    This is a sufficient existence bound, not an optimal dimension theorem.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    k = max(1, math.floor(2.0 * math.log(max(1, n - 1)) / (alpha * alpha)) + 1)
    while hoeffding_one_sided_failure_bound(n, k, alpha) >= 1.0:
        k += 1
    return k


def guaranteed_noise_radius(alpha: float) -> float:
    """If all nontrivial inner products <= alpha, half pair distance is this."""
    return math.sqrt((1.0 - alpha) / 2.0)


def random_search(
    n: int,
    k: int,
    trials: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Find a small-coherence frequency multiset by deterministic random search."""
    if trials < 1:
        raise ValueError("trials must be >= 1")
    rng = np.random.default_rng(seed)
    best_f: np.ndarray | None = None
    best_metrics: dict[str, float | int] | None = None
    best_corr = float("inf")
    for _ in range(trials):
        f = rng.integers(0, n, size=k, dtype=np.int64)
        metrics = geometric_metrics(n, f)
        corr = float(metrics["max_nontrivial_inner_product"])
        if corr < best_corr:
            best_corr = corr
            best_f = f.copy()
            best_metrics = metrics
    assert best_f is not None and best_metrics is not None
    return best_f, best_metrics


@dataclass(frozen=True)
class ComparisonRow:
    n: int
    method: str
    real_dimension: int
    noise_radius: float
    extra: dict[str, object]


def compare(n: int, k: int, trials: int, seed: int) -> list[ComparisonRow]:
    # Single-frequency phase orbit.
    phase_radius = math.sin(math.pi / n)

    # Harmonic-frame search.
    f, hm = random_search(n, k, trials=trials, seed=seed)

    # Regular simplex reference: N unit vectors in R^(N-1), half pair distance.
    simplex_radius = math.sqrt(n / (2.0 * (n - 1)))

    return [
        ComparisonRow(
            n=n,
            method="single_phase",
            real_dimension=2,
            noise_radius=phase_radius,
            extra={"frequency": 1},
        ),
        ComparisonRow(
            n=n,
            method="harmonic_random_search",
            real_dimension=int(hm["real_dimension"]),
            noise_radius=float(hm["nearest_prototype_noise_radius"]),
            extra={
                "frequencies": f.tolist(),
                "max_nontrivial_inner_product": hm["max_nontrivial_inner_product"],
                "trials": trials,
                "seed": seed,
                "cycle_error": verify_cycle(n, f),
            },
        ),
        ComparisonRow(
            n=n,
            method="regular_simplex",
            real_dimension=n - 1,
            noise_radius=simplex_radius,
            extra={"reference_only": True},
        ),
    ]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cyclic harmonic-frame state tracker: exact rotations versus dimension/margin"
    )
    p.add_argument("--n", nargs="+", type=int, default=[31, 101, 1009])
    p.add_argument("--k", type=int, default=16, help="number of complex Fourier modes")
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    payload: dict[str, object] = {
        "alpha": args.alpha,
        "guaranteed_noise_radius_if_bound_met": guaranteed_noise_radius(args.alpha),
        "rows": {},
    }
    for n in args.n:
        payload["rows"][str(n)] = {
            "comparison": [asdict(r) for r in compare(n, args.k, args.trials, args.seed + n)],
            "existence_bound": {
                "complex_modes_k": existence_k(n, args.alpha),
                "real_dimension": 2 * existence_k(n, args.alpha),
                "union_failure_bound": hoeffding_one_sided_failure_bound(
                    n, existence_k(n, args.alpha), args.alpha
                ),
            },
        }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        f"one-sided target alpha={args.alpha}; guaranteed radius >= "
        f"{guaranteed_noise_radius(args.alpha):.6g}"
    )
    for n in args.n:
        rows = payload["rows"][str(n)]
        print(f"\nC_{n}")
        for row in rows["comparison"]:
            print(
                f"  {row['method']:24s} d={row['real_dimension']:4d} "
                f"radius={row['noise_radius']:.6g}"
            )
        eb = rows["existence_bound"]
        print(
            f"  elementary existence bound: k={eb['complex_modes_k']} "
            f"(real d={eb['real_dimension']}), union bound={eb['union_failure_bound']:.4g}"
        )


if __name__ == "__main__":
    main()
