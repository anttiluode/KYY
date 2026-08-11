from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


def max_robust_radius_volume_bound(n_states: int, dimension: int, radius: float = 1.0) -> float:
    """Elementary Euclidean volume-packing upper bound on robust radius epsilon.

    If N state points lie in a d-ball of radius R and closed epsilon-balls around
    them are disjoint, those small balls fit inside the concentric ball of radius
    R+epsilon.  Hence N*epsilon^d <= (R+epsilon)^d and

        epsilon <= R / (N^(1/d) - 1).

    This is a necessary condition, not generally achievable.
    """
    if n_states < 2:
        raise ValueError("n_states must be >= 2")
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    if radius <= 0:
        raise ValueError("radius must be positive")
    return radius / (n_states ** (1.0 / dimension) - 1.0)


def minimum_dimension_volume_bound(
    n_states: int,
    robust_radius: float,
    radius: float = 1.0,
) -> int:
    """Necessary dimension from N <= (1 + R/epsilon)^d."""
    if n_states < 2:
        raise ValueError("n_states must be >= 2")
    if robust_radius <= 0 or radius <= 0:
        raise ValueError("radii must be positive")
    denom = math.log1p(radius / robust_radius)
    return max(1, math.ceil(math.log(n_states) / denom - 1e-12))


def packing_capacity_bits(dimension: int, robust_radius: float, radius: float = 1.0) -> float:
    """Log2 of the elementary volume-packing upper bound on distinguishable states.

    This is a geometric resolution budget, not literal ADC/thermodynamic bits.
    """
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    if robust_radius <= 0 or radius <= 0:
        raise ValueError("radii must be positive")
    return dimension * math.log2(1.0 + radius / robust_radius)


def cyclic_orbit_radius(n_states: int, radius: float = 1.0) -> float:
    return radius * math.sin(math.pi / n_states)


def scalar_counter_radius(n_states: int, radius: float = 0.5) -> float:
    """Optimal robust radius for N equally spaced points in [-R,R]."""
    return radius / (n_states - 1)


def simplex_radius(n_states: int, radius: float = 1.0) -> float:
    """Half pair distance for regular N-vertex simplex on sphere radius R."""
    if n_states < 2:
        raise ValueError("n_states must be >= 2")
    return radius * math.sqrt(n_states / (2.0 * (n_states - 1)))


@dataclass(frozen=True)
class Example:
    name: str
    n_states: int
    dimension: int
    radius: float
    robust_radius: float
    volume_bound_radius: float
    capacity_bits_upper_bound: float
    state_information_bits: float
    packing_slack_ratio: float


def make_example(
    name: str,
    n_states: int,
    dimension: int,
    radius: float,
    robust_radius: float,
) -> Example:
    bound = max_robust_radius_volume_bound(n_states, dimension, radius)
    return Example(
        name=name,
        n_states=n_states,
        dimension=dimension,
        radius=radius,
        robust_radius=robust_radius,
        volume_bound_radius=bound,
        capacity_bits_upper_bound=packing_capacity_bits(dimension, robust_radius, radius),
        state_information_bits=math.log2(n_states),
        packing_slack_ratio=bound / robust_radius,
    )


def standard_examples(n: int) -> list[Example]:
    if n < 3:
        raise ValueError("n must be >= 3")
    return [
        make_example(
            "normalized_scalar_counter",
            n_states=n,
            dimension=1,
            radius=0.5,
            robust_radius=scalar_counter_radius(n, radius=0.5),
        ),
        make_example(
            "2d_cyclic_orbit",
            n_states=n,
            dimension=2,
            radius=1.0,
            robust_radius=cyclic_orbit_radius(n, radius=1.0),
        ),
        make_example(
            "regular_simplex",
            n_states=n,
            dimension=n - 1,
            radius=1.0,
            robust_radius=simplex_radius(n, radius=1.0),
        ),
    ]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Elementary bounded-state dimension/precision packing floor"
    )
    p.add_argument("--n", nargs="+", type=int, default=[10, 100, 1000])
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    payload = {str(n): [asdict(x) for x in standard_examples(n)] for n in args.n}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    for n in args.n:
        print(f"\nN={n}, information={math.log2(n):.4f} bits")
        print(" representation             d     eps        bound      cap-bits   slack")
        print("-------------------------  ----  ---------  ---------  ---------  -------")
        for x in standard_examples(n):
            print(
                f"{x.name:25s} {x.dimension:4d}  {x.robust_radius:9.3g}  "
                f"{x.volume_bound_radius:9.3g}  {x.capacity_bits_upper_bound:9.3f}  "
                f"{x.packing_slack_ratio:7.3f}"
            )


if __name__ == "__main__":
    main()
