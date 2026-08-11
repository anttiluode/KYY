from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import numpy as np


def rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def prototypes(n: int) -> np.ndarray:
    if n < 3:
        raise ValueError("n must be >= 3")
    angles = 2.0 * math.pi * np.arange(n, dtype=np.float64) / n
    return np.stack((np.cos(angles), np.sin(angles)), axis=1)


def nearest_prototype(z: np.ndarray, proto: np.ndarray) -> int:
    # Equal-norm prototypes: max inner product == nearest Euclidean prototype.
    return int(np.argmax(proto @ z))


@dataclass(frozen=True)
class OrbitResult:
    n: int
    recurrent_real_dimension: int
    exact_cycle_state_error: float
    exact_reset_state_error: float
    minimum_pair_distance: float
    nearest_prototype_noise_radius: float
    unit_logit_margin: float
    predicted_noise_radius: float
    predicted_logit_margin: float


def probe(n: int) -> OrbitResult:
    proto = prototypes(n)
    theta = 2.0 * math.pi / n
    C = rotation(theta)
    v0 = proto[0]

    cycle_err = 0.0
    reset_err = 0.0
    for k in range(n):
        cycle_target = proto[(k + 1) % n]
        cycle_err = max(cycle_err, float(np.linalg.norm(C @ proto[k] - cycle_target)))

        # Affine reset z' = A_R z + b_R with A_R=0 and b_R=v0.
        reset_out = v0
        reset_err = max(reset_err, float(np.linalg.norm(reset_out - v0)))

    pair_distances = []
    for i in range(n):
        for j in range(i):
            pair_distances.append(float(np.linalg.norm(proto[i] - proto[j])))
    min_distance = min(pair_distances)

    # For nearest-prototype / equal-norm linear decoding, the Euclidean distance
    # from a prototype to its nearest decision hyperplane is sin(pi/n).
    noise_radius = math.sin(math.pi / n)
    logit_margin = 1.0 - math.cos(2.0 * math.pi / n)

    # Validate all orbit points decode to the intended state.
    for k in range(n):
        decoded = nearest_prototype(proto[k], proto)
        if decoded != k:
            raise AssertionError(f"prototype {k} decoded as {decoded}")

    return OrbitResult(
        n=n,
        recurrent_real_dimension=2,
        exact_cycle_state_error=cycle_err,
        exact_reset_state_error=reset_err,
        minimum_pair_distance=min_distance,
        nearest_prototype_noise_radius=noise_radius,
        unit_logit_margin=logit_margin,
        predicted_noise_radius=math.sin(math.pi / n),
        predicted_logit_margin=1.0 - math.cos(2.0 * math.pi / n),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Exact 2D group-orbit realization of C_n plus affine reset"
    )
    p.add_argument("--n", nargs="+", type=int, default=[3, 5, 10, 100, 1000])
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = [probe(n) for n in args.n]
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
        return

    print(" n | dim | cycle err | min pair dist | noise radius | unit logit margin")
    print("---+-----+-----------+---------------+--------------+------------------")
    for row in rows:
        print(
            f"{row.n:4d} | {row.recurrent_real_dimension:3d} | "
            f"{row.exact_cycle_state_error:9.2e} | {row.minimum_pair_distance:13.6g} | "
            f"{row.nearest_prototype_noise_radius:12.6g} | {row.unit_logit_margin:16.6g}"
        )


if __name__ == "__main__":
    main()
