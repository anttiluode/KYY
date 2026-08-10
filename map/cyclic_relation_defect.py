from __future__ import annotations

import argparse
import json
import math

import numpy as np


def wrapped_phase(x: np.ndarray) -> np.ndarray:
    """Wrap radians to [-pi, pi)."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def mode_relation_defects(n: int, angles: np.ndarray | list[float], wraps: int = 1) -> np.ndarray:
    """Per-mode ||R(theta)^(n*wraps) - I||_2 defects.

    For a 2D rotation R(phi), ||R(phi)-I||_2 = 2 |sin(phi/2)|.
    Exact C_n characters theta=2*pi*f/n therefore give zero at every integer wrap.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if wraps < 1:
        raise ValueError("wraps must be >= 1")
    a = np.asarray(angles, dtype=np.float64).reshape(-1)
    if a.size < 1:
        raise ValueError("at least one angle is required")
    phi = float(n * wraps) * a
    return 2.0 * np.abs(np.sin(0.5 * phi))


def operator_relation_defect(n: int, angles: np.ndarray | list[float], wraps: int = 1) -> float:
    """Spectral norm of A^(n*wraps)-I for a block-diagonal rotary bank."""
    return float(np.max(mode_relation_defects(n, angles, wraps=wraps)))


def state_wrap_defect(n: int, angles: np.ndarray | list[float], wraps: int = 1) -> float:
    """Distance between normalized phase-bank states S and S+n*wraps.

    With k equal-amplitude modes, this is

        2 * sqrt(mean_i sin^2(n*wraps*theta_i/2)).

    It is independent of the starting count S.
    """
    d = mode_relation_defects(n, angles, wraps=wraps)
    return float(np.sqrt(np.mean(d * d)))


def nearest_character_residuals(n: int, angles: np.ndarray | list[float]) -> np.ndarray:
    """Wrapped angular residual delta_i from the nearest exact C_n character."""
    a = np.asarray(angles, dtype=np.float64).reshape(-1)
    f = np.rint(n * a / (2.0 * math.pi))
    exact = 2.0 * math.pi * f / n
    return wrapped_phase(a - exact)


def small_defect_wrap_slope(n: int, angles: np.ndarray | list[float]) -> float:
    """First-order state-defect growth per symbolic wrap.

    If delta_i is the angular error from the nearest exact character, then for
    q*n*|delta_i| << 1,

        D(q) ~= q * n * sqrt(mean_i delta_i^2).
    """
    delta = nearest_character_residuals(n, angles)
    return float(n * np.sqrt(np.mean(delta * delta)))


def exact_character_angles(n: int, frequencies: np.ndarray | list[int]) -> np.ndarray:
    f = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    return 2.0 * math.pi * f / n


def payload(n: int, angles: np.ndarray, max_wraps: int) -> dict[str, object]:
    return {
        "n": n,
        "angles": angles.tolist(),
        "operator_relation_defect_one_wrap": operator_relation_defect(n, angles),
        "state_wrap_defect_one_wrap": state_wrap_defect(n, angles),
        "small_defect_wrap_slope": small_defect_wrap_slope(n, angles),
        "wrap_curve": {
            str(q): {
                "operator": operator_relation_defect(n, angles, wraps=q),
                "state": state_wrap_defect(n, angles, wraps=q),
            }
            for q in range(1, max_wraps + 1)
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Algebraic relation-defect audit for a cyclic rotary state code")
    p.add_argument("--n", type=int, default=31)
    p.add_argument("--frequencies", nargs="+", type=int, default=[1, 3, 7, 11])
    p.add_argument("--angle-error", type=float, default=0.0, help="additive radians per unit increment on every mode")
    p.add_argument("--max-wraps", type=int, default=16)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    angles = exact_character_angles(args.n, args.frequencies) + args.angle_error
    out = payload(args.n, angles, args.max_wraps)
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    print(f"C_{args.n}; modes={len(args.frequencies)}; eta={args.angle_error:g}")
    print(f"one-wrap operator defect = {out['operator_relation_defect_one_wrap']:.8g}")
    print(f"one-wrap state defect    = {out['state_wrap_defect_one_wrap']:.8g}")
    print(f"small-error slope/wrap   = {out['small_defect_wrap_slope']:.8g}")
    for q, row in out["wrap_curve"].items():
        print(f"wraps={int(q):3d} operator={row['operator']:.8g} state={row['state']:.8g}")


if __name__ == "__main__":
    main()
