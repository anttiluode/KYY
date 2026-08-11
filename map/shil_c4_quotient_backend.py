from __future__ import annotations

import argparse
import json
import math

import numpy as np

TAU = 2.0 * math.pi


def wrap(x: np.ndarray) -> np.ndarray:
    return (x + math.pi) % TAU - math.pi


def c4_phase(q: np.ndarray) -> np.ndarray:
    return (np.asarray(q, dtype=np.float64) % 4.0) * (math.pi / 2.0)


def decode_c4(phi: np.ndarray) -> np.ndarray:
    return (np.floor(((np.asarray(phi) % TAU) + math.pi / 4.0) / (math.pi / 2.0)).astype(np.int64)) % 4


def merge_symbolic(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.int64) % 4
    return np.where(q < 2, 0, 2)


def deterministic_merge_margin(alpha: float) -> float:
    """Worst angular margin for C4 pair-merge via N=2 well alpha then C4 relock.

    For 0<alpha<pi/4:
      - q=1's distance to the N=2 basin boundary is alpha;
      - q=0's N=2 input margin is pi/2-alpha (never limiting here);
      - merged well alpha's distance to the C4 q=0 separatrix is pi/4-alpha.
    By symmetry the other pair has the same margins.
    """
    a = float(alpha)
    if not (0.0 < a < math.pi / 4.0):
        return 0.0
    return min(a, math.pi / 2.0 - a, math.pi / 4.0 - a)


def best_alpha_grid(points: int = 10001) -> tuple[float, float]:
    xs = np.linspace(1e-9, math.pi / 4.0 - 1e-9, points)
    margins = np.asarray([deterministic_merge_margin(x) for x in xs])
    i = int(np.argmax(margins))
    return float(xs[i]), float(margins[i])


def relax_phase(
    phi: np.ndarray,
    n_wells: int,
    alpha: float,
    kappa: float,
    diffusion: float,
    dt: float,
    steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Euler-Maruyama phase reduction for an N-SHIL-like locking potential."""
    p = np.asarray(phi, dtype=np.float64).copy()
    noise_scale = math.sqrt(2.0 * diffusion * dt)
    for _ in range(int(steps)):
        drift = -kappa * np.sin(n_wells * (p - alpha))
        p += drift * dt
        if diffusion > 0.0:
            p += noise_scale * rng.normal(size=p.shape)
        p = p % TAU
    return p


def one_merge_trial(
    alpha: float,
    diffusion: float,
    samples_per_state: int,
    seed: int,
    kappa: float = 4.0,
    dt: float = 0.002,
    merge_time: float = 0.5,
    relock_time: float = 0.5,
    initial_jitter: float = 0.0,
) -> dict:
    rng = np.random.default_rng(seed)
    q = np.repeat(np.arange(4, dtype=np.int64), samples_per_state)
    phi = c4_phase(q)
    if initial_jitter > 0.0:
        phi = (phi + rng.normal(scale=initial_jitter, size=phi.shape)) % TAU

    merge_steps = max(1, int(round(merge_time / dt)))
    relock_steps = max(1, int(round(relock_time / dt)))

    phi_mid = relax_phase(phi, 2, alpha, kappa, diffusion, dt, merge_steps, rng)
    # Return to the original four-well digital code.
    phi_out = relax_phase(phi_mid, 4, 0.0, kappa, diffusion, dt, relock_steps, rng)
    pred = decode_c4(phi_out)
    target = merge_symbolic(q)
    per_state = {str(i): float(np.mean(pred[q == i] == target[q == i])) for i in range(4)}
    return {
        "alpha": float(alpha),
        "deterministic_margin": deterministic_merge_margin(alpha),
        "accuracy": float(np.mean(pred == target)),
        "per_source_state_accuracy": per_state,
        "mean_midpoint_distance_to_target_two_well": float(
            np.mean(np.minimum(np.abs(wrap(phi_mid - alpha)), np.abs(wrap(phi_mid - (alpha + math.pi)))))
        ),
    }


def alpha_sweep(
    diffusion: float,
    alphas: list[float],
    samples_per_state: int,
    seed: int,
    **kwargs,
) -> list[dict]:
    return [
        one_merge_trial(a, diffusion, samples_per_state, seed + i * 101, **kwargs)
        for i, a in enumerate(alphas)
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Compile a C4 partial merge into a two-well SHIL phase potential")
    p.add_argument("--diffusions", nargs="+", type=float, default=[0.0, 0.002, 0.005, 0.01, 0.02])
    p.add_argument("--samples-per-state", type=int, default=4000)
    p.add_argument("--seed", type=int, default=4800)
    p.add_argument("--kappa", type=float, default=4.0)
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--merge-time", type=float, default=0.5)
    p.add_argument("--relock-time", type=float, default=0.5)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    analytic_alpha = math.pi / 8.0
    grid_alpha, grid_margin = best_alpha_grid()
    compare_alphas = [
        math.pi / 16.0,
        analytic_alpha,
        3.0 * math.pi / 16.0,
        math.pi / 4.0,
    ]

    payload = {
        "config": vars(args),
        "analytic_compiler": {
            "optimal_alpha": analytic_alpha,
            "optimal_alpha_degrees": 22.5,
            "worst_basin_margin": math.pi / 8.0,
            "grid_optimal_alpha": grid_alpha,
            "grid_optimal_margin": grid_margin,
            "reason": "maximize min(alpha, pi/4-alpha) for pair capture versus C4 re-entry; optimum alpha=pi/8",
        },
        "sweeps": {},
        "interpretation": {
            "digital": "C4 basin identity is the discrete state carried by a continuous oscillator phase.",
            "merge": "Switching temporarily from four phase wells to two shifted wells implements the quotient {0,1}->0 and {2,3}->2 after returning to four-well locking.",
            "midpoint_failure": "alpha=pi/4 puts the merged two-well attractors on C4 separatrices; restoring four-well locking makes the final state noise-selected rather than deterministic.",
            "prior_art": "Multi-phase SHIL/Potts oscillator hardware is prior art. This probe only instantiates the KYY partial-merge contract in that known phase-potential geometry.",
        },
    }
    for i, d in enumerate(args.diffusions):
        payload["sweeps"][str(d)] = alpha_sweep(
            d,
            compare_alphas,
            args.samples_per_state,
            args.seed + i * 10000,
            kappa=args.kappa,
            dt=args.dt,
            merge_time=args.merge_time,
            relock_time=args.relock_time,
        )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
