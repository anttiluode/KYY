from __future__ import annotations

import argparse
import json
import math

import numpy as np


def phase_bank(n: int, frequencies: list[int], q: int, t: np.ndarray) -> np.ndarray:
    """Concatenated planar character coordinates at state offset q and time t."""
    cols = []
    for f in frequencies:
        phase = 2.0 * math.pi * f * (t + q) / n
        cols.extend([np.cos(phase), np.sin(phase)])
    return np.stack(cols, axis=-1)


def integrated_quadratic_energy(n: int, frequencies: list[int], w: np.ndarray, q: int, window: int) -> float:
    t = np.arange(window, dtype=np.float64)
    h = phase_bank(n, frequencies, q, t)
    y = h @ np.asarray(w, dtype=np.float64)
    return float(np.sum(y * y))


def full_period_energies(n: int, frequencies: list[int], w: np.ndarray) -> np.ndarray:
    return np.asarray([integrated_quadratic_energy(n, frequencies, w, q, n) for q in range(n)])


def instantaneous_prototypes(n: int, frequencies: list[int]) -> np.ndarray:
    """Exact current-state prototypes; row q is the phase bank at t=0."""
    return np.stack([phase_bank(n, frequencies, q, np.asarray([0.0]))[0] for q in range(n)])


def instantaneous_accuracy(n: int, frequencies: list[int]) -> float:
    proto = instantaneous_prototypes(n, frequencies)
    logits = proto @ proto.T
    return float(np.mean(logits.argmax(axis=1) == np.arange(n)))


def partial_window_spread(n: int, frequencies: list[int], w: np.ndarray, window: int) -> float:
    vals = np.asarray([integrated_quadratic_energy(n, frequencies, w, q, window) for q in range(n)])
    return float(vals.max() - vals.min())


def main() -> None:
    p = argparse.ArgumentParser(description="Audit energy readout against exact cyclic phase-state semantics")
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--frequencies", nargs="+", type=int, default=[16,18,19,20,25,28,30,31])
    p.add_argument("--seed", type=int, default=4701)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    w = rng.normal(size=2 * len(args.frequencies))
    full = full_period_energies(args.n, args.frequencies, w)
    payload = {
        "config": vars(args),
        "instantaneous_exact_prototype_accuracy": instantaneous_accuracy(args.n, args.frequencies),
        "full_period_energy_min": float(full.min()),
        "full_period_energy_max": float(full.max()),
        "full_period_energy_spread": float(full.max() - full.min()),
        "partial_window_spread": {
            "1": partial_window_spread(args.n, args.frequencies, w, 1),
            "8": partial_window_spread(args.n, args.frequencies, w, 8),
            "16": partial_window_spread(args.n, args.frequencies, w, 16),
            str(args.n): partial_window_spread(args.n, args.frequencies, w, args.n),
        },
        "interpretation": {
            "full_period": "For h_q(t)=h_0(t+q), summing any fixed quadratic output over a complete n-sample period is invariant to q because q only permutes the summation index.",
            "partial_window": "A truncated energy window can be phase-sensitive, but then the output depends on the chosen observation window and is not a time-translation-invariant terminal-state decoder.",
            "backend": "The 2026 metacircuit's integrated output-energy classifier is therefore not a natural direct port for KYY cyclic phase-state tracking without a phase-sensitive readout modification.",
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
