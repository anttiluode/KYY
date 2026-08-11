from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict

import numpy as np


def states(n: int, delta: float, cycles: int) -> tuple[np.ndarray, np.ndarray]:
    """Physical oscillator states for each symbolic q across winding histories.

    One nominal increment uses theta=2*pi/n+delta.  A symbolic state q can be
    reached after t=q+k*n increments.  If delta != 0, the physical body retains
    the winding count k even though the symbolic machine has forgotten it.
    """
    theta = 2.0 * math.pi / n + delta
    xs, ys = [], []
    for q in range(n):
        for k in range(cycles):
            t = q + k * n
            phase = t * theta
            xs.append([math.cos(phase), math.sin(phase)])
            ys.append(q)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64)


def fit_least_squares_port(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Fit an affine static port by least squares to one-hot class targets."""
    xa = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    target = np.eye(n, dtype=np.float64)[y]
    # xa @ beta ~= target; returned rows are class affine coefficients.
    beta = np.linalg.lstsq(xa, target, rcond=None)[0]
    return beta.T


def predict_port(port: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xa = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    logits = xa @ port.T
    pred = logits.argmax(axis=1)
    return pred, logits


def accuracy_margin(port: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    pred, logits = predict_port(port, x)
    rows = np.arange(len(y))
    true = logits[rows, y]
    competitor = logits.copy()
    competitor[rows, y] = -np.inf
    margin = true - competitor.max(axis=1)
    return float(np.mean(pred == y)), float(np.min(margin))


def min_interclass_distance(x: np.ndarray, y: np.ndarray) -> float:
    best = np.inf
    for i in range(len(x)):
        mask = y != y[i]
        d = np.linalg.norm(x[mask] - x[i], axis=1)
        if len(d):
            best = min(best, float(np.min(d)))
    return float(best)


def exact_prototype_port(n: int) -> np.ndarray:
    rows = []
    for q in range(n):
        phase = 2.0 * math.pi * q / n
        rows.append([math.cos(phase), math.sin(phase), 0.0])
    return np.asarray(rows, dtype=np.float64)


@dataclass
class HorizonRow:
    cycles: int
    original_port_accuracy: float
    original_port_min_margin: float
    calibrated_port_accuracy: float
    calibrated_port_min_margin: float
    min_interclass_distance: float
    legalized_accuracy: float
    legalized_min_margin: float


def run(n: int, delta: float, train_cycles: int, horizons: list[int]):
    xt, yt = states(n, delta, train_cycles)
    calibrated = fit_least_squares_port(xt, yt, n)
    original = exact_prototype_port(n)
    rows = []
    for cycles in horizons:
        x, y = states(n, delta, cycles)
        oa, om = accuracy_margin(original, x, y)
        ca, cm = accuracy_margin(calibrated, x, y)
        xl, yl = states(n, 0.0, cycles)
        la, lm = accuracy_margin(original, xl, yl)
        rows.append(
            HorizonRow(
                cycles=cycles,
                original_port_accuracy=oa,
                original_port_min_margin=om,
                calibrated_port_accuracy=ca,
                calibrated_port_min_margin=cm,
                min_interclass_distance=min_interclass_distance(x, y),
                legalized_accuracy=la,
                legalized_min_margin=lm,
            )
        )
    return calibrated, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--train-cycles", type=int, default=16)
    p.add_argument("--horizons", nargs="+", type=int, default=[16, 64, 256, 1024, 4096])
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    port, rows = run(args.n, args.delta, args.train_cycles, args.horizons)
    payload = {
        "config": vars(args),
        "cycle_relation_residual_phase": args.n * args.delta,
        "calibrated_port": port.tolist(),
        "results": [asdict(r) for r in rows],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("cycles original_acc calibrated_acc calibrated_margin min_interclass legalized_acc")
        for r in rows:
            print(f"{r.cycles:6d} {r.original_port_accuracy:.6f} {r.calibrated_port_accuracy:.6f} "
                  f"{r.calibrated_port_min_margin:+.3e} {r.min_interclass_distance:.3e} "
                  f"{r.legalized_accuracy:.6f}")


if __name__ == "__main__":
    main()
