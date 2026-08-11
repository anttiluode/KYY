from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_local(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


backend = load_local("metacircuit_backend_nonideal", ROOT / "map" / "metacircuit_cyclic_backend.py")
design = load_local("metacircuit_design_nonideal", ROOT / "map" / "metacircuit_frequency_design.py")


def ideal_theta(n: int, frequency: int) -> float:
    return 2.0 * math.pi * int(frequency) / int(n)


def ideal_ratio(n: int, frequency: int) -> float:
    return backend.required_admittance_over_fdnr(ideal_theta(n, frequency))


def ideal_initial_companion(n: int, frequency: int) -> np.ndarray:
    # phase=[1,0] and T h=phase gives h=[1, cos(theta)].
    theta = ideal_theta(n, frequency)
    return np.asarray([1.0, math.cos(theta)], dtype=np.float64)


def physical_features(
    n: int,
    frequencies: list[int],
    cycles: int,
    relative_ratio_error: np.ndarray | None = None,
    relative_radius_error: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return companion-state features for q+k*n increment histories.

    The ideal per-mode recurrence has roots exp(+-i theta).  A relative error in
    D^-1Y perturbs theta.  A separate pole-radius error r-1 is a generic toy for
    uncompensated loss/gain (Q/radius) mismatch; it is not claimed to be a full
    circuit model of the 2026 metacircuit.
    """
    m = len(frequencies)
    ratio_error = np.zeros(m) if relative_ratio_error is None else np.asarray(relative_ratio_error, dtype=np.float64)
    radius_error = np.zeros(m) if relative_radius_error is None else np.asarray(relative_radius_error, dtype=np.float64)
    if ratio_error.shape != (m,) or radius_error.shape != (m,):
        raise ValueError("nonideality vectors must have one value per character")

    times = np.asarray([q + k * n for q in range(n) for k in range(cycles)], dtype=np.float64)
    x = np.empty((len(times), 2 * m), dtype=np.float64)

    for j, f in enumerate(frequencies):
        ratio = ideal_ratio(n, f) * (1.0 + float(ratio_error[j]))
        cphi = 1.0 - ratio / 2.0
        if not (-1.0 < cphi < 1.0):
            raise ValueError(f"mode f={f} left the stable oscillatory interior: ratio={ratio}")
        phi = math.acos(cphi)
        sphi = math.sin(phi)
        r = 1.0 + float(radius_error[j])
        if r <= 0.0:
            raise ValueError("pole radius must remain positive")

        h0 = ideal_initial_companion(n, f)
        alpha = h0[0]
        # u_t = r^t [alpha cos(t phi) + beta sin(t phi)].
        # At t=-1, u_-1=r^-1[alpha cos(phi)-beta sin(phi)].
        beta = (alpha * math.cos(phi) - r * h0[1]) / sphi

        log_r = math.log(r)
        amp = np.exp(times * log_r)
        u_t = amp * (alpha * np.cos(times * phi) + beta * np.sin(times * phi))
        tm1 = times - 1.0
        amp_m1 = np.exp(tm1 * log_r)
        u_m1 = amp_m1 * (alpha * np.cos(tm1 * phi) + beta * np.sin(tm1 * phi))
        x[:, 2 * j] = u_t
        x[:, 2 * j + 1] = u_m1

    y = np.repeat(np.arange(n, dtype=np.int64), cycles)
    return x, y


def exact_companion_port(n: int, frequencies: list[int]) -> np.ndarray:
    """Exact matched-filter port in ideal companion coordinates, with bias column."""
    m = len(frequencies)
    port = np.zeros((n, 2 * m + 1), dtype=np.float64)
    for q in range(n):
        for j, f in enumerate(frequencies):
            theta = ideal_theta(n, f)
            phase_proto = np.asarray([math.cos(theta * q), math.sin(theta * q)])
            port[q, 2 * j : 2 * j + 2] = phase_proto @ backend.phase_from_companion(theta)
    return port


def fit_affine_port(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    xa = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    target = np.eye(n, dtype=np.float64)[y]
    beta = np.linalg.lstsq(xa, target, rcond=None)[0]
    return beta.T


def accuracy_margin(port: np.ndarray, x: np.ndarray, y: np.ndarray, chunk: int = 16384) -> tuple[float, float]:
    correct = 0
    minimum_margin = np.inf
    for start in range(0, len(x), chunk):
        xb = x[start : start + chunk]
        yb = y[start : start + chunk]
        xa = np.concatenate((xb, np.ones((len(xb), 1))), axis=1)
        logits = xa @ port.T
        pred = logits.argmax(axis=1)
        correct += int(np.sum(pred == yb))
        rows = np.arange(len(yb))
        true = logits[rows, yb]
        logits[rows, yb] = -np.inf
        minimum_margin = min(minimum_margin, float(np.min(true - logits.max(axis=1))))
    return float(correct / len(x)), float(minimum_margin)


def orthogonal_sensor(dim: int, seed: int) -> np.ndarray:
    """Pure invertible port-coordinate distortion; no body error."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q @ np.diag(signs)


def relation_defect_for_errors(n: int, frequencies: list[int], relative_ratio_error: np.ndarray) -> float:
    defects = []
    for f, eps in zip(frequencies, relative_ratio_error):
        ratio = ideal_ratio(n, f) * (1.0 + float(eps))
        a = np.asarray([[2.0 - ratio, -1.0], [1.0, 0.0]], dtype=np.float64)
        defects.append(np.linalg.norm(np.linalg.matrix_power(a, n) - np.eye(2), ord="fro"))
    return float(max(defects))


@dataclass
class HorizonStats:
    horizon_cycles: int
    mean_accuracy: float
    min_accuracy: float
    mean_min_margin: float


@dataclass
class SweepRow:
    relative_ratio_sigma: float
    mean_max_relation_defect: float
    horizons: list[HorizonStats]


@dataclass
class RadiusRow:
    relative_radius_sigma: float
    horizons: list[HorizonStats]


def summarize_trials(records: dict[int, list[tuple[float, float]]]) -> list[HorizonStats]:
    out = []
    for horizon in sorted(records):
        vals = records[horizon]
        out.append(
            HorizonStats(
                horizon_cycles=int(horizon),
                mean_accuracy=float(np.mean([v[0] for v in vals])),
                min_accuracy=float(np.min([v[0] for v in vals])),
                mean_min_margin=float(np.mean([v[1] for v in vals])),
            )
        )
    return out


def calibrated_trials(
    n: int,
    frequencies: list[int],
    train_cycles: int,
    horizons: list[int],
    trials: int,
    sigma: float,
    seed: int,
    sensor: np.ndarray,
    mode: str,
) -> tuple[list[HorizonStats], float]:
    records = {h: [] for h in horizons}
    defects = []
    m = len(frequencies)
    for trial in range(trials):
        rng = np.random.default_rng(seed + trial)
        z = rng.normal(size=m)
        ratio_error = sigma * z if mode == "ratio" else np.zeros(m)
        radius_error = sigma * z if mode == "radius" else np.zeros(m)
        defects.append(relation_defect_for_errors(n, frequencies, ratio_error))

        xt, yt = physical_features(n, frequencies, train_cycles, ratio_error, radius_error)
        port = fit_affine_port(xt @ sensor.T, yt, n)
        for horizon in horizons:
            x, y = physical_features(n, frequencies, horizon, ratio_error, radius_error)
            records[horizon].append(accuracy_margin(port, x @ sensor.T, y))
    return summarize_trials(records), float(np.mean(defects))


def sensor_only_control(n: int, frequencies: list[int], train_cycles: int, horizon: int, sensor: np.ndarray):
    xt, yt = physical_features(n, frequencies, train_cycles)
    x, y = physical_features(n, frequencies, horizon)
    original = exact_companion_port(n, frequencies)
    calibrated = fit_affine_port(xt @ sensor.T, yt, n)
    original_acc, original_margin = accuracy_margin(original, x @ sensor.T, y)
    calibrated_acc, calibrated_margin = accuracy_margin(calibrated, x @ sensor.T, y)
    return {
        "sensor_condition": float(np.linalg.cond(sensor)),
        "uncalibrated_accuracy": original_acc,
        "uncalibrated_min_margin": original_margin,
        "calibrated_accuracy": calibrated_acc,
        "calibrated_min_margin": calibrated_margin,
    }


def trimmed_control(n: int, frequencies: list[int], train_cycles: int, horizon: int, sensor: np.ndarray):
    xt, yt = physical_features(n, frequencies, train_cycles)
    port = fit_affine_port(xt @ sensor.T, yt, n)
    x, y = physical_features(n, frequencies, horizon)
    return accuracy_margin(port, x @ sensor.T, y)


def main():
    p = argparse.ArgumentParser(description="Separate port-only distortion from resonator-body relation errors")
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--condition-cap", type=float, default=2.0)
    p.add_argument("--train-cycles", type=int, default=16)
    p.add_argument("--horizons", nargs="+", type=int, default=[16, 64, 256, 1024])
    p.add_argument("--ratio-sigmas", nargs="+", type=float, default=[1e-5, 2e-5, 5e-5, 1e-4])
    p.add_argument("--radius-sigmas", nargs="+", type=float, default=[1e-5, 5e-5])
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--seed", type=int, default=4600)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    unconstrained = design.greedy_design(args.n, args.modes, None)
    conditioned = design.greedy_design(args.n, args.modes, args.condition_cap)
    banks = {
        "unconstrained": unconstrained.frequencies,
        "conditioned": conditioned.frequencies,
    }
    sensor = orthogonal_sensor(2 * args.modes, args.seed + 900)

    payload = {
        "config": vars(args),
        "banks": {
            "unconstrained": asdict(unconstrained),
            "conditioned": asdict(conditioned),
        },
        "sensor_only": {},
        "ratio_mismatch": {},
        "radius_mismatch_negative_control": {},
        "ideal_body_trim": {},
        "interpretation": {
            "sensor_only": "A static invertible output-coordinate distortion belongs to the port and should be absorbable by static calibration.",
            "ratio_mismatch": "Static D^-1Y mismatch changes recurrent phase/relation semantics; finite-horizon port recalibration is not body legalization.",
            "radius_mismatch": "Generic pole-radius/Q mismatch is a separate body nonideality; the frequency-condition cap is not expected to solve every hardware error.",
            "trim": "Exact ratio trim is an upper bound for relation-aware physical tuning, not a claim that a real circuit can be tuned perfectly.",
        },
    }

    for name, freqs in banks.items():
        payload["sensor_only"][name] = sensor_only_control(
            args.n, freqs, args.train_cycles, max(args.horizons), sensor
        )
        ratio_rows = []
        for sigma in args.ratio_sigmas:
            stats, defect = calibrated_trials(
                args.n, freqs, args.train_cycles, args.horizons, args.trials,
                sigma, args.seed, sensor, "ratio"
            )
            ratio_rows.append(asdict(SweepRow(float(sigma), defect, stats)))
        payload["ratio_mismatch"][name] = ratio_rows

        radius_rows = []
        for sigma in args.radius_sigmas:
            stats, _ = calibrated_trials(
                args.n, freqs, args.train_cycles, args.horizons, args.trials,
                sigma, args.seed + 200, sensor, "radius"
            )
            radius_rows.append(asdict(RadiusRow(float(sigma), stats)))
        payload["radius_mismatch_negative_control"][name] = radius_rows

        trim_acc, trim_margin = trimmed_control(
            args.n, freqs, args.train_cycles, max(args.horizons), sensor
        )
        payload["ideal_body_trim"][name] = {
            "horizon_cycles": int(max(args.horizons)),
            "accuracy": trim_acc,
            "min_margin": trim_margin,
        }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("banks:", banks)
        for name in banks:
            print(name, "sensor", payload["sensor_only"][name])
            for row in payload["ratio_mismatch"][name]:
                h = row["horizons"][-1]
                print(name, "ratio", row["relative_ratio_sigma"], "relation", row["mean_max_relation_defect"],
                      "Lmax", h["mean_accuracy"], h["min_accuracy"])
            for row in payload["radius_mismatch_negative_control"][name]:
                h = row["horizons"][-1]
                print(name, "radius", row["relative_radius_sigma"], "Lmax", h["mean_accuracy"], h["min_accuracy"])


if __name__ == "__main__":
    main()
