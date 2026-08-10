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
MODULE_NAME = "cyclic_harmonic_state_oracle_for_drift"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "cyclic_harmonic_state_oracle.py"
)
assert SPEC is not None and SPEC.loader is not None
harmonic = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = harmonic
SPEC.loader.exec_module(harmonic)


def defect_operator(
    n: int,
    frequencies: np.ndarray | list[int],
    phase_error: float,
) -> np.ndarray:
    """Increment operator with the same additive angle defect on every 2D block."""
    f = np.asarray(frequencies, dtype=np.int64).reshape(-1)
    k = f.size
    A = np.zeros((2 * k, 2 * k), dtype=np.float64)
    for idx, freq in enumerate(f):
        theta = 2.0 * math.pi * int(freq) / n + phase_error
        c, s = math.cos(theta), math.sin(theta)
        A[2 * idx : 2 * idx + 2, 2 * idx : 2 * idx + 2] = np.array(
            [[c, -s], [s, c]], dtype=np.float64
        )
    return A


def common_defect_distance(steps: int, phase_error: float) -> float:
    """Exact ideal-vs-defective state distance under common block phase error."""
    return 2.0 * abs(math.sin(0.5 * steps * phase_error))


def safe_steps_from_radius(radius: float, phase_error: float) -> int:
    """Largest initial token count guaranteed by the nearest-prototype radius.

    This is the first monotone lobe only; it is the useful finite-horizon bound,
    not a claim that later periodic re-entry cannot occur.
    """
    eta = abs(phase_error)
    if eta == 0.0:
        return 2**63 - 1
    if not 0.0 < radius <= 1.0:
        raise ValueError("radius must lie in (0,1]")
    boundary = 2.0 * math.asin(min(1.0, radius / 2.0)) / eta
    # Require strict distance < radius.  Subtract a tiny numerical guard before floor.
    return max(0, math.ceil(boundary - 1e-12) - 1)


def nearest_state(z: np.ndarray, prototypes: np.ndarray) -> int:
    # All prototypes have equal norm; nearest Euclidean == maximum dot product.
    return int(np.argmax(prototypes @ z))


def first_decoder_failure(
    n: int,
    frequencies: np.ndarray | list[int],
    phase_error: float,
    max_steps: int,
) -> int | None:
    """Return first t>=1 whose defective rollout decodes to the wrong C_n state."""
    proto = harmonic.harmonic_prototypes(n, frequencies)
    A_bad = defect_operator(n, frequencies, phase_error)
    z = proto[0].copy()
    for t in range(1, max_steps + 1):
        z = A_bad @ z
        expected = t % n
        if nearest_state(z, proto) != expected:
            return t
    return None


@dataclass(frozen=True)
class DriftRow:
    n: int
    method: str
    real_dimension: int
    noise_radius: float
    phase_error_per_block_rad: float
    guaranteed_safe_steps: int
    first_decoder_failure: int | None
    frequencies: list[int]


def compare(
    n: int,
    harmonic_k: int,
    phase_error: float,
    search_trials: int,
    seed: int,
    max_steps: int,
) -> list[DriftRow]:
    single_f = np.array([1], dtype=np.int64)
    single_metrics = harmonic.geometric_metrics(n, single_f)

    f, hm = harmonic.random_search(n, harmonic_k, trials=search_trials, seed=seed)

    rows = []
    for name, freqs, metrics in (
        ("single_phase", single_f, single_metrics),
        ("harmonic", f, hm),
    ):
        radius = float(metrics["nearest_prototype_noise_radius"])
        rows.append(
            DriftRow(
                n=n,
                method=name,
                real_dimension=int(metrics["real_dimension"]),
                noise_radius=radius,
                phase_error_per_block_rad=phase_error,
                guaranteed_safe_steps=safe_steps_from_radius(radius, phase_error),
                first_decoder_failure=first_decoder_failure(
                    n, freqs, phase_error, max_steps=max_steps
                ),
                frequencies=np.asarray(freqs, dtype=int).tolist(),
            )
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase-defect horizon: single phase versus harmonic cyclic state code"
    )
    p.add_argument("--n", nargs="+", type=int, default=[31, 101, 1009])
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--phase-error", type=float, default=1e-4)
    p.add_argument("--search-trials", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    payload = {
        str(n): [
            asdict(row)
            for row in compare(
                n,
                harmonic_k=args.k,
                phase_error=args.phase_error,
                search_trials=args.search_trials,
                seed=args.seed + n,
                max_steps=args.max_steps,
            )
        ]
        for n in args.n
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"common per-step block phase defect eta={args.phase_error:g} rad")
    for n in args.n:
        print(f"\nC_{n}")
        for row in payload[str(n)]:
            print(
                f"  {row['method']:12s} d={row['real_dimension']:4d} "
                f"radius={row['noise_radius']:.6g} "
                f"guaranteed={row['guaranteed_safe_steps']:6d} "
                f"first_fail={row['first_decoder_failure']}"
            )


if __name__ == "__main__":
    main()
