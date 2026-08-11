from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def companion_block(theta: float) -> np.ndarray:
    """Central-difference harmonic-oscillator state [u_t, u_{t-1}]."""
    c = math.cos(theta)
    return np.array([[2.0 * c, -1.0], [1.0, 0.0]], dtype=np.float64)


def phase_from_companion(theta: float, tol: float = 1e-12) -> np.ndarray:
    """p=[cos psi,sin psi] = T [u_t,u_{t-1}] for u_t=cos psi."""
    c, s = math.cos(theta), math.sin(theta)
    if abs(s) <= tol:
        raise ValueError("theta is at a companion-coordinate degeneracy (0 or pi mod 2pi)")
    return np.array([[1.0, 0.0], [-c / s, 1.0 / s]], dtype=np.float64)


def required_admittance_over_fdnr(theta: float, dt: float = 1.0) -> float:
    """For u[t+1]=(2-dt^2 D^-1 Y)u[t]-u[t-1], return scalar D^-1 Y."""
    return 2.0 * (1.0 - math.cos(theta)) / (dt * dt)


def prototype_margin(n: int, frequency: int) -> float:
    """Unit-amplitude one-character matched-filter minimum score gap."""
    f = int(frequency) % n
    gaps = [1.0 - math.cos(2.0 * math.pi * f * d / n) for d in range(1, n)]
    return float(min(gaps))


@dataclass
class ModeLowering:
    n: int
    frequency: int
    centered_degree: int
    theta: float
    relation_defect: float
    admittance_over_fdnr: float
    phase_map_condition: float
    phase_map_norm: float
    absolute_phase_sensitivity_to_ratio: float
    relative_phase_sensitivity_to_ratio: float
    positive_relative_stability_headroom: float
    one_mode_symbolic_margin: float
    stable_interior: bool
    degenerate: bool


def lower_mode(n: int, frequency: int, dt: float = 1.0, tol: float = 1e-10) -> ModeLowering:
    f = int(frequency) % int(n)
    theta = 2.0 * math.pi * f / n
    a = companion_block(theta)
    rel = float(np.linalg.norm(np.linalg.matrix_power(a, n) - np.eye(2), ord="fro"))
    ratio = required_admittance_over_fdnr(theta, dt)
    s_signed = math.sin(theta)
    s = abs(s_signed)
    degenerate = bool(s <= 1e-12)
    stability_limit = 4.0 / (dt * dt)
    if degenerate:
        cond = float("inf")
        tnorm = float("inf")
        abs_sens = float("inf")
        rel_sens = float("inf") if ratio > tol else 0.0
    else:
        t = phase_from_companion(theta)
        cond = float(np.linalg.cond(t))
        tnorm = float(np.linalg.norm(t, ord=2))
        # cos(theta)=1-dt^2*ratio/2, hence |d theta / d ratio|.
        abs_sens = float((dt * dt) / (2.0 * s))
        rel_sens = float(ratio * abs_sens)
        # Verify the exact similarity used for port transport.
        sim = t @ a @ np.linalg.inv(t)
        if np.linalg.norm(sim - rotation(theta), ord="fro") > 1e-9:
            raise AssertionError("companion-to-phase similarity failed")
    if ratio <= tol:
        headroom = float("inf")
    else:
        headroom = float(max(0.0, stability_limit - ratio) / ratio)
    return ModeLowering(
        n=int(n),
        frequency=f,
        centered_degree=int(min(f, n - f)),
        theta=float(theta),
        relation_defect=rel,
        admittance_over_fdnr=float(ratio),
        phase_map_condition=cond,
        phase_map_norm=tnorm,
        absolute_phase_sensitivity_to_ratio=abs_sens,
        relative_phase_sensitivity_to_ratio=rel_sens,
        positive_relative_stability_headroom=headroom,
        one_mode_symbolic_margin=prototype_margin(n, f),
        stable_interior=bool(ratio > tol and ratio < stability_limit - tol),
        degenerate=degenerate,
    )


def transport_phase_port_to_companion(weight_rows: np.ndarray, theta: float) -> np.ndarray:
    """If p=T h and logits=W_p p, physical companion port is W_h=W_p T."""
    return np.asarray(weight_rows, dtype=np.float64) @ phase_from_companion(theta)


def best_faithful_frequency(n: int, dt: float = 1.0) -> ModeLowering:
    rows = [lower_mode(n, f, dt) for f in range(1, n) if math.gcd(n, f) == 1]
    rows = [r for r in rows if not r.degenerate and r.stable_interior]
    return min(rows, key=lambda r: (r.phase_map_condition, abs(r.admittance_over_fdnr - 2.0)))


def audit_existing_c101(dt: float = 1.0):
    path = ROOT / "results" / "cyclic_c101_equivariant_port.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in payload["results"]:
        modes = [lower_mode(101, f, dt) for f in row["projected_frequencies"]]
        worst = max(modes, key=lambda x: x.phase_map_condition)
        tightest = min(modes, key=lambda x: x.positive_relative_stability_headroom)
        out.append(
            {
                "seed": int(row["seed"]),
                "frequencies": row["projected_frequencies"],
                "max_phase_map_condition": max(m.phase_map_condition for m in modes),
                "max_phase_map_norm": max(m.phase_map_norm for m in modes),
                "max_relative_phase_sensitivity": max(m.relative_phase_sensitivity_to_ratio for m in modes),
                "min_positive_relative_stability_headroom": min(m.positive_relative_stability_headroom for m in modes),
                "worst_frequency": worst.frequency,
                "worst_centered_degree": worst.centered_degree,
                "tightest_stability_frequency": tightest.frequency,
                "admittance_over_fdnr_range": [
                    min(m.admittance_over_fdnr for m in modes),
                    max(m.admittance_over_fdnr for m in modes),
                ],
                "max_relation_defect": max(m.relation_defect for m in modes),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Lower exact C_n characters to a second-order resonator companion form")
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--frequencies", nargs="+", type=int, default=[1, 4, 25, 49, 50])
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    modes = [lower_mode(args.n, f, args.dt) for f in args.frequencies]
    best = best_faithful_frequency(args.n, args.dt)
    payload = {
        "config": vars(args),
        "modes": [asdict(x) for x in modes],
        "best_faithful_frequency_by_companion_condition": asdict(best),
        "existing_c101_seed_audit": audit_existing_c101(args.dt) if args.n == 101 else [],
        "interpretation": {
            "physical_ratio": "D^-1 Y = 2(1-cos(theta))/dt^2",
            "port_transport": "W_companion = W_phase T(theta)",
            "absolute_phase_sensitivity": "|d theta / d(D^-1Y)| = dt^2/(2|sin(theta)|)",
            "relative_phase_sensitivity": "|(D^-1Y) d theta/d(D^-1Y)| = |tan(theta/2)| for this recurrence",
            "positive_stability_headroom": "(4/dt^2 - D^-1Y)/(D^-1Y)",
            "degeneracy": "sin(theta)=0 makes the displacement/lag companion representation non-diagonalizable at theta=0 or pi",
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("f degree margin D^-1Y cond(T) relsens +headroom relation")
        for x in modes:
            print(f"{x.frequency:3d} {x.centered_degree:6d} {x.one_mode_symbolic_margin:.6e} "
                  f"{x.admittance_over_fdnr:.6f} {x.phase_map_condition:9.3f} "
                  f"{x.relative_phase_sensitivity_to_ratio:8.3f} "
                  f"{x.positive_relative_stability_headroom:9.3e} {x.relation_defect:.2e}")
        print("best faithful:", best.frequency, best.phase_map_condition)


if __name__ == "__main__":
    main()
