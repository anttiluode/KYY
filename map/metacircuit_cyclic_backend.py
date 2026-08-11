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
    one_mode_symbolic_margin: float
    stable_interior: bool
    degenerate: bool


def lower_mode(n: int, frequency: int, dt: float = 1.0, tol: float = 1e-10) -> ModeLowering:
    f = int(frequency) % int(n)
    theta = 2.0 * math.pi * f / n
    a = companion_block(theta)
    rel = float(np.linalg.norm(np.linalg.matrix_power(a, n) - np.eye(2), ord="fro"))
    ratio = required_admittance_over_fdnr(theta, dt)
    s = abs(math.sin(theta))
    degenerate = bool(s <= 1e-12)
    if degenerate:
        cond = float("inf")
        tnorm = float("inf")
    else:
        t = phase_from_companion(theta)
        cond = float(np.linalg.cond(t))
        tnorm = float(np.linalg.norm(t, ord=2))
        # Verify the exact similarity used for port transport.
        sim = t @ a @ np.linalg.inv(t)
        if np.linalg.norm(sim - rotation(theta), ord="fro") > 1e-9:
            raise AssertionError("companion-to-phase similarity failed")
    return ModeLowering(
        n=int(n),
        frequency=f,
        centered_degree=int(min(f, n - f)),
        theta=float(theta),
        relation_defect=rel,
        admittance_over_fdnr=float(ratio),
        phase_map_condition=cond,
        phase_map_norm=tnorm,
        one_mode_symbolic_margin=prototype_margin(n, f),
        stable_interior=bool(ratio > tol and ratio < 4.0 / (dt * dt) - tol),
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
        out.append(
            {
                "seed": int(row["seed"]),
                "frequencies": row["projected_frequencies"],
                "max_phase_map_condition": max(m.phase_map_condition for m in modes),
                "max_phase_map_norm": max(m.phase_map_norm for m in modes),
                "worst_frequency": worst.frequency,
                "worst_centered_degree": worst.centered_degree,
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
            "degeneracy": "sin(theta)=0 makes the displacement/lag companion representation non-diagonalizable at theta=0 or pi",
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("f degree margin D^-1Y cond(T) ||T|| relation")
        for x in modes:
            print(f"{x.frequency:3d} {x.centered_degree:6d} {x.one_mode_symbolic_margin:.6e} "
                  f"{x.admittance_over_fdnr:.6f} {x.phase_map_condition:9.3f} "
                  f"{x.phase_map_norm:9.3f} {x.relation_defect:.2e}")
        print("best faithful:", best.frequency, best.phase_map_condition)


if __name__ == "__main__":
    main()
