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
MODULE_NAME = "metacircuit_cyclic_backend_for_design"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "metacircuit_cyclic_backend.py")
assert SPEC is not None and SPEC.loader is not None
backend = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = backend
SPEC.loader.exec_module(backend)


def equal_weight_margin(n: int, frequencies: list[int]) -> float:
    if not frequencies:
        return 0.0
    return float(
        min(
            sum(1.0 - math.cos(2.0 * math.pi * f * d / n) for f in frequencies)
            for d in range(1, n)
        )
    )


def character_gcd(n: int, frequencies: list[int]) -> int:
    g = int(n)
    for f in frequencies:
        g = math.gcd(g, int(f))
    return abs(g)


@dataclass
class Design:
    n: int
    modes: int
    condition_cap: float | None
    frequencies: list[int]
    equal_weight_min_margin: float
    character_gcd: int
    certified_faithful: bool
    max_phase_map_condition: float
    max_phase_map_norm: float
    min_positive_relative_stability_headroom: float
    max_relative_phase_sensitivity: float
    admittance_over_fdnr_range: list[float]


def canonical_candidates(n: int):
    # f and n-f have identical cos(theta), conditioning and positive-kernel gap,
    # so keep one representative of each conjugate pair.
    return list(range(1, n // 2 + 1))


def greedy_design(n: int, modes: int, condition_cap: float | None = None) -> Design:
    candidates = []
    for f in canonical_candidates(n):
        row = backend.lower_mode(n, f)
        if row.degenerate or not row.stable_interior:
            continue
        if condition_cap is not None and row.phase_map_condition > condition_cap:
            continue
        candidates.append(f)
    if len(candidates) < modes:
        raise ValueError("not enough backend-legal character candidates for requested mode count")

    selected: list[int] = []
    for _ in range(modes):
        best = None
        for f in candidates:
            if f in selected:
                continue
            trial = selected + [f]
            margin = equal_weight_margin(n, trial)
            row = backend.lower_mode(n, f)
            # Maximize symbolic margin; prefer better-conditioned new character on ties.
            key = (margin, -row.phase_map_condition, row.positive_relative_stability_headroom)
            if best is None or key > best[0]:
                best = (key, f)
        assert best is not None
        selected.append(int(best[1]))

    rows = [backend.lower_mode(n, f) for f in selected]
    g = character_gcd(n, selected)
    return Design(
        n=int(n),
        modes=int(modes),
        condition_cap=None if condition_cap is None else float(condition_cap),
        frequencies=selected,
        equal_weight_min_margin=equal_weight_margin(n, selected),
        character_gcd=g,
        certified_faithful=bool(g == 1),
        max_phase_map_condition=max(r.phase_map_condition for r in rows),
        max_phase_map_norm=max(r.phase_map_norm for r in rows),
        min_positive_relative_stability_headroom=min(r.positive_relative_stability_headroom for r in rows),
        max_relative_phase_sensitivity=max(r.relative_phase_sensitivity_to_ratio for r in rows),
        admittance_over_fdnr_range=[
            min(r.admittance_over_fdnr for r in rows),
            max(r.admittance_over_fdnr for r in rows),
        ],
    )


def main():
    p = argparse.ArgumentParser(description="Choose exact cyclic characters with an explicit resonator-backend conditioning budget")
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--caps", nargs="+", type=float, default=[2.0, 3.0, 5.0, 10.0])
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    unconstrained = greedy_design(args.n, args.modes, None)
    constrained = [greedy_design(args.n, args.modes, cap) for cap in args.caps]
    payload = {
        "config": vars(args),
        "unconstrained": asdict(unconstrained),
        "constrained": [asdict(x) for x in constrained],
        "note": "Greedy search is a reproducible design heuristic, not a global optimum proof.",
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("cap margin maxcond minheadroom frequencies")
        rows = [unconstrained] + constrained
        for row in rows:
            cap = "none" if row.condition_cap is None else f"{row.condition_cap:g}"
            print(f"{cap:>5} {row.equal_weight_min_margin:8.4f} {row.max_phase_map_condition:7.3f} "
                  f"{row.min_positive_relative_stability_headroom:9.4f} {row.frequencies}")


if __name__ == "__main__":
    main()
