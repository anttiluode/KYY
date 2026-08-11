from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

TAU = 2.0 * math.pi


@dataclass
class BlockPlan:
    block: int
    source_states: list[int]
    representative_state: int
    representative_phase: float
    coarse_attractor_phase: float
    capture_margin: float
    fine_reentry_margin: float
    worst_composition_margin: float


@dataclass
class QuotientPlan:
    n: int
    m: int
    block_size: int
    fine_spacing: float
    shil_order: int
    coarse_well_spacing: float
    parity: str
    certified_margin: float
    blocks: list[BlockPlan]


def compile_equal_block_quotient(n: int, m: int) -> QuotientPlan:
    """Compile consecutive equal C_n blocks to one m-well SHIL stage then back to C_n.

    Canonical quotient blocks are
        {jr, jr+1, ..., jr+r-1},  j=0..m-1,
    where r=n/m.

    The temporary m-well attractor for each block is placed to maximize the
    minimum of (i) coarse-basin capture margin for every source fine state and
    (ii) re-entry margin into the chosen representative C_n basin.
    """
    n = int(n); m = int(m)
    if n <= 0 or m <= 0 or n % m != 0:
        raise ValueError("require positive m dividing n")
    r = n // m
    delta = TAU / n
    coarse_spacing = TAU / m
    plans = []

    if r % 2 == 1:
        local_rep = (r - 1) // 2
        local_center = local_rep * delta
        local_attractor = local_center
        margin = delta / 2.0
        parity = "odd"
    else:
        # Choose the lower of the two central fine states.  The block center is
        # half a fine spacing to its right, so minimax composition puts the
        # temporary attractor halfway between them: +delta/4.
        local_rep = r // 2 - 1
        local_center = (r - 1) * delta / 2.0
        local_attractor = local_rep * delta + delta / 4.0
        margin = delta / 4.0
        parity = "even"

    half_coarse_basin = coarse_spacing / 2.0
    for j in range(m):
        base = j * r
        src = [(base + k) % n for k in range(r)]
        rep = (base + local_rep) % n
        rep_phase = rep * delta
        attractor = (j * coarse_spacing + local_attractor) % TAU

        # Unwrap this canonical block locally around j*coarse_spacing.
        local_sources = np.arange(r, dtype=np.float64) * delta
        a = local_attractor
        capture = min(
            float(local_sources[0] - (a - half_coarse_basin)),
            float((a + half_coarse_basin) - local_sources[-1]),
        )
        fine_reentry = delta / 2.0 - abs(a - local_rep * delta)
        plans.append(
            BlockPlan(
                block=j,
                source_states=src,
                representative_state=rep,
                representative_phase=float(rep_phase % TAU),
                coarse_attractor_phase=float(attractor % TAU),
                capture_margin=float(capture),
                fine_reentry_margin=float(fine_reentry),
                worst_composition_margin=float(min(capture, fine_reentry)),
            )
        )

    cert = min(p.worst_composition_margin for p in plans)
    if abs(cert - margin) > 1e-10:
        raise AssertionError("analytic margin check failed")
    return QuotientPlan(
        n=n,
        m=m,
        block_size=r,
        fine_spacing=delta,
        shil_order=m,
        coarse_well_spacing=coarse_spacing,
        parity=parity,
        certified_margin=float(cert),
        blocks=plans,
    )


def cyclic_runs(mapping: list[int]) -> list[tuple[int, int]]:
    """Return (label,length) runs on a cyclic sequence, merging first/last if equal."""
    if not mapping:
        return []
    runs = []
    cur = mapping[0]; length = 1
    for x in mapping[1:]:
        if x == cur:
            length += 1
        else:
            runs.append((cur, length)); cur = x; length = 1
    runs.append((cur, length))
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs[0] = (runs[0][0], runs[0][1] + runs[-1][1])
        runs.pop()
    return runs


def single_uniform_shil_realizable(mapping: list[int]) -> dict:
    """Audit whether a cyclic quotient kernel is equal contiguous blocks.

    A uniform m-well phase potential has m equal contiguous attraction basins.
    Up to rotation and output-label permutation, a deterministic quotient sampled
    on equally spaced C_n wells must therefore have exactly m cyclic runs, each
    of size n/m, with each output label appearing in exactly one run.
    """
    mapping = [int(x) for x in mapping]
    n = len(mapping)
    if n == 0:
        return {"realizable": False, "reason": "empty mapping"}
    runs = cyclic_runs(mapping)
    labels = [r[0] for r in runs]
    lengths = [r[1] for r in runs]
    m = len(set(mapping))
    ok = (
        len(runs) == m
        and len(set(labels)) == m
        and n % m == 0
        and all(length == n // m for length in lengths)
    )
    return {
        "realizable": bool(ok),
        "n": n,
        "m": m,
        "runs": [{"label": lab, "length": ln} for lab, ln in runs],
        "required_block_size": (n // m if m and n % m == 0 else None),
        "reason": (
            "equal contiguous cyclic kernel classes match one uniform m-well SHIL basin partition"
            if ok else
            "one uniform SHIL stage requires one equal contiguous cyclic run per quotient class"
        ),
    }


def canonical_mapping(n: int, m: int) -> list[int]:
    if n % m != 0:
        raise ValueError("m must divide n")
    r = n // m
    return [k // r for k in range(n)]


def main() -> None:
    p = argparse.ArgumentParser(description="Compile/audit cyclic quotients for a uniform multi-phase SHIL backend")
    p.add_argument("--examples", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    cases = [(4,2),(12,4),(12,3),(15,5),(16,4),(100,10)]
    payload = {
        "plans": [asdict(compile_equal_block_quotient(n,m)) for n,m in cases],
        "realizability_controls": {
            "c4_pairs": single_uniform_shil_realizable([0,0,1,1]),
            "c4_alternating": single_uniform_shil_realizable([0,1,0,1]),
            "c12_three_blocks": single_uniform_shil_realizable(canonical_mapping(12,4)),
            "unequal_runs": single_uniform_shil_realizable([0,0,0,1,1,2]),
        },
        "analytic_rule": {
            "odd_block_size": "representative = middle fine state; temporary attractor = representative; margin = Delta/2",
            "even_block_size": "representative = either central fine state; temporary attractor = representative + Delta/4 toward block center; margin = Delta/4",
            "Delta": "2*pi/n",
        },
        "scope": "This is a compiler law for the ideal uniform phase-potential abstraction. Phase-shifted SHIL/Potts hardware and staged phase memory are prior art.",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
