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
MODULE_NAME = "cyclic_harmonic_state_oracle_for_mixed_radix"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "cyclic_harmonic_state_oracle.py"
)
assert SPEC is not None and SPEC.loader is not None
harmonic = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = harmonic
SPEC.loader.exec_module(harmonic)


I_TOKEN = 0
INC_TOKEN = 1
RESET_TOKEN = 2


def symbolic_step(state: tuple[int, int], token: int, p: int, q: int) -> tuple[int, int]:
    """Two-digit mixed-radix counter, low digit first.

    On INC, the high digit receives carry iff the OLD low digit is p-1.
    On RESET, both digits reset to zero.
    """
    low, high = state
    if token == I_TOKEN:
        return state
    if token == RESET_TOKEN:
        return (0, 0)
    if token != INC_TOKEN:
        raise ValueError("token must be I=0, INC=1, RESET=2")
    carry = low == p - 1
    return ((low + 1) % p, (high + int(carry)) % q)


def symbolic_value(state: tuple[int, int], p: int, q: int) -> int:
    low, high = state
    return low + p * high


def run_symbolic(tokens: list[int], p: int, q: int) -> list[tuple[int, int]]:
    state = (0, 0)
    out = [state]
    for token in tokens:
        state = symbolic_step(state, token, p, q)
        out.append(state)
    return out


def verify_increment_cycle(p: int, q: int) -> bool:
    state = (0, 0)
    seen = []
    for _ in range(p * q):
        seen.append(symbolic_value(state, p, q))
        state = symbolic_step(state, INC_TOKEN, p, q)
    return state == (0, 0) and seen == list(range(p * q))


@dataclass(frozen=True)
class CodeResource:
    name: str
    behavioral_states: int
    real_dimension: int
    total_state_norm: float
    nearest_prototype_radius_native_scale: float
    nearest_prototype_radius_equal_total_norm: float
    intercell_state_edges: int
    raw_predecessor_real_channels: int
    sufficient_structured_carry_bits: int
    explicit_controller_transition_entries: int
    notes: str


def harmonic_search_resource(
    n: int,
    k: int,
    trials: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    return harmonic.random_search(n, k, trials=trials, seed=seed)


def compare_resources(
    p: int,
    q: int,
    modes_per_factor: int,
    trials: int,
    seed: int,
) -> tuple[list[CodeResource], dict[str, object]]:
    if p < 2 or q < 2:
        raise ValueError("p,q must be >=2")
    n = p * q
    total_modes = 2 * modes_per_factor

    f_mono, mono = harmonic_search_resource(n, total_modes, trials, seed)
    f_p, mp = harmonic_search_resource(p, modes_per_factor, trials, seed + 1)
    f_q, mq = harmonic_search_resource(q, modes_per_factor, trials, seed + 2)

    mono_radius = float(mono["nearest_prototype_noise_radius"])
    rp = float(mp["nearest_prototype_noise_radius"])
    rq = float(mq["nearest_prototype_noise_radius"])
    factor_local_radius = min(rp, rq)

    # Each factor code has unit norm, so concatenating the two cells has norm sqrt(2).
    # If the total state is rescaled to unit norm for a fair dynamic-range/energy
    # comparison, distances and robustness radii scale by 1/sqrt(2).
    factor_equal_norm_radius = factor_local_radius / math.sqrt(2.0)

    # Explicit one-hot Sigma controller table, counting one target entry for every
    # (component state, input context).  The actual carry rule has an O(1)
    # structured description, so report both viewpoints rather than conflating them.
    alphabet = 3
    explicit_sigma_entries = p * alphabet + q * alphabet * p

    resources = [
        CodeResource(
            name="monolithic_single_phase",
            behavioral_states=n,
            real_dimension=2,
            total_state_norm=1.0,
            nearest_prototype_radius_native_scale=math.sin(math.pi / n),
            nearest_prototype_radius_equal_total_norm=math.sin(math.pi / n),
            intercell_state_edges=0,
            raw_predecessor_real_channels=0,
            sufficient_structured_carry_bits=0,
            explicit_controller_transition_entries=3 * n,
            notes="Exact C_pq phase orbit + affine reset; tiny dimension, shrinking margin.",
        ),
        CodeResource(
            name="monolithic_harmonic",
            behavioral_states=n,
            real_dimension=int(mono["real_dimension"]),
            total_state_norm=1.0,
            nearest_prototype_radius_native_scale=mono_radius,
            nearest_prototype_radius_equal_total_norm=mono_radius,
            intercell_state_edges=0,
            raw_predecessor_real_channels=0,
            sufficient_structured_carry_bits=0,
            explicit_controller_transition_entries=3 * n,
            notes="Same total complex-mode budget as factorized harmonic code.",
        ),
        CodeResource(
            name="sigma_factor_harmonic",
            behavioral_states=n,
            real_dimension=int(mp["real_dimension"]) + int(mq["real_dimension"]),
            total_state_norm=math.sqrt(2.0),
            nearest_prototype_radius_native_scale=factor_local_radius,
            nearest_prototype_radius_equal_total_norm=factor_equal_norm_radius,
            intercell_state_edges=1,
            raw_predecessor_real_channels=int(mp["real_dimension"]),
            sufficient_structured_carry_bits=1,
            explicit_controller_transition_entries=explicit_sigma_entries,
            notes=(
                "Two local cyclic factors. High factor increments iff old low==p-1. "
                "One carry bit suffices if that predicate is decoded at the boundary; "
                "otherwise the raw predecessor harmonic state crosses the edge."
            ),
        ),
        CodeResource(
            name="sigma_factor_one_hot",
            behavioral_states=n,
            real_dimension=p + q,
            total_state_norm=math.sqrt(2.0),
            nearest_prototype_radius_native_scale=1.0 / math.sqrt(2.0),
            nearest_prototype_radius_equal_total_norm=0.5,
            intercell_state_edges=1,
            raw_predecessor_real_channels=p,
            sufficient_structured_carry_bits=1,
            explicit_controller_transition_entries=explicit_sigma_entries,
            notes="Robust one-hot factor baseline; structured carry can still be one bit.",
        ),
    ]

    detail = {
        "p": p,
        "q": q,
        "behavioral_states": n,
        "modes_per_factor": modes_per_factor,
        "monolithic_total_modes": total_modes,
        "monolithic_frequencies": f_mono.tolist(),
        "low_frequencies": f_p.tolist(),
        "high_frequencies": f_q.tolist(),
        "monolithic_max_inner_product": mono["max_nontrivial_inner_product"],
        "low_max_inner_product": mp["max_nontrivial_inner_product"],
        "high_max_inner_product": mq["max_nontrivial_inner_product"],
        "increment_cycle_verified": verify_increment_cycle(p, q),
        "structured_controller": {
            "low": "INC -> +1 mod p; RESET -> 0; else identity",
            "high": "INC and old_low==p-1 -> +1 mod q; RESET -> 0; else identity",
            "carry_message_bits_if_locally_decoded": 1,
        },
    }
    return resources, detail


def main() -> None:
    p = argparse.ArgumentParser(
        description="End-to-end resource comparison for a two-factor Sigma-local mixed-radix counter"
    )
    p.add_argument("--p", type=int, default=31)
    p.add_argument("--q", type=int, default=29)
    p.add_argument("--modes-per-factor", type=int, default=8)
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    resources, detail = compare_resources(
        args.p, args.q, args.modes_per_factor, args.trials, args.seed
    )
    payload = {"detail": detail, "resources": [asdict(r) for r in resources]}

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"mixed radix C_{args.p} -> C_{args.q}, total states={args.p*args.q}")
    print(
        "representation              d  norm   native-r  equal-norm-r  edges raw-ch carry"
    )
    print("-------------------------  ---- -----  --------  ------------  ----- ------ -----")
    for r in resources:
        print(
            f"{r.name:25s} {r.real_dimension:4d} {r.total_state_norm:5.2f} "
            f"{r.nearest_prototype_radius_native_scale:9.4f} "
            f"{r.nearest_prototype_radius_equal_total_norm:12.4f} "
            f"{r.intercell_state_edges:5d} {r.raw_predecessor_real_channels:6d} "
            f"{r.sufficient_structured_carry_bits:5d}"
        )


if __name__ == "__main__":
    main()
