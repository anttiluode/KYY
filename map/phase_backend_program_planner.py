from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

from map.phase_kernel_lowering_classifier import canonicalize, classify_kernel


@dataclass
class TransitionPlan:
    name: str
    mapping: list[int]
    lowering: str
    standing_character_frequency: int | None
    runtime_instruction: str | None
    faithful_embedding: int | None
    reject_reason: str | None


@dataclass
class ProgramPlan:
    n: int
    transitions: list[TransitionPlan]
    precarried_character_bank: list[int]
    precarried_carrier_count: int
    runtime_harmonic_character_frequencies_if_not_precarried: list[int]
    shil_stage_count: int
    rejected_transition_count: int
    exact_standing_character_lower_bound: int
    lower_bound_scope: str


def simplest_nontrivial_character(candidates: list[int], n: int) -> int:
    nonzero = sorted({int(f) % n for f in candidates if int(f) % n != 0})
    if not nonzero:
        raise ValueError("no nonzero character candidate")
    # Frequencies with the same gcd have the same kernel. Prefer the smallest
    # representative because this planner is algebraic; hardware may price them differently.
    return nonzero[0]


def kernel_class_count_from_frequency(n: int, f: int) -> int:
    return n // math.gcd(n, int(f) % n)


def plan_program(n: int, named_mappings: list[tuple[str, list[int]]]) -> ProgramPlan:
    n = int(n)
    if n <= 1:
        raise ValueError("n must be >1")

    # f=1 is the canonical faithful carrier. Any unit would do algebraically.
    standing = {1}
    runtime_harmonics: set[int] = set()
    plans: list[TransitionPlan] = []
    shil_count = 0
    rejected = 0
    distinct_nontrivial_character_kernels: dict[tuple[int, ...], int] = {}

    for name, raw in named_mappings:
        if len(raw) != n:
            raise ValueError(f"transition {name!r} has length {len(raw)}, expected {n}")
        target = canonicalize(raw)
        c = classify_kernel(target)

        if c.class_count == n:
            plans.append(TransitionPlan(name, list(target), "identity/no-op", None, None, None, None))
            continue
        if c.class_count == 1:
            plans.append(TransitionPlan(name, list(target), "universal collapse", None, "one-well collapse or hard reset", None, None))
            continue

        if c.character_candidates:
            f = simplest_nontrivial_character(c.character_candidates, n)
            standing.add(f)
            runtime_harmonics.add(f)
            distinct_nontrivial_character_kernels[target] = f
            plans.append(
                TransitionPlan(
                    name=name,
                    mapping=list(target),
                    lowering="quotient-aligned character",
                    standing_character_frequency=f,
                    runtime_instruction=(
                        f"precarry f={f} and retire distinguishing carriers; otherwise synthesize/transfer to harmonic character f={f}"
                    ),
                    faithful_embedding=None,
                    reject_reason=None,
                )
            )
            continue

        if c.shil_faithful_embeddings:
            f = min(c.shil_faithful_embeddings)
            shil_count += 1
            plans.append(
                TransitionPlan(
                    name=name,
                    mapping=list(target),
                    lowering="uniform SHIL basin collapse",
                    standing_character_frequency=None,
                    runtime_instruction="uniform multi-well relaxation",
                    faithful_embedding=f,
                    reject_reason=None,
                )
            )
            continue

        rejected += 1
        plans.append(
            TransitionPlan(
                name=name,
                mapping=list(target),
                lowering="reject current phase library",
                standing_character_frequency=None,
                runtime_instruction=None,
                faithful_embedding=None,
                reject_reason=c.reject_reason,
            )
        )

    # Within the restricted strategy "make every exact congruence quotient a
    # directly readable single character coordinate", distinct kernels require
    # distinct character gcds. One faithful coordinate is also required to
    # distinguish all pre-quotient C_n states.
    lb = 1 + len(distinct_nontrivial_character_kernels)

    return ProgramPlan(
        n=n,
        transitions=plans,
        precarried_character_bank=sorted(standing),
        precarried_carrier_count=len(standing),
        runtime_harmonic_character_frequencies_if_not_precarried=sorted(runtime_harmonics),
        shil_stage_count=shil_count,
        rejected_transition_count=rejected,
        exact_standing_character_lower_bound=lb,
        lower_bound_scope=(
            "Exact only for the restricted direct-coordinate strategy: one faithful cyclic character preserves the full state, "
            "and every distinct nontrivial congruence kernel must be directly available as a single retained character coordinate. "
            "More general encodings, nonlinear decoders or extra state can change the resource count."
        ),
    )


def congruence_mapping(n: int, class_count: int) -> list[int]:
    """Canonical character kernel with the requested class count, when class_count divides n."""
    if n % class_count:
        raise ValueError("class_count must divide n")
    f = n // class_count
    raw = [(f * q) % n for q in range(n)]
    return list(canonicalize(raw))


def contiguous_block_mapping(n: int, class_count: int) -> list[int]:
    if n % class_count:
        raise ValueError("class_count must divide n")
    r = n // class_count
    return [q // r for q in range(n)]


def demo_plan() -> ProgramPlan:
    n = 12
    transitions = [
        ("keep parity-like C2 congruence", congruence_mapping(n, 2)),
        ("keep C3 congruence", congruence_mapping(n, 3)),
        ("contiguous C4 block quotient", contiguous_block_mapping(n, 4)),
        ("equal-size wrong-topology control", [0, 0, 1, 0, 1, 1, 2, 2, 3, 2, 3, 3]),
        ("unequal control", [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5]),
    ]
    return plan_program(n, transitions)


def main() -> None:
    p = argparse.ArgumentParser(description="Plan standing character versus runtime instruction resources for a cyclic phase backend")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    print(json.dumps(asdict(demo_plan()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
