from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass
class KernelClassification:
    mapping: list[int]
    n: int
    class_count: int
    class_sizes: list[int]
    equal_class_sizes: bool
    character_candidates: list[int]
    shil_faithful_embeddings: list[int]
    primary_lowering: str
    standing_cost: str
    runtime_cost: str
    reject_reason: str | None


def canonicalize(mapping: Iterable[int]) -> tuple[int, ...]:
    table: dict[int, int] = {}
    nxt = 0
    out: list[int] = []
    for x0 in mapping:
        x = int(x0)
        if x not in table:
            table[x] = nxt
            nxt += 1
        out.append(table[x])
    return tuple(out)


def class_sizes(mapping: tuple[int, ...]) -> list[int]:
    return sorted([mapping.count(lab) for lab in set(mapping)])


def character_kernel(n: int, frequency: int) -> tuple[int, ...]:
    """Equality kernel of q -> exp(2 pi i frequency q / n).

    We represent the phase by its discrete phase index frequency*q mod n and
    canonicalize output labels, so only the induced partition remains.
    """
    f = int(frequency) % n
    return canonicalize((f * q) % n for q in range(n))


def character_candidates(mapping: tuple[int, ...]) -> list[int]:
    n = len(mapping)
    target = canonicalize(mapping)
    return [f for f in range(n) if character_kernel(n, f) == target]


def cyclic_runs(mapping: tuple[int, ...]) -> list[tuple[int, int]]:
    if not mapping:
        return []
    runs: list[tuple[int, int]] = []
    cur = mapping[0]
    length = 1
    for x in mapping[1:]:
        if x == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = x
            length = 1
    runs.append((cur, length))
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs[0] = (runs[0][0], runs[0][1] + runs[-1][1])
        runs.pop()
    return runs


def uniform_shil_realizable_in_order(mapping: tuple[int, ...]) -> bool:
    """Whether labels are one equal contiguous cyclic run per class."""
    n = len(mapping)
    if n == 0:
        return False
    m = len(set(mapping))
    runs = cyclic_runs(mapping)
    return bool(
        m > 0
        and n % m == 0
        and len(runs) == m
        and len({lab for lab, _ in runs}) == m
        and all(length == n // m for _, length in runs)
    )


def units_mod_n(n: int) -> list[int]:
    return [f for f in range(1, n) if math.gcd(f, n) == 1]


def labels_in_physical_phase_order(mapping: tuple[int, ...], faithful_frequency: int) -> tuple[int, ...]:
    """Read abstract q-labels around the physical circle of character f.

    A faithful character embeds q at phase slot p=f*q mod n.  Walking physical
    phase slots p=0..n-1 therefore visits q=f^{-1}p mod n.
    """
    n = len(mapping)
    f = int(faithful_frequency) % n
    if math.gcd(f, n) != 1:
        raise ValueError("physical re-encoding frequency must be faithful")
    inv = pow(f, -1, n)
    return tuple(mapping[(inv * p) % n] for p in range(n))


def shil_faithful_embeddings(mapping: tuple[int, ...]) -> list[int]:
    target = canonicalize(mapping)
    out: list[int] = []
    for f in units_mod_n(len(target)):
        physical = labels_in_physical_phase_order(target, f)
        if uniform_shil_realizable_in_order(physical):
            out.append(f)
    return out


def classify_kernel(mapping: Iterable[int]) -> KernelClassification:
    target = canonicalize(mapping)
    n = len(target)
    if n == 0:
        raise ValueError("mapping must be nonempty")
    sizes = class_sizes(target)
    m = len(sizes)
    chars = character_candidates(target)
    shil = shil_faithful_embeddings(target)
    equal = len(set(sizes)) == 1

    if m == n:
        primary = "identity/no-op"
        standing = "none"
        runtime = "none"
        reject = None
    elif m == 1:
        primary = "universal collapse"
        standing = "constant character f=0 is available if desired"
        runtime = "one uniform one-well relaxation is also possible"
        reject = None
    elif chars:
        primary = "quotient-aligned non-faithful character"
        standing = (
            "carry this quotient character beside a faithful state code, then retire modes "
            "that distinguish within the quotient class"
        )
        runtime = (
            "if the quotient character is not already present, harmonic/carrier conversion "
            "can synthesize the same congruence kernel at runtime"
        )
        reject = None
    elif shil:
        primary = "uniform SHIL basin collapse after faithful phase re-encoding"
        standing = (
            "choose one of the listed faithful characters as the physical C_n embedding; "
            "this preserves full pre-transition state"
        )
        runtime = "one uniform multi-well relaxation stage"
        reject = None
    elif not equal:
        primary = "unsupported by current uniform phase library"
        standing = "none of the current exact character or uniform-SHIL primitives matches this kernel"
        runtime = "requires nonuniform forcing, extra state, a richer nonlinear map, or another embedding"
        reject = "kernel classes have unequal sizes"
    else:
        primary = "unsupported by current uniform phase library"
        standing = "equal class size is necessary but not sufficient"
        runtime = "requires nonuniform/order-changing forcing, extra state, or another physical instruction"
        reject = "equal-size kernel has neither cyclic-contiguous nor cyclic-congruence geometry"

    return KernelClassification(
        mapping=list(target),
        n=n,
        class_count=m,
        class_sizes=sizes,
        equal_class_sizes=equal,
        character_candidates=chars,
        shil_faithful_embeddings=shil,
        primary_lowering=primary,
        standing_cost=standing,
        runtime_cost=runtime,
        reject_reason=reject,
    )


def set_partitions_rgs(n: int):
    """Generate every set partition of {0,...,n-1} once as a restricted-growth string."""
    n = int(n)
    if n <= 0:
        return
    a = [0] * n

    def rec(i: int, max_label: int):
        if i == n:
            yield tuple(a)
            return
        for label in range(max_label + 2):
            a[i] = label
            yield from rec(i + 1, max(max_label, label))

    a[0] = 0
    yield from rec(1, 0)


def coverage_summary(n: int) -> dict:
    counts = {
        "trivial": 0,
        "character_only": 0,
        "shil_only": 0,
        "both_nontrivial": 0,
        "unsupported_equal_size": 0,
        "unsupported_unequal_size": 0,
    }
    examples: dict[str, list[int] | None] = {k: None for k in counts}
    total = 0
    for p in set_partitions_rgs(n):
        total += 1
        c = classify_kernel(p)
        trivial = c.class_count in (1, n)
        char = bool(c.character_candidates)
        shil = bool(c.shil_faithful_embeddings)
        if trivial:
            key = "trivial"
        elif char and shil:
            key = "both_nontrivial"
        elif char:
            key = "character_only"
        elif shil:
            key = "shil_only"
        elif c.equal_class_sizes:
            key = "unsupported_equal_size"
        else:
            key = "unsupported_unequal_size"
        counts[key] += 1
        if examples[key] is None:
            examples[key] = list(p)

    supported_nontrivial = counts["character_only"] + counts["shil_only"] + counts["both_nontrivial"]
    nontrivial = total - counts["trivial"]
    return {
        "n": n,
        "bell_partition_count": total,
        "nontrivial_partition_count": nontrivial,
        "counts": counts,
        "supported_nontrivial": supported_nontrivial,
        "supported_nontrivial_fraction": (supported_nontrivial / nontrivial if nontrivial else 1.0),
        "first_examples": examples,
    }


def verify_nontrivial_character_shil_disjoint(n_max: int = 16) -> dict:
    """Finite audit of a general group-theoretic fact.

    A nontrivial proper character kernel is a coset partition of the unique
    subgroup of C_n of that order.  Any faithful character is a C_n
    automorphism and maps that subgroup to itself, so its non-singleton cosets
    remain interleaved rather than becoming contiguous arcs.
    """
    collisions: list[dict] = []
    for n in range(2, n_max + 1):
        seen = set()
        for k in range(1, n):
            p = character_kernel(n, k)
            if p in seen:
                continue
            seen.add(p)
            m = len(set(p))
            if 1 < m < n:
                sh = shil_faithful_embeddings(p)
                if sh:
                    collisions.append({"n": n, "character_frequency": k, "partition": list(p), "shil_embeddings": sh})
    return {
        "n_max": n_max,
        "collision_count": len(collisions),
        "collisions": collisions,
        "statement": (
            "Nontrivial proper cyclic character/congruence kernels and uniform contiguous-SHIL kernels "
            "are disjoint under faithful one-circle re-encoding."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Finite lowering classifier for the current cyclic phase backend library")
    p.add_argument("--n-max", type=int, default=8)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    controls = {
        "c4_adjacent": asdict(classify_kernel([0, 0, 1, 1])),
        "c4_alternating": asdict(classify_kernel([0, 1, 0, 1])),
        "c4_unequal": asdict(classify_kernel([0, 0, 1, 2])),
        "c6_equal_but_neither": asdict(classify_kernel([0, 0, 1, 0, 1, 1])),
    }
    payload = {
        "library": [
            "faithful cyclic phase re-encoding",
            "uniform equal-basin SHIL collapse",
            "pre-carried quotient-aligned character",
            "runtime harmonic/carrier conversion for congruence kernels",
        ],
        "decision_order": [
            "prefer already-carried quotient character when present",
            "otherwise use a faithful embedding plus uniform SHIL when legal",
            "otherwise a congruence kernel may use runtime harmonic/carrier conversion",
            "otherwise reject current library and name the missing physical resource",
        ],
        "controls": controls,
        "coverage": [coverage_summary(n) for n in range(2, args.n_max + 1)],
        "character_vs_shil_disjoint_audit": verify_nontrivial_character_shil_disjoint(max(16, args.n_max)),
        "scope": (
            "This closes only the declared phase instruction library as a finite decision procedure. "
            "It is not a completeness theorem for physical quotient implementations."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
