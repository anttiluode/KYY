from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import reduce
from math import gcd


@dataclass(frozen=True)
class NormalForm:
    kind: str  # "rotation" or "constant"
    index: int


def reduce_word(n: int, word: list[int | str]) -> NormalForm:
    """Reduce a C_n + reset word without constructing its state map.

    Integers are cyclic increments. The string ``R`` is reset-to-zero.
    Reading left to right:

    - before the last reset, increments affect the incoming state;
    - after a reset, the incoming state is gone forever and only the suffix sum
      matters.

    Hence every word has one of exactly two normal-form shapes:

        c^k      (rotation / permutation)
        R c^k    (constant map to k).
    """
    if n <= 1:
        raise ValueError("n must be > 1")
    k = 0
    reset_seen = False
    for symbol in word:
        if symbol == "R":
            reset_seen = True
            k = 0
        elif isinstance(symbol, int):
            k = (k + symbol) % n
        else:
            raise ValueError(f"unknown symbol {symbol!r}")
    return NormalForm("constant" if reset_seen else "rotation", int(k))


def active_character_gcd(n: int, frequencies: list[int]) -> int:
    return abs(reduce(gcd, [int(n)] + [int(f) for f in frequencies]))


@dataclass(frozen=True)
class CyclicResetMonoidCertificate:
    n: int
    frequencies: list[int]
    character_gcd: int
    faithful_cycle_orbit: bool
    reset_is_constant_by_compiler_contract: bool
    reset_is_idempotent_by_compiler_contract: bool
    reset_erases_prefix_by_normal_form: bool
    group_normal_forms: int
    constant_normal_forms: int
    predicted_transformation_monoid_size: int
    certified: bool


def certificate(n: int, frequencies: list[int]) -> CyclicResetMonoidCertificate:
    """Structural certificate for the compiled C_n + exact-reset machine.

    Assumptions encoded by the compiler contract:
      1. the cycle generator has already been snapped to exact C_n characters;
      2. reset is implemented as the literal constant overwrite h -> h0;
      3. the output port distinguishes the legal harmonic orbit separately
         (e.g. by the positive-kernel certificate).

    The only remaining issue for the hidden transition action is whether the
    chosen characters give a faithful C_n orbit. For a harmonic C_n direct sum,
    that is gcd(n,f_1,...,f_m)=1.

    If faithful, the generated transformation monoid has n distinct rotations
    and n distinct constant maps. They are disjoint for n>1 because rotations
    are bijections while constants have rank one.
    """
    if n <= 1:
        raise ValueError("n must be > 1")
    f = [int(x) % n for x in frequencies]
    g = active_character_gcd(n, f)
    faithful = g == 1
    return CyclicResetMonoidCertificate(
        n=int(n),
        frequencies=f,
        character_gcd=int(g),
        faithful_cycle_orbit=faithful,
        reset_is_constant_by_compiler_contract=True,
        reset_is_idempotent_by_compiler_contract=True,
        reset_erases_prefix_by_normal_form=True,
        group_normal_forms=int(n if faithful else n // g),
        constant_normal_forms=int(n if faithful else n // g),
        predicted_transformation_monoid_size=int(2 * (n // g)),
        certified=bool(faithful),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Normal-form certificate for a compiled cyclic permutation-reset monoid")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--frequencies", nargs="+", type=int, required=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    report = certificate(args.n, args.frequencies)
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(report)


if __name__ == "__main__":
    main()
