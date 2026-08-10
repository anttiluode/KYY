from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from functools import reduce
from math import gcd

import numpy as np


def active_character_gcd(
    n: int,
    frequencies: np.ndarray | list[int],
    squared_amplitudes: np.ndarray | list[float] | None = None,
    *,
    tol: float = 1e-15,
) -> int:
    """Return gcd(n, active character frequencies).

    A mode is active when its squared seed amplitude is > tol. If amplitudes
    are omitted, every listed character is treated as active.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    f = np.asarray(frequencies, dtype=np.int64).reshape(-1)
    if squared_amplitudes is None:
        active = f
    else:
        a = np.asarray(squared_amplitudes, dtype=np.float64).reshape(-1)
        if len(a) != len(f):
            raise ValueError("frequency/amplitude counts must match")
        active = f[a > tol]
    values = [int(n)] + [int(x) for x in active.tolist()]
    return abs(reduce(gcd, values))


def prototype_margin_at_displacement(
    n: int,
    frequencies: np.ndarray | list[int],
    displacement: int,
    squared_amplitudes: np.ndarray | list[float] | None = None,
) -> float:
    """Exact matched-filter margin between state k and class k+d.

    For z_k = rho(k) z_0 and prototype weights w_j=z_j with equal-norm bias,

        score(k,k) - score(k,k+d)
          = sum_i a_i [1-cos(2*pi*f_i*d/n)],

    where a_i is the squared norm of seed mode i.
    """
    f = np.asarray(frequencies, dtype=np.int64).reshape(-1)
    if squared_amplitudes is None:
        a = np.full(len(f), 1.0 / max(len(f), 1), dtype=np.float64)
    else:
        a = np.asarray(squared_amplitudes, dtype=np.float64).reshape(-1)
        if len(a) != len(f):
            raise ValueError("frequency/amplitude counts must match")
    phase = 2.0 * math.pi * f.astype(np.float64) * int(displacement) / int(n)
    return float(np.sum(a * (1.0 - np.cos(phase))))


def prototype_correctness_certificate(
    n: int,
    frequencies: np.ndarray | list[int],
    squared_amplitudes: np.ndarray | list[float] | None = None,
) -> tuple[bool, int]:
    """Certificate for unique exact C_n prototype decoding.

    The matched-filter margin is a sum of nonnegative terms. It vanishes at a
    displacement d iff every active character satisfies f_i d = 0 mod n.
    Thus every nonzero displacement has strictly positive margin iff the joint
    character map is faithful, which for C_n is equivalent to

        gcd(n, active f_1, ..., active f_m) = 1.

    Returns (certified, kernel_size). The kernel size equals that gcd.
    """
    g = active_character_gcd(n, frequencies, squared_amplitudes)
    return g == 1, g


@dataclass
class CertificateReport:
    n: int
    frequencies: list[int]
    character_gcd: int
    certified_unique: bool
    predicted_orbit_size: int
    exhaustive_distinct_displacements: int
    exhaustive_zero_margin_displacements: list[int]
    exhaustive_min_nonzero_margin: float


def audit(
    n: int,
    frequencies: np.ndarray | list[int],
    squared_amplitudes: np.ndarray | list[float] | None = None,
) -> CertificateReport:
    """Small-n/expository check of the algebraic certificate by enumeration."""
    f = np.asarray(frequencies, dtype=np.int64).reshape(-1)
    ok, g = prototype_correctness_certificate(n, f, squared_amplitudes)
    margins = np.asarray(
        [prototype_margin_at_displacement(n, f, d, squared_amplitudes) for d in range(1, n)],
        dtype=np.float64,
    )
    zero = [int(i + 1) for i, x in enumerate(margins) if abs(float(x)) < 1e-12]
    positive = margins[margins > 1e-12]
    minpos = float(np.min(positive)) if len(positive) else 0.0
    return CertificateReport(
        n=int(n),
        frequencies=[int(x % n) for x in f.tolist()],
        character_gcd=int(g),
        certified_unique=bool(ok),
        predicted_orbit_size=int(n // g),
        exhaustive_distinct_displacements=int((n - 1) - len(zero)),
        exhaustive_zero_margin_displacements=zero,
        exhaustive_min_nonzero_margin=minpos,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="O(m) exact correctness certificate for a cyclic prototype decoder")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--frequencies", nargs="+", type=int, required=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    report = audit(args.n, args.frequencies)
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(report)


if __name__ == "__main__":
    main()
