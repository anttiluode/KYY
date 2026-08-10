from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np


def edge_generator(n: int, i: int, j: int) -> np.ndarray:
    """Skew-symmetric plane-rotation generator X_ij = E_ij - E_ji."""
    a = np.zeros((n, n), dtype=np.float64)
    a[i, j] = 1.0
    a[j, i] = -1.0
    return a


def _vec_skew(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    return np.asarray([a[i, j] for i in range(n) for j in range(i + 1, n)])


def _numerical_rank(columns: list[np.ndarray], rtol: float, atol: float) -> int:
    if not columns:
        return 0
    mat = np.stack(columns, axis=1)
    s = np.linalg.svd(mat, compute_uv=False)
    if s.size == 0:
        return 0
    threshold = max(atol, rtol * float(s[0]))
    return int(np.sum(s > threshold))


def lie_closure_dimension(
    generators: list[np.ndarray],
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> int:
    """Numerically compute dim Lie(generators) for real skew matrices.

    Pairwise commutators are repeatedly added until the span stops growing.
    Each candidate is normalized before a fresh SVD rank test.  This is slower
    than one-pass Gram-Schmidt, but avoids a failure mode we actually hit: tiny
    roundoff remnants were being promoted to fake Lie directions for the N=12
    ring.  For the small map sizes used here, robustness is worth the cost.
    """
    if not generators:
        return 0

    n = generators[0].shape[0]
    target = n * (n - 1) // 2
    basis: list[np.ndarray] = []
    vectors: list[np.ndarray] = []

    def add(a: np.ndarray) -> bool:
        a = 0.5 * (a - a.T)
        norm = float(np.linalg.norm(a))
        if norm <= atol:
            return False
        a = a / norm
        v = _vec_skew(a)
        old_rank = _numerical_rank(vectors, rtol, atol)
        new_rank = _numerical_rank(vectors + [v], rtol, atol)
        if new_rank <= old_rank:
            return False
        basis.append(a)
        vectors.append(v)
        return True

    for g in generators:
        add(g)

    while True:
        old = list(basis)
        changed = False
        for i in range(len(old)):
            for j in range(i):
                comm = old[i] @ old[j] - old[j] @ old[i]
                if add(comm):
                    changed = True
                    if len(basis) == target:
                        return target
        if not changed:
            return len(basis)


def path_drift(n: int) -> np.ndarray:
    return sum((edge_generator(n, i, i + 1) for i in range(n - 1)), np.zeros((n, n)))


def ring_drift(n: int) -> np.ndarray:
    edges = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]
    return sum((edge_generator(n, i, j) for i, j in edges), np.zeros((n, n)))


@dataclass
class Row:
    n: int
    so_n: int
    path_drift_plus_one_local: int
    ring_drift_plus_one_local: int
    ring_drift_plus_two_local: int


def measure(n: int) -> Row:
    b01 = edge_generator(n, 0, 1)
    b12 = edge_generator(n, 1, 2) if n >= 3 else b01
    return Row(
        n=n,
        so_n=n * (n - 1) // 2,
        path_drift_plus_one_local=lie_closure_dimension([path_drift(n), b01]),
        ring_drift_plus_one_local=lie_closure_dimension([ring_drift(n), b01]),
        ring_drift_plus_two_local=lie_closure_dimension([ring_drift(n), b01, b12]),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="KYY local-geometry dynamical Lie-algebra probe")
    p.add_argument("--n-min", type=int, default=3)
    p.add_argument("--n-max", type=int, default=12)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = [measure(n) for n in range(args.n_min, args.n_max + 1)]
    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
        return

    print(" n | so(n) | path G+B | ring G+B | ring G+B1+B2")
    print("---+-------+----------+----------+-------------")
    for r in rows:
        print(
            f"{r.n:2d} | {r.so_n:5d} | {r.path_drift_plus_one_local:8d} | "
            f"{r.ring_drift_plus_one_local:8d} | {r.ring_drift_plus_two_local:11d}"
        )


if __name__ == "__main__":
    main()
