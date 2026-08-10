from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np


def nullspace(a: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    _, s, vh = np.linalg.svd(a, full_matrices=True)
    rank = int(np.sum(s > tol))
    return vh[rank:].T.copy()


@dataclass
class LoweringResult:
    kind: str
    exact: bool
    residual: float
    dependency_violation: float
    operator: list[list[float]]
    bias: list[float] | None
    continuous_rank: int


def dependency_violation(code: np.ndarray, targets: np.ndarray, tol: float = 1e-10) -> float:
    """Measure failure of ker(code) subset ker(targets), columns are states."""
    ns = nullspace(code, tol=tol)
    if ns.shape[1] == 0:
        return 0.0
    return float(np.linalg.norm(targets @ ns, ord=2))


def linear_lowering(code: np.ndarray, targets: np.ndarray, tol: float = 1e-9) -> LoweringResult:
    """Find the minimum-Frobenius linear map A with A Z approximately Zx.

    Exact solvability of A Z = Zx is equivalent to every linear dependency
    among the code columns also being a dependency among the target columns:

        ker Z subset ker Zx.
    """
    z = np.asarray(code, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if z.ndim != 2 or y.ndim != 2 or z.shape[1] != y.shape[1]:
        raise ValueError("code and targets must be 2D with matching state columns")
    A = y @ np.linalg.pinv(z)
    residual = float(np.linalg.norm(A @ z - y, ord=2))
    dep = dependency_violation(z, y)
    rank = int(np.linalg.matrix_rank(A, tol=tol))
    return LoweringResult(
        kind="linear",
        exact=bool(residual <= tol and dep <= tol),
        residual=residual,
        dependency_violation=dep,
        operator=A.tolist(),
        bias=None,
        continuous_rank=rank,
    )


def affine_lowering(code: np.ndarray, targets: np.ndarray, tol: float = 1e-9) -> LoweringResult:
    """Find [A b] such that A Z + b 1^T = Zx when possible.

    This is the same linear-realization test after homogeneous augmentation of
    the code with one constant coordinate.
    """
    z = np.asarray(code, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    aug = np.vstack((z, np.ones((1, z.shape[1]), dtype=np.float64)))
    L = y @ np.linalg.pinv(aug)
    A, b = L[:, :-1], L[:, -1]
    residual = float(np.linalg.norm(A @ z + b[:, None] - y, ord=2))
    dep = dependency_violation(aug, y)
    rank = int(np.linalg.matrix_rank(A, tol=tol))
    return LoweringResult(
        kind="affine",
        exact=bool(residual <= tol and dep <= tol),
        residual=residual,
        dependency_violation=dep,
        operator=A.tolist(),
        bias=b.tolist(),
        continuous_rank=rank,
    )


def choose_lowering(code: np.ndarray, targets: np.ndarray, tol: float = 1e-9) -> LoweringResult:
    linear = linear_lowering(code, targets, tol=tol)
    if linear.exact:
        return linear
    affine = affine_lowering(code, targets, tol=tol)
    if affine.exact:
        return affine
    return affine


def cyclic_code(n: int, frequency: int = 1) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    phase = 2.0 * np.pi * int(frequency) * k / int(n)
    return np.vstack((np.cos(phase), np.sin(phase)))


def transition_targets(code: np.ndarray, mapping: list[int]) -> np.ndarray:
    z = np.asarray(code, dtype=np.float64)
    if len(mapping) != z.shape[1]:
        raise ValueError("mapping length must equal number of code states")
    return z[:, np.asarray(mapping, dtype=np.int64)]


def demo() -> dict[str, object]:
    z = cyclic_code(4)
    cases = {
        "cycle": [1, 2, 3, 0],
        "partial_merge": [0, 0, 2, 2],
        "total_reset": [0, 0, 0, 0],
    }
    out: dict[str, object] = {}
    for name, mapping in cases.items():
        y = transition_targets(z, mapping)
        out[name] = {
            "linear": asdict(linear_lowering(z, y)),
            "affine": asdict(affine_lowering(z, y)),
            "chosen": asdict(choose_lowering(z, y)),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Audit whether task transitions lower linearly or affinely on a geometric state code")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    payload = demo()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for name, row in payload.items():
            chosen = row["chosen"]
            print(
                f"{name:16s} -> {chosen['kind']:6s} exact={chosen['exact']} "
                f"residual={chosen['residual']:.3e} rank={chosen['continuous_rank']}"
            )


if __name__ == "__main__":
    main()
