from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class FiberLoweringResult:
    realizable: bool
    residual: float
    dependency_residual: float
    operator: np.ndarray
    generator_rank: int
    generator_columns: int


def build_fiber_generators(
    centers: np.ndarray,
    tangents: Sequence[np.ndarray],
    transition: Sequence[int],
    tangent_maps: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Build source/target generator matrices for affine fibers.

    centers[:, q] is the base point c_q.
    tangents[q] is d x k_q and spans the continuous fiber at q.
    transition[q] is the target digital state tau(q).
    tangent_maps[q] maps source fiber coordinates to coordinates in the
    target state's tangent basis.

    A global linear token operator must satisfy both point and tangent rules:

        A c_q = c_tau(q)
        A V_q = V_tau(q) L_q.

    These constraints are concatenated into A X = Y.
    """
    c = np.asarray(centers, dtype=float)
    if c.ndim != 2:
        raise ValueError("centers must be d x n")
    d, n = c.shape
    if len(tangents) != n or len(transition) != n or len(tangent_maps) != n:
        raise ValueError("one tangent, transition and tangent map are required per digital state")

    xs = [c]
    ys = [c[:, np.asarray(transition, dtype=int)]]
    for q in range(n):
        v = np.asarray(tangents[q], dtype=float)
        if v.ndim == 1:
            v = v[:, None]
        if v.shape[0] != d:
            raise ValueError("tangent ambient dimension mismatch")
        target_q = int(transition[q])
        vt = np.asarray(tangents[target_q], dtype=float)
        if vt.ndim == 1:
            vt = vt[:, None]
        lq = np.asarray(tangent_maps[q], dtype=float)
        if lq.ndim == 0:
            lq = np.asarray([[float(lq)]])
        if lq.ndim == 1:
            lq = np.diag(lq)
        if lq.shape[0] != vt.shape[1] or lq.shape[1] != v.shape[1]:
            raise ValueError("tangent map shape mismatch")
        xs.append(v)
        ys.append(vt @ lq)
    return np.concatenate(xs, axis=1), np.concatenate(ys, axis=1)


def audit_linear_fiber_lowering(
    centers: np.ndarray,
    tangents: Sequence[np.ndarray],
    transition: Sequence[int],
    tangent_maps: Sequence[np.ndarray],
    *,
    tol: float = 1e-10,
) -> FiberLoweringResult:
    """Test exact global-linear realizability of point+tangent semantics.

    For X,Y built above, A X = Y has a solution iff every linear dependency
    among source generators is also a dependency among target generators:

        ker(X) subset ker(Y).

    Numerically, I-X^+X projects coefficient space onto ker(X). The returned
    A=Y X^+ is the minimum-Frobenius-norm exact lowering when realizable.
    """
    x, y = build_fiber_generators(centers, tangents, transition, tangent_maps)
    pinv = np.linalg.pinv(x)
    dependency_projector = np.eye(x.shape[1]) - pinv @ x
    dep_resid = float(np.linalg.norm(y @ dependency_projector, ord="fro"))
    a = y @ pinv
    residual = float(np.linalg.norm(a @ x - y, ord="fro"))
    scale = max(float(np.linalg.norm(y, ord="fro")), 1.0)
    return FiberLoweringResult(
        realizable=bool(dep_resid <= tol * scale and residual <= tol * scale),
        residual=residual,
        dependency_residual=dep_resid,
        operator=a,
        generator_rank=int(np.linalg.matrix_rank(x)),
        generator_columns=int(x.shape[1]),
    )


def demo() -> None:
    centers = np.array(
        [[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0]]
    )
    v = np.array([[0.0], [0.0], [1.0]])
    tangents = [v.copy() for _ in range(4)]

    merge = [0, 0, 2, 2]
    preserve = [np.eye(1) for _ in range(4)]
    good = audit_linear_fiber_lowering(centers, tangents, merge, preserve)

    # Same source tangent vector appears for q=0 and q=1, but the requested
    # target tangent images disagree. No single global linear A can do both.
    mode_dependent = [np.array([[1.0]]), np.array([[0.5]]), np.array([[1.0]]), np.array([[1.0]])]
    bad = audit_linear_fiber_lowering(centers, tangents, merge, mode_dependent)

    print("shared-tangent merge preserving analog:", good.realizable, good.residual)
    print("same shared tangent, source-mode-dependent analog action:", bad.realizable, bad.residual)


if __name__ == "__main__":
    demo()
