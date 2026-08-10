from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "dihedral_joint_for_generator_transport"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "dihedral_joint_legalization_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
joint = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = joint
SPEC.loader.exec_module(joint)
base = joint.base


def linear_superoperator(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """4x4 row-major operator K with vec(L M R^T) = K vec(M)."""
    dtype = torch.promote_types(left.dtype, right.dtype)
    left = left.to(dtype)
    right = right.to(dtype)
    cols = []
    for j in range(4):
        e = torch.zeros(2, 2, dtype=dtype)
        e.reshape(-1)[j] = 1.0
        cols.append((left @ e @ right.T).reshape(-1))
    return torch.stack(cols, dim=1)


def matrix_power_sum(k: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (K^n, I + K + ... + K^(n-1)) in O(log n) matrix products."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    eye = torch.eye(k.shape[0], dtype=k.dtype, device=k.device)
    zero = torch.zeros_like(k)
    if n == 0:
        return eye, zero
    if n == 1:
        return k, eye
    if n % 2 == 0:
        p, s = matrix_power_sum(k, n // 2)
        return p @ p, s + p @ s
    p, s = matrix_power_sum(k, n - 1)
    return p @ k, s + p


def rotation_cross_covariance_from_generators(
    n: int,
    learned_angle: float,
    projected_angle: float,
    seed_vector: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Compute sum_k x*_k x_k^T without enumerating orbit states.

    x*_k = R(theta*)^k v, x_k = R(theta)^k v.
    """
    dtype = torch.float64
    r_star = joint.rotation_matrix(projected_angle, dtype=dtype)
    r_learned = joint.rotation_matrix(learned_angle, dtype=dtype)
    v = torch.as_tensor(seed_vector, dtype=dtype).reshape(2, 1)
    m0 = v @ v.T
    k = linear_superoperator(r_star, r_learned)
    _, s = matrix_power_sum(k, n)
    return (s @ m0.reshape(-1)).reshape(2, 2)


def procrustes_from_cross(cross: torch.Tensor) -> torch.Tensor:
    """Orthogonal Q minimizing ||X Q - Y|| when cross = X^T Y."""
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    return u @ vh


def generator_quotient_block_port(
    n: int,
    learned_angles: np.ndarray | torch.Tensor,
    projected_angles: np.ndarray | torch.Tensor,
    learned_reflections: np.ndarray | torch.Tensor,
    projected_reflections: np.ndarray | torch.Tensor,
    h0: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compile the two D_n quotient-conditioned O(2)^m port maps from generators.

    No legal state labels or orbit enumeration are used. For each mode, first
    compute the rotation-coset cross covariance directly from a 4x4 geometric
    series. If C = sum x*_k x_k^T, then reflected states satisfy

        y*_k = S* x*_k,   y_k = S~ x_k,

    so their cross covariance is exactly

        C_ref = S* C S~^T.
    """
    learned_a = np.asarray(learned_angles, dtype=np.float64).reshape(-1)
    projected_a = np.asarray(projected_angles, dtype=np.float64).reshape(-1)
    learned_s = np.asarray(learned_reflections, dtype=np.float64).reshape(-1, 2, 2)
    projected_s = np.asarray(projected_reflections, dtype=np.float64).reshape(-1, 2, 2)
    seed = torch.as_tensor(h0, dtype=torch.float64).reshape(-1, 2)
    modes = len(learned_a)
    if not (len(projected_a) == len(learned_s) == len(projected_s) == seed.shape[0] == modes):
        raise ValueError("mode counts must match")

    q_rot = []
    q_ref = []
    for i in range(modes):
        c = rotation_cross_covariance_from_generators(
            n, learned_a[i], projected_a[i], seed[i]
        )
        s_star = torch.as_tensor(projected_s[i], dtype=torch.float64)
        s_learned = torch.as_tensor(learned_s[i], dtype=torch.float64)
        c_ref = s_star @ c @ s_learned.T
        q_rot.append(procrustes_from_cross(c))
        q_ref.append(procrustes_from_cross(c_ref))
    return torch.block_diag(*q_rot), torch.block_diag(*q_ref)


def orbit_block_port_for_check(
    n: int,
    learned_angles: np.ndarray,
    projected_angles: np.ndarray,
    learned_reflections: np.ndarray,
    projected_reflections: np.ndarray,
    h0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_learned = joint.canonical_orbit(n, learned_angles, learned_reflections, h0)
    z_exact = joint.canonical_orbit(n, projected_angles, projected_reflections, h0)
    return joint.quotient_block_port(z_exact, z_learned, n)


@dataclass
class TransportRun:
    seed: int
    raw_accuracy: float
    generator_transport_accuracy: float
    generator_transport_min_margin: float
    max_q_difference_vs_orbit: float
    generator_alignment_error: float
    orbit_alignment_error: float
    post_relation_max_defect: float


def train_and_probe(
    *,
    n: int,
    modes: int,
    seed: int,
    train_length: int,
    train_steps: int,
    batch_size: int,
    max_increment: int,
    reflection_probability: float,
    reflection_scale: float,
    lr: float,
    random_start: bool,
) -> TransportRun:
    model = joint.train_model(
        n=n, modes=modes, seed=seed, train_length=train_length,
        train_steps=train_steps, batch_size=batch_size,
        max_increment=max_increment, reflection_probability=reflection_probability,
        lr=lr, random_start=random_start, reflection_scale=reflection_scale,
    )
    learned_a = model.angles.detach().cpu().numpy().astype(np.float64)
    learned_s = model.reflection_matrices().detach().cpu().numpy().astype(np.float64)
    projected_a, _ = base.project_angles_to_dn_characters(n, learned_a)
    projected_s = joint.project_reflections(learned_s)
    post = joint.relation_defects(n, projected_a, projected_s)
    h0 = model.h0.detach().cpu()
    W = model.readout.weight.detach().cpu()
    b = model.readout.bias.detach().cpu()

    z_learned = joint.canonical_orbit(n, learned_a, learned_s, h0)
    z_exact = joint.canonical_orbit(n, projected_a, projected_s, h0)
    raw_acc, _, _ = joint.metrics(z_exact, W, b)

    qg0, qg1 = generator_quotient_block_port(
        n, learned_a, projected_a, learned_s, projected_s, h0
    )
    qo0, qo1 = joint.quotient_block_port(z_exact, z_learned, n)
    z_gen = joint.apply_quotient_port(z_exact, n, qg0, qg1)
    z_orb = joint.apply_quotient_port(z_exact, n, qo0, qo1)
    gen_acc, gen_margin, _ = joint.metrics(z_gen, W, b)

    denom = max(float(torch.linalg.matrix_norm(z_learned).item()), 1e-12)
    gen_err = float(torch.linalg.matrix_norm(z_gen - z_learned).item()) / denom
    orb_err = float(torch.linalg.matrix_norm(z_orb - z_learned).item()) / denom
    qdiff = max(
        float(torch.linalg.matrix_norm(qg0 - qo0).item()),
        float(torch.linalg.matrix_norm(qg1 - qo1).item()),
    )
    return TransportRun(
        seed=seed,
        raw_accuracy=float(raw_acc),
        generator_transport_accuracy=float(gen_acc),
        generator_transport_min_margin=float(gen_margin),
        max_q_difference_vs_orbit=qdiff,
        generator_alignment_error=gen_err,
        orbit_alignment_error=orb_err,
        post_relation_max_defect=float(max(post)),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Compile D_n quotient block ports directly from pre/post generators")
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=2200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-increment", type=int, default=4)
    p.add_argument("--reflection-probability", type=float, default=0.25)
    p.add_argument("--reflection-scale", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--random-start", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = [train_and_probe(
        n=args.n, modes=args.modes, seed=seed, train_length=args.train_length,
        train_steps=args.train_steps, batch_size=args.batch_size,
        max_increment=args.max_increment, reflection_probability=args.reflection_probability,
        reflection_scale=args.reflection_scale, lr=args.lr, random_start=args.random_start,
    ) for seed in args.seeds]
    payload = {"config": vars(args), "results": [asdict(x) for x in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("seed raw generator-acc margin q-diff gen-align orbit-align post-defect")
    for x in rows:
        print(
            f"{x.seed:4d} {x.raw_accuracy:5.3f} {x.generator_transport_accuracy:13.3f} "
            f"{x.generator_transport_min_margin:+8.3f} {x.max_q_difference_vs_orbit:.3e} "
            f"{x.generator_alignment_error:.5f} {x.orbit_alignment_error:.5f} "
            f"{x.post_relation_max_defect:.2e}"
        )

if __name__ == "__main__":
    main()
