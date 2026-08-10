from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "dihedral_joint_for_stabilizer_certificate"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "dihedral_joint_legalization_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
joint = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = joint
SPEC.loader.exec_module(joint)


def prototype_decoder(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    W = z.to(torch.float64).clone()
    b = -0.5 * torch.sum(W * W, dim=1)
    return W, b


def _crt_pair(a: int, m: int, b: int, q: int) -> tuple[int, int] | None:
    """Intersect x=a mod m and x=b mod q. Return canonical (r,lcm)."""
    if m <= 0 or q <= 0:
        raise ValueError("moduli must be positive")
    g = math.gcd(m, q)
    delta = b - a
    if delta % g:
        return None
    m1, q1 = m // g, q // g
    # m1 and q1 are coprime. Handle q1=1 separately because pow(...,-1,1)
    # is not portable across all supported Python versions.
    if q1 == 1:
        t = 0
    else:
        t = ((delta // g) * pow(m1, -1, q1)) % q1
    lcm = m * q1
    return (a + m * t) % lcm, lcm


def solve_linear_congruence(f: int, t: int, n: int) -> tuple[int, int] | None:
    """Solve f*k=t (mod n) as one residue class k=a (mod m)."""
    f %= n
    t %= n
    d = math.gcd(f, n)
    if t % d:
        return None
    if f == 0:
        # Then t must be zero (handled by divisibility); all k are solutions.
        return 0, 1
    fp, tp, np_ = f // d, t // d, n // d
    inv = pow(fp, -1, np_)
    return (inv * tp) % np_, np_


def reflection_axis(matrix: np.ndarray | torch.Tensor) -> float:
    """Axis angle beta (mod pi) for a 2D orthogonal reflection."""
    s = np.asarray(matrix, dtype=np.float64).reshape(2, 2)
    # Reflection form [[cos 2b, sin 2b],[sin 2b,-cos 2b]].
    return 0.5 * math.atan2(float(s[0, 1] + s[1, 0]), float(s[0, 0] - s[1, 1]))


@dataclass
class StabilizerCertificate:
    rotation_kernel_size: int
    reflection_stabilizer_exists: bool
    reflection_candidate_k: int | None
    min_reflection_grid_gap: float
    max_reflection_grid_gap: float
    trivial_stabilizer_certified: bool
    active_modes: int


def dihedral_stabilizer_certificate(
    n: int,
    frequencies: np.ndarray | list[int],
    reflections: np.ndarray | torch.Tensor,
    h0: torch.Tensor,
    *,
    grid_tol: float = 1e-8,
    amplitude_tol: float = 1e-15,
) -> StabilizerCertificate:
    """O(m log n) stabilizer certificate for block-planar D_n representations.

    Rotations act as R(2*pi*f_i/n), each S_i is a 2D reflection, and the seed
    has mode vectors u_i.

    Rotation stabilizer:
        r^k fixes all active modes iff f_i k = 0 (mod n) for all i.
        The kernel size is gcd(n, active frequencies).

    Reflection stabilizer:
        S_i R(2*pi*f_i*k/n) u_i = u_i
    iff, writing beta_i for the reflection-axis angle and gamma_i for the seed
    angle,
        f_i k = n (beta_i-gamma_i)/pi  (mod n).
    The left side is integral mod n. If the right side is off the integer grid
    for any active mode there is no reflected stabilizer. Otherwise solve and
    intersect the resulting modular congruences without enumerating k.
    """
    f = np.asarray(frequencies, dtype=np.int64).reshape(-1) % int(n)
    s = np.asarray(reflections, dtype=np.float64).reshape(-1, 2, 2)
    u = h0.detach().cpu().to(torch.float64).numpy().reshape(-1, 2)
    if len(f) != len(s) or len(f) != len(u):
        raise ValueError("mode counts must match")
    amp2 = np.sum(u * u, axis=1)
    active = np.where(amp2 > amplitude_tol)[0]
    if len(active) == 0:
        return StabilizerCertificate(
            rotation_kernel_size=n,
            reflection_stabilizer_exists=True,
            reflection_candidate_k=0,
            min_reflection_grid_gap=0.0,
            max_reflection_grid_gap=0.0,
            trivial_stabilizer_certified=False,
            active_modes=0,
        )

    rot_gcd = abs(reduce(gcd, [int(n)] + [int(f[i]) for i in active]))

    residue, modulus = 0, 1
    gaps: list[float] = []
    reflection_possible = True
    for i in active:
        beta = reflection_axis(s[i])
        gamma = math.atan2(float(u[i, 1]), float(u[i, 0]))
        target = (n * (beta - gamma) / math.pi) % n
        nearest = int(round(target))
        # Circular distance to the integer lattice modulo n. Since nearest may
        # be n, reduce only after the real-valued gap is measured.
        gap = abs(target - nearest)
        gap = min(gap, abs(gap - n))
        gaps.append(float(gap))
        if gap > grid_tol:
            reflection_possible = False
            break
        eq = solve_linear_congruence(int(f[i]), nearest % n, n)
        if eq is None:
            reflection_possible = False
            break
        merged = _crt_pair(residue, modulus, eq[0], eq[1])
        if merged is None:
            reflection_possible = False
            break
        residue, modulus = merged

    candidate = int(residue % n) if reflection_possible else None
    return StabilizerCertificate(
        rotation_kernel_size=int(rot_gcd),
        reflection_stabilizer_exists=bool(reflection_possible),
        reflection_candidate_k=candidate,
        min_reflection_grid_gap=float(min(gaps)) if gaps else 0.0,
        max_reflection_grid_gap=float(max(gaps)) if gaps else 0.0,
        trivial_stabilizer_certified=bool(rot_gcd == 1 and not reflection_possible),
        active_modes=int(len(active)),
    )


@dataclass
class Run:
    seed: int
    frequencies: list[int]
    relation_rotation_defect: float
    relation_reflection_defect: float
    relation_conjugation_defect: float
    relation_orthogonality_defect: float
    prototype_accuracy: float
    prototype_min_margin: float
    prototype_mistakes: int
    certificate: dict[str, object]


def run_one(
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
) -> Run:
    model = joint.train_model(
        n=n,
        modes=modes,
        seed=seed,
        train_length=train_length,
        train_steps=train_steps,
        batch_size=batch_size,
        max_increment=max_increment,
        reflection_probability=reflection_probability,
        lr=lr,
        random_start=random_start,
        reflection_scale=reflection_scale,
    )
    learned_a = model.angles.detach().cpu().numpy().astype(np.float64)
    projected_a, frequencies = joint.base.project_angles_to_dn_characters(n, learned_a)
    learned_s = model.reflection_matrices().detach().cpu().numpy().astype(np.float64)
    projected_s = joint.project_reflections(learned_s)
    defects = joint.relation_defects(n, projected_a, projected_s)

    z = joint.canonical_orbit(n, projected_a, projected_s, model.h0.detach().cpu())
    W, b = prototype_decoder(z)
    acc, margin, mistakes = joint.metrics(z, W, b)
    cert = dihedral_stabilizer_certificate(
        n,
        frequencies,
        projected_s,
        model.h0.detach().cpu(),
    )
    return Run(
        seed=int(seed),
        frequencies=[int(x) for x in frequencies.tolist()],
        relation_rotation_defect=float(defects[0]),
        relation_reflection_defect=float(defects[1]),
        relation_conjugation_defect=float(defects[2]),
        relation_orthogonality_defect=float(defects[3]),
        prototype_accuracy=float(acc),
        prototype_min_margin=float(margin),
        prototype_mistakes=int(mistakes),
        certificate=asdict(cert),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Stabilizer-based certificate for legalized harmonic D_n prototype ports")
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
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
    rows = [
        run_one(
            n=args.n,
            modes=args.modes,
            seed=seed,
            train_length=args.train_length,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            max_increment=args.max_increment,
            reflection_probability=args.reflection_probability,
            reflection_scale=args.reflection_scale,
            lr=args.lr,
            random_start=args.random_start,
        )
        for seed in args.seeds
    ]
    payload = {"config": vars(args), "results": [asdict(x) for x in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("seed proto margin rotkernel reflstab cert mingap")
        for x in rows:
            c = x.certificate
            print(
                f"{x.seed:4d} {x.prototype_accuracy:5.3f} {x.prototype_min_margin:+.3f} "
                f"{int(c['rotation_kernel_size']):9d} {str(c['reflection_stabilizer_exists']):>8s} "
                f"{str(c['trivial_stabilizer_certified']):>5s} {float(c['min_reflection_grid_gap']):.3e}"
            )


if __name__ == "__main__":
    main()
