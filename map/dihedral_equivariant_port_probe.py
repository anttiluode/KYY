from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "dihedral_stabilizer_for_equivariant_port"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "dihedral_stabilizer_certificate.py"
)
assert SPEC is not None and SPEC.loader is not None
certmod = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = certmod
SPEC.loader.exec_module(certmod)
joint = certmod.joint


def representation_matrix(
    angles: np.ndarray | torch.Tensor,
    reflections: np.ndarray | torch.Tensor,
    k: int,
    branch: int,
) -> torch.Tensor:
    """Block-diagonal rho(r^k) or rho(s r^k) in column-vector convention."""
    a = np.asarray(angles, dtype=np.float64).reshape(-1)
    s = np.asarray(reflections, dtype=np.float64).reshape(-1, 2, 2)
    blocks: list[torch.Tensor] = []
    for theta, refl in zip(a, s):
        r = joint.rotation_matrix(float(k) * float(theta), dtype=torch.float64)
        if branch:
            block = torch.as_tensor(refl, dtype=torch.float64) @ r
        else:
            block = r
        blocks.append(block)
    return torch.block_diag(*blocks)


def project_dihedral_equivariant_decoder(
    n: int,
    angles: np.ndarray | torch.Tensor,
    reflections: np.ndarray | torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthogonal projection onto Hom_{D_n}(rho, regular-output action).

    With class convention [r^k, s r^k], exact equivariance requires

        w_g = rho(g) w_e,
        b_g = constant.

    Therefore the least-squares base template is the group average

        w_e = (1/|G|) sum_g rho(g)^T w_g.
    """
    W = weight.to(torch.float64)
    b = bias.to(torch.float64)
    if W.shape[0] != 2 * n:
        raise ValueError("decoder must have 2n rows")
    d = W.shape[1]
    w0 = torch.zeros(d, dtype=torch.float64)
    idx = 0
    for branch in (0, 1):
        for k in range(n):
            rho = representation_matrix(angles, reflections, k, branch)
            w0 += rho.T @ W[idx]
            idx += 1
    w0 /= float(2 * n)

    rows: list[torch.Tensor] = []
    for branch in (0, 1):
        for k in range(n):
            rho = representation_matrix(angles, reflections, k, branch)
            rows.append(rho @ w0)
    W_eq = torch.stack(rows, dim=0)
    b_eq = torch.full((2 * n,), float(b.mean().item()), dtype=torch.float64)
    return W_eq, b_eq, w0


def project_positive_orbit_kernel(
    n: int,
    angles: np.ndarray | torch.Tensor,
    reflections: np.ndarray | torch.Tensor,
    h0: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
    """Project the equivariant base template onto nonnegative seed rays per mode."""
    _, b_eq, w0 = project_dihedral_equivariant_decoder(
        n, angles, reflections, weight, bias
    )
    u = h0.to(torch.float64).reshape(-1, 2)
    w = w0.reshape(-1, 2)
    alpha: list[float] = []
    blocks: list[torch.Tensor] = []
    for ui, wi in zip(u, w):
        denom = float(torch.dot(ui, ui).item())
        a = 0.0 if denom <= 1e-15 else max(0.0, float(torch.dot(wi, ui).item()) / denom)
        alpha.append(a)
        blocks.append(a * ui)
    w0_pos = torch.stack(blocks, dim=0).reshape(-1)
    rows: list[torch.Tensor] = []
    for branch in (0, 1):
        for k in range(n):
            rho = representation_matrix(angles, reflections, k, branch)
            rows.append(rho @ w0_pos)
    return torch.stack(rows, dim=0), b_eq, np.asarray(alpha, dtype=np.float64), w0_pos


def equivariance_defect(
    n: int,
    angles: np.ndarray | torch.Tensor,
    reflections: np.ndarray | torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> float:
    W = W.to(torch.float64)
    b = b.to(torch.float64)
    w0 = W[0]
    b0 = b[0]
    defect = 0.0
    idx = 0
    for branch in (0, 1):
        for k in range(n):
            rho = representation_matrix(angles, reflections, k, branch)
            defect = max(defect, float(torch.max(torch.abs(W[idx] - rho @ w0)).item()))
            defect = max(defect, abs(float((b[idx] - b0).item())))
            idx += 1
    return defect


@dataclass
class Run:
    seed: int
    raw_accuracy: float
    raw_min_margin: float
    equivariant_accuracy: float
    equivariant_min_margin: float
    positive_kernel_accuracy: float
    positive_kernel_min_margin: float
    prototype_accuracy: float
    prototype_min_margin: float
    decoder_projection_relative_error: float
    learned_equivariance_defect: float
    projected_equivariance_defect: float
    alpha: list[float]
    active_positive_modes: int
    min_positive_alpha: float
    stabilizer_certificate: dict[str, object]
    raw_parameter_count: int
    equivariant_parameter_count: int
    positive_kernel_parameter_count: int
    projected_frequencies: list[int]


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
    z = joint.canonical_orbit(n, projected_a, projected_s, model.h0.detach().cpu())
    W = model.readout.weight.detach().cpu().to(torch.float64)
    b = model.readout.bias.detach().cpu().to(torch.float64)

    raw = joint.metrics(z, W, b)
    W_eq, b_eq, _ = project_dihedral_equivariant_decoder(
        n, projected_a, projected_s, W, b
    )
    eq = joint.metrics(z, W_eq, b_eq)
    W_pos, b_pos, alpha, _ = project_positive_orbit_kernel(
        n, projected_a, projected_s, model.h0.detach().cpu(), W, b
    )
    pos = joint.metrics(z, W_pos, b_pos)
    W_proto, b_proto = certmod.prototype_decoder(z)
    proto = joint.metrics(z, W_proto, b_proto)
    stab = certmod.dihedral_stabilizer_certificate(
        n, frequencies, projected_s, model.h0.detach().cpu()
    )

    denom = max(float(torch.linalg.matrix_norm(W).item()), 1e-12)
    projerr = float(torch.linalg.matrix_norm(W_eq - W).item()) / denom
    positive = alpha[alpha > 1e-12]
    return Run(
        seed=int(seed),
        raw_accuracy=float(raw[0]),
        raw_min_margin=float(raw[1]),
        equivariant_accuracy=float(eq[0]),
        equivariant_min_margin=float(eq[1]),
        positive_kernel_accuracy=float(pos[0]),
        positive_kernel_min_margin=float(pos[1]),
        prototype_accuracy=float(proto[0]),
        prototype_min_margin=float(proto[1]),
        decoder_projection_relative_error=projerr,
        learned_equivariance_defect=equivariance_defect(n, projected_a, projected_s, W, b),
        projected_equivariance_defect=equivariance_defect(n, projected_a, projected_s, W_eq, b_eq),
        alpha=[float(x) for x in alpha.tolist()],
        active_positive_modes=int(len(positive)),
        min_positive_alpha=float(np.min(positive)) if len(positive) else 0.0,
        stabilizer_certificate=asdict(stab),
        raw_parameter_count=int(W.numel() + b.numel()),
        equivariant_parameter_count=int(2 * modes + 1),
        positive_kernel_parameter_count=int(modes + 1),
        projected_frequencies=[int(x) for x in frequencies.tolist()],
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Compress jointly legalized D_n learned output ports by exact group equivariance")
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
        print("seed raw eq positive proto margins(eq/positive) active cert")
        for x in rows:
            cert = x.stabilizer_certificate
            print(
                f"{x.seed:4d} {x.raw_accuracy:5.3f} {x.equivariant_accuracy:5.3f} "
                f"{x.positive_kernel_accuracy:8.3f} {x.prototype_accuracy:5.3f} "
                f"{x.equivariant_min_margin:+.3f}/{x.positive_kernel_min_margin:+.3f} "
                f"{x.active_positive_modes:6d} {str(cert['trivial_stabilizer_certified']):>5s}"
            )


if __name__ == "__main__":
    main()
