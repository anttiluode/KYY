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
MODULE_NAME = "cyclic_equivariant_port_for_positive_kernel"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "cyclic_equivariant_port_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
base = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = base
SPEC.loader.exec_module(base)


def positive_kernel_projection(
    n: int,
    angles: np.ndarray,
    state_zero: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
    """Project a decoder into the nonnegative correlation-kernel cone.

    First group-average the decoder into the exact C_n-equivariant subspace.
    Then, mode by mode, project the base template onto the nonnegative ray
    spanned by the state-0 vector for that mode.

    The resulting relative score kernel has form

        q(d) = b + sum_i alpha_i ||u_i||^2 cos(f_i d), alpha_i >= 0.

    Thus q(0)-q(d) is a sum of nonnegative 1-cos terms.
    """
    _, b_eq, w0 = base.project_cyclic_equivariant_decoder(n, angles, weight, bias)
    u = state_zero.to(torch.float64).reshape(-1, 2)
    w = w0.to(torch.float64).reshape(-1, 2)
    alpha: list[float] = []
    blocks: list[torch.Tensor] = []
    for ui, wi in zip(u, w):
        denom = float(torch.dot(ui, ui).item())
        if denom <= 1e-15:
            a = 0.0
        else:
            a = max(0.0, float(torch.dot(wi, ui).item()) / denom)
        alpha.append(a)
        blocks.append(a * ui)
    w0_pos = torch.stack(blocks, dim=0).reshape(-1)
    rows = [base.block_rotation(angles, j) @ w0_pos for j in range(n)]
    W_pos = torch.stack(rows, dim=0)
    return W_pos, b_eq, np.asarray(alpha, dtype=np.float64), w0_pos


def active_gcd(n: int, frequencies: np.ndarray, alpha: np.ndarray, tol: float = 1e-12) -> int:
    active = [int(f) for f, a in zip(frequencies.tolist(), alpha.tolist()) if a > tol]
    return abs(reduce(gcd, [int(n)] + active))


def positive_kernel_certificate(
    n: int,
    frequencies: np.ndarray,
    alpha: np.ndarray,
    *,
    tol: float = 1e-12,
) -> tuple[bool, int]:
    """Return exact correctness certificate for the positive-kernel port.

    q(0)-q(d) = sum_i c_i (1-cos(2*pi*f_i*d/n)), c_i >= 0.
    It is strictly positive for every nonidentity d iff the active characters
    have trivial common kernel. For C_n that is gcd(n, active frequencies)=1.
    """
    if np.any(alpha < -tol):
        return False, n
    g = active_gcd(n, frequencies, alpha, tol=tol)
    return g == 1, g


@dataclass
class PositiveKernelRun:
    seed: int
    inherited_accuracy: float
    equivariant_accuracy: float
    positive_kernel_accuracy: float
    prototype_accuracy: float
    positive_kernel_min_margin: float
    equivariant_min_margin: float
    prototype_min_margin: float
    alpha: list[float]
    active_modes: int
    min_positive_alpha: float
    character_gcd: int
    algebraically_certified: bool
    raw_parameter_count: int
    equivariant_parameter_count: int
    positive_kernel_parameter_count: int
    compression_raw_to_positive: float
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
    lr: float,
    random_start: bool,
) -> PositiveKernelRun:
    model = base.train_learned_model(
        n=n,
        modes=modes,
        seed=seed,
        train_length=train_length,
        train_steps=train_steps,
        batch_size=batch_size,
        max_increment=max_increment,
        lr=lr,
        random_start=random_start,
    )
    learned = model.angles.detach().cpu().numpy().astype(np.float64)
    projected, frequencies = base.base.project_angles_to_characters(n, learned)
    q = base.midpoint_port(n, learned, projected)
    z = base.exact_orbit(n, projected, model.h0.detach().cpu()) @ q
    W = model.readout.weight.detach().cpu().to(torch.float64)
    b = model.readout.bias.detach().cpu().to(torch.float64)

    inherited = base.readout_metrics(z, W, b)
    W_eq, b_eq, _ = base.project_cyclic_equivariant_decoder(n, projected, W, b)
    eq = base.readout_metrics(z, W_eq, b_eq)
    W_pos, b_pos, alpha, _ = positive_kernel_projection(
        n, projected, z[0], W, b
    )
    pos = base.readout_metrics(z, W_pos, b_pos)
    W_proto, b_proto = base.prototype_decoder(z)
    proto = base.readout_metrics(z, W_proto, b_proto)
    certified, g = positive_kernel_certificate(n, frequencies, alpha)

    positive = alpha[alpha > 1e-12]
    raw_params = int(W.numel() + b.numel())
    eq_params = int(2 * modes + 1)
    pos_params = int(modes + 1)
    return PositiveKernelRun(
        seed=int(seed),
        inherited_accuracy=float(inherited[0]),
        equivariant_accuracy=float(eq[0]),
        positive_kernel_accuracy=float(pos[0]),
        prototype_accuracy=float(proto[0]),
        positive_kernel_min_margin=float(pos[1]),
        equivariant_min_margin=float(eq[1]),
        prototype_min_margin=float(proto[1]),
        alpha=[float(x) for x in alpha.tolist()],
        active_modes=int(len(positive)),
        min_positive_alpha=float(np.min(positive)) if len(positive) else 0.0,
        character_gcd=int(g),
        algebraically_certified=bool(certified),
        raw_parameter_count=raw_params,
        equivariant_parameter_count=eq_params,
        positive_kernel_parameter_count=pos_params,
        compression_raw_to_positive=float(raw_params / pos_params),
        projected_frequencies=[int(x) for x in frequencies.tolist()],
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Project C_n output ports into an algebraically certifiable positive correlation kernel")
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=2200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-increment", type=int, default=4)
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
            lr=args.lr,
            random_start=args.random_start,
        )
        for seed in args.seeds
    ]
    payload = {"config": vars(args), "results": [asdict(x) for x in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("seed inherited equivariant positive prototype cert gcd active margin")
        for x in rows:
            print(
                f"{x.seed:4d} {x.inherited_accuracy:9.3f} {x.equivariant_accuracy:11.3f} "
                f"{x.positive_kernel_accuracy:8.3f} {x.prototype_accuracy:9.3f} "
                f"{str(x.algebraically_certified):>5s} {x.character_gcd:4d} "
                f"{x.active_modes:6d} {x.positive_kernel_min_margin:+.3f}"
            )


if __name__ == "__main__":
    main()
