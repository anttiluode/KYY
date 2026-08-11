from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "harmonic_training_for_equivariant_port"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "harmonic_training_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
base = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = base
SPEC.loader.exec_module(base)


def block_rotation(angles: np.ndarray | torch.Tensor, power: int = 1) -> torch.Tensor:
    """Column-vector block representation rho(power) = diag R(power*theta_i)."""
    a = torch.as_tensor(angles, dtype=torch.float64).reshape(-1)
    blocks: list[torch.Tensor] = []
    for theta in a:
        t = float(power) * theta
        c, s = torch.cos(t), torch.sin(t)
        blocks.append(torch.stack((torch.stack((c, -s)), torch.stack((s, c)))))
    return torch.block_diag(*blocks)


def midpoint_port(n: int, learned: np.ndarray, projected: np.ndarray) -> torch.Tensor:
    """Pass-34 zero-label cyclic recentering in row-vector convention."""
    phi = -0.5 * (n - 1) * (np.asarray(projected) - np.asarray(learned))
    phi = np.arctan2(np.sin(phi), np.cos(phi))
    blocks = []
    for p in phi:
        c, s = math.cos(float(p)), math.sin(float(p))
        blocks.append(torch.tensor([[c, s], [-s, c]], dtype=torch.float64))
    return torch.block_diag(*blocks)


def exact_orbit(n: int, angles: np.ndarray | torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """Rows are the n exact cyclic hidden states."""
    a = torch.as_tensor(angles, dtype=torch.float64).reshape(1, -1)
    seed = h0.to(torch.float64).reshape(1, -1, 2)
    k = torch.arange(n, dtype=torch.float64).reshape(-1, 1)
    phase = k * a
    c, s = torch.cos(phase), torch.sin(phase)
    x0, y0 = seed[..., 0], seed[..., 1]
    x = c * x0 - s * y0
    y = s * x0 + c * y0
    return torch.stack((x, y), dim=-1).reshape(n, -1)


def readout_metrics(
    z: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> tuple[float, float, int]:
    n = z.shape[0]
    W = weight.to(torch.float64)
    b = bias.to(torch.float64)
    logits = z @ W.T + b
    labels = torch.arange(n)
    pred = logits.argmax(dim=-1)
    rows = torch.arange(n)
    true = logits[rows, labels]
    competitor = logits.clone()
    competitor[rows, labels] = -torch.inf
    margin = true - competitor.max(dim=-1).values
    correct = int((pred == labels).sum().item())
    return correct / n, float(margin.min().item()), n - correct


def project_cyclic_equivariant_decoder(
    n: int,
    angles: np.ndarray | torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthogonal projection of a linear C_n decoder onto the equivariant subspace.

    Hidden columns transform by rho(j). Output classes transform by the regular
    cyclic shift. Exact equivariance requires class-j weight columns

        w_j = rho(j) w_0

    and one shared bias. The least-squares projection therefore has

        w_0 = mean_j rho(j)^T w_j.

    Returns (W_eq, b_eq, w0), where W_eq has the same explicit n x d shape as
    the learned decoder but is generated from only d+1 free scalars.
    """
    W = weight.to(torch.float64)
    b = bias.to(torch.float64)
    if W.shape[0] != n:
        raise ValueError("decoder row count must equal n")
    d = W.shape[1]
    w0 = torch.zeros(d, dtype=torch.float64)
    for j in range(n):
        rho = block_rotation(angles, j)
        w0 += rho.T @ W[j]
    w0 /= float(n)

    rows = []
    for j in range(n):
        rho = block_rotation(angles, j)
        rows.append(rho @ w0)
    W_eq = torch.stack(rows, dim=0)
    b_eq = torch.full((n,), float(b.mean().item()), dtype=torch.float64)
    return W_eq, b_eq, w0


def circulant_logit_defect(z: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> float:
    logits = z.to(torch.float64) @ W.to(torch.float64).T + b.to(torch.float64)
    ref = logits[0]
    defect = 0.0
    for k in range(logits.shape[0]):
        defect = max(defect, float(torch.max(torch.abs(logits[k] - torch.roll(ref, k))).item()))
    return defect


def prototype_decoder(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Matched-filter decoder for an equal-norm orbit."""
    W = z.to(torch.float64).clone()
    norms = torch.sum(W * W, dim=1)
    # Nearest-prototype logits: <z,w_j> - 1/2 ||w_j||^2.
    b = -0.5 * norms
    return W, b


def train_learned_model(
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
) -> base.RotaryModTracker:
    base.seed_everything(seed)
    rng = np.random.default_rng(seed + 1009 * n)
    initial = rng.uniform(-math.pi, math.pi, size=modes)
    model = base.RotaryModTracker(n, initial, learn_angles=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(train_steps):
        x, y = base.generate_batch(
            n, batch_size, train_length, max_increment, random_start=random_start
        )
        logits = model(x)
        loss = criterion(logits.reshape(-1, n), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


@dataclass
class PortRun:
    seed: int
    inherited_accuracy: float
    inherited_min_margin: float
    inherited_mistakes: int
    equivariant_accuracy: float
    equivariant_min_margin: float
    equivariant_mistakes: int
    prototype_accuracy: float
    prototype_min_margin: float
    prototype_mistakes: int
    decoder_projection_relative_error: float
    inherited_circulant_logit_defect: float
    equivariant_circulant_logit_defect: float
    bias_std: float
    projected_frequencies: list[int]
    max_character_degree: int
    fourier_support_bound: int
    raw_decoder_parameter_count: int
    equivariant_parameter_count: int
    compression_ratio: float


def train_and_probe(
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
) -> PortRun:
    model = train_learned_model(
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
    projected, frequencies = base.project_angles_to_characters(n, learned)
    q = midpoint_port(n, learned, projected)
    z = exact_orbit(n, projected, model.h0.detach().cpu()) @ q
    W = model.readout.weight.detach().cpu().to(torch.float64)
    b = model.readout.bias.detach().cpu().to(torch.float64)

    inherited_acc, inherited_margin, inherited_bad = readout_metrics(z, W, b)
    W_eq, b_eq, _ = project_cyclic_equivariant_decoder(n, projected, W, b)
    eq_acc, eq_margin, eq_bad = readout_metrics(z, W_eq, b_eq)
    W_proto, b_proto = prototype_decoder(z)
    proto_acc, proto_margin, proto_bad = readout_metrics(z, W_proto, b_proto)

    denom = max(float(torch.linalg.matrix_norm(W).item()), 1e-12)
    proj_err = float(torch.linalg.matrix_norm(W_eq - W).item()) / denom
    raw_params = int(W.numel() + b.numel())
    eq_params = int(W.shape[1] + 1)
    f = np.asarray(frequencies, dtype=np.int64) % n
    canonical_degree = np.minimum(f, n - f)
    return PortRun(
        seed=seed,
        inherited_accuracy=float(inherited_acc),
        inherited_min_margin=float(inherited_margin),
        inherited_mistakes=int(inherited_bad),
        equivariant_accuracy=float(eq_acc),
        equivariant_min_margin=float(eq_margin),
        equivariant_mistakes=int(eq_bad),
        prototype_accuracy=float(proto_acc),
        prototype_min_margin=float(proto_margin),
        prototype_mistakes=int(proto_bad),
        decoder_projection_relative_error=proj_err,
        inherited_circulant_logit_defect=circulant_logit_defect(z, W, b),
        equivariant_circulant_logit_defect=circulant_logit_defect(z, W_eq, b_eq),
        bias_std=float(torch.std(b, unbiased=False).item()),
        projected_frequencies=[int(x) for x in f.tolist()],
        max_character_degree=int(np.max(canonical_degree)) if len(canonical_degree) else 0,
        fourier_support_bound=int(2 * modes + 1),
        raw_decoder_parameter_count=raw_params,
        equivariant_parameter_count=eq_params,
        compression_ratio=float(raw_params / max(eq_params, 1)),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Audit and compress the cyclic learned output port by exact C_n symmetry")
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
        train_and_probe(
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
        return
    print("seed inherited eq prototype projerr circ(raw/eq) degree params(raw/eq)")
    for x in rows:
        print(
            f"{x.seed:4d} {x.inherited_accuracy:8.3f} {x.equivariant_accuracy:5.3f} "
            f"{x.prototype_accuracy:9.3f} {x.decoder_projection_relative_error:7.3f} "
            f"{x.inherited_circulant_logit_defect:.2e}/{x.equivariant_circulant_logit_defect:.2e} "
            f"{x.max_character_degree:6d} "
            f"{x.raw_decoder_parameter_count}/{x.equivariant_parameter_count}"
        )


if __name__ == "__main__":
    main()
