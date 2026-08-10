from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def project_angles_to_dn_characters(n: int, angles: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Snap rotation blocks onto exact n-th roots of unity."""
    a = np.asarray(angles, dtype=np.float64).reshape(-1)
    f = np.rint(n * a / (2.0 * math.pi)).astype(np.int64)
    return 2.0 * math.pi * f.astype(np.float64) / n, np.mod(f, n)


def rotation_relation_defect(n: int, angles: np.ndarray | torch.Tensor) -> float:
    a = np.asarray(angles, dtype=np.float64).reshape(-1)
    return float(np.max(2.0 * np.abs(np.sin(0.5 * n * a))))


def default_h0(modes: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generic equal-amplitude seed not fixed by the reflection generator."""
    i = torch.arange(modes, dtype=dtype)
    gamma = 2.0 * math.pi * (i + 0.37) / (2.0 * modes + 1.0)
    scale = 1.0 / math.sqrt(modes)
    return torch.stack((torch.cos(gamma), torch.sin(gamma)), dim=-1) * scale


def generate_batch(
    n: int,
    batch_size: int,
    length: int,
    max_increment: int,
    reflection_probability: float,
    *,
    random_start: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random words in D_n with left multiplication by rotations/reflection.

    State convention is g = r^k for branch b=0 and g = s r^k for b=1.
    Rotation token j left-multiplies by r^j, hence k += j on b=0 and
    k -= j on b=1. Reflection token n left-multiplies by s and toggles b.

    With random_start, the first token is a uniformly random rotation r^j.
    This exposes every rotational coordinate at short horizon without adding
    any learned token-specific operator beyond the common generator angles.
    """
    if not 0.0 <= reflection_probability <= 1.0:
        raise ValueError("reflection_probability must lie in [0,1]")
    if length < 1:
        raise ValueError("length must be >= 1")

    rot = torch.randint(0, max_increment + 1, (batch_size, length))
    is_ref = torch.rand(batch_size, length) < reflection_probability
    is_ref[:, 0] = False
    if random_start:
        rot[:, 0] = torch.randint(0, n, (batch_size,))
    tokens = torch.where(is_ref, torch.full_like(rot, n), rot)

    branch = torch.zeros(batch_size, dtype=torch.long)
    k = torch.zeros(batch_size, dtype=torch.long)
    y = torch.empty((batch_size, length), dtype=torch.long)
    for t in range(length):
        tok = tokens[:, t]
        ref = tok == n
        inc = torch.where(ref, torch.zeros_like(tok), tok)
        signed = torch.where(branch == 0, inc, -inc)
        k = torch.where(ref, k, (k + signed).remainder(n))
        branch = torch.where(ref, 1 - branch, branch)
        y[:, t] = branch * n + k
    return tokens, y


class DihedralHarmonicTracker(nn.Module):
    """Harmonic D_n tracker with exact reflection relation and learned rotations.

    Each complex mode carries a planar representation. Rotation tokens apply
    R(j theta_i); the reflection token applies F(x,y)=(x,-y). Therefore

        F^2 = I
        F R(theta) F = R(-theta)

    identically for every learned theta. Only the finite-order relation
    R(theta)^n = I needs post-training legalization.
    """

    def __init__(self, n: int, angles: np.ndarray, learn_angles: bool = True):
        super().__init__()
        self.n = int(n)
        a = torch.tensor(np.asarray(angles), dtype=torch.float32)
        if learn_angles:
            self.angles = nn.Parameter(a.clone())
        else:
            self.register_buffer("angles", a)
        self.modes = int(a.numel())
        self.register_buffer("h0", default_h0(self.modes))
        self.readout = nn.Linear(2 * self.modes, 2 * self.n)

    def step(self, h: torch.Tensor, token: torch.Tensor, angle_error: float = 0.0) -> torch.Tensor:
        ref = token == self.n
        inc = torch.where(ref, torch.zeros_like(token), token).float().unsqueeze(-1)
        theta = inc * (self.angles + float(angle_error)).unsqueeze(0)
        c, s = torch.cos(theta), torch.sin(theta)
        x, y = h[..., 0], h[..., 1]
        rot = torch.stack((c * x - s * y, s * x + c * y), dim=-1)
        refl = torch.stack((x, -y), dim=-1)
        return torch.where(ref[:, None, None], refl, rot)

    def forward(self, tokens: torch.Tensor, angle_error: float = 0.0, port: torch.Tensor | None = None) -> torch.Tensor:
        bsz, length = tokens.shape
        h = self.h0.unsqueeze(0).expand(bsz, -1, -1)
        outs: list[torch.Tensor] = []
        for t in range(length):
            h = self.step(h, tokens[:, t], angle_error=angle_error)
            flat = h.reshape(bsz, -1)
            if port is not None:
                flat = flat @ port
            outs.append(self.readout(flat))
        return torch.stack(outs, dim=1)


def orbit_prototypes(n: int, angles: np.ndarray | torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """Canonical complete orbit [r^k, s r^k], k=0..n-1."""
    a = torch.as_tensor(angles, dtype=torch.float64).reshape(1, -1)
    seed = h0.to(torch.float64).reshape(1, -1, 2)
    k = torch.arange(n, dtype=torch.float64).reshape(-1, 1)
    theta = k * a
    c, s = torch.cos(theta), torch.sin(theta)
    x0 = seed[..., 0]
    y0 = seed[..., 1]
    x = c * x0 - s * y0
    y = s * x0 + c * y0
    rot = torch.stack((x, y), dim=-1)
    refl = torch.stack((x, -y), dim=-1)
    return torch.cat((rot, refl), dim=0).reshape(2 * n, -1)


def readout_metrics(z: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[float, float, int]:
    nstates = z.shape[0]
    W = weight.to(torch.float64)
    b = bias.to(torch.float64)
    logits = z @ W.T + b
    labels = torch.arange(nstates)
    pred = logits.argmax(dim=-1)
    rows = torch.arange(nstates)
    true = logits[rows, labels]
    competitor = logits.clone()
    competitor[rows, labels] = -torch.inf
    margin = true - competitor.max(dim=-1).values
    correct = int((pred == labels).sum().item())
    return correct / nstates, float(margin.min().item()), nstates - correct


def midpoint_mode_port(n: int, learned: np.ndarray, projected: np.ndarray) -> torch.Tensor:
    """Cyclic Pass-34 recentering, deliberately tested as a non-Abelian control."""
    phi = -0.5 * (n - 1) * (np.asarray(projected) - np.asarray(learned))
    phi = np.arctan2(np.sin(phi), np.cos(phi))
    blocks = []
    for p in phi:
        c, s = math.cos(float(p)), math.sin(float(p))
        # Row-vector convention: [x,y] @ Q equals active rotation by +p.
        blocks.append(torch.tensor([[c, s], [-s, c]], dtype=torch.float64))
    return torch.block_diag(*blocks)


def orthogonal_procrustes_port(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return orthogonal Q minimizing ||source Q - target||_F."""
    cross = source.T @ target
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    return u @ vh


def block_orthogonal_procrustes_port(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape != target.shape or source.shape[1] % 2:
        raise ValueError("source/target must have equal even feature dimension")
    blocks = []
    for j in range(source.shape[1] // 2):
        x = source[:, 2 * j : 2 * j + 2]
        y = target[:, 2 * j : 2 * j + 2]
        blocks.append(orthogonal_procrustes_port(x, y))
    return torch.block_diag(*blocks)


def evaluate(
    model: DihedralHarmonicTracker,
    *,
    n: int,
    lengths: list[int],
    batch_size: int,
    max_increment: int,
    reflection_probability: float,
    random_start: bool,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    with torch.no_grad():
        for length in lengths:
            x, y = generate_batch(
                n,
                batch_size,
                length,
                max_increment,
                reflection_probability,
                random_start=random_start,
            )
            pred = model(x).argmax(dim=-1)
            out[str(length)] = float((pred == y).float().mean().item())
    return out


@dataclass
class DihedralRun:
    seed: int
    n: int
    modes: int
    pre_clean_accuracy: dict[str, float]
    pre_relation_defect: float
    post_relation_defect: float
    projected_frequencies: list[int]
    learned_orbit_accuracy: float
    raw_projected_accuracy: float
    raw_projected_min_margin: float
    midpoint_accuracy: float
    midpoint_min_margin: float
    block_procrustes_accuracy: float
    block_procrustes_min_margin: float
    full_procrustes_accuracy: float
    full_procrustes_min_margin: float
    block_alignment_error: float
    full_alignment_error: float


def train_and_probe(
    *,
    n: int,
    modes: int,
    seed: int,
    train_length: int,
    train_steps: int,
    batch_size: int,
    eval_batch_size: int,
    max_increment: int,
    reflection_probability: float,
    lr: float,
    random_start: bool,
) -> DihedralRun:
    seed_everything(seed)
    rng = np.random.default_rng(seed + 1009 * n)
    initial = rng.uniform(-math.pi, math.pi, size=modes)
    model = DihedralHarmonicTracker(n, initial, learn_angles=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(train_steps):
        x, y = generate_batch(
            n,
            batch_size,
            train_length,
            max_increment,
            reflection_probability,
            random_start=random_start,
        )
        logits = model(x)
        loss = criterion(logits.reshape(-1, 2 * n), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    pre_clean = evaluate(
        model,
        n=n,
        lengths=[train_length, 64, 256, 1024],
        batch_size=eval_batch_size,
        max_increment=max_increment,
        reflection_probability=reflection_probability,
        random_start=random_start,
    )

    learned = model.angles.detach().cpu().numpy().astype(np.float64)
    projected, frequencies = project_angles_to_dn_characters(n, learned)
    pre_defect = rotation_relation_defect(n, learned)
    post_defect = rotation_relation_defect(n, projected)
    h0 = model.h0.detach().cpu()
    W = model.readout.weight.detach().cpu()
    b = model.readout.bias.detach().cpu()

    z_learned = orbit_prototypes(n, learned, h0)
    z_projected = orbit_prototypes(n, projected, h0)
    learned_acc, _, _ = readout_metrics(z_learned, W, b)
    raw_acc, raw_margin, _ = readout_metrics(z_projected, W, b)

    midpoint_q = midpoint_mode_port(n, learned, projected)
    z_mid = z_projected @ midpoint_q
    mid_acc, mid_margin, _ = readout_metrics(z_mid, W, b)

    block_q = block_orthogonal_procrustes_port(z_projected, z_learned)
    z_block = z_projected @ block_q
    block_acc, block_margin, _ = readout_metrics(z_block, W, b)

    full_q = orthogonal_procrustes_port(z_projected, z_learned)
    z_full = z_projected @ full_q
    full_acc, full_margin, _ = readout_metrics(z_full, W, b)

    denom = float(torch.linalg.matrix_norm(z_learned).item())
    block_err = float(torch.linalg.matrix_norm(z_block - z_learned).item()) / max(denom, 1e-12)
    full_err = float(torch.linalg.matrix_norm(z_full - z_learned).item()) / max(denom, 1e-12)

    return DihedralRun(
        seed=seed,
        n=n,
        modes=modes,
        pre_clean_accuracy=pre_clean,
        pre_relation_defect=float(pre_defect),
        post_relation_defect=float(post_defect),
        projected_frequencies=[int(x) for x in frequencies.tolist()],
        learned_orbit_accuracy=float(learned_acc),
        raw_projected_accuracy=float(raw_acc),
        raw_projected_min_margin=float(raw_margin),
        midpoint_accuracy=float(mid_acc),
        midpoint_min_margin=float(mid_margin),
        block_procrustes_accuracy=float(block_acc),
        block_procrustes_min_margin=float(block_margin),
        full_procrustes_accuracy=float(full_acc),
        full_procrustes_min_margin=float(full_margin),
        block_alignment_error=block_err,
        full_alignment_error=full_err,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Non-Abelian D_n harmonic legalization and zero-label port alignment")
    p.add_argument("--n", type=int, default=31)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=1800)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-increment", type=int, default=4)
    p.add_argument("--reflection-probability", type=float, default=0.25)
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
            eval_batch_size=args.eval_batch_size,
            max_increment=args.max_increment,
            reflection_probability=args.reflection_probability,
            lr=args.lr,
            random_start=args.random_start,
        )
        for seed in args.seeds
    ]
    payload = {"config": vars(args), "results": [asdict(x) for x in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("seed preL16 preL1024 rel-def learned-orbit raw midpoint block-proc full-proc")
    for x in rows:
        print(
            f"{x.seed:4d} {x.pre_clean_accuracy[str(args.train_length)]:7.3f} "
            f"{x.pre_clean_accuracy['1024']:9.3f} {x.pre_relation_defect:7.3f} "
            f"{x.learned_orbit_accuracy:12.3f} {x.raw_projected_accuracy:5.3f} "
            f"{x.midpoint_accuracy:8.3f} {x.block_procrustes_accuracy:10.3f} "
            f"{x.full_procrustes_accuracy:9.3f}"
        )


if __name__ == "__main__":
    main()
