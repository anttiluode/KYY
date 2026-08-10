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
EQ_NAME = "cyclic_equivariant_for_partial_merge"
EQ_SPEC = importlib.util.spec_from_file_location(
    EQ_NAME, ROOT / "map" / "cyclic_equivariant_port_probe.py"
)
assert EQ_SPEC is not None and EQ_SPEC.loader is not None
eq = importlib.util.module_from_spec(EQ_SPEC)
sys.modules[EQ_NAME] = eq
EQ_SPEC.loader.exec_module(eq)

POS_NAME = "cyclic_positive_for_partial_merge"
POS_SPEC = importlib.util.spec_from_file_location(
    POS_NAME, ROOT / "map" / "cyclic_positive_kernel_port_probe.py"
)
assert POS_SPEC is not None and POS_SPEC.loader is not None
pos = importlib.util.module_from_spec(POS_SPEC)
sys.modules[POS_NAME] = pos
POS_SPEC.loader.exec_module(pos)

N = 4
MERGE_TOKEN = 4
EXACT_MERGE = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float64)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def merge_state(q: torch.Tensor) -> torch.Tensor:
    """Exact behavioral pinch: {0,1}->0 and {2,3}->2."""
    return torch.where(q < 2, torch.zeros_like(q), torch.full_like(q, 2))


def generate_batch(
    batch_size: int,
    length: int,
    merge_probability: float,
    *,
    random_start: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if length < 1:
        raise ValueError("length must be positive")
    # 0..3 are C4 increments; 4 is the partial merge.
    x = torch.randint(0, 4, (batch_size, length))
    use_merge = torch.rand(batch_size, length) < float(merge_probability)
    x[use_merge] = MERGE_TOKEN
    if random_start:
        x[:, 0] = torch.randint(0, 4, (batch_size,))

    q = torch.zeros(batch_size, dtype=torch.long)
    ys: list[torch.Tensor] = []
    for t in range(length):
        tok = x[:, t]
        is_merge = tok == MERGE_TOKEN
        q = torch.where(is_merge, merge_state(q), (q + tok) % 4)
        ys.append(q.clone())
    return x, torch.stack(ys, dim=1)


class SoftPartialMergeTracker(nn.Module):
    """One learned planar oscillator plus one learned full-rank merge map."""

    def __init__(self, seed: int):
        super().__init__()
        rng = np.random.default_rng(7919 + seed)
        # Intentionally near, but not on, the legal C4 generator.
        theta0 = math.pi / 2 + float(rng.uniform(-0.35, 0.35))
        self.angle = nn.Parameter(torch.tensor(theta0, dtype=torch.float32))
        # Start from a visibly full-rank soft version of the singular pinch.
        base = np.array([[1.15, 0.85], [0.12, 0.28]], dtype=np.float32)
        base += rng.normal(scale=0.08, size=(2, 2)).astype(np.float32)
        self.merge = nn.Parameter(torch.tensor(base, dtype=torch.float32))
        self.register_buffer("h0", torch.tensor([1.0, 0.0], dtype=torch.float32))
        self.readout = nn.Linear(2, 4)

    def forward(self, tokens: torch.Tensor, *, return_hidden: bool = False):
        bsz, length = tokens.shape
        h = self.h0.unsqueeze(0).expand(bsz, -1).clone()
        outs: list[torch.Tensor] = []
        hidden: list[torch.Tensor] = []
        for t in range(length):
            tok = tokens[:, t]
            is_merge = (tok == MERGE_TOKEN).view(-1, 1)
            inc = torch.where(tok == MERGE_TOKEN, torch.zeros_like(tok), tok)
            phase = inc.float() * self.angle
            c, s = torch.cos(phase), torch.sin(phase)
            x, y = h[:, 0], h[:, 1]
            rotated = torch.stack((c * x - s * y, s * x + c * y), dim=-1)
            merged = h @ self.merge.T
            h = torch.where(is_merge, merged, rotated)
            hidden.append(h)
            outs.append(self.readout(h))
        logits = torch.stack(outs, dim=1)
        if return_hidden:
            return logits, torch.stack(hidden, dim=1)
        return logits


def train_model(
    model: SoftPartialMergeTracker,
    *,
    steps: int,
    train_length: int,
    batch_size: int,
    merge_probability: float,
    lr: float,
    random_start: bool,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        x, y = generate_batch(
            batch_size, train_length, merge_probability, random_start=random_start
        )
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, 4), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def learned_accuracy(
    model: SoftPartialMergeTracker,
    lengths: list[int],
    batch_size: int,
    merge_probability: float,
    random_start: bool,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for length in lengths:
        x, y = generate_batch(
            batch_size, length, merge_probability, random_start=random_start
        )
        pred = model(x).argmax(dim=-1)
        out[str(length)] = float((pred == y).float().mean().item())
    return out


def compiled_runtime(
    tokens: torch.Tensor,
    angle: float,
    W: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    h = torch.tensor([1.0, 0.0], dtype=torch.float64).view(1, 2).expand(tokens.shape[0], -1).clone()
    outs: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    merge = EXACT_MERGE
    for t in range(tokens.shape[1]):
        tok = tokens[:, t]
        is_merge = (tok == MERGE_TOKEN).view(-1, 1)
        inc = torch.where(tok == MERGE_TOKEN, torch.zeros_like(tok), tok)
        phase = inc.to(torch.float64) * float(angle)
        c, s = torch.cos(phase), torch.sin(phase)
        x, y = h[:, 0], h[:, 1]
        rotated = torch.stack((c * x - s * y, s * x + c * y), dim=-1)
        merged = h @ merge.T
        h = torch.where(is_merge, merged, rotated)
        states.append(h)
        outs.append(h @ W.to(torch.float64).T + b.to(torch.float64))
    return torch.stack(outs, dim=1), torch.stack(states, dim=1)


@torch.no_grad()
def compiled_accuracy(
    *,
    angle: float,
    W: torch.Tensor,
    b: torch.Tensor,
    lengths: list[int],
    batch_size: int,
    merge_probability: float,
    random_start: bool,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for length in lengths:
        x, y = generate_batch(batch_size, length, merge_probability, random_start=random_start)
        pred = compiled_runtime(x, angle, W, b)[0].argmax(dim=-1)
        out[str(length)] = float((pred == y).to(torch.float64).mean().item())
    return out


def exact_orbit(angle: float) -> torch.Tensor:
    return eq.exact_orbit(4, np.asarray([angle], dtype=np.float64), torch.tensor([[1.0, 0.0]], dtype=torch.float64))


def merge_generator_certificate(angle: float) -> dict[str, object]:
    """Certify the C4 cycle and partial merge from a basis and legal orbit.

    For the square code, v0=(1,0), v1=(0,1) form a basis.  The exact pinch is
    therefore determined by M v0=v0 and M v1=v0.  Linearity then forces
    M v2=v2 and M v3=v2 because v2=-v0 and v3=-v1.
    """
    orbit = exact_orbit(angle)
    M = EXACT_MERGE
    cycle = eq.block_rotation(np.asarray([angle]), 1)
    cycle4 = torch.linalg.matrix_power(cycle, 4)
    cycle_defect = float(torch.linalg.matrix_norm(cycle4 - torch.eye(2, dtype=torch.float64)).item())
    basis_defect = max(
        float(torch.linalg.vector_norm(M @ orbit[0] - orbit[0]).item()),
        float(torch.linalg.vector_norm(M @ orbit[1] - orbit[0]).item()),
    )
    targets = torch.stack((orbit[0], orbit[0], orbit[2], orbit[2]), dim=0)
    merged = orbit @ M.T
    orbit_action_defect = float(torch.max(torch.linalg.vector_norm(merged - targets, dim=-1)).item())
    idem_defect = float(torch.linalg.matrix_norm(M @ M - M).item())
    s = torch.linalg.svdvals(M)
    return {
        "cycle_relation_defect": cycle_defect,
        "merge_basis_constraint_defect": basis_defect,
        "merge_legal_orbit_action_defect": orbit_action_defect,
        "merge_idempotence_defect": idem_defect,
        "merge_continuous_rank": int(torch.linalg.matrix_rank(M).item()),
        "merge_singular_values": [float(x) for x in s.tolist()],
        "behavioral_image_states": [0, 2],
        "behavioral_kernel_blocks": [[0, 1], [2, 3]],
        "behavioral_rank": 2,
        "certified": bool(cycle_defect < 1e-10 and basis_defect < 1e-10 and idem_defect < 1e-10),
    }


def paired_leakage_sequences(batch_size: int, continuation_length: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Prefixes end in 0 vs 1, then both receive M, then identical cycle-only future."""
    a0 = torch.zeros((batch_size, 1), dtype=torch.long)
    b0 = torch.ones((batch_size, 1), dtype=torch.long)
    merge = torch.full((batch_size, 1), MERGE_TOKEN, dtype=torch.long)
    future = torch.randint(0, 4, (batch_size, continuation_length), dtype=torch.long)
    return torch.cat((a0, merge, future), 1), torch.cat((b0, merge, future), 1), 1


def tv(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.abs(p - q), dim=-1)


@torch.no_grad()
def learned_leakage(model: SoftPartialMergeTracker, batch_size: int, continuation_length: int) -> dict[str, float]:
    a, b, merge_index = paired_leakage_sequences(batch_size, continuation_length)
    la, ha = model(a, return_hidden=True)
    lb, hb = model(b, return_hidden=True)
    pa, pb = torch.softmax(la, -1), torch.softmax(lb, -1)
    curve_tv = tv(pa[:, merge_index:], pb[:, merge_index:])
    hdiff = torch.linalg.vector_norm(ha[:, merge_index:] - hb[:, merge_index:], dim=-1)
    mismatch = la[:, merge_index:].argmax(-1) != lb[:, merge_index:].argmax(-1)
    return {
        "hidden_difference_at_merge": float(hdiff[:, 0].mean().item()),
        "hidden_difference_max_future": float(hdiff.max().item()),
        "probability_tv_at_merge": float(curve_tv[:, 0].mean().item()),
        "probability_tv_max_future": float(curve_tv.max().item()),
        "prediction_mismatch_max_rate": float(mismatch.float().mean(dim=0).max().item()),
    }


@torch.no_grad()
def compiled_leakage(angle: float, W: torch.Tensor, b: torch.Tensor, batch_size: int, continuation_length: int) -> dict[str, float]:
    a, bb, merge_index = paired_leakage_sequences(batch_size, continuation_length)
    la, ha = compiled_runtime(a, angle, W, b)
    lb, hb = compiled_runtime(bb, angle, W, b)
    pa, pb = torch.softmax(la, -1), torch.softmax(lb, -1)
    curve_tv = tv(pa[:, merge_index:], pb[:, merge_index:])
    hdiff = torch.linalg.vector_norm(ha[:, merge_index:] - hb[:, merge_index:], dim=-1)
    mismatch = la[:, merge_index:].argmax(-1) != lb[:, merge_index:].argmax(-1)
    return {
        "hidden_difference_at_merge": float(hdiff[:, 0].mean().item()),
        "hidden_difference_max_future": float(hdiff.max().item()),
        "probability_tv_at_merge": float(curve_tv[:, 0].mean().item()),
        "probability_tv_max_future": float(curve_tv.max().item()),
        "prediction_mismatch_max_rate": float(mismatch.float().mean(dim=0).max().item()),
    }


@dataclass
class Run:
    seed: int
    learned_accuracy: dict[str, float]
    learned_angle: float
    projected_frequency: int
    projected_angle: float
    learned_cycle_relation_defect: float
    learned_merge_determinant: float
    learned_merge_singular_values: list[float]
    learned_merge_rank: int
    learned_merge_distance_to_exact: float
    learned_merge_kernel_direction_residual: float
    learned_leakage: dict[str, float]
    compiled_inherited_accuracy: dict[str, float]
    compiled_equivariant_accuracy: dict[str, float]
    compiled_positive_accuracy: dict[str, float]
    compiled_equivariant_orbit_margin: float
    compiled_positive_orbit_margin: float
    positive_alpha: list[float]
    positive_port_certified: bool
    generator_certificate: dict[str, object]
    compiled_leakage: dict[str, float]


def run_one(
    *,
    seed: int,
    steps: int,
    train_length: int,
    test_lengths: list[int],
    batch_size: int,
    eval_batch_size: int,
    merge_probability: float,
    lr: float,
    random_start: bool,
    leakage_batch_size: int,
    leakage_continuation: int,
) -> Run:
    seed_everything(seed)
    model = SoftPartialMergeTracker(seed)
    train_model(
        model,
        steps=steps,
        train_length=train_length,
        batch_size=batch_size,
        merge_probability=merge_probability,
        lr=lr,
        random_start=random_start,
    )
    lacc = learned_accuracy(model, test_lengths, eval_batch_size, merge_probability, random_start)
    learned_angle = float(model.angle.detach().cpu().item())
    projected, freqs = eq.base.project_angles_to_characters(4, np.asarray([learned_angle]))
    projected_angle = float(projected[0])
    freq = int(freqs[0])
    before, _ = eq.base.relation_defects(4, np.asarray([learned_angle]))

    B = model.merge.detach().cpu().to(torch.float64)
    sv = torch.linalg.svdvals(B)
    kernel_direction = torch.tensor([1.0, -1.0], dtype=torch.float64) / math.sqrt(2.0)
    kernel_resid = float(torch.linalg.vector_norm(B @ kernel_direction).item())

    W = model.readout.weight.detach().cpu().to(torch.float64)
    b = model.readout.bias.detach().cpu().to(torch.float64)
    orbit = exact_orbit(projected_angle)
    W_eq, b_eq, _ = eq.project_cyclic_equivariant_decoder(4, np.asarray([projected_angle]), W, b)
    eq_metrics = eq.readout_metrics(orbit, W_eq, b_eq)
    W_pos, b_pos, alpha, _ = pos.positive_kernel_projection(4, np.asarray([projected_angle]), orbit[0], W, b)
    pos_metrics = eq.readout_metrics(orbit, W_pos, b_pos)
    pos_cert, _ = pos.positive_kernel_certificate(4, freqs, alpha)

    return Run(
        seed=int(seed),
        learned_accuracy=lacc,
        learned_angle=learned_angle,
        projected_frequency=freq,
        projected_angle=projected_angle,
        learned_cycle_relation_defect=float(before),
        learned_merge_determinant=float(torch.linalg.det(B).item()),
        learned_merge_singular_values=[float(x) for x in sv.tolist()],
        learned_merge_rank=int(torch.linalg.matrix_rank(B).item()),
        learned_merge_distance_to_exact=float(torch.linalg.matrix_norm(B - EXACT_MERGE).item()),
        learned_merge_kernel_direction_residual=kernel_resid,
        learned_leakage=learned_leakage(model, leakage_batch_size, leakage_continuation),
        compiled_inherited_accuracy=compiled_accuracy(angle=projected_angle, W=W, b=b, lengths=test_lengths, batch_size=eval_batch_size, merge_probability=merge_probability, random_start=random_start),
        compiled_equivariant_accuracy=compiled_accuracy(angle=projected_angle, W=W_eq, b=b_eq, lengths=test_lengths, batch_size=eval_batch_size, merge_probability=merge_probability, random_start=random_start),
        compiled_positive_accuracy=compiled_accuracy(angle=projected_angle, W=W_pos, b=b_pos, lengths=test_lengths, batch_size=eval_batch_size, merge_probability=merge_probability, random_start=random_start),
        compiled_equivariant_orbit_margin=float(eq_metrics[1]),
        compiled_positive_orbit_margin=float(pos_metrics[1]),
        positive_alpha=[float(x) for x in alpha.tolist()],
        positive_port_certified=bool(pos_cert),
        generator_certificate=merge_generator_certificate(projected_angle),
        compiled_leakage=compiled_leakage(projected_angle, W_pos, b_pos, leakage_batch_size, leakage_continuation),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train a soft full-rank partial merge, then compile the exact C4 pinch and port")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--test-lengths", nargs="+", type=int, default=[16,64,256,1024])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--merge-probability", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--random-start", action="store_true")
    p.add_argument("--leakage-batch-size", type=int, default=256)
    p.add_argument("--leakage-continuation", type=int, default=64)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    rows = [
        run_one(
            seed=seed,
            steps=args.steps,
            train_length=args.train_length,
            test_lengths=args.test_lengths,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            merge_probability=args.merge_probability,
            lr=args.lr,
            random_start=args.random_start,
            leakage_batch_size=args.leakage_batch_size,
            leakage_continuation=args.leakage_continuation,
        )
        for seed in args.seeds
    ]
    payload = {"config": vars(args), "results": [asdict(x) for x in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("seed learned1024 freq det kres leakTV eq1024 pos1024 cert margin")
    for r in rows:
        print(
            f"{r.seed:4d} {r.learned_accuracy.get('1024', float('nan')):11.3f} "
            f"{r.projected_frequency:4d} {r.learned_merge_determinant:+.3e} "
            f"{r.learned_merge_kernel_direction_residual:.3e} "
            f"{r.learned_leakage['probability_tv_max_future']:.3e} "
            f"{r.compiled_equivariant_accuracy.get('1024', float('nan')):6.3f} "
            f"{r.compiled_positive_accuracy.get('1024', float('nan')):7.3f} "
            f"{str(r.generator_certificate['certified'] and r.positive_port_certified):>5s} "
            f"{r.compiled_positive_orbit_margin:+.3f}"
        )


if __name__ == "__main__":
    main()
