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

EQ_NAME = "cyclic_equivariant_for_reset_compiler"
EQ_SPEC = importlib.util.spec_from_file_location(
    EQ_NAME, ROOT / "map" / "cyclic_equivariant_port_probe.py"
)
assert EQ_SPEC is not None and EQ_SPEC.loader is not None
eq = importlib.util.module_from_spec(EQ_SPEC)
sys.modules[EQ_NAME] = eq
EQ_SPEC.loader.exec_module(eq)

POS_NAME = "cyclic_positive_for_reset_compiler"
POS_SPEC = importlib.util.spec_from_file_location(
    POS_NAME, ROOT / "map" / "cyclic_positive_kernel_port_probe.py"
)
assert POS_SPEC is not None and POS_SPEC.loader is not None
pos = importlib.util.module_from_spec(POS_SPEC)
sys.modules[POS_NAME] = pos
POS_SPEC.loader.exec_module(pos)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SoftResetRotaryTracker(nn.Module):
    """Learned harmonic counter with a deliberately non-exact reset primitive.

    Inputs 0..n-1 are cyclic increments. Token n is reset. During training,
    reset is a residual affine blend

        h' = g*h + (1-g)*r,

    with one learned 0<g_i<1 per mode and a learned reset target r_i.
    Thus reset can become strongly contractive but is never exactly singular.

    The compiler later replaces it by the exact overwrite h' = h0 while
    snapping the rotation generator onto C_n characters.
    """

    def __init__(self, n: int, modes: int, seed: int):
        super().__init__()
        self.n = int(n)
        self.modes = int(modes)
        rng = np.random.default_rng(seed + 1009 * n)
        angles = rng.uniform(-math.pi, math.pi, size=modes)
        self.angles = nn.Parameter(torch.tensor(angles, dtype=torch.float32))

        h0 = torch.zeros(modes, 2, dtype=torch.float32)
        h0[:, 0] = 1.0 / math.sqrt(modes)
        self.register_buffer("h0", h0)

        # Start as a visibly soft reset rather than an almost-exact overwrite.
        self.reset_gate_logits = nn.Parameter(torch.zeros(modes, dtype=torch.float32))
        target = h0 + 0.15 * torch.randn_like(h0)
        self.reset_target = nn.Parameter(target)
        self.readout = nn.Linear(2 * modes, n)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, length = tokens.shape
        h = self.h0.unsqueeze(0).expand(bsz, -1, -1).clone()
        outs: list[torch.Tensor] = []
        gate = torch.sigmoid(self.reset_gate_logits).view(1, -1, 1)
        target = self.reset_target.view(1, self.modes, 2)
        for t in range(length):
            token = tokens[:, t]
            reset_mask = (token == self.n).view(-1, 1, 1)
            increment = torch.where(token == self.n, torch.zeros_like(token), token)
            theta = increment.float().unsqueeze(-1) * self.angles.unsqueeze(0)
            c, s = torch.cos(theta), torch.sin(theta)
            x, y = h[..., 0], h[..., 1]
            rotated = torch.stack((c * x - s * y, s * x + c * y), dim=-1)
            soft_reset = gate * h + (1.0 - gate) * target
            h = torch.where(reset_mask, soft_reset, rotated)
            outs.append(self.readout(h.reshape(bsz, -1)))
        return torch.stack(outs, dim=1)


def generate_batch(
    n: int,
    batch_size: int,
    length: int,
    max_increment: int,
    reset_probability: float,
    *,
    random_start: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if length < 1:
        raise ValueError("length must be positive")
    tokens = torch.randint(0, max_increment + 1, (batch_size, length))
    reset = torch.rand(batch_size, length) < float(reset_probability)
    tokens[reset] = n
    if random_start:
        # Expose the full orbit before local mixed cycle/reset continuation.
        tokens[:, 0] = torch.randint(0, n, (batch_size,))

    state = torch.zeros(batch_size, dtype=torch.long)
    ys: list[torch.Tensor] = []
    for t in range(length):
        token = tokens[:, t]
        is_reset = token == n
        state = torch.where(is_reset, torch.zeros_like(state), (state + token) % n)
        ys.append(state.clone())
    return tokens, torch.stack(ys, dim=1)


def train_model(
    model: SoftResetRotaryTracker,
    *,
    steps: int,
    train_length: int,
    batch_size: int,
    max_increment: int,
    reset_probability: float,
    lr: float,
    random_start: bool,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        x, y = generate_batch(
            model.n,
            batch_size,
            train_length,
            max_increment,
            reset_probability,
            random_start=random_start,
        )
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, model.n), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def learned_accuracy(
    model: SoftResetRotaryTracker,
    lengths: list[int],
    batch_size: int,
    max_increment: int,
    reset_probability: float,
    random_start: bool,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for length in lengths:
        x, y = generate_batch(
            model.n,
            batch_size,
            length,
            max_increment,
            reset_probability,
            random_start=random_start,
        )
        pred = model(x).argmax(dim=-1)
        out[str(length)] = float((pred == y).float().mean().item())
    return out


def compiled_runtime(
    tokens: torch.Tensor,
    n: int,
    projected_angles: np.ndarray,
    h0: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Exact compiled runtime: character rotations plus singular reset overwrite."""
    angles = torch.as_tensor(projected_angles, dtype=torch.float64)
    h = h0.to(torch.float64).unsqueeze(0).expand(tokens.shape[0], -1, -1).clone()
    outs: list[torch.Tensor] = []
    for t in range(tokens.shape[1]):
        token = tokens[:, t]
        reset_mask = (token == n).view(-1, 1, 1)
        inc = torch.where(token == n, torch.zeros_like(token), token)
        theta = inc.to(torch.float64).unsqueeze(-1) * angles.unsqueeze(0)
        c, s = torch.cos(theta), torch.sin(theta)
        x, y = h[..., 0], h[..., 1]
        rotated = torch.stack((c * x - s * y, s * x + c * y), dim=-1)
        exact_reset = h0.to(torch.float64).view(1, -1, 2).expand_as(h)
        h = torch.where(reset_mask, exact_reset, rotated)
        outs.append(h.reshape(tokens.shape[0], -1) @ W.to(torch.float64).T + b.to(torch.float64))
    return torch.stack(outs, dim=1)


@torch.no_grad()
def compiled_accuracy(
    *,
    n: int,
    projected_angles: np.ndarray,
    h0: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    lengths: list[int],
    batch_size: int,
    max_increment: int,
    reset_probability: float,
    random_start: bool,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for length in lengths:
        x, y = generate_batch(
            n,
            batch_size,
            length,
            max_increment,
            reset_probability,
            random_start=random_start,
        )
        pred = compiled_runtime(x, n, projected_angles, h0, W, b).argmax(dim=-1)
        out[str(length)] = float((pred == y).to(torch.float64).mean().item())
    return out


def reset_relation_defects(
    n: int,
    projected_angles: np.ndarray,
    h0: torch.Tensor,
) -> dict[str, float]:
    """Numerically audit the defining compiled cycle/reset relations.

    On hidden vectors the exact reset map is Z(h)=h0. We report:
    - r^n = I operator defect;
    - Z^2 = Z (zero by construction);
    - Z o r = Z (zero by construction as Z ignores input);
    - distance between r^k h0 and h0 for k=1..n-1, useful as the
      orbit-faithfulness/margin-side diagnostic rather than a reset relation.
    """
    op_defect, state_defect = eq.base.relation_defects(n, projected_angles)
    z = eq.exact_orbit(n, projected_angles, h0.to(torch.float64))
    d = torch.linalg.vector_norm(z[1:] - z[0], dim=-1)
    return {
        "cycle_operator_relation_defect": float(op_defect),
        "cycle_state_relation_defect": float(state_defect),
        "reset_idempotence_defect": 0.0,
        "reset_after_cycle_defect": 0.0,
        "minimum_nonidentity_orbit_distance": float(d.min().item()),
    }


@dataclass
class Run:
    seed: int
    learned_accuracy: dict[str, float]
    compiled_inherited_accuracy: dict[str, float]
    compiled_equivariant_accuracy: dict[str, float]
    compiled_positive_accuracy: dict[str, float]
    learned_reset_gates: list[float]
    mean_learned_reset_gate: float
    reset_target_to_h0: float
    character_operator_defect_before: float
    character_operator_defect_after: float
    projected_frequencies: list[int]
    equivariant_orbit_accuracy: float
    equivariant_orbit_min_margin: float
    positive_orbit_accuracy: float
    positive_orbit_min_margin: float
    positive_alpha: list[float]
    active_positive_modes: int
    positive_character_gcd: int
    positive_port_certified: bool
    compiled_relation_defects: dict[str, float]


def run_one(
    *,
    n: int,
    modes: int,
    seed: int,
    steps: int,
    train_length: int,
    test_lengths: list[int],
    batch_size: int,
    eval_batch_size: int,
    max_increment: int,
    reset_probability: float,
    lr: float,
    random_start: bool,
) -> Run:
    seed_everything(seed)
    model = SoftResetRotaryTracker(n, modes, seed)
    train_model(
        model,
        steps=steps,
        train_length=train_length,
        batch_size=batch_size,
        max_increment=max_increment,
        reset_probability=reset_probability,
        lr=lr,
        random_start=random_start,
    )
    learned_acc = learned_accuracy(
        model,
        test_lengths,
        eval_batch_size,
        max_increment,
        reset_probability,
        random_start,
    )

    learned_angles = model.angles.detach().cpu().numpy().astype(np.float64)
    projected, frequencies = eq.base.project_angles_to_characters(n, learned_angles)
    before, _ = eq.base.relation_defects(n, learned_angles)
    after, _ = eq.base.relation_defects(n, projected)
    h0 = model.h0.detach().cpu().to(torch.float64)
    W = model.readout.weight.detach().cpu().to(torch.float64)
    b = model.readout.bias.detach().cpu().to(torch.float64)
    orbit = eq.exact_orbit(n, projected, h0)

    inherited_orbit = eq.readout_metrics(orbit, W, b)
    W_eq, b_eq, _ = eq.project_cyclic_equivariant_decoder(n, projected, W, b)
    eq_orbit = eq.readout_metrics(orbit, W_eq, b_eq)
    W_pos, b_pos, alpha, _ = pos.positive_kernel_projection(
        n, projected, orbit[0], W, b
    )
    pos_orbit = eq.readout_metrics(orbit, W_pos, b_pos)
    pos_cert, char_gcd = pos.positive_kernel_certificate(n, frequencies, alpha)

    inherited_acc = compiled_accuracy(
        n=n,
        projected_angles=projected,
        h0=h0,
        W=W,
        b=b,
        lengths=test_lengths,
        batch_size=eval_batch_size,
        max_increment=max_increment,
        reset_probability=reset_probability,
        random_start=random_start,
    )
    eq_acc = compiled_accuracy(
        n=n,
        projected_angles=projected,
        h0=h0,
        W=W_eq,
        b=b_eq,
        lengths=test_lengths,
        batch_size=eval_batch_size,
        max_increment=max_increment,
        reset_probability=reset_probability,
        random_start=random_start,
    )
    pos_acc = compiled_accuracy(
        n=n,
        projected_angles=projected,
        h0=h0,
        W=W_pos,
        b=b_pos,
        lengths=test_lengths,
        batch_size=eval_batch_size,
        max_increment=max_increment,
        reset_probability=reset_probability,
        random_start=random_start,
    )

    gates = torch.sigmoid(model.reset_gate_logits.detach()).cpu().numpy().astype(np.float64)
    target_delta = float(
        torch.linalg.vector_norm(model.reset_target.detach().cpu().to(torch.float64) - h0).item()
    )
    return Run(
        seed=int(seed),
        learned_accuracy=learned_acc,
        compiled_inherited_accuracy=inherited_acc,
        compiled_equivariant_accuracy=eq_acc,
        compiled_positive_accuracy=pos_acc,
        learned_reset_gates=[float(x) for x in gates.tolist()],
        mean_learned_reset_gate=float(np.mean(gates)),
        reset_target_to_h0=target_delta,
        character_operator_defect_before=float(before),
        character_operator_defect_after=float(after),
        projected_frequencies=[int(x) for x in frequencies.tolist()],
        equivariant_orbit_accuracy=float(eq_orbit[0]),
        equivariant_orbit_min_margin=float(eq_orbit[1]),
        positive_orbit_accuracy=float(pos_orbit[0]),
        positive_orbit_min_margin=float(pos_orbit[1]),
        positive_alpha=[float(x) for x in alpha.tolist()],
        active_positive_modes=int(np.sum(alpha > 1e-12)),
        positive_character_gcd=int(char_gcd),
        positive_port_certified=bool(pos_cert),
        compiled_relation_defects=reset_relation_defects(n, projected, h0),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train a soft cyclic/reset machine, then compile it into an exact permutation-reset monoid")
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--test-lengths", nargs="+", type=int, default=[16, 64, 256, 1024])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--max-increment", type=int, default=4)
    p.add_argument("--reset-probability", type=float, default=0.12)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--random-start", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = [
        run_one(
            n=args.n,
            modes=args.modes,
            seed=seed,
            steps=args.steps,
            train_length=args.train_length,
            test_lengths=args.test_lengths,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            max_increment=args.max_increment,
            reset_probability=args.reset_probability,
            lr=args.lr,
            random_start=args.random_start,
        )
        for seed in args.seeds
    ]
    payload = {"config": vars(args), "results": [asdict(row) for row in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("seed learned1024 inherited1024 eq1024 pos1024 gate opdef cert margin")
    for row in rows:
        print(
            f"{row.seed:4d} {row.learned_accuracy.get('1024', float('nan')):11.3f} "
            f"{row.compiled_inherited_accuracy.get('1024', float('nan')):13.3f} "
            f"{row.compiled_equivariant_accuracy.get('1024', float('nan')):6.3f} "
            f"{row.compiled_positive_accuracy.get('1024', float('nan')):7.3f} "
            f"{row.mean_learned_reset_gate:5.3f} "
            f"{row.character_operator_defect_before:6.3f} "
            f"{str(row.positive_port_certified):>5s} {row.positive_orbit_min_margin:+.3f}"
        )


if __name__ == "__main__":
    main()
