"""Nonphysical mixture control for the KYY mixed-maturity deadline result.

Phase-supervised ``geom_scatter`` uses one shared readout after every homogeneous
checkerboard phase. The deadline experiment then found that this readout also works on
asynchronous mixed-maturity states that were never shown during training.

Before attributing that robustness to the local causal schedule, this script asks a
harsher question:

    Does the same readout also tolerate *nonphysical* mixtures of phase states?

Controls
--------
``coordinate_splice``
    For every sample and hidden coordinate independently, choose its value from a
    randomly selected homogeneous phase state. This mixture need not correspond to any
    legal execution history.

``coordinate_splice_with_pre``
    Same, but the untouched pre-token state is also an allowed source. This creates a
    broader maturity mixture.

``convex_mix``
    Random convex combination of whole homogeneous phase states. With a linear shared
    readout and already aligned phase logits, this is expected to be easy and serves as
    an algebraic sanity control.

Interpretation
--------------
If nonphysical splices are highly readable, then asynchronous locality cannot claim
credit for *semantic mixture robustness*. Local scheduling may still determine which
partial states become physically available at which deadline, but the robust public
projection comes from the training/representation geometry.

Run:
    python experiments/mixed_state_splice_control.py --quick
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kyy import TASKS, build_model, generate_batch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def phase_edges(model):
    for sweep in range(model.sweeps):
        yield sweep, model.phase0
        yield sweep, model.phase1


def train_phase_model(
    *,
    task: str,
    seed: int,
    state_dim: int,
    topology: str,
    length: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
):
    seed_everything(seed)
    spec = TASKS[task]
    model = build_model(
        "geom_scatter",
        spec.vocab_size,
        spec.n_classes,
        state_dim,
        topology=topology,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(1, steps + 1):
        x, y = generate_batch(task, batch_size, length, device)
        h = model.h0.unsqueeze(0).expand(batch_size, -1)
        losses = []
        final_logits = []

        for t in range(length):
            tok = x[:, t]
            target = y[:, t]
            for sweep, edge_ids in phase_edges(model):
                h = model._scatter_phase(h, tok, sweep, edge_ids)
                losses.append(criterion(model.readout(h), target))
            final_logits.append(model.readout(h))

        loss = torch.stack(losses).mean()
        logits = torch.stack(final_logits, dim=1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            acc = float((logits.argmax(-1) == y).float().mean().detach())
            print(
                f"seed={seed} step={step:4d}/{steps} loss={float(loss.detach()):.4f} "
                f"final-acc={acc:.3f}",
                flush=True,
            )

    return model


@torch.no_grad()
def final_token_phase_states(model, x: torch.Tensor) -> list[torch.Tensor]:
    """Return pre-token state plus each homogeneous phase state for final token."""
    bsz, length = x.shape
    h = model.h0.unsqueeze(0).expand(bsz, -1)

    for t in range(length - 1):
        tok = x[:, t]
        for sweep, edge_ids in phase_edges(model):
            h = model._scatter_phase(h, tok, sweep, edge_ids)

    states = [h]
    tok = x[:, -1]
    for sweep, edge_ids in phase_edges(model):
        h = model._scatter_phase(h, tok, sweep, edge_ids)
        states.append(h)
    return states


@torch.no_grad()
def random_coordinate_splice(states: list[torch.Tensor], rng: torch.Generator) -> torch.Tensor:
    """Choose a phase source independently for every hidden coordinate."""
    stacked = torch.stack(states, dim=2)  # [batch, dim, n_states]
    bsz, dim, n_states = stacked.shape
    idx = torch.randint(
        n_states,
        (bsz, dim, 1),
        generator=rng,
        device=stacked.device,
    )
    return torch.gather(stacked, 2, idx).squeeze(2)


@torch.no_grad()
def random_convex_mix(states: list[torch.Tensor], rng: torch.Generator) -> torch.Tensor:
    stacked = torch.stack(states, dim=2)
    bsz, _dim, n_states = stacked.shape
    weights = torch.rand(
        bsz,
        n_states,
        generator=rng,
        device=stacked.device,
    )
    weights = weights / weights.sum(dim=1, keepdim=True)
    return (stacked * weights[:, None, :]).sum(dim=2)


@torch.no_grad()
def evaluate(
    model,
    *,
    task: str,
    batches: int,
    batch_size: int,
    length: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(seed)
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 99)

    correct_phase = None
    correct_splice = 0
    correct_splice_pre = 0
    correct_convex = 0
    correct_convex_pre = 0
    total = 0

    for _ in range(batches):
        x, y = generate_batch(task, batch_size, length, device)
        target = y[:, -1]
        states = final_token_phase_states(model, x)  # [pre, p1, p2, ...]
        phase_states = states[1:]

        if correct_phase is None:
            correct_phase = np.zeros(len(phase_states), dtype=np.float64)

        for i, state in enumerate(phase_states):
            correct_phase[i] += float((model.readout(state).argmax(-1) == target).sum())

        splice = random_coordinate_splice(phase_states, generator)
        splice_pre = random_coordinate_splice(states, generator)
        convex = random_convex_mix(phase_states, generator)
        convex_pre = random_convex_mix(states, generator)

        correct_splice += int((model.readout(splice).argmax(-1) == target).sum())
        correct_splice_pre += int((model.readout(splice_pre).argmax(-1) == target).sum())
        correct_convex += int((model.readout(convex).argmax(-1) == target).sum())
        correct_convex_pre += int((model.readout(convex_pre).argmax(-1) == target).sum())
        total += batch_size

    assert correct_phase is not None
    return {
        "phase_accuracy": (correct_phase / total).tolist(),
        "coordinate_splice_accuracy": correct_splice / total,
        "coordinate_splice_with_pre_accuracy": correct_splice_pre / total,
        "convex_mix_accuracy": correct_convex / total,
        "convex_mix_with_pre_accuracy": correct_convex_pre / total,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="KYY nonphysical mixed-state splice control")
    p.add_argument("--task", choices=sorted(TASKS), default="perm3")
    p.add_argument("--state-dim", type=int, default=32)
    p.add_argument("--topology", choices=["ring", "path", "matching", "disconnected"], default="ring")
    p.add_argument("--train-length", type=int, default=32)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--eval-length", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.steps = min(args.steps, 150)
        args.model_seeds = [args.model_seeds[0]]
        args.eval_batches = min(args.eval_batches, 3)
        args.eval_batch_size = min(args.eval_batch_size, 128)
        args.eval_length = min(args.eval_length, 24)
        args.log_every = min(args.log_every, 30)

    device = torch.device(args.device)
    rows = []

    for model_seed in args.model_seeds:
        model = train_phase_model(
            task=args.task,
            seed=model_seed,
            state_dim=args.state_dim,
            topology=args.topology,
            length=args.train_length,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            log_every=args.log_every,
        )
        row = evaluate(
            model,
            task=args.task,
            batches=args.eval_batches,
            batch_size=args.eval_batch_size,
            length=args.eval_length,
            seed=70_000 + model_seed,
            device=device,
        )
        row["model_seed"] = model_seed
        rows.append(row)

        print(f"\nseed={model_seed}")
        print("phase accuracy              " + " ".join(f"{x:.3f}" for x in row["phase_accuracy"]))
        print(f"coordinate splice           {row['coordinate_splice_accuracy']:.3f}")
        print(f"coordinate splice + pre     {row['coordinate_splice_with_pre_accuracy']:.3f}")
        print(f"convex mix                  {row['convex_mix_accuracy']:.3f}")
        print(f"convex mix + pre            {row['convex_mix_with_pre_accuracy']:.3f}")

    scalar_keys = [
        "coordinate_splice_accuracy",
        "coordinate_splice_with_pre_accuracy",
        "convex_mix_accuracy",
        "convex_mix_with_pre_accuracy",
    ]
    means = {
        key: float(np.mean([row[key] for row in rows]))
        for key in scalar_keys
    }
    means["phase_accuracy"] = np.asarray(
        [row["phase_accuracy"] for row in rows], dtype=float
    ).mean(axis=0).tolist()

    print("\n=== means ===")
    print("phase accuracy              " + " ".join(f"{x:.4f}" for x in means["phase_accuracy"]))
    for key in scalar_keys:
        print(f"{key:34s} {means[key]:.4f}")

    print("\nGuardrail")
    print("---------")
    print("High accuracy on nonphysical splices means semantic mixture robustness cannot be")
    print("credited to the causal local schedule. Locality may still buy physical availability")
    print("and latency; the public-subspace training buys the robust readout.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"mixed-state-splice-{stamp}.json"
    path.write_text(json.dumps({"config": vars(args), "rows": rows, "means": means}, indent=2, default=str))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
