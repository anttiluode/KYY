"""Ordered frontier control for KYY mixed-maturity states.

This attacks one remaining ambiguity in the partial-maturity work.

The existing nonphysical splice control asks whether a shared readout tolerates an
*unordered* coordinate-wise mixture of homogeneous phase states.  It does.  That
kills the claim that semantic mixture robustness itself requires a physical async
schedule.

But a propagating frontier has another property that the random splice destroys:
spatial order.  This script therefore holds the maturity histogram and its maximum
span fixed while changing only the assignment of maturity to physical coordinates.

For every final-token example we first compute the ordinary homogeneous states:

    pre, phase_1, phase_2, ..., final

Then we make three coordinate-wise states using EXACTLY the same multiset of source
phase labels:

    forward  : phase labels increase around the physical coordinate order
    reverse  : the same labels in the opposite order
    shuffled : the same labels randomly permuted across coordinates

For ring topology we average forward/reverse over every cyclic origin, preventing an
arbitrary coordinate zero from creating a result.  Shuffled controls preserve the
same histogram exactly and are redrawn per origin.

Interpretation:

* forward ~= reverse ~= shuffled
    Ordered maturity carries no special readout advantage here.  The proposed
    `frontier direction` quantity does not survive this gate.

* forward ~= reverse > shuffled
    Spatial ORDER matters, but direction sign/chirality does not.

* forward != reverse, both controlled against shuffled
    The learned substrate/readout is sensitive to orientation of the maturity
    frontier.  This is the strongest outcome, but still only a property of this
    toy substrate--not evidence for a brain mechanism.

No direction-specific state is shown during training.  The model receives the same
phase supervision used by the earlier splice control.

Run:
    python experiments/frontier_direction_control.py --quick
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from kyy import TASKS, generate_batch
from mixed_state_splice_control import (
    final_token_phase_states,
    seed_everything,
    train_phase_model,
)


def balanced_maturity_labels(dim: int, n_states: int) -> np.ndarray:
    """Monotone 0..n_states-1 labels with near-equal occupancy."""
    if dim < 1 or n_states < 2:
        raise ValueError("need dim>=1 and at least two maturity states")
    labels = np.floor(np.arange(dim, dtype=np.float64) * n_states / dim).astype(np.int64)
    labels = np.clip(labels, 0, n_states - 1)
    labels[-1] = n_states - 1
    return labels


@torch.no_grad()
def splice_from_labels(states: list[torch.Tensor], labels: np.ndarray) -> torch.Tensor:
    """Take coordinate j from states[labels[j]], identically for every sample."""
    stacked = torch.stack(states, dim=2)  # [batch, dim, n_states]
    bsz, dim, n_states = stacked.shape
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (dim,):
        raise ValueError(f"label shape {labels.shape} != {(dim,)}")
    if labels.min() < 0 or labels.max() >= n_states:
        raise ValueError("maturity labels outside state range")
    idx = torch.as_tensor(labels, dtype=torch.long, device=stacked.device)
    idx = idx.view(1, dim, 1).expand(bsz, -1, -1)
    return torch.gather(stacked, 2, idx).squeeze(2)


def label_roughness(labels: np.ndarray, ring: bool = True) -> float:
    """Mean absolute adjacent maturity jump; diagnostic only."""
    labels = np.asarray(labels, dtype=np.float64)
    diffs = np.abs(np.diff(labels))
    if ring and len(labels) > 1:
        diffs = np.concatenate([diffs, [abs(labels[0] - labels[-1])]])
    return float(diffs.mean()) if len(diffs) else 0.0


@torch.no_grad()
def evaluate(
    model,
    *,
    task: str,
    batches: int,
    batch_size: int,
    length: int,
    seed: int,
    shuffle_draws: int,
    include_pre: bool,
    all_origins: bool,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(seed)
    model.eval()
    rng = np.random.default_rng(seed + 771)

    dim = int(model.state_dim)
    offsets = list(range(dim)) if all_origins else [0]

    forward_correct = np.zeros(len(offsets), dtype=np.float64)
    reverse_correct = np.zeros(len(offsets), dtype=np.float64)
    shuffled_correct = np.zeros((len(offsets), shuffle_draws), dtype=np.float64)
    phase_correct = None
    total = 0

    forward_roughness = []
    reverse_roughness = []
    shuffled_roughness = []
    maturity_histogram = None

    for _ in range(batches):
        x, y = generate_batch(task, batch_size, length, device)
        target = y[:, -1]
        all_states = final_token_phase_states(model, x)  # [pre, p1, ..., final]
        states = all_states if include_pre else all_states[1:]

        if phase_correct is None:
            phase_correct = np.zeros(len(all_states), dtype=np.float64)
            base = balanced_maturity_labels(dim, len(states))
            maturity_histogram = np.bincount(base, minlength=len(states)).tolist()

        for i, state in enumerate(all_states):
            phase_correct[i] += float((model.readout(state).argmax(-1) == target).sum())

        base = balanced_maturity_labels(dim, len(states))
        reverse_base = base[::-1].copy()

        for oi, offset in enumerate(offsets):
            forward_labels = np.roll(base, offset)
            reverse_labels = np.roll(reverse_base, offset)

            f_state = splice_from_labels(states, forward_labels)
            r_state = splice_from_labels(states, reverse_labels)
            forward_correct[oi] += float((model.readout(f_state).argmax(-1) == target).sum())
            reverse_correct[oi] += float((model.readout(r_state).argmax(-1) == target).sum())

            if total == 0:
                forward_roughness.append(label_roughness(forward_labels, ring=True))
                reverse_roughness.append(label_roughness(reverse_labels, ring=True))

            for draw in range(shuffle_draws):
                shuffled_labels = rng.permutation(forward_labels)
                s_state = splice_from_labels(states, shuffled_labels)
                shuffled_correct[oi, draw] += float(
                    (model.readout(s_state).argmax(-1) == target).sum()
                )
                if total == 0:
                    shuffled_roughness.append(label_roughness(shuffled_labels, ring=True))

        total += batch_size

    assert phase_correct is not None and maturity_histogram is not None
    forward_acc = forward_correct / total
    reverse_acc = reverse_correct / total
    shuffled_acc = shuffled_correct / total
    phase_acc = phase_correct / total

    return {
        "include_pre": bool(include_pre),
        "maturity_states": int(len(phase_acc) if include_pre else len(phase_acc) - 1),
        "maturity_histogram": maturity_histogram,
        "phase_accuracy_pre_through_final": phase_acc.tolist(),
        "forward_by_origin": forward_acc.tolist(),
        "reverse_by_origin": reverse_acc.tolist(),
        "shuffled_by_origin_draw": shuffled_acc.tolist(),
        "forward_mean": float(forward_acc.mean()),
        "reverse_mean": float(reverse_acc.mean()),
        "shuffled_mean": float(shuffled_acc.mean()),
        "forward_minus_shuffled": float(forward_acc.mean() - shuffled_acc.mean()),
        "reverse_minus_shuffled": float(reverse_acc.mean() - shuffled_acc.mean()),
        "forward_minus_reverse": float(forward_acc.mean() - reverse_acc.mean()),
        "forward_origin_sd": float(forward_acc.std()),
        "reverse_origin_sd": float(reverse_acc.std()),
        "shuffled_sd": float(shuffled_acc.std()),
        "forward_roughness_mean": float(np.mean(forward_roughness)),
        "reverse_roughness_mean": float(np.mean(reverse_roughness)),
        "shuffled_roughness_mean": float(np.mean(shuffled_roughness)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="KYY ordered maturity-frontier control")
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
    p.add_argument("--shuffle-draws", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.steps = min(args.steps, 120)
        args.model_seeds = [args.model_seeds[0]]
        args.eval_batches = min(args.eval_batches, 3)
        args.eval_batch_size = min(args.eval_batch_size, 128)
        args.eval_length = min(args.eval_length, 24)
        args.shuffle_draws = min(args.shuffle_draws, 2)
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

        for include_pre in (False, True):
            row = evaluate(
                model,
                task=args.task,
                batches=args.eval_batches,
                batch_size=args.eval_batch_size,
                length=args.eval_length,
                seed=90_000 + model_seed + (1000 if include_pre else 0),
                shuffle_draws=args.shuffle_draws,
                include_pre=include_pre,
                all_origins=(args.topology == "ring"),
                device=device,
            )
            row["model_seed"] = model_seed
            rows.append(row)

            tag = "with-pre" if include_pre else "phase-only"
            print(f"\nseed={model_seed} {tag}")
            print(
                "  phase acc pre..final: "
                + " ".join(f"{x:.3f}" for x in row["phase_accuracy_pre_through_final"])
            )
            print(f"  forward mean          {row['forward_mean']:.4f}")
            print(f"  reverse mean          {row['reverse_mean']:.4f}")
            print(f"  shuffled mean         {row['shuffled_mean']:.4f}")
            print(f"  forward - shuffled    {row['forward_minus_shuffled']:+.4f}")
            print(f"  reverse - shuffled    {row['reverse_minus_shuffled']:+.4f}")
            print(f"  forward - reverse     {row['forward_minus_reverse']:+.4f}")
            print(
                "  roughness F/R/S       "
                f"{row['forward_roughness_mean']:.3f} / "
                f"{row['reverse_roughness_mean']:.3f} / "
                f"{row['shuffled_roughness_mean']:.3f}"
            )

    print("\n=== interpretation gate ===")
    for include_pre in (False, True):
        subset = [r for r in rows if r["include_pre"] == include_pre]
        f = float(np.mean([r["forward_mean"] for r in subset]))
        r = float(np.mean([r["reverse_mean"] for r in subset]))
        s = float(np.mean([r_["shuffled_mean"] for r_ in subset]))
        tag = "with-pre" if include_pre else "phase-only"
        print(f"{tag:10s} forward={f:.4f} reverse={r:.4f} shuffled={s:.4f}")
        print(f"           ordered advantage mean={0.5*(f+r)-s:+.4f} orientation gap={f-r:+.4f}")

    print("\nGuardrail")
    print("---------")
    print("The histogram and maximum maturity span are identical across F/R/shuffled arms.")
    print("Only coordinate order changes. A positive ordered advantage would show that the")
    print("fixed KYY readout preserves information about maturity ordering; it would not by")
    print("itself establish a useful physical architecture or a biological mechanism.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"frontier-direction-{stamp}.json"
    path.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2, default=str))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
