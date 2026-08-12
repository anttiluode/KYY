"""Control for the perm3 identity-token artifact in the KYY deadline gate.

The original deadline curves include token 0, the identity generator. Before any work
on the final token has completed, the previous state is therefore already correct on
roughly one third of examples. That makes a synchronized pre-barrier plateau around
0.33 unsurprising and should not be quoted as evidence about partial computation.

This script reruns the phase-legible geom_scatter deadline experiment while evaluating
ONLY examples whose final token is non-identity (token 1 or 2). For those examples the
pre-token state is necessarily the wrong exact group state, so a synchronized system
that has not crossed its first barrier should score approximately zero.

The control does not establish a geometry advantage. It asks only whether the async
critical-path advantage survives removal of the degenerate identity mass.

Run:

    python experiments/deadline_nonidentity_control.py --quick

Fuller compact sweep:

    python experiments/deadline_nonidentity_control.py \
        --model-seeds 0 1 2 --delay-seeds 0 1 2 --steps 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from kyy import generate_batch
from deadline_scatter_gate import (
    async_states_at_deadlines,
    build_schedule,
    canonical_step,
    make_edge_durations,
    seed_everything,
    settled_prefix_state,
    sync_states_at_deadlines,
    train_model,
)


@torch.no_grad()
def evaluate_nonidentity(
    model,
    *,
    eval_seed: int,
    delay_seed: int,
    delay_log_sd: float,
    sequence_length: int,
    batch_size: int,
    batches: int,
    deadline_fractions: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate perm3 final-token deadlines conditioned on final token != identity."""
    model.eval()
    edge_durations = make_edge_durations(model, delay_seed, delay_log_sd)
    schedule = build_schedule(model, edge_durations)
    deadlines = deadline_fractions * schedule.sync_full_time

    async_correct = np.zeros(len(deadlines), dtype=np.float64)
    sync_correct = np.zeros(len(deadlines), dtype=np.float64)
    total = 0
    max_state_error = 0.0

    seed_everything(eval_seed)

    for _ in range(batches):
        x, y = generate_batch("perm3", batch_size, sequence_length, device)
        mask = x[:, -1] != 0
        if not bool(mask.any()):
            continue

        x = x[mask]
        target = y[mask, -1]
        tok = x[:, -1]

        if sequence_length > 1:
            h0 = settled_prefix_state(model, x[:, :-1])
        else:
            h0 = model.h0.unsqueeze(0).expand(x.shape[0], -1)

        canonical_final = canonical_step(model, h0, tok)
        async_states = async_states_at_deadlines(model, h0, tok, schedule, deadlines)
        sync_states = sync_states_at_deadlines(model, h0, tok, schedule, deadlines)

        completed_async = async_states_at_deadlines(
            model,
            h0,
            tok,
            schedule,
            np.asarray([schedule.async_full_time + 1e-9]),
        )[0]
        max_state_error = max(
            max_state_error,
            float((completed_async - canonical_final).abs().max()),
        )

        for i, state in enumerate(async_states):
            pred = model.readout(state).argmax(dim=-1)
            async_correct[i] += float((pred == target).sum())
        for i, state in enumerate(sync_states):
            pred = model.readout(state).argmax(dim=-1)
            sync_correct[i] += float((pred == target).sum())

        total += int(target.numel())

    if max_state_error > 2e-5:
        raise AssertionError(
            f"completed async state differs from canonical state: {max_state_error}"
        )

    async_acc = async_correct / max(1, total)
    sync_acc = sync_correct / max(1, total)

    def at(values: np.ndarray, fraction: float) -> float:
        idx = int(np.argmin(np.abs(deadline_fractions - fraction)))
        return float(values[idx])

    return {
        "delay_seed": delay_seed,
        "n_examples": total,
        "async_full_over_sync_full": schedule.async_full_time / schedule.sync_full_time,
        "deadline_fractions": deadline_fractions.tolist(),
        "async_accuracy": async_acc.tolist(),
        "sync_accuracy": sync_acc.tolist(),
        "auc_async": float(np.trapz(async_acc, deadline_fractions)),
        "auc_sync": float(np.trapz(sync_acc, deadline_fractions)),
        "async_acc_10pct": at(async_acc, 0.10),
        "sync_acc_10pct": at(sync_acc, 0.10),
        "async_acc_20pct": at(async_acc, 0.20),
        "sync_acc_20pct": at(sync_acc, 0.20),
        "async_acc_50pct": at(async_acc, 0.50),
        "sync_acc_50pct": at(sync_acc, 0.50),
        "final_accuracy": float(async_acc[-1]),
        "max_completed_state_abs_error": max_state_error,
    }


def mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    p = argparse.ArgumentParser(description="KYY perm3 non-identity deadline control")
    p.add_argument("--state-dim", type=int, default=32)
    p.add_argument("--topology", choices=["ring", "path", "matching", "disconnected"], default="ring")
    p.add_argument("--train-length", type=int, default=32)
    p.add_argument("--eval-length", type=int, default=32)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--delay-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--delay-log-sd", type=float, default=0.55)
    p.add_argument("--deadline-points", type=int, default=21)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.steps = min(args.steps, 160)
        args.model_seeds = [args.model_seeds[0]]
        args.delay_seeds = args.delay_seeds[:2]
        args.eval_batches = min(args.eval_batches, 4)

    device = torch.device(args.device)
    fractions = np.linspace(0.0, 1.0, args.deadline_points)
    rows: list[dict[str, object]] = []

    for model_seed in args.model_seeds:
        model, _seconds = train_model(
            training_mode="phase",
            task="perm3",
            seed=model_seed,
            state_dim=args.state_dim,
            topology=args.topology,
            train_length=args.train_length,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            log_every=max(1, args.steps // 5),
        )

        for delay_seed in args.delay_seeds:
            row = evaluate_nonidentity(
                model,
                eval_seed=100_000 + 1000 * model_seed + delay_seed,
                delay_seed=delay_seed,
                delay_log_sd=args.delay_log_sd,
                sequence_length=args.eval_length,
                batch_size=args.eval_batch_size,
                batches=args.eval_batches,
                deadline_fractions=fractions,
                device=device,
            )
            row["model_seed"] = model_seed
            rows.append(row)
            print(
                f"model={model_seed} delay={delay_seed} "
                f"AUC={row['auc_async']:.3f}/{row['auc_sync']:.3f} "
                f"@10={row['async_acc_10pct']:.3f}/{row['sync_acc_10pct']:.3f} "
                f"@20={row['async_acc_20pct']:.3f}/{row['sync_acc_20pct']:.3f} "
                f"final={row['final_accuracy']:.3f}"
            )

    summary = {
        key: mean(rows, key)
        for key in (
            "auc_async",
            "auc_sync",
            "async_acc_10pct",
            "sync_acc_10pct",
            "async_acc_20pct",
            "sync_acc_20pct",
            "async_acc_50pct",
            "sync_acc_50pct",
            "final_accuracy",
            "async_full_over_sync_full",
        )
    }

    print("\nmeans")
    print("-----")
    for key, value in summary.items():
        print(f"{key:>28s}: {value:.4f}")

    print("\nGuardrail:")
    print("  sync near zero before its first barrier is expected here: identity tokens were removed.")
    print("  any remaining async advantage is a scheduling result, not yet a local-geometry result.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"deadline-nonidentity-{stamp}.json"
    path.write_text(json.dumps({"config": vars(args), "rows": rows, "mean": summary}, indent=2, default=str))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
