"""Separate monotone maturity order from mere spatial contiguity.

The replicated frontier_direction_control found:

* phase-only: forward/reverse contiguous maturity blocks modestly beat coordinate shuffle;
* with-pre: no ordered advantage;
* no forward-vs-reverse orientation effect in either case.

But coordinate shuffle destroys TWO properties at once:

    temporal order
    spatial block contiguity

This gate keeps each maturity state in one contiguous coordinate block while permuting
the ORDER of the phase blocks.  It therefore asks whether the small phase-only effect
needs monotone maturity order, or whether generic spatial coherence is enough.

Arms, with identical phase histograms:

    monotone        0,1,2,3 around the ring (forward/reverse, cyclic origins averaged)
    block-permuted  contiguous phase blocks, but non-monotone phase order
    coord-shuffled  same labels scattered coordinate-wise

If block-permuted ~= monotone > coord-shuffled, the residual is a contiguity effect,
not a temporal-order effect.

Run:
    PYTHONPATH=. python experiments/frontier_order_vs_contiguity.py --quick
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from kyy import TASKS, generate_batch
from mixed_state_splice_control import final_token_phase_states, seed_everything, train_phase_model
from frontier_direction_control import balanced_maturity_labels, splice_from_labels


def is_dihedral_monotone(order: tuple[int, ...]) -> bool:
    """True for cyclic rotations of forward or reverse phase order."""
    n = len(order)
    f = tuple(range(n))
    r = tuple(reversed(f))
    dihedral = set()
    for base in (f, r):
        for shift in range(n):
            dihedral.add(base[shift:] + base[:shift])
    return order in dihedral


def relabel_blocks(base_labels: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
    """Keep the same coordinate blocks; replace each block's phase identity."""
    out = np.empty_like(base_labels)
    for block_index, phase_label in enumerate(order):
        out[base_labels == block_index] = phase_label
    return out


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
    device: torch.device,
) -> dict[str, object]:
    seed_everything(seed)
    rng = np.random.default_rng(seed + 991)
    model.eval()

    dim = int(model.state_dim)
    totals = {
        "forward": 0.0,
        "reverse": 0.0,
        "block_permuted": 0.0,
        "coord_shuffled": 0.0,
    }
    denom = {k: 0 for k in totals}
    block_orders_used: list[tuple[int, ...]] | None = None
    total_examples = 0

    for _ in range(batches):
        x, y = generate_batch(task, batch_size, length, device)
        target = y[:, -1]
        all_states = final_token_phase_states(model, x)
        states = all_states if include_pre else all_states[1:]
        n_states = len(states)
        base = balanced_maturity_labels(dim, n_states)

        if block_orders_used is None:
            perms = [p for p in itertools.permutations(range(n_states)) if not is_dihedral_monotone(p)]
            block_orders_used = perms

        reverse_base = base[::-1].copy()

        # Average monotone arms over every ring origin.
        for offset in range(dim):
            for name, labels in (
                ("forward", np.roll(base, offset)),
                ("reverse", np.roll(reverse_base, offset)),
            ):
                state = splice_from_labels(states, labels)
                totals[name] += float((model.readout(state).argmax(-1) == target).sum())
                denom[name] += batch_size

        # Every non-dihedral contiguous block order, also averaged over origins.
        assert block_orders_used is not None
        for order in block_orders_used:
            labels0 = relabel_blocks(base, order)
            for offset in range(dim):
                labels = np.roll(labels0, offset)
                state = splice_from_labels(states, labels)
                totals["block_permuted"] += float(
                    (model.readout(state).argmax(-1) == target).sum()
                )
                denom["block_permuted"] += batch_size

        # Coordinate shuffle preserves histogram but destroys contiguity.
        for _draw in range(shuffle_draws * dim):
            labels = rng.permutation(base)
            state = splice_from_labels(states, labels)
            totals["coord_shuffled"] += float(
                (model.readout(state).argmax(-1) == target).sum()
            )
            denom["coord_shuffled"] += batch_size

        total_examples += batch_size

    acc = {k: totals[k] / denom[k] for k in totals}
    monotone = 0.5 * (acc["forward"] + acc["reverse"])
    return {
        "include_pre": include_pre,
        "n_states": n_states,
        "block_orders": len(block_orders_used or []),
        "eval_examples": total_examples,
        **acc,
        "monotone_mean": monotone,
        "monotone_minus_block": monotone - acc["block_permuted"],
        "block_minus_shuffle": acc["block_permuted"] - acc["coord_shuffled"],
        "monotone_minus_shuffle": monotone - acc["coord_shuffled"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=sorted(TASKS), default="perm3")
    p.add_argument("--state-dim", type=int, default=32)
    p.add_argument("--topology", choices=["ring", "path", "matching", "disconnected"], default="ring")
    p.add_argument("--train-length", type=int, default=32)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--eval-batches", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--eval-length", type=int, default=32)
    p.add_argument("--shuffle-draws", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.steps = 140
        args.model_seeds = [args.model_seeds[0]]
        args.eval_batches = 2
        args.eval_batch_size = 128
        args.eval_length = 24
        args.shuffle_draws = 1
        args.log_every = 70

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
                seed=120_000 + model_seed + int(include_pre) * 1000,
                shuffle_draws=args.shuffle_draws,
                include_pre=include_pre,
                device=device,
            )
            row["model_seed"] = model_seed
            rows.append(row)
            tag = "with-pre" if include_pre else "phase-only"
            print(f"\nseed={model_seed} {tag}")
            print(f"  monotone mean          {row['monotone_mean']:.4f}")
            print(f"  block-permuted         {row['block_permuted']:.4f}")
            print(f"  coordinate-shuffled    {row['coord_shuffled']:.4f}")
            print(f"  monotone - block       {row['monotone_minus_block']:+.4f}")
            print(f"  block - shuffle        {row['block_minus_shuffle']:+.4f}")
            print(f"  monotone - shuffle     {row['monotone_minus_shuffle']:+.4f}")

    print("\n=== aggregate ===")
    for include_pre in (False, True):
        subset = [r for r in rows if r["include_pre"] == include_pre]
        tag = "with-pre" if include_pre else "phase-only"
        for key in (
            "monotone_mean",
            "block_permuted",
            "coord_shuffled",
            "monotone_minus_block",
            "block_minus_shuffle",
            "monotone_minus_shuffle",
        ):
            print(f"{tag:10s} {key:24s} {np.mean([r[key] for r in subset]):+.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"frontier-order-contiguity-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2, default=str))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
