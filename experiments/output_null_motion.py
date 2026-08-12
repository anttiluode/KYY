"""Does phase-legible KYY move ongoing computation into readout-null directions?

Motivation
----------
The deadline/partial-maturity gate showed that removing synchronization barriers makes
intermediate local computation physically available, but an ordinary final readout can
misinterpret those states badly. Supervising the same shared readout after each
checkerboard phase makes mixed-maturity states much more legible.

This script asks *how* that happens geometrically.

For the trained linear readout

    logits = W h + b

state changes ``delta_h`` can be decomposed into

    output-potent motion : projection onto row(W)
    output-null motion   : projection onto null(W)

Only the potent component can change logits instantaneously under this readout.

We compare two otherwise identical ``geom_scatter`` models:

``final``
    ordinary KYY training; supervise after each complete token transition.

``phase``
    same shared readout supervised after every homogeneous checkerboard phase.
    No extra heads and no asynchronous states during training.

If phase supervision makes later internal computation predominantly output-null while
keeping the public logits stable, that is a concrete mechanism for maturity-consistent
legibility.  It is not a novelty claim: output-null/output-potent population geometry
is established neuroscience, and deep supervision is established ML prior art.

Run
---

    python experiments/output_null_motion.py --quick

or

    python experiments/output_null_motion.py --model-seeds 0 1 2 --steps 300

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


def train_model(
    *,
    mode: str,
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
    if mode not in {"final", "phase"}:
        raise ValueError(mode)

    seed_everything(seed)
    spec = TASKS[task]
    model = build_model(
        "geom_scatter",
        spec.vocab_size,
        spec.n_classes,
        state_dim,
        topology=topology,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(1, steps + 1):
        x, y = generate_batch(task, batch_size, length, device)

        if mode == "final":
            logits = model(x)
            loss = criterion(logits.reshape(-1, spec.n_classes), y.reshape(-1))
            final_acc = float((logits.argmax(-1) == y).float().mean().detach())
        else:
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
            final_acc = float((logits.argmax(-1) == y).float().mean().detach())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % log_every == 0 or step == steps:
            print(
                f"[{mode:5s}] seed={seed} step={step:4d}/{steps} "
                f"loss={float(loss.detach()):.4f} final-acc={final_acc:.3f}",
                flush=True,
            )

    return model


@torch.no_grad()
def readout_row_basis(model, tol: float = 1e-6) -> tuple[torch.Tensor, int]:
    """Orthonormal basis V whose columns span row(W) in hidden-state coordinates."""
    W = model.readout.weight.detach()
    _U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    rank = int((S > tol).sum().item())
    V = Vh[:rank].T.contiguous()
    return V, rank


@torch.no_grad()
def measure_model(
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
    V, rank = readout_row_basis(model)
    n_phases = 2 * model.sweeps

    total_energy = np.zeros(n_phases, dtype=np.float64)
    potent_energy = np.zeros(n_phases, dtype=np.float64)
    logit_delta_energy = np.zeros(n_phases, dtype=np.float64)
    correct = np.zeros(n_phases, dtype=np.float64)
    samples = 0

    for _ in range(batches):
        x, y = generate_batch(task, batch_size, length, device)
        h = model.h0.unsqueeze(0).expand(batch_size, -1)

        for t in range(length):
            tok = x[:, t]
            target = y[:, t]
            phase_index = 0

            for sweep, edge_ids in phase_edges(model):
                h_prev = h
                logits_prev = model.readout(h_prev)
                h = model._scatter_phase(h, tok, sweep, edge_ids)
                logits = model.readout(h)

                delta = h - h_prev
                potent_coord = delta @ V
                total_energy[phase_index] += float((delta * delta).sum())
                potent_energy[phase_index] += float((potent_coord * potent_coord).sum())
                logit_delta_energy[phase_index] += float(
                    ((logits - logits_prev) ** 2).sum()
                )
                correct[phase_index] += float((logits.argmax(-1) == target).sum())
                phase_index += 1

            samples += batch_size

    potent_fraction = potent_energy / np.maximum(total_energy, 1e-12)
    null_fraction = 1.0 - potent_fraction

    return {
        "readout_rank": rank,
        "phase_accuracy": (correct / samples).tolist(),
        "potent_motion_fraction": potent_fraction.tolist(),
        "null_motion_fraction": null_fraction.tolist(),
        "mean_logit_delta_energy": (logit_delta_energy / samples).tolist(),
        "mean_hidden_delta_energy": (total_energy / samples).tolist(),
    }


def mean_curve(rows: list[dict[str, object]], key: str) -> list[float]:
    return np.asarray([row[key] for row in rows], dtype=float).mean(axis=0).tolist()


def main() -> None:
    p = argparse.ArgumentParser(description="KYY output-null partial-maturity diagnostic")
    p.add_argument("--task", choices=sorted(TASKS), default="perm3")
    p.add_argument("--state-dim", type=int, default=32)
    p.add_argument("--topology", choices=["ring", "path", "matching", "disconnected"], default="ring")
    p.add_argument("--train-length", type=int, default=32)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--measure-batches", type=int, default=8)
    p.add_argument("--measure-batch-size", type=int, default=256)
    p.add_argument("--measure-length", type=int, default=24)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.steps = min(args.steps, 150)
        args.model_seeds = [args.model_seeds[0]]
        args.measure_batches = min(args.measure_batches, 3)
        args.measure_batch_size = min(args.measure_batch_size, 128)
        args.measure_length = min(args.measure_length, 16)
        args.log_every = min(args.log_every, 30)

    device = torch.device(args.device)
    rows: dict[str, list[dict[str, object]]] = {"final": [], "phase": []}

    for seed in args.model_seeds:
        for mode in ("final", "phase"):
            model = train_model(
                mode=mode,
                task=args.task,
                seed=seed,
                state_dim=args.state_dim,
                topology=args.topology,
                length=args.train_length,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                log_every=args.log_every,
            )
            measured = measure_model(
                model,
                task=args.task,
                batches=args.measure_batches,
                batch_size=args.measure_batch_size,
                length=args.measure_length,
                seed=50_000 + seed,
                device=device,
            )
            measured["model_seed"] = seed
            rows[mode].append(measured)

            print(f"\n{mode} seed={seed}")
            print("  phase accuracy : " + " ".join(f"{x:.3f}" for x in measured["phase_accuracy"]))
            print("  null fraction  : " + " ".join(f"{x:.3f}" for x in measured["null_motion_fraction"]))
            print("  logit delta E  : " + " ".join(f"{x:.3f}" for x in measured["mean_logit_delta_energy"]))

    means: dict[str, dict[str, object]] = {}
    for mode in ("final", "phase"):
        means[mode] = {
            "phase_accuracy": mean_curve(rows[mode], "phase_accuracy"),
            "potent_motion_fraction": mean_curve(rows[mode], "potent_motion_fraction"),
            "null_motion_fraction": mean_curve(rows[mode], "null_motion_fraction"),
            "mean_logit_delta_energy": mean_curve(rows[mode], "mean_logit_delta_energy"),
            "mean_hidden_delta_energy": mean_curve(rows[mode], "mean_hidden_delta_energy"),
        }

    print("\n=== means ===")
    for mode in ("final", "phase"):
        print(f"\n{mode}")
        print("  phase accuracy : " + " ".join(f"{x:.4f}" for x in means[mode]["phase_accuracy"]))
        print("  null fraction  : " + " ".join(f"{x:.4f}" for x in means[mode]["null_motion_fraction"]))
        print("  logit delta E  : " + " ".join(f"{x:.4f}" for x in means[mode]["mean_logit_delta_energy"]))

    print("\nInterpretation guardrail")
    print("------------------------")
    print("A high null-motion fraction means ongoing hidden-state motion has little direct")
    print("effect on the current linear readout. It does not prove a biological mechanism")
    print("or a KYY advantage; it characterizes how phase legibility is implemented here.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"output-null-motion-{stamp}.json"
    path.write_text(json.dumps({"config": vars(args), "rows": rows, "means": means}, indent=2, default=str))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
