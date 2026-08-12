"""Controls for circularity in the KYY output-null interpretation.

Two questions are attacked.

1. Receiver specificity
----------------------
Phase supervision directly optimizes one shared readout W. If later motion becomes
more null to W, that may simply restate the objective. We therefore compare the null
fraction of the same hidden-state motion relative to:

    trained receiver W
    random never-trained rank-matched receiver W'

If only W changes, the effect is receiver-specific alignment rather than a global
reorganization of hidden dynamics.

2. Held-out maturity surface
----------------------------
Train the shared head after all ordinary checkerboard phases EXCEPT one phase index.
Then evaluate the omitted homogeneous phase. If semantic alignment generalizes there,
that would be stronger evidence than measuring a directly supervised phase. If it
fails, the direct deep-supervision explanation wins.

This is an engineering diagnostic. Do not claim that brains literally deep-supervise
checkerboard phases.

Run:

    python experiments/output_null_transfer_control.py --quick

Fuller:

    python experiments/output_null_transfer_control.py \
        --model-seeds 0 1 2 --steps 300 --held-out-phase 2
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


def phases(model):
    for sweep in range(model.sweeps):
        yield sweep, model.phase0
        yield sweep, model.phase1


def train(
    *,
    mode: str,
    task: str,
    seed: int,
    state_dim: int,
    topology: str,
    train_length: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    held_out_phase: int | None = None,
):
    """Train final-only, all-phase, or phase supervision with one phase omitted."""
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

    for _step in range(steps):
        x, y = generate_batch(task, batch_size, train_length, device)

        if mode == "final":
            logits = model(x)
            loss = criterion(logits.reshape(-1, spec.n_classes), y.reshape(-1))
        else:
            h = model.h0.unsqueeze(0).expand(batch_size, -1)
            losses = []
            for t in range(train_length):
                tok = x[:, t]
                target = y[:, t]
                phase_index = 0
                for sweep, edge_ids in phases(model):
                    h = model._scatter_phase(h, tok, sweep, edge_ids)
                    if held_out_phase is None or phase_index != held_out_phase:
                        losses.append(criterion(model.readout(h), target))
                    phase_index += 1
            loss = torch.stack(losses).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return model


@torch.no_grad()
def row_basis(W: torch.Tensor, tol: float = 1e-6) -> tuple[torch.Tensor, int]:
    _U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    rank = int((S > tol).sum().item())
    return Vh[:rank].T.contiguous(), rank


@torch.no_grad()
def random_basis(state_dim: int, rank: int, seed: int, device: torch.device, dtype) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    raw = torch.randn(state_dim, rank, generator=g, dtype=dtype).to(device)
    Q, _R = torch.linalg.qr(raw, mode="reduced")
    return Q


@torch.no_grad()
def measure(
    model,
    *,
    task: str,
    seed: int,
    length: int,
    batch_size: int,
    batches: int,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(seed)
    model.eval()
    W_basis, rank = row_basis(model.readout.weight.detach())
    R_basis = random_basis(
        model.state_dim,
        rank,
        seed=seed + 9_999_991,
        device=device,
        dtype=model.readout.weight.dtype,
    )
    n_phases = 2 * model.sweeps

    correct = np.zeros(n_phases, dtype=np.float64)
    total_energy = np.zeros(n_phases, dtype=np.float64)
    trained_potent = np.zeros(n_phases, dtype=np.float64)
    random_potent = np.zeros(n_phases, dtype=np.float64)
    samples = 0

    for _ in range(batches):
        x, y = generate_batch(task, batch_size, length, device)
        h = model.h0.unsqueeze(0).expand(batch_size, -1)

        for t in range(length):
            tok = x[:, t]
            target = y[:, t]
            phase_index = 0
            for sweep, edge_ids in phases(model):
                h_prev = h
                h = model._scatter_phase(h, tok, sweep, edge_ids)
                delta = h - h_prev
                energy = (delta * delta).sum(dim=-1)
                trained = ((delta @ W_basis) ** 2).sum(dim=-1)
                random_e = ((delta @ R_basis) ** 2).sum(dim=-1)

                total_energy[phase_index] += float(energy.sum())
                trained_potent[phase_index] += float(trained.sum())
                random_potent[phase_index] += float(random_e.sum())
                correct[phase_index] += float(
                    (model.readout(h).argmax(dim=-1) == target).sum()
                )
                phase_index += 1

            samples += batch_size

    trained_null = 1.0 - trained_potent / np.maximum(total_energy, 1e-12)
    random_null = 1.0 - random_potent / np.maximum(total_energy, 1e-12)

    return {
        "readout_rank": rank,
        "isotropic_null_baseline": 1.0 - rank / model.state_dim,
        "phase_accuracy": (correct / samples).tolist(),
        "trained_receiver_null_fraction": trained_null.tolist(),
        "random_receiver_null_fraction": random_null.tolist(),
    }


def curve_mean(rows: list[dict[str, object]], key: str) -> list[float]:
    return np.asarray([r[key] for r in rows], dtype=float).mean(axis=0).tolist()


def main() -> None:
    p = argparse.ArgumentParser(description="KYY output-null transfer/circularity controls")
    p.add_argument("--task", choices=sorted(TASKS), default="perm3")
    p.add_argument("--state-dim", type=int, default=32)
    p.add_argument("--topology", choices=["ring", "path", "matching", "disconnected"], default="ring")
    p.add_argument("--train-length", type=int, default=32)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--held-out-phase", type=int, default=2, help="0-based homogeneous phase index")
    p.add_argument("--measure-length", type=int, default=24)
    p.add_argument("--measure-batch-size", type=int, default=256)
    p.add_argument("--measure-batches", type=int, default=6)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if not (0 <= args.held_out_phase < 4):
        p.error("--held-out-phase must be 0..3 for the default two-sweep geom_scatter")

    if args.quick:
        args.steps = min(args.steps, 140)
        args.model_seeds = [args.model_seeds[0]]
        args.measure_batches = min(args.measure_batches, 3)
        args.measure_batch_size = min(args.measure_batch_size, 128)
        args.measure_length = min(args.measure_length, 16)

    device = torch.device(args.device)
    rows: dict[str, list[dict[str, object]]] = {
        "final": [],
        "phase": [],
        "heldout": [],
    }

    for seed in args.model_seeds:
        conditions = [
            ("final", "final", None),
            ("phase", "phase", None),
            ("heldout", "phase", args.held_out_phase),
        ]
        for label, mode, holdout in conditions:
            model = train(
                mode=mode,
                task=args.task,
                seed=seed,
                state_dim=args.state_dim,
                topology=args.topology,
                train_length=args.train_length,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                held_out_phase=holdout,
            )
            m = measure(
                model,
                task=args.task,
                seed=50_000 + seed,
                length=args.measure_length,
                batch_size=args.measure_batch_size,
                batches=args.measure_batches,
                device=device,
            )
            m["model_seed"] = seed
            rows[label].append(m)
            print(f"\n{label} seed={seed}")
            print("  accuracy     : " + " ".join(f"{v:.3f}" for v in m["phase_accuracy"]))
            print("  null trained : " + " ".join(f"{v:.3f}" for v in m["trained_receiver_null_fraction"]))
            print("  null random  : " + " ".join(f"{v:.3f}" for v in m["random_receiver_null_fraction"]))

    means = {}
    for label, label_rows in rows.items():
        means[label] = {
            "phase_accuracy": curve_mean(label_rows, "phase_accuracy"),
            "trained_receiver_null_fraction": curve_mean(label_rows, "trained_receiver_null_fraction"),
            "random_receiver_null_fraction": curve_mean(label_rows, "random_receiver_null_fraction"),
            "isotropic_null_baseline": float(
                np.mean([r["isotropic_null_baseline"] for r in label_rows])
            ),
        }

    print("\n=== means ===")
    for label, m in means.items():
        print(f"\n{label}")
        print("  accuracy     : " + " ".join(f"{v:.4f}" for v in m["phase_accuracy"]))
        print("  null trained : " + " ".join(f"{v:.4f}" for v in m["trained_receiver_null_fraction"]))
        print("  null random  : " + " ".join(f"{v:.4f}" for v in m["random_receiver_null_fraction"]))
        print(f"  isotropic ref: {m['isotropic_null_baseline']:.4f}")

    print("\nInterpretation guardrail")
    print("------------------------")
    print("If null motion rises only relative to the trained receiver, that is receiver-specific")
    print("alignment. If the omitted phase remains illegible, direct phase supervision is doing")
    print("the semantic work. Neither result is evidence that brains use this training rule.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"output-null-transfer-{stamp}.json"
    path.write_text(json.dumps({"config": vars(args), "rows": rows, "means": means}, indent=2, default=str))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
