from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from kyy import MODEL_NAMES, TASKS, build_model, generate_batch, parameter_count


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, task: str, length: int, batch_size: int, batches: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    final_correct = 0
    final_total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for _ in range(batches):
        x, y = generate_batch(task, batch_size, length, device)
        logits = model(x)
        loss_sum += float(criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)))
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum())
        total += y.numel()
        final_correct += int((pred[:, -1] == y[:, -1]).sum())
        final_total += batch_size
    return {
        "accuracy": correct / max(1, total),
        "final_accuracy": final_correct / max(1, final_total),
        "loss": loss_sum / max(1, total),
    }


def train_one(
    model_name: str,
    task: str,
    seed: int,
    state_dim: int,
    train_length: int,
    test_lengths: List[int],
    steps: int,
    batch_size: int,
    eval_batches: int,
    lr: float,
    topology: str,
    device: torch.device,
    log_every: int,
) -> Dict[str, object]:
    seed_everything(seed)
    spec = TASKS[task]
    model = build_model(model_name, spec.vocab_size, spec.n_classes, state_dim, topology=topology).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    started = time.perf_counter()
    model.train()
    train_curve = []
    for step in range(1, steps + 1):
        x, y = generate_batch(task, batch_size, train_length, device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, spec.n_classes), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            with torch.no_grad():
                acc = float((logits.argmax(dim=-1) == y).float().mean())
            train_curve.append({"step": step, "loss": float(loss.detach()), "accuracy": acc})
            print(
                f"[{task:8s}] {model_name:12s} seed={seed} "
                f"step={step:4d}/{steps} loss={float(loss.detach()):.4f} acc={acc:.3f}",
                flush=True,
            )

    train_seconds = time.perf_counter() - started
    evals = {}
    for length in test_lengths:
        evals[str(length)] = evaluate(model, task, length, batch_size, eval_batches, device)

    return {
        "model": model_name,
        "task": task,
        "seed": seed,
        "state_dim": state_dim,
        "topology": topology if model_name in {"geom_wave", "geom_scatter"} else None,
        "parameters": parameter_count(model),
        "train_length": train_length,
        "steps": steps,
        "batch_size": batch_size,
        "train_seconds": train_seconds,
        "train_curve": train_curve,
        "eval": evals,
        "operator": model.operator_summary(),
    }


def flatten_rows(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for r in results:
        for length, metrics in r["eval"].items():
            rows.append(
                {
                    "model": r["model"],
                    "task": r["task"],
                    "seed": r["seed"],
                    "state_dim": r["state_dim"],
                    "topology": r["topology"],
                    "parameters": r["parameters"],
                    "train_length": r["train_length"],
                    "test_length": int(length),
                    "accuracy": metrics["accuracy"],
                    "final_accuracy": metrics["final_accuracy"],
                    "loss": metrics["loss"],
                    "train_seconds": r["train_seconds"],
                }
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="KYY: geometry-derived recurrent operator benchmark")
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=list(TASKS))
    p.add_argument("--state-dim", type=int, default=32)
    p.add_argument("--train-length", type=int, default=32)
    p.add_argument("--test-lengths", nargs="+", type=int, default=[32, 64, 128, 256])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--topology", choices=["ring", "path", "matching", "disconnected"], default="ring")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--quick", action="store_true", help="small one-seed smoke benchmark")
    args = p.parse_args()

    if args.state_dim % 2 and any(m in {"complex_diag", "geom_wave", "geom_scatter"} for m in args.models):
        p.error("--state-dim must be even when using complex_diag, geom_wave, or geom_scatter")

    if args.quick:
        args.steps = min(args.steps, 250)
        args.seeds = [args.seeds[0]]
        args.test_lengths = [args.train_length, args.train_length * 2, args.train_length * 4]
        args.eval_batches = min(args.eval_batches, 5)
        args.log_every = min(args.log_every, 50)

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["device"] = str(device)
    print(json.dumps(config, indent=2))

    results = []
    for task in args.tasks:
        for model_name in args.models:
            for seed in args.seeds:
                results.append(
                    train_one(
                        model_name=model_name,
                        task=task,
                        seed=seed,
                        state_dim=args.state_dim,
                        train_length=args.train_length,
                        test_lengths=args.test_lengths,
                        steps=args.steps,
                        batch_size=args.batch_size,
                        eval_batches=args.eval_batches,
                        lr=args.lr,
                        topology=args.topology,
                        device=device,
                        log_every=args.log_every,
                    )
                )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"kyy-{stamp}.json"
    csv_path = out_dir / f"kyy-{stamp}.csv"
    json_path.write_text(json.dumps({"config": config, "results": results}, indent=2))

    rows = flatten_rows(results)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")

    print("\n=== final-token accuracy ===")
    for row in rows:
        print(
            f"{row['task']:8s} {row['model']:12s} seed={row['seed']} "
            f"L={row['test_length']:4d} final={row['final_accuracy']:.3f} "
            f"all={row['accuracy']:.3f} params={row['parameters']}"
        )


if __name__ == "__main__":
    main()
