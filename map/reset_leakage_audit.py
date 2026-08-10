from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from kyy import TASKS, build_model, generate_batch
from kyy.tasks import permreset3_targets


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_reset_pairs(
    batch_size: int,
    prefix_length: int,
    continuation_length: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Create paired permreset3 sequences that differ before one shared reset.

    Prefixes contain only I/C tokens, so changing one I<->C changes the
    pre-reset state by exactly +/-1 mod 3.  The reset and every later token are
    identical in the pair.

    Returns `(a, b, reset_index)` where both have shape [batch, length].
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if prefix_length <= 0:
        raise ValueError("prefix_length must be positive")
    if continuation_length < 0:
        raise ValueError("continuation_length must be non-negative")

    device = torch.device(device)
    prefix_a = torch.randint(0, 2, (batch_size, prefix_length), device=device)
    prefix_b = prefix_a.clone()

    # Flip one I/C event in every row.  With no reset in the prefix this changes
    # the pre-reset state by one modulo 3 and guarantees different behavior.
    positions = torch.randint(0, prefix_length, (batch_size,), device=device)
    rows = torch.arange(batch_size, device=device)
    prefix_b[rows, positions] = 1 - prefix_b[rows, positions]

    reset_column = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)
    continuation = torch.randint(0, 3, (batch_size, continuation_length), device=device)

    a = torch.cat((prefix_a, reset_column, continuation), dim=1)
    b = torch.cat((prefix_b, reset_column, continuation), dim=1)
    return a, b, prefix_length


def validate_reset_pairs(a: torch.Tensor, b: torch.Tensor, reset_index: int) -> None:
    """Raise if the pair does not satisfy the behavioral-reset construction."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("paired sequences must have equal [batch,length] shape")
    if not 0 <= reset_index < a.shape[1]:
        raise ValueError("invalid reset index")
    if not torch.all(a[:, reset_index] == 2) or not torch.all(b[:, reset_index] == 2):
        raise AssertionError("both sequences must reset at reset_index")
    if not torch.equal(a[:, reset_index:], b[:, reset_index:]):
        raise AssertionError("reset token and continuation must be identical")

    ya = permreset3_targets(a)
    yb = permreset3_targets(b)
    if torch.any(ya[:, reset_index - 1] == yb[:, reset_index - 1]):
        raise AssertionError("prefixes must end in different behavioral states")
    if not torch.equal(ya[:, reset_index:], yb[:, reset_index:]):
        raise AssertionError("exact behavior must coincide from the reset onward")


def total_variation(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 0.5 * (p - q).abs().sum(dim=-1)


def leakage_curve_from_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    reset_index: int,
) -> list[dict[str, float]]:
    """Port-visible divergence at reset and each later continuation step."""
    if logits_a.shape != logits_b.shape or logits_a.ndim != 3:
        raise ValueError("logits must have equal [batch,length,classes] shape")
    if not 0 <= reset_index < logits_a.shape[1]:
        raise ValueError("invalid reset index")

    pa = torch.softmax(logits_a, dim=-1)
    pb = torch.softmax(logits_b, dim=-1)
    pred_a = logits_a.argmax(dim=-1)
    pred_b = logits_b.argmax(dim=-1)

    out: list[dict[str, float]] = []
    for t in range(reset_index, logits_a.shape[1]):
        da = logits_a[:, t]
        db = logits_b[:, t]
        logit_l2 = torch.linalg.vector_norm(da - db, dim=-1)
        tv = total_variation(pa[:, t], pb[:, t])
        out.append(
            {
                "lag": float(t - reset_index),
                "mean_logit_l2": float(logit_l2.mean()),
                "max_logit_l2": float(logit_l2.max()),
                "mean_probability_tv": float(tv.mean()),
                "max_probability_tv": float(tv.max()),
                "prediction_mismatch_rate": float((pred_a[:, t] != pred_b[:, t]).float().mean()),
            }
        )
    return out


def summarize_curve(curve: list[dict[str, float]]) -> dict[str, float]:
    if not curve:
        return {
            "mean_probability_tv": 0.0,
            "max_probability_tv": 0.0,
            "max_prediction_mismatch_rate": 0.0,
        }
    return {
        "mean_probability_tv": float(np.mean([x["mean_probability_tv"] for x in curve])),
        "max_probability_tv": float(max(x["max_probability_tv"] for x in curve)),
        "max_prediction_mismatch_rate": float(max(x["prediction_mismatch_rate"] for x in curve)),
    }


def train_model(
    model: nn.Module,
    steps: int,
    train_length: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    spec = TASKS["permreset3"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        x, y = generate_batch("permreset3", batch_size, train_length, device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, spec.n_classes), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@torch.no_grad()
def accuracy(
    model: nn.Module,
    length: int,
    batch_size: int,
    batches: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = total = final_correct = final_total = 0
    for _ in range(batches):
        x, y = generate_batch("permreset3", batch_size, length, device)
        pred = model(x).argmax(dim=-1)
        correct += int((pred == y).sum())
        total += y.numel()
        final_correct += int((pred[:, -1] == y[:, -1]).sum())
        final_total += batch_size
    return {
        "accuracy": correct / max(1, total),
        "final_accuracy": final_correct / max(1, final_total),
    }


@torch.no_grad()
def audit_leakage(
    model: nn.Module,
    pair_batches: int,
    batch_size: int,
    prefix_length: int,
    continuation_length: int,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Average the leakage curve over independently generated paired batches."""
    model.eval()
    accum: list[dict[str, float]] | None = None
    for _ in range(pair_batches):
        a, b, reset_index = make_reset_pairs(
            batch_size, prefix_length, continuation_length, device=device
        )
        validate_reset_pairs(a, b, reset_index)
        curve = leakage_curve_from_logits(model(a), model(b), reset_index)
        if accum is None:
            accum = [dict(row) for row in curve]
        else:
            for dst, src in zip(accum, curve):
                for key in (
                    "mean_logit_l2",
                    "max_logit_l2",
                    "mean_probability_tv",
                    "max_probability_tv",
                    "prediction_mismatch_rate",
                ):
                    dst[key] += src[key]

    assert accum is not None
    for row in accum:
        for key in (
            "mean_logit_l2",
            "max_logit_l2",
            "mean_probability_tv",
            "max_probability_tv",
            "prediction_mismatch_rate",
        ):
            row[key] /= pair_batches
    return accum, summarize_curve(accum)


@dataclass
class AuditRow:
    model: str
    seed: int
    state_dim: int
    topology: str | None
    accuracy: dict[str, dict[str, float]]
    leakage_summary: dict[str, float]
    leakage_curve: list[dict[str, float]]


def main() -> None:
    p = argparse.ArgumentParser(
        description="KYY post-reset behavioral leakage audit for existing recurrent models"
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["diag_signed", "complex_diag", "householder2", "geom_wave", "geom_scatter", "gru"],
    )
    p.add_argument("--state-dim", type=int, default=8)
    p.add_argument("--topology", default="ring")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--train-length", type=int, default=32)
    p.add_argument("--test-lengths", nargs="+", type=int, default=[32, 128, 512])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--pair-batches", type=int, default=10)
    p.add_argument("--prefix-length", type=int, default=24)
    p.add_argument("--continuation-length", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    spec = TASKS["permreset3"]
    rows: list[AuditRow] = []

    for name in args.models:
        for seed in args.seeds:
            seed_everything(seed)
            model = build_model(
                name,
                vocab_size=spec.vocab_size,
                n_classes=spec.n_classes,
                state_dim=args.state_dim,
                topology=args.topology,
            ).to(device)
            train_model(model, args.steps, args.train_length, args.batch_size, args.lr, device)
            evals = {
                str(length): accuracy(model, length, args.batch_size, args.eval_batches, device)
                for length in args.test_lengths
            }
            curve, leakage_summary = audit_leakage(
                model,
                args.pair_batches,
                args.batch_size,
                args.prefix_length,
                args.continuation_length,
                device,
            )
            row = AuditRow(
                model=name,
                seed=seed,
                state_dim=args.state_dim,
                topology=args.topology if name in {"geom_wave", "geom_scatter"} else None,
                accuracy=evals,
                leakage_summary=leakage_summary,
                leakage_curve=curve,
            )
            rows.append(row)
            print(json.dumps(asdict(row), indent=2))

    payload = {"config": vars(args), "rows": [asdict(row) for row in rows]}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
