from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "partial_merge_for_reification_baseline"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "partial_merge_compiler_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
pm = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = pm
SPEC.loader.exec_module(pm)


def legal_prototypes(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=dtype
    )


def nearest_prototype(h: torch.Tensor, prototypes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d2 = torch.sum((h[:, None, :] - prototypes[None, :, :]) ** 2, dim=-1)
    idx = d2.argmin(dim=-1)
    return prototypes[idx], idx


@torch.no_grad()
def reified_runtime(
    model: pm.SoftPartialMergeTracker,
    tokens: torch.Tensor,
    *,
    reify_every_step: bool,
    reify_merge_only: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if reify_every_step and reify_merge_only:
        raise ValueError("choose at most one reification schedule")
    dtype = model.angle.dtype
    prototypes = legal_prototypes(dtype=dtype).to(model.angle.device)
    h = model.h0.unsqueeze(0).expand(tokens.shape[0], -1).clone()
    outs: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    ids: list[torch.Tensor] = []
    for t in range(tokens.shape[1]):
        tok = tokens[:, t]
        is_merge = (tok == pm.MERGE_TOKEN).view(-1, 1)
        inc = torch.where(tok == pm.MERGE_TOKEN, torch.zeros_like(tok), tok)
        phase = inc.float() * model.angle
        c, s = torch.cos(phase), torch.sin(phase)
        x, y = h[:, 0], h[:, 1]
        rotated = torch.stack((c * x - s * y, s * x + c * y), dim=-1)
        merged = h @ model.merge.T
        candidate = torch.where(is_merge, merged, rotated)

        do_reify = torch.ones_like(is_merge) if reify_every_step else is_merge if reify_merge_only else torch.zeros_like(is_merge)
        snapped, snap_id = nearest_prototype(candidate, prototypes)
        h = torch.where(do_reify, snapped, candidate)
        # Record the nearest legal-state interpretation even if no snap was requested.
        _, nearest_id = nearest_prototype(h, prototypes)
        states.append(h)
        ids.append(nearest_id)
        outs.append(model.readout(h))
    return torch.stack(outs, dim=1), torch.stack(states, dim=1), torch.stack(ids, dim=1)


@torch.no_grad()
def accuracy(
    model: pm.SoftPartialMergeTracker,
    *,
    lengths: list[int],
    batch_size: int,
    merge_probability: float,
    random_start: bool,
    schedule: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for length in lengths:
        x, y = pm.generate_batch(batch_size, length, merge_probability, random_start=random_start)
        logits, _, _ = reified_runtime(
            model,
            x,
            reify_every_step=schedule == "every",
            reify_merge_only=schedule == "merge",
        )
        out[str(length)] = float((logits.argmax(-1) == y).float().mean().item())
    return out


@torch.no_grad()
def leakage(
    model: pm.SoftPartialMergeTracker,
    *,
    batch_size: int,
    continuation_length: int,
    schedule: str,
) -> dict[str, float]:
    a, b, merge_index = pm.paired_leakage_sequences(batch_size, continuation_length)
    la, ha, ia = reified_runtime(
        model, a,
        reify_every_step=schedule == "every",
        reify_merge_only=schedule == "merge",
    )
    lb, hb, ib = reified_runtime(
        model, b,
        reify_every_step=schedule == "every",
        reify_merge_only=schedule == "merge",
    )
    pa, pb = torch.softmax(la, -1), torch.softmax(lb, -1)
    curve_tv = pm.tv(pa[:, merge_index:], pb[:, merge_index:])
    hdiff = torch.linalg.vector_norm(ha[:, merge_index:] - hb[:, merge_index:], dim=-1)
    mismatch = la[:, merge_index:].argmax(-1) != lb[:, merge_index:].argmax(-1)
    id_mismatch = ia[:, merge_index:] != ib[:, merge_index:]
    return {
        "hidden_difference_at_merge": float(hdiff[:, 0].mean().item()),
        "hidden_difference_max_future": float(hdiff.max().item()),
        "probability_tv_at_merge": float(curve_tv[:, 0].mean().item()),
        "probability_tv_max_future": float(curve_tv.max().item()),
        "prediction_mismatch_max_rate": float(mismatch.float().mean(dim=0).max().item()),
        "nearest_state_mismatch_max_rate": float(id_mismatch.float().mean(dim=0).max().item()),
    }


@dataclass
class ReificationRun:
    seed: int
    learned_accuracy: dict[str, float]
    merge_only_accuracy: dict[str, float]
    every_step_accuracy: dict[str, float]
    compiled_accuracy: dict[str, float]
    learned_leakage: dict[str, float]
    merge_only_leakage: dict[str, float]
    every_step_leakage: dict[str, float]
    learned_merge_rank: int
    learned_merge_smallest_singular: float


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
) -> ReificationRun:
    pm.seed_everything(seed)
    model = pm.SoftPartialMergeTracker(seed)
    pm.train_model(
        model,
        steps=steps,
        train_length=train_length,
        batch_size=batch_size,
        merge_probability=merge_probability,
        lr=lr,
        random_start=random_start,
    )
    learned = pm.learned_accuracy(
        model, test_lengths, eval_batch_size, merge_probability, random_start
    )
    merge_only = accuracy(
        model, lengths=test_lengths, batch_size=eval_batch_size,
        merge_probability=merge_probability, random_start=random_start, schedule="merge"
    )
    every = accuracy(
        model, lengths=test_lengths, batch_size=eval_batch_size,
        merge_probability=merge_probability, random_start=random_start, schedule="every"
    )

    learned_angle = float(model.angle.detach().cpu().item())
    projected, _ = pm.eq.base.project_angles_to_characters(4, [learned_angle])
    angle = float(projected[0])
    W = model.readout.weight.detach().cpu().to(torch.float64)
    b = model.readout.bias.detach().cpu().to(torch.float64)
    compiled = pm.compiled_accuracy(
        angle=angle, W=W, b=b, lengths=test_lengths,
        batch_size=eval_batch_size, merge_probability=merge_probability,
        random_start=random_start,
    )
    sv = torch.linalg.svdvals(model.merge.detach().cpu().to(torch.float64))
    return ReificationRun(
        seed=seed,
        learned_accuracy=learned,
        merge_only_accuracy=merge_only,
        every_step_accuracy=every,
        compiled_accuracy=compiled,
        learned_leakage=pm.learned_leakage(model, leakage_batch_size, leakage_continuation),
        merge_only_leakage=leakage(
            model, batch_size=leakage_batch_size,
            continuation_length=leakage_continuation, schedule="merge"
        ),
        every_step_leakage=leakage(
            model, batch_size=leakage_batch_size,
            continuation_length=leakage_continuation, schedule="every"
        ),
        learned_merge_rank=int(torch.linalg.matrix_rank(model.merge.detach()).item()),
        learned_merge_smallest_singular=float(sv[-1].item()),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Nearest-state reification baseline for Pass-44 partial merge")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--test-lengths", nargs="+", type=int, default=[16,64,256,1024])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--merge-probability", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--random-start", action="store_true")
    p.add_argument("--leakage-batch-size", type=int, default=512)
    p.add_argument("--leakage-continuation", type=int, default=64)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    rows = [run_one(
        seed=s, steps=args.steps, train_length=args.train_length,
        test_lengths=args.test_lengths, batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size, merge_probability=args.merge_probability,
        lr=args.lr, random_start=args.random_start,
        leakage_batch_size=args.leakage_batch_size,
        leakage_continuation=args.leakage_continuation,
    ) for s in args.seeds]
    payload = {"config": vars(args), "results": [asdict(r) for r in rows]}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
