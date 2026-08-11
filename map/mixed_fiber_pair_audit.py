from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "mixed_fiber_compiler_for_pair_audit"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "mixed_fiber_compiler_probe.py")
assert SPEC is not None and SPEC.loader is not None
base = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = base
SPEC.loader.exec_module(base)


def paired_tokens(batch: int, continuation: int, scale_probability: float):
    """Same analog payload; digital histories 0 vs 1; then M and common no-merge suffix."""
    q_a = torch.zeros(batch, dtype=torch.long)
    q_b = torch.ones(batch, dtype=torch.long)
    a0 = torch.empty(batch).uniform_(-1.5, 1.5)
    first = torch.full((batch, 1), base.MERGE, dtype=torch.long)
    future = torch.randint(0, 4, (batch, continuation), dtype=torch.long)
    use_scale = torch.rand(batch, continuation) < scale_probability
    future[use_scale] = base.SCALE
    tokens = torch.cat((first, future), 1)
    return tokens, q_a, q_b, a0


@torch.no_grad()
def reified_pair_audit(model, tokens: torch.Tensor, q_a: torch.Tensor, q_b: torch.Tensor, a0: torch.Tensor):
    ha = model.init_hidden(q_a, a0)
    hb = model.init_hidden(q_b, a0)
    q_mismatch: list[torch.Tensor] = []
    analog_gap: list[torch.Tensor] = []
    hidden_gap: list[torch.Tensor] = []
    for t in range(tokens.shape[1]):
        tok = tokens[:, t]
        ha = model.step(ha, tok)
        hb = model.step(hb, tok)
        ha, qa, aa = base.nearest_fiber(model, ha)
        hb, qb, ab = base.nearest_fiber(model, hb)
        q_mismatch.append(qa != qb)
        analog_gap.append(torch.abs(aa - ab))
        hidden_gap.append(torch.linalg.vector_norm(ha - hb, dim=-1))
    qm = torch.stack(q_mismatch, 1)
    ag = torch.stack(analog_gap, 1)
    hg = torch.stack(hidden_gap, 1)
    return {
        "q_mismatch_at_merge": float(qm[:, 0].to(torch.float64).mean().item()),
        "q_mismatch_max_future_rate": float(qm.to(torch.float64).mean(0).max().item()),
        "analog_gap_at_merge_mean": float(ag[:, 0].mean().item()),
        "analog_gap_at_merge_max": float(ag[:, 0].max().item()),
        "analog_gap_max_future": float(ag.max().item()),
        "hidden_gap_at_merge_mean": float(hg[:, 0].mean().item()),
        "hidden_gap_max_future": float(hg.max().item()),
    }


@torch.no_grad()
def learned_output_pair_audit(model, tokens: torch.Tensor, q_a: torch.Tensor, q_b: torch.Tensor, a0: torch.Tensor):
    la, aa = model(tokens, q_a, a0)
    lb, ab = model(tokens, q_b, a0)
    qmis = la.argmax(-1) != lb.argmax(-1)
    agap = torch.abs(aa - ab)
    return {
        "q_output_mismatch_at_merge": float(qmis[:, 0].to(torch.float64).mean().item()),
        "q_output_mismatch_max_future_rate": float(qmis.to(torch.float64).mean(0).max().item()),
        "analog_output_gap_at_merge_mean": float(agap[:, 0].mean().item()),
        "analog_output_gap_max_future": float(agap.max().item()),
    }


@torch.no_grad()
def compiled_pair_audit(model, tokens: torch.Tensor, q_a: torch.Tensor, q_b: torch.Tensor, a0: torch.Tensor):
    la, aa = base.compiled_runtime(model, tokens, q_a, a0)
    lb, ab = base.compiled_runtime(model, tokens, q_b, a0)
    qmis = la.argmax(-1) != lb.argmax(-1)
    agap = torch.abs(aa - ab)
    return {
        "q_output_mismatch_at_merge": float(qmis[:, 0].to(torch.float64).mean().item()),
        "q_output_mismatch_max_future_rate": float(qmis.to(torch.float64).mean(0).max().item()),
        "analog_gap_at_merge_mean": float(agap[:, 0].mean().item()),
        "analog_gap_max_future": float(agap.max().item()),
    }


@dataclass
class Run:
    seed: int
    learned_merge_epsilon: float
    learned_tangent_history_gap: float
    learned_outputs: dict[str, float]
    fiber_reified: dict[str, float]
    compiled: dict[str, float]


def run_one(seed: int, args) -> Run:
    model = base.train_model(
        seed,
        args.steps,
        args.train_length,
        args.batch_size,
        args.merge_probability,
        args.scale_probability,
    )
    tokens, qa, qb, a0 = paired_tokens(args.audit_batch_size, args.continuation, args.scale_probability)
    # In canonical coordinates, equal-a histories q=0 and q=1 differ after M
    # in the analog coordinate by r0-r1. This is a legal along-fiber memory channel.
    tangent_gap = abs(float((model.r0 - model.r1).detach().item()))
    return Run(
        seed=seed,
        learned_merge_epsilon=float(torch.sigmoid(model.eps_logit).detach().item()),
        learned_tangent_history_gap=tangent_gap,
        learned_outputs=learned_output_pair_audit(model, tokens, qa, qb, a0),
        fiber_reified=reified_pair_audit(model, tokens, qa, qb, a0),
        compiled=compiled_pair_audit(model, tokens, qa, qb, a0),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--merge-probability", type=float, default=0.05)
    p.add_argument("--scale-probability", type=float, default=0.08)
    p.add_argument("--audit-batch-size", type=int, default=256)
    p.add_argument("--continuation", type=int, default=128)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    rows = [run_one(seed, args) for seed in args.seeds]
    payload = {"config": vars(args), "results": [asdict(r) for r in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("seed tangent-gap reified-q@M reified-a@M reified-a-max compiled-a-max")
    for r in rows:
        print(
            f"{r.seed:3d} {r.learned_tangent_history_gap:.3e} "
            f"{r.fiber_reified['q_mismatch_at_merge']:.3f} "
            f"{r.fiber_reified['analog_gap_at_merge_mean']:.3e} "
            f"{r.fiber_reified['analog_gap_max_future']:.3e} "
            f"{r.compiled['analog_gap_max_future']:.3e}"
        )


if __name__ == "__main__":
    main()
