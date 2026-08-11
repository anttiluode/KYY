from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "mixed_fiber_compiler_for_noise_boundary"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "mixed_fiber_compiler_probe.py")
assert SPEC is not None and SPEC.loader is not None
base = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = base
SPEC.loader.exec_module(base)


def exact_targets(tok: torch.Tensor, q0: torch.Tensor, a0: torch.Tensor):
    q = q0.clone()
    a = a0.clone().to(torch.float64)
    qs, aa = [], []
    for t in range(tok.shape[1]):
        x = tok[:, t]
        q = torch.where(x == base.MERGE, base.merge_q(q), torch.where(x == base.SCALE, q, (q + x) % 4))
        a = torch.where(x == base.SCALE, 0.9 * a, a)
        qs.append(q.clone())
        aa.append(a.clone())
    return torch.stack(qs, 1), torch.stack(aa, 1)


def generate_eval(batch: int, length: int, merge_probability: float, scale_probability: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    tok = torch.randint(0, 4, (batch, length), generator=g)
    r = torch.rand((batch, length), generator=g)
    tok[r < merge_probability] = base.MERGE
    tok[(r >= merge_probability) & (r < merge_probability + scale_probability)] = base.SCALE
    q0 = torch.randint(0, 4, (batch,), generator=g)
    a0 = torch.empty(batch).uniform_(-1.5, 1.5, generator=g)
    qy, ay = exact_targets(tok, q0, a0)
    return tok, q0, a0, qy, ay


def compiled_noisy_runtime(model, tok, q0, a0, sigma: float, *, reify: bool, seed: int):
    cycle, merge, scale = base.synthesize_exact_operators(model)
    powers = torch.stack([torch.linalg.matrix_power(cycle, k) for k in range(4)])
    h = model.init_hidden(q0, a0).to(torch.float64)
    wq, bq, wa = base.exact_ports(model)
    g = torch.Generator().manual_seed(seed)
    logits, analog = [], []
    for t in range(tok.shape[1]):
        x = tok[:, t]
        inc = torch.where(x < 4, x, torch.zeros_like(x))
        rotated = torch.einsum("bij,bj->bi", powers[inc], h)
        merged = h @ merge.T
        scaled = h @ scale.T
        h = torch.where((x == base.MERGE)[:, None], merged, torch.where((x == base.SCALE)[:, None], scaled, rotated))
        if sigma > 0:
            h = h + sigma * torch.randn(h.shape, dtype=torch.float64, generator=g)
        if reify:
            h32 = h.to(torch.float32)
            h32, _, _ = base.nearest_fiber(model, h32)
            h = h32.to(torch.float64)
        logits.append(h @ wq.T + bq)
        analog.append(h @ wa)
    return torch.stack(logits, 1), torch.stack(analog, 1)


def explicit_hybrid_runtime(tok, q0, a0, sigma: float, seed: int):
    """Reference Q x R implementation: q is exact digital state; only a is noisy."""
    q = q0.clone()
    a = a0.clone().to(torch.float64)
    g = torch.Generator().manual_seed(seed)
    qs, aa = [], []
    for t in range(tok.shape[1]):
        x = tok[:, t]
        q = torch.where(x == base.MERGE, base.merge_q(q), torch.where(x == base.SCALE, q, (q + x) % 4))
        a = torch.where(x == base.SCALE, 0.9 * a, a)
        if sigma > 0:
            a = a + sigma * torch.randn(a.shape, dtype=torch.float64, generator=g)
        qs.append(q.clone())
        aa.append(a.clone())
    return torch.stack(qs, 1), torch.stack(aa, 1)


def metrics(qp, ap, qy, ay):
    return {
        "q_accuracy": float((qp == qy).to(torch.float64).mean().item()),
        "q_final_accuracy": float((qp[:, -1] == qy[:, -1]).to(torch.float64).mean().item()),
        "analog_rmse": float(torch.sqrt(torch.mean((ap - ay) ** 2)).item()),
        "analog_final_rmse": float(torch.sqrt(torch.mean((ap[:, -1] - ay[:, -1]) ** 2)).item()),
    }


@dataclass
class NoiseRow:
    sigma: float
    compiled_continuous: dict[str, float]
    reified_continuous: dict[str, float]
    explicit_hybrid: dict[str, float]


def run(args):
    model = base.SoftFiberTracker(0)
    tok, q0, a0, qy, ay = generate_eval(
        args.batch_size, args.length, args.merge_probability, args.scale_probability, args.seed
    )
    rows = []
    for i, sigma in enumerate(args.sigmas):
        lc, ac = compiled_noisy_runtime(model, tok, q0, a0, sigma, reify=False, seed=args.seed + 100 + i)
        lr, ar = compiled_noisy_runtime(model, tok, q0, a0, sigma, reify=True, seed=args.seed + 100 + i)
        qh, ah = explicit_hybrid_runtime(tok, q0, a0, sigma, seed=args.seed + 200 + i)
        rows.append(
            NoiseRow(
                sigma=float(sigma),
                compiled_continuous=metrics(lc.argmax(-1), ac, qy, ay),
                reified_continuous=metrics(lr.argmax(-1), ar, qy, ay),
                explicit_hybrid=metrics(qh, ah, qy, ay),
            )
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--merge-probability", type=float, default=0.05)
    p.add_argument("--scale-probability", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    rows = run(args)
    if args.json:
        print(json.dumps({"config": vars(args), "results": [asdict(r) for r in rows]}, indent=2, sort_keys=True))
    else:
        print("sigma cont_q reify_q hybrid_q cont_a reify_a hybrid_a")
        for r in rows:
            print(f"{r.sigma:.1e} {r.compiled_continuous['q_accuracy']:.6f} "
                  f"{r.reified_continuous['q_accuracy']:.6f} {r.explicit_hybrid['q_accuracy']:.6f} "
                  f"{r.compiled_continuous['analog_rmse']:.3e} {r.reified_continuous['analog_rmse']:.3e} "
                  f"{r.explicit_hybrid['analog_rmse']:.3e}")


if __name__ == "__main__":
    main()
