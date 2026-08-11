from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "mixed_fiber_noise_for_relock"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "mixed_fiber_noise_boundary.py")
assert SPEC is not None and SPEC.loader is not None
noise = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = noise
SPEC.loader.exec_module(noise)
base = noise.base


def periodic_relock_runtime(model, tok, q0, a0, sigma: float, interval: int | None, seed: int):
    cycle, merge, scale = base.synthesize_exact_operators(model)
    powers = torch.stack([torch.linalg.matrix_power(cycle, k) for k in range(4)])
    h = model.init_hidden(q0, a0).to(torch.float64)
    wq, bq, wa = base.exact_ports(model)
    g = torch.Generator().manual_seed(seed)
    logits, analog = [], []
    relocks = 0
    for t in range(tok.shape[1]):
        x = tok[:, t]
        inc = torch.where(x < 4, x, torch.zeros_like(x))
        rotated = torch.einsum("bij,bj->bi", powers[inc], h)
        merged = h @ merge.T
        scaled = h @ scale.T
        h = torch.where((x == base.MERGE)[:, None], merged, torch.where((x == base.SCALE)[:, None], scaled, rotated))
        if sigma > 0:
            h = h + sigma * torch.randn(h.shape, dtype=torch.float64, generator=g)
        if interval is not None and interval > 0 and (t + 1) % interval == 0:
            h32, _, _ = base.nearest_fiber(model, h.to(torch.float32))
            h = h32.to(torch.float64)
            relocks += 1
        logits.append(h @ wq.T + bq)
        analog.append(h @ wa)
    return torch.stack(logits, 1), torch.stack(analog, 1), relocks


@dataclass
class RelockRow:
    interval: int | None
    relocks_per_sequence: int
    q_accuracy: float
    q_final_accuracy: float
    analog_rmse: float
    analog_final_rmse: float


def run(args):
    model = base.SoftFiberTracker(0)
    tok, q0, a0, qy, ay = noise.generate_eval(
        args.batch_size, args.length, args.merge_probability, args.scale_probability, args.seed
    )
    rows = []
    intervals = [None if x <= 0 else x for x in args.intervals]
    for i, interval in enumerate(intervals):
        logits, analog, relocks = periodic_relock_runtime(
            model, tok, q0, a0, args.sigma, interval, args.seed + 100 + i
        )
        m = noise.metrics(logits.argmax(-1), analog, qy, ay)
        rows.append(
            RelockRow(
                interval=interval,
                relocks_per_sequence=relocks,
                q_accuracy=m["q_accuracy"],
                q_final_accuracy=m["q_final_accuracy"],
                analog_rmse=m["analog_rmse"],
                analog_final_rmse=m["analog_final_rmse"],
            )
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=0.03)
    p.add_argument("--intervals", nargs="+", type=int, default=[1, 4, 16, 64, 256, 1024, 0])
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--merge-probability", type=float, default=0.05)
    p.add_argument("--scale-probability", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=2345)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    rows = run(args)
    if args.json:
        print(json.dumps({"config": vars(args), "results": [asdict(r) for r in rows]}, indent=2, sort_keys=True))
    else:
        print("interval relocks q_acc q_final analog_rmse")
        for r in rows:
            name = "never" if r.interval is None else str(r.interval)
            print(f"{name:>8} {r.relocks_per_sequence:7d} {r.q_accuracy:.6f} {r.q_final_accuracy:.6f} {r.analog_rmse:.3e}")


if __name__ == "__main__":
    main()
