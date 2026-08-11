from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "mixed_fiber_relock_for_frontier"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, ROOT / "map" / "mixed_fiber_relock_boundary.py")
assert SPEC is not None and SPEC.loader is not None
relock = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = relock
SPEC.loader.exec_module(relock)
base = relock.base
noise = relock.noise


@dataclass
class FrontierRow:
    sigma: float
    interval: int
    q_accuracy: float
    q_final_accuracy: float
    analog_rmse: float


def run(args):
    model = base.SoftFiberTracker(0)
    tok, q0, a0, qy, ay = noise.generate_eval(
        args.batch_size, args.length, args.merge_probability, args.scale_probability, args.seed
    )
    rows: list[FrontierRow] = []
    physical_noise_seed = args.seed + 100
    for sigma in args.sigmas:
        for interval in args.intervals:
            logits, analog, _ = relock.periodic_relock_runtime(
                model, tok, q0, a0, float(sigma), int(interval), physical_noise_seed
            )
            m = noise.metrics(logits.argmax(-1), analog, qy, ay)
            rows.append(
                FrontierRow(
                    sigma=float(sigma),
                    interval=int(interval),
                    q_accuracy=m["q_accuracy"],
                    q_final_accuracy=m["q_final_accuracy"],
                    analog_rmse=m["analog_rmse"],
                )
            )
    return rows


def summarize(rows: list[FrontierRow], overall_threshold: float, final_threshold: float):
    out = []
    for sigma in sorted(set(r.sigma for r in rows)):
        group = [r for r in rows if r.sigma == sigma]
        safe = [r for r in group if r.q_accuracy >= overall_threshold and r.q_final_accuracy >= final_threshold]
        best = max(safe, key=lambda r: r.interval) if safe else None
        out.append(
            {
                "sigma": sigma,
                "largest_tested_safe_interval": None if best is None else best.interval,
                "safe_relocks_per_1024": None if best is None else 1024 // best.interval,
            }
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.01, 0.015, 0.02, 0.03, 0.04, 0.05])
    p.add_argument("--intervals", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--merge-probability", type=float, default=0.05)
    p.add_argument("--scale-probability", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=3456)
    p.add_argument("--overall-threshold", type=float, default=0.999)
    p.add_argument("--final-threshold", type=float, default=0.99)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    rows = run(args)
    payload = {
        "config": vars(args),
        "frontier": summarize(rows, args.overall_threshold, args.final_threshold),
        "results": [asdict(r) for r in rows],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for x in payload["frontier"]:
            print(x)


if __name__ == "__main__":
    main()
