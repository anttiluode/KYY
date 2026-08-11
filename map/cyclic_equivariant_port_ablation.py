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
MODULE_NAME = "cyclic_equivariant_port_base_for_ablation"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "cyclic_equivariant_port_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
base = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = base
SPEC.loader.exec_module(base)


@dataclass
class AblationRun:
    seed: int
    raw_inherited_accuracy: float
    raw_inherited_min_margin: float
    raw_equivariant_accuracy: float
    raw_equivariant_min_margin: float
    midpoint_inherited_accuracy: float
    midpoint_inherited_min_margin: float
    midpoint_equivariant_accuracy: float
    midpoint_equivariant_min_margin: float
    projected_decoder_relative_error: float
    projected_frequencies: list[int]


def run_one(
    *,
    n: int,
    modes: int,
    seed: int,
    train_length: int,
    train_steps: int,
    batch_size: int,
    max_increment: int,
    lr: float,
    random_start: bool,
) -> AblationRun:
    model = base.train_learned_model(
        n=n,
        modes=modes,
        seed=seed,
        train_length=train_length,
        train_steps=train_steps,
        batch_size=batch_size,
        max_increment=max_increment,
        lr=lr,
        random_start=random_start,
    )
    learned = model.angles.detach().cpu().numpy().astype(np.float64)
    projected, frequencies = base.base.project_angles_to_characters(n, learned)
    W = model.readout.weight.detach().cpu().to(torch.float64)
    b = model.readout.bias.detach().cpu().to(torch.float64)

    z_raw = base.exact_orbit(n, projected, model.h0.detach().cpu())
    q = base.midpoint_port(n, learned, projected)
    z_mid = z_raw @ q
    W_eq, b_eq, _ = base.project_cyclic_equivariant_decoder(n, projected, W, b)

    raw_inh = base.readout_metrics(z_raw, W, b)
    raw_eq = base.readout_metrics(z_raw, W_eq, b_eq)
    mid_inh = base.readout_metrics(z_mid, W, b)
    mid_eq = base.readout_metrics(z_mid, W_eq, b_eq)
    denom = max(float(torch.linalg.matrix_norm(W).item()), 1e-12)
    proj_err = float(torch.linalg.matrix_norm(W_eq - W).item()) / denom

    return AblationRun(
        seed=int(seed),
        raw_inherited_accuracy=float(raw_inh[0]),
        raw_inherited_min_margin=float(raw_inh[1]),
        raw_equivariant_accuracy=float(raw_eq[0]),
        raw_equivariant_min_margin=float(raw_eq[1]),
        midpoint_inherited_accuracy=float(mid_inh[0]),
        midpoint_inherited_min_margin=float(mid_inh[1]),
        midpoint_equivariant_accuracy=float(mid_eq[0]),
        midpoint_equivariant_min_margin=float(mid_eq[1]),
        projected_decoder_relative_error=proj_err,
        projected_frequencies=[int(x) for x in frequencies.tolist()],
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Ablate midpoint recentering against full cyclic decoder equivariance projection")
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=2200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-increment", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--random-start", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    rows = [
        run_one(
            n=args.n,
            modes=args.modes,
            seed=seed,
            train_length=args.train_length,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            max_increment=args.max_increment,
            lr=args.lr,
            random_start=args.random_start,
        )
        for seed in args.seeds
    ]
    payload = {"config": vars(args), "results": [asdict(x) for x in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("seed raw_inh raw_eq mid_inh mid_eq margins(raweq/mideq)")
        for x in rows:
            print(
                f"{x.seed:4d} {x.raw_inherited_accuracy:7.3f} {x.raw_equivariant_accuracy:6.3f} "
                f"{x.midpoint_inherited_accuracy:7.3f} {x.midpoint_equivariant_accuracy:6.3f} "
                f"{x.raw_equivariant_min_margin:+.3f}/{x.midpoint_equivariant_min_margin:+.3f}"
            )


if __name__ == "__main__":
    main()
