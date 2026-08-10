from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "phase_readout_for_midpoint_compensation"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "phase_readout_legalization_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
phase = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = phase
SPEC.loader.exec_module(phase)
base = phase.base


def wrap_phase(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def midpoint_phase_compensation(
    n: int,
    learned_angles: np.ndarray,
    projected_angles: np.ndarray,
) -> np.ndarray:
    """Zero-label phase adapter that centers snap-induced drift over states 0..n-1.

    The pre/post phase mismatch for mode i at state s is

        s * (theta*_i - theta_i).

    Adding a readout-only phase phi_i makes it

        (s-c) * (theta*_i - theta_i)

    for phi_i = -c*(theta*_i-theta_i).  The midpoint c=(n-1)/2
    minimizes the maximum absolute linear mismatch over the finite interval.
    """
    learned = np.asarray(learned_angles, dtype=np.float64)
    projected = np.asarray(projected_angles, dtype=np.float64)
    if learned.shape != projected.shape:
        raise ValueError("learned/projected angle shapes must match")
    center = 0.5 * (n - 1)
    return wrap_phase(-center * (projected - learned))


def apply_mode_phase(z: torch.Tensor, phi: np.ndarray | torch.Tensor) -> torch.Tensor:
    q = z.reshape(z.shape[0], -1, 2)
    p = torch.as_tensor(phi, dtype=z.dtype, device=z.device).reshape(1, -1)
    c, s = torch.cos(p), torch.sin(p)
    x, y = q[..., 0], q[..., 1]
    return torch.stack((c * x - s * y, s * x + c * y), dim=-1).reshape_as(z)


def phase_mismatch_stats(
    n: int,
    learned_angles: np.ndarray,
    projected_angles: np.ndarray,
    phi: np.ndarray,
) -> tuple[float, float]:
    states = np.arange(n, dtype=np.float64)[:, None]
    delta = projected_angles[None, :] - learned_angles[None, :]
    mismatch = wrap_phase(states * delta + phi[None, :])
    abs_mismatch = np.abs(mismatch)
    return float(abs_mismatch.max()), float(abs_mismatch.mean())


@dataclass
class MidpointRun:
    seed: int
    state_relation_defect: float
    projected_frequencies: list[int]
    pre_orbit_accuracy: float
    projected_accuracy: float
    projected_min_margin: float
    compensated_accuracy: float
    compensated_min_margin: float
    compensated_eta_1e3_accuracy: float
    max_phase_mismatch_before: float
    max_phase_mismatch_after: float
    mean_phase_mismatch_before: float
    mean_phase_mismatch_after: float
    phase_adapter_norm: float


def train_and_probe(
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
) -> MidpointRun:
    base.seed_everything(seed)
    initial, _, _ = base.make_angles("learned", n, modes, trials=1, seed=seed)
    model = base.RotaryModTracker(n, initial, learn_angles=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(train_steps):
        x, y = base.generate_batch(
            n,
            batch_size,
            train_length,
            max_increment,
            random_start=random_start,
        )
        logits = model(x)
        loss = criterion(logits.reshape(-1, n), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    learned = model.angles.detach().cpu().numpy().astype(np.float64)
    _, state_rel = base.relation_defects(n, learned)
    projected, frequencies = base.project_angles_to_characters(n, learned)
    weight = model.readout.weight.detach().cpu().to(torch.float64)
    bias = model.readout.bias.detach().cpu().to(torch.float64)

    z_pre = phase.phase_shifted_prototypes(n, learned, 0.0)
    pre_acc, _, _ = phase.orbit_readout_metrics(z_pre, weight, bias)

    z_proj = phase.phase_shifted_prototypes(n, projected, 0.0)
    proj_acc, proj_margin, _ = phase.orbit_readout_metrics(z_proj, weight, bias)

    phi = midpoint_phase_compensation(n, learned, projected)
    z_comp = apply_mode_phase(z_proj, phi)
    comp_acc, comp_margin, _ = phase.orbit_readout_metrics(z_comp, weight, bias)

    noisy = phase.phase_shifted_prototypes(n, projected + 1e-3, 0.0)
    noisy_comp = apply_mode_phase(noisy, phi)
    noisy_acc, _, _ = phase.orbit_readout_metrics(noisy_comp, weight, bias)

    zero = np.zeros_like(phi)
    max0, mean0 = phase_mismatch_stats(n, learned, projected, zero)
    max1, mean1 = phase_mismatch_stats(n, learned, projected, phi)

    return MidpointRun(
        seed=seed,
        state_relation_defect=float(state_rel),
        projected_frequencies=[int(x) for x in frequencies.tolist()],
        pre_orbit_accuracy=float(pre_acc),
        projected_accuracy=float(proj_acc),
        projected_min_margin=float(proj_margin),
        compensated_accuracy=float(comp_acc),
        compensated_min_margin=float(comp_margin),
        compensated_eta_1e3_accuracy=float(noisy_acc),
        max_phase_mismatch_before=max0,
        max_phase_mismatch_after=max1,
        mean_phase_mismatch_before=mean0,
        mean_phase_mismatch_after=mean1,
        phase_adapter_norm=float(np.linalg.norm(phi)),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Zero-label midpoint phase compensation after cyclic operator snapping")
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4,5,6,7,8,9])
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-increment", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--random-start", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = [
        train_and_probe(
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
        return

    print("seed pre projected compensated eta1e-3 max-phase-before max-phase-after")
    for x in rows:
        print(
            f"{x.seed:4d} {x.pre_orbit_accuracy:7.3f} {x.projected_accuracy:9.3f} "
            f"{x.compensated_accuracy:11.3f} {x.compensated_eta_1e3_accuracy:9.3f} "
            f"{x.max_phase_mismatch_before:16.4f} {x.max_phase_mismatch_after:15.4f}"
        )


if __name__ == "__main__":
    main()
