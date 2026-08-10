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
MODULE_NAME = "harmonic_training_probe_for_phase_readout"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "harmonic_training_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
base = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = base
SPEC.loader.exec_module(base)


def phase_shifted_prototypes(
    n: int,
    projected_angles: np.ndarray | torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Exact C_n orbit sampled at a fractional generator-time offset tau.

    If a legal mode has generator angle theta_i, delaying/advancing the common
    observation time by tau generator units changes its phase by tau*theta_i.
    This is the harmonic/Fourier time-shift rule.  We deliberately use the
    nearest projected angle itself rather than frequencies modulo n, because
    fractional interpolation depends on the chosen physical generator branch.
    """
    a = torch.as_tensor(projected_angles, dtype=torch.float64).reshape(1, -1)
    s = torch.arange(n, dtype=torch.float64).reshape(-1, 1)
    phase = (s + float(tau)) * a
    scale = 1.0 / math.sqrt(a.shape[1])
    return torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1).reshape(n, -1) * scale


def orbit_readout_metrics(
    prototypes: torch.Tensor,
    readout_weight: np.ndarray | torch.Tensor,
    readout_bias: np.ndarray | torch.Tensor,
) -> tuple[float, float, int]:
    n = prototypes.shape[0]
    W = torch.as_tensor(readout_weight, dtype=torch.float64)
    b = torch.as_tensor(readout_bias, dtype=torch.float64).reshape(-1)
    logits = prototypes @ W.T + b
    labels = torch.arange(n)
    pred = logits.argmax(dim=-1)
    rows = torch.arange(n)
    true = logits[rows, labels]
    competitor = logits.clone()
    competitor[rows, labels] = -torch.inf
    margin = true - competitor.max(dim=-1).values
    correct = int((pred == labels).sum().item())
    return correct / n, float(margin.min().item()), n - correct


@dataclass(frozen=True)
class PhaseSearchResult:
    baseline_accuracy: float
    baseline_min_margin: float
    baseline_wrong_states: int
    best_tau: float
    best_accuracy: float
    best_min_margin: float
    best_wrong_states: int
    exhaustive_rescued: bool
    margin_improvement: float


def search_phase_offset(
    n: int,
    projected_angles: np.ndarray | torch.Tensor,
    readout_weight: np.ndarray | torch.Tensor,
    readout_bias: np.ndarray | torch.Tensor,
    *,
    half_span: float = 0.5,
    grid: int = 2001,
) -> PhaseSearchResult:
    """Search one scalar readout-time offset with operator and decoder frozen.

    Selection is lexicographic: maximize exhaustive legal-orbit accuracy, then
    maximize the minimum true-class margin.  This prevents a phase with a large
    margin on most states from hiding a newly broken symbolic state.
    """
    if grid < 3:
        raise ValueError("grid must be >= 3")
    if half_span <= 0:
        raise ValueError("half_span must be > 0")

    z0 = phase_shifted_prototypes(n, projected_angles, 0.0)
    acc0, margin0, wrong0 = orbit_readout_metrics(z0, readout_weight, readout_bias)

    best_tau = 0.0
    best_acc = acc0
    best_margin = margin0
    best_wrong = wrong0

    for tau in np.linspace(-half_span, half_span, grid):
        z = phase_shifted_prototypes(n, projected_angles, float(tau))
        acc, margin, wrong = orbit_readout_metrics(z, readout_weight, readout_bias)
        if (acc > best_acc + 1e-15) or (
            abs(acc - best_acc) <= 1e-15 and margin > best_margin
        ):
            best_tau = float(tau)
            best_acc = float(acc)
            best_margin = float(margin)
            best_wrong = int(wrong)

    return PhaseSearchResult(
        baseline_accuracy=float(acc0),
        baseline_min_margin=float(margin0),
        baseline_wrong_states=int(wrong0),
        best_tau=best_tau,
        best_accuracy=best_acc,
        best_min_margin=best_margin,
        best_wrong_states=best_wrong,
        exhaustive_rescued=bool(wrong0 > 0 and best_wrong == 0 and best_margin > 0.0),
        margin_improvement=float(best_margin - margin0),
    )


@dataclass
class PhaseRun:
    seed: int
    n: int
    modes: int
    train_length: int
    steps: int
    pre_l16_accuracy: float
    pre_l1024_accuracy: float
    state_relation_defect: float
    projected_frequencies: list[int]
    phase_search: dict[str, float | int | bool]


def train_and_phase_search(
    *,
    n: int,
    modes: int,
    seed: int,
    train_length: int,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    max_increment: int,
    lr: float,
    random_start: bool,
    half_span: float,
    grid: int,
) -> PhaseRun:
    base.seed_everything(seed)
    initial, _, _ = base.make_angles("learned", n, modes, trials=1, seed=seed)
    model = base.RotaryModTracker(n, initial, learn_angles=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(steps):
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

    clean, _ = base.evaluate(
        model,
        n,
        [train_length, 1024],
        eval_batch_size,
        max_increment,
        0.0,
        random_start=random_start,
    )

    learned = model.angles.detach().cpu().numpy().astype(np.float64)
    _, state_rel = base.relation_defects(n, learned)
    projected, frequencies = base.project_angles_to_characters(n, learned)

    phase = search_phase_offset(
        n,
        projected,
        model.readout.weight.detach().cpu(),
        model.readout.bias.detach().cpu(),
        half_span=half_span,
        grid=grid,
    )

    return PhaseRun(
        seed=seed,
        n=n,
        modes=modes,
        train_length=train_length,
        steps=steps,
        pre_l16_accuracy=float(clean[str(train_length)]),
        pre_l1024_accuracy=float(clean["1024"]),
        state_relation_defect=float(state_rel),
        projected_frequencies=[int(x) for x in frequencies.tolist()],
        phase_search=asdict(phase),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="One-parameter phase-timed readout test after exact cyclic legalization"
    )
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-increment", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--random-start", action="store_true")
    p.add_argument("--half-span", type=float, default=0.5)
    p.add_argument("--grid", type=int, default=2001)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = [
        train_and_phase_search(
            n=args.n,
            modes=args.modes,
            seed=seed,
            train_length=args.train_length,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            max_increment=args.max_increment,
            lr=args.lr,
            random_start=args.random_start,
            half_span=args.half_span,
            grid=args.grid,
        )
        for seed in args.seeds
    ]
    payload = {"config": vars(args), "results": [asdict(row) for row in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("seed rel-def base-correct best-correct best-tau base-margin best-margin rescued")
    for row in rows:
        q = row.phase_search
        print(
            f"{row.seed:4d} {row.state_relation_defect:7.4f} "
            f"{row.n-int(q['baseline_wrong_states']):12d} "
            f"{row.n-int(q['best_wrong_states']):12d} "
            f"{float(q['best_tau']):8.4f} "
            f"{float(q['baseline_min_margin']):11.5f} "
            f"{float(q['best_min_margin']):11.5f} "
            f"{str(q['exhaustive_rescued']):>7s}"
        )


if __name__ == "__main__":
    main()
