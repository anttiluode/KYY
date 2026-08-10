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
PHASE_MODULE = "phase_readout_for_port_calibration"
SPEC = importlib.util.spec_from_file_location(
    PHASE_MODULE, ROOT / "map" / "phase_readout_legalization_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
phase = importlib.util.module_from_spec(SPEC)
sys.modules[PHASE_MODULE] = phase
SPEC.loader.exec_module(phase)
base = phase.base


def mode_phase_transform(z: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Apply one readout-only phase offset per complex harmonic mode."""
    n, d = z.shape
    if d % 2:
        raise ValueError("real harmonic state dimension must be even")
    k = d // 2
    if phi.numel() != k:
        raise ValueError(f"expected {k} mode phases, got {phi.numel()}")
    q = z.reshape(n, k, 2)
    c = torch.cos(phi).reshape(1, k)
    s = torch.sin(phi).reshape(1, k)
    x, y = q[..., 0], q[..., 1]
    out = torch.stack((c * x - s * y, s * x + c * y), dim=-1)
    return out.reshape(n, d)


def low_rank_hidden_transform(z: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Readout-side residual adapter z -> z (I + U V^T)."""
    if u.shape != v.shape:
        raise ValueError("u and v must have the same [dimension, rank] shape")
    return z + (z @ v) @ u.T


def full_hidden_transform(z: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Readout-side residual adapter z -> z (I + Delta)."""
    if delta.shape != (z.shape[1], z.shape[1]):
        raise ValueError("delta must be square with the hidden dimension")
    return z + z @ delta


def subset_score(
    z: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    states: torch.Tensor,
) -> tuple[float, float]:
    logits = z[states] @ weight.T + bias
    labels = states
    pred = logits.argmax(dim=-1)
    rows = torch.arange(states.numel())
    true = logits[rows, labels]
    competitor = logits.clone()
    competitor[rows, labels] = -torch.inf
    margin = true - competitor.max(dim=-1).values
    return float((pred == labels).double().mean().item()), float(margin.min().item())


def full_score(
    z: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[float, float, int]:
    return phase.orbit_readout_metrics(z, weight, bias)


def search_tau_on_subset(
    n: int,
    projected_angles: np.ndarray,
    weight: torch.Tensor,
    bias: torch.Tensor,
    states: torch.Tensor,
    *,
    half_span: float,
    grid: int,
) -> float:
    best_tau = 0.0
    z0 = phase.phase_shifted_prototypes(n, projected_angles, 0.0)
    best_acc, best_margin = subset_score(z0, weight, bias, states)
    for tau in np.linspace(-half_span, half_span, grid):
        z = phase.phase_shifted_prototypes(n, projected_angles, float(tau))
        acc, margin = subset_score(z, weight, bias, states)
        if (acc > best_acc + 1e-15) or (
            abs(acc - best_acc) <= 1e-15 and margin > best_margin
        ):
            best_tau = float(tau)
            best_acc = acc
            best_margin = margin
    return best_tau


def _ce(z: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    logits = z[states] @ weight.T + bias
    return nn.functional.cross_entropy(logits, states)


def fit_mode_phase(
    z: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    states: torch.Tensor,
    *,
    steps: int,
    lr: float,
    l2: float,
) -> torch.Tensor:
    k = z.shape[1] // 2
    phi = torch.zeros(k, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([phi], lr=lr)
    for _ in range(steps):
        out = mode_phase_transform(z, phi)
        loss = _ce(out, weight, bias, states) + l2 * phi.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return phi.detach()


def fit_low_rank_hidden(
    z: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    states: torch.Tensor,
    *,
    rank: int,
    steps: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = z.shape[1]
    g = torch.Generator().manual_seed(seed)
    u = (1e-3 * torch.randn(d, rank, generator=g, dtype=torch.float64)).requires_grad_()
    v = (1e-3 * torch.randn(d, rank, generator=g, dtype=torch.float64)).requires_grad_()
    opt = torch.optim.Adam([u, v], lr=lr)
    for _ in range(steps):
        out = low_rank_hidden_transform(z, u, v)
        delta = u @ v.T
        loss = _ce(out, weight, bias, states) + l2 * delta.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return u.detach(), v.detach()


def fit_full_hidden(
    z: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    states: torch.Tensor,
    *,
    steps: int,
    lr: float,
    l2: float,
) -> torch.Tensor:
    d = z.shape[1]
    delta = torch.zeros(d, d, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    for _ in range(steps):
        out = full_hidden_transform(z, delta)
        loss = _ce(out, weight, bias, states) + l2 * delta.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return delta.detach()


def fit_full_readout(
    z: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    states: torch.Tensor,
    *,
    steps: int,
    lr: float,
    l2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dw = torch.zeros_like(weight, dtype=torch.float64, requires_grad=True)
    db = torch.zeros_like(bias, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([dw, db], lr=lr)
    for _ in range(steps):
        w = weight + dw
        b = bias + db
        loss = _ce(z, w, b, states) + l2 * (dw.square().mean() + db.square().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return dw.detach(), db.detach()


@dataclass
class RepairRow:
    seed: int
    calibration_size: int
    repeat: int
    method: str
    parameter_count: int
    full_accuracy: float
    full_min_margin: float
    wrong_states: int
    systematic_angle_error_accuracy: dict[str, float]
    timing_tau: float | None
    update_norm: float


@dataclass
class SeedBaseline:
    seed: int
    state_relation_defect: float
    projected_frequencies: list[int]
    inherited_accuracy: float
    inherited_min_margin: float
    inherited_wrong_states: int
    oracle_tau: float
    oracle_tau_accuracy: float
    oracle_tau_min_margin: float


def make_calibration_states(n: int, size: int, seed: int) -> torch.Tensor:
    if not 1 <= size <= n:
        raise ValueError("calibration size must lie in [1,n]")
    if size == n:
        return torch.arange(n, dtype=torch.long)
    rng = np.random.default_rng(seed)
    states = np.sort(rng.choice(n, size=size, replace=False))
    return torch.tensor(states, dtype=torch.long)


def train_seed(
    *,
    n: int,
    modes: int,
    seed: int,
    train_length: int,
    steps: int,
    batch_size: int,
    max_increment: int,
    lr: float,
    random_start: bool,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, float, list[int]]:
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

    learned = model.angles.detach().cpu().numpy().astype(np.float64)
    _, state_rel = base.relation_defects(n, learned)
    projected, frequencies = base.project_angles_to_characters(n, learned)
    weight = model.readout.weight.detach().cpu().to(torch.float64)
    bias = model.readout.bias.detach().cpu().to(torch.float64)
    return projected, weight, bias, float(state_rel), [int(x) for x in frequencies.tolist()]


def evaluate_method(
    *,
    method: str,
    n: int,
    projected: np.ndarray,
    weight: torch.Tensor,
    bias: torch.Tensor,
    states: torch.Tensor,
    repair_steps: int,
    repair_lr: float,
    l2: float,
    half_span: float,
    tau_grid: int,
    angle_errors: list[float],
    fit_seed: int,
) -> tuple[float, float, int, dict[str, float], float | None, float, int]:
    d = weight.shape[1]
    k = d // 2
    z = phase.phase_shifted_prototypes(n, projected, 0.0)
    tau: float | None = None
    update_norm = 0.0

    def score_for_error(eta: float) -> tuple[float, float, int]:
        noisy_angles = np.asarray(projected, dtype=np.float64) + float(eta)
        if method == "tau":
            zn = phase.phase_shifted_prototypes(n, noisy_angles, float(tau))
            return full_score(zn, weight, bias)
        if method == "mode_phase":
            zn = phase.phase_shifted_prototypes(n, noisy_angles, 0.0)
            return full_score(mode_phase_transform(zn, phi), weight, bias)
        if method == "rank1_hidden":
            zn = phase.phase_shifted_prototypes(n, noisy_angles, 0.0)
            return full_score(low_rank_hidden_transform(zn, u, v), weight, bias)
        if method == "tau_rank1_hidden":
            zn = phase.phase_shifted_prototypes(n, noisy_angles, float(tau))
            return full_score(low_rank_hidden_transform(zn, u, v), weight, bias)
        if method == "full_hidden":
            zn = phase.phase_shifted_prototypes(n, noisy_angles, 0.0)
            return full_score(full_hidden_transform(zn, delta), weight, bias)
        if method == "full_readout":
            zn = phase.phase_shifted_prototypes(n, noisy_angles, 0.0)
            return full_score(zn, weight + dw, bias + db)
        raise ValueError(method)

    if method == "tau":
        tau = search_tau_on_subset(
            n, projected, weight, bias, states, half_span=half_span, grid=tau_grid
        )
        parameter_count = 1
    elif method == "mode_phase":
        phi = fit_mode_phase(
            z, weight, bias, states, steps=repair_steps, lr=repair_lr, l2=l2
        )
        update_norm = float(torch.linalg.vector_norm(phi).item())
        parameter_count = k
    elif method == "rank1_hidden":
        u, v = fit_low_rank_hidden(
            z,
            weight,
            bias,
            states,
            rank=1,
            steps=repair_steps,
            lr=repair_lr,
            l2=l2,
            seed=fit_seed,
        )
        update_norm = float(torch.linalg.matrix_norm(u @ v.T).item())
        parameter_count = 2 * d
    elif method == "tau_rank1_hidden":
        tau = search_tau_on_subset(
            n, projected, weight, bias, states, half_span=half_span, grid=tau_grid
        )
        ztau = phase.phase_shifted_prototypes(n, projected, tau)
        u, v = fit_low_rank_hidden(
            ztau,
            weight,
            bias,
            states,
            rank=1,
            steps=repair_steps,
            lr=repair_lr,
            l2=l2,
            seed=fit_seed,
        )
        update_norm = float(torch.linalg.matrix_norm(u @ v.T).item())
        parameter_count = 1 + 2 * d
    elif method == "full_hidden":
        delta = fit_full_hidden(
            z, weight, bias, states, steps=repair_steps, lr=repair_lr, l2=l2
        )
        update_norm = float(torch.linalg.matrix_norm(delta).item())
        parameter_count = d * d
    elif method == "full_readout":
        dw, db = fit_full_readout(
            z, weight, bias, states, steps=repair_steps, lr=repair_lr, l2=l2
        )
        update_norm = float(
            torch.sqrt(torch.linalg.vector_norm(dw).square() + torch.linalg.vector_norm(db).square()).item()
        )
        parameter_count = weight.numel() + bias.numel()
    else:
        raise ValueError(f"unknown method {method}")

    acc, margin, wrong = score_for_error(0.0)
    robustness = {
        f"{eta:.6g}": float(score_for_error(float(eta))[0]) for eta in angle_errors
    }
    return acc, margin, wrong, robustness, tau, update_norm, parameter_count


def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure calibration and parameter cost of repairing frozen ports after cyclic legalization"
    )
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 5, 6, 7, 9])
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-increment", type=int, default=4)
    p.add_argument("--train-lr", type=float, default=3e-3)
    p.add_argument("--random-start", action="store_true")
    p.add_argument("--calibration-sizes", nargs="+", type=int, default=[4, 8, 16, 32, 64, 101])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["tau", "mode_phase", "rank1_hidden", "tau_rank1_hidden", "full_hidden", "full_readout"],
    )
    p.add_argument("--repair-steps", type=int, default=500)
    p.add_argument("--repair-lr", type=float, default=5e-2)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--half-span", type=float, default=0.5)
    p.add_argument("--tau-grid", type=int, default=1001)
    p.add_argument("--angle-errors", nargs="+", type=float, default=[1e-4, 5e-4, 1e-3])
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    baselines: list[SeedBaseline] = []
    rows: list[RepairRow] = []

    for seed in args.seeds:
        projected, weight, bias, state_rel, frequencies = train_seed(
            n=args.n,
            modes=args.modes,
            seed=seed,
            train_length=args.train_length,
            steps=args.train_steps,
            batch_size=args.batch_size,
            max_increment=args.max_increment,
            lr=args.train_lr,
            random_start=args.random_start,
        )
        z = phase.phase_shifted_prototypes(args.n, projected, 0.0)
        inherited_acc, inherited_margin, inherited_wrong = full_score(z, weight, bias)
        oracle = phase.search_phase_offset(
            args.n,
            projected,
            weight,
            bias,
            half_span=args.half_span,
            grid=max(args.tau_grid, 1001),
        )
        baselines.append(
            SeedBaseline(
                seed=seed,
                state_relation_defect=state_rel,
                projected_frequencies=frequencies,
                inherited_accuracy=inherited_acc,
                inherited_min_margin=inherited_margin,
                inherited_wrong_states=inherited_wrong,
                oracle_tau=float(oracle.best_tau),
                oracle_tau_accuracy=float(oracle.best_accuracy),
                oracle_tau_min_margin=float(oracle.best_min_margin),
            )
        )

        for size in args.calibration_sizes:
            for repeat in range(args.repeats):
                subset_seed = seed * 1_000_000 + size * 100 + repeat
                states = make_calibration_states(args.n, size, subset_seed)
                for mi, method in enumerate(args.methods):
                    acc, margin, wrong, robustness, tau, norm, count = evaluate_method(
                        method=method,
                        n=args.n,
                        projected=projected,
                        weight=weight,
                        bias=bias,
                        states=states,
                        repair_steps=args.repair_steps,
                        repair_lr=args.repair_lr,
                        l2=args.l2,
                        half_span=args.half_span,
                        tau_grid=args.tau_grid,
                        angle_errors=args.angle_errors,
                        fit_seed=subset_seed + 7919 * (mi + 1),
                    )
                    rows.append(
                        RepairRow(
                            seed=seed,
                            calibration_size=size,
                            repeat=repeat,
                            method=method,
                            parameter_count=count,
                            full_accuracy=float(acc),
                            full_min_margin=float(margin),
                            wrong_states=int(wrong),
                            systematic_angle_error_accuracy=robustness,
                            timing_tau=None if tau is None else float(tau),
                            update_norm=float(norm),
                        )
                    )

    payload = {
        "config": vars(args),
        "baselines": [asdict(x) for x in baselines],
        "repairs": [asdict(x) for x in rows],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("baselines")
    print("seed inherited oracle-tau oracle-acc inherited-margin oracle-margin")
    for x in baselines:
        print(
            f"{x.seed:4d} {x.inherited_accuracy:9.3f} {x.oracle_tau:+10.4f} "
            f"{x.oracle_tau_accuracy:10.3f} {x.inherited_min_margin:+16.4f} "
            f"{x.oracle_tau_min_margin:+13.4f}"
        )
    print("\nrepairs")
    print("seed calib rep method params full-acc wrong min-margin eta1e-3")
    for x in rows:
        print(
            f"{x.seed:4d} {x.calibration_size:5d} {x.repeat:3d} {x.method:18s} "
            f"{x.parameter_count:6d} {x.full_accuracy:8.3f} {x.wrong_states:5d} "
            f"{x.full_min_margin:+10.4f} {x.systematic_angle_error_accuracy.get('0.001', float('nan')):9.3f}"
        )


if __name__ == "__main__":
    main()
