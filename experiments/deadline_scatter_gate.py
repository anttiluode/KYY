"""KYY deadline gate: same local scatter computation, different synchronization.

This experiment asks a deliberately narrow question motivated by PresentMoment:

    If the local 2-port operations of ``geom_scatter`` have heterogeneous physical
    durations, what is readable at a hard deadline when we either

      ASYNC  - let each local gate start as soon as its two endpoint dependencies
               are complete, or
      SYNC   - retain KYY's checkerboard phase barriers and expose a readout only
               after each complete phase?

The final computation is identical.  We do not add, remove, or reorder any pair of
operations that share a state coordinate.  Disjoint operations may finish in a
physical-time order different from the canonical checkerboard order, which is safe
because they commute.  The script asserts that the fully completed ASYNC state,
SYNC state, and ordinary KYY canonical state agree numerically.

Two training modes are intentionally contrasted:

``final``
    The ordinary KYY objective: supervise the shared readout only after a complete
    token transition.

``phase``
    Deep-supervise the *same shared readout* after every ordinary homogeneous
    checkerboard phase.  There are no extra exit heads, no deadline labels, and no
    asynchronous mixed-maturity states during training.

The interesting test in ``phase`` mode is therefore OOD: at evaluation the readout
is applied to mixtures in which different local coordinates have progressed through
unequal amounts of the transition.

This is NOT yet a geometry win.  Anytime classifiers, early-exit networks,
asynchronous/event-driven networks, delayed reservoirs, and adaptive-compute models
are established prior art.  This gate isolates one property of KYY's existing local
operator before stronger cross-architecture controls are added.

Example
-------

    python experiments/deadline_scatter_gate.py --quick

Fuller run:

    python experiments/deadline_scatter_gate.py \
        --training-modes final phase \
        --model-seeds 0 1 2 \
        --delay-seeds 0 1 2 \
        --task perm3 --state-dim 32 --train-length 32 --steps 300

"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from kyy import TASKS, build_model, generate_batch, parameter_count


@dataclass(frozen=True)
class LocalOp:
    """One declared physical 2-port scatter operation."""

    sweep: int
    phase: int
    edge: int
    src: int
    dst: int
    end_time: float


@dataclass(frozen=True)
class Schedule:
    async_ops: tuple[LocalOp, ...]
    sync_phase_ends: tuple[float, ...]
    async_full_time: float
    sync_full_time: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def phase_edge_lists(model) -> list[tuple[int, int, torch.Tensor]]:
    """Return canonical (sweep, phase, edge_ids) sequence."""
    out: list[tuple[int, int, torch.Tensor]] = []
    for sweep in range(model.sweeps):
        out.append((sweep, 0, model.phase0))
        out.append((sweep, 1, model.phase1))
    return out


def canonical_phase_states(model, h: torch.Tensor, tok: torch.Tensor) -> list[torch.Tensor]:
    """State before the transition and after each ordinary checkerboard phase."""
    states = [h]
    for sweep, _phase, edge_ids in phase_edge_lists(model):
        h = model._scatter_phase(h, tok, sweep, edge_ids)
        states.append(h)
    return states


def canonical_step(model, h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
    return canonical_phase_states(model, h, tok)[-1]


def settled_prefix_state(model, tokens: torch.Tensor) -> torch.Tensor:
    """Fully process all supplied tokens with ordinary KYY semantics."""
    bsz = tokens.shape[0]
    h = model.h0.unsqueeze(0).expand(bsz, -1)
    for t in range(tokens.shape[1]):
        h = canonical_step(model, h, tokens[:, t])
    return h


def make_edge_durations(model, delay_seed: int, log_sd: float) -> np.ndarray:
    """One fixed physical duration per declared edge, reused across sweeps."""
    rng = np.random.default_rng(delay_seed)
    return np.exp(rng.normal(0.0, log_sd, size=model.n_edges)).astype(np.float64)


def build_schedule(model, edge_durations: np.ndarray) -> Schedule:
    """Build a local dataflow schedule and a globally synchronized schedule.

    ASYNC dependencies are exactly the preceding canonical operation touching either
    endpoint.  This preserves the order of all non-commuting/overlapping gates while
    allowing disjoint local gates to execute concurrently.
    """
    if edge_durations.shape != (model.n_edges,):
        raise ValueError("edge_durations has wrong shape")
    if np.any(edge_durations <= 0.0):
        raise ValueError("edge durations must be positive")

    src = model.src.detach().cpu().numpy()
    dst = model.dst.detach().cpu().numpy()

    last_end = np.zeros(model.state_dim, dtype=np.float64)
    ops: list[LocalOp] = []

    for sweep, phase, edge_ids_t in phase_edge_lists(model):
        for edge_t in edge_ids_t.detach().cpu().tolist():
            edge = int(edge_t)
            i, j = int(src[edge]), int(dst[edge])
            start = max(last_end[i], last_end[j])
            end = start + float(edge_durations[edge])
            ops.append(LocalOp(sweep, phase, edge, i, j, end))
            last_end[i] = end
            last_end[j] = end

    # Operations sorted by completion time are a valid topological execution order:
    # every dependency has strictly smaller end time. Independent operations commute.
    async_ops = tuple(sorted(ops, key=lambda op: op.end_time))
    async_full = max((op.end_time for op in async_ops), default=0.0)

    sync_ends: list[float] = []
    sync_time = 0.0
    for _sweep, _phase, edge_ids_t in phase_edge_lists(model):
        edge_ids = np.asarray(edge_ids_t.detach().cpu().tolist(), dtype=int)
        phase_duration = float(edge_durations[edge_ids].max()) if len(edge_ids) else 0.0
        sync_time += phase_duration
        sync_ends.append(sync_time)

    return Schedule(
        async_ops=async_ops,
        sync_phase_ends=tuple(sync_ends),
        async_full_time=float(async_full),
        sync_full_time=float(sync_time),
    )


def apply_one_edge(model, h: torch.Tensor, tok: torch.Tensor, op: LocalOp) -> torch.Tensor:
    edge_ids = torch.tensor([op.edge], dtype=torch.long, device=h.device)
    return model._scatter_phase(h, tok, op.sweep, edge_ids)


def async_states_at_deadlines(
    model,
    h0: torch.Tensor,
    tok: torch.Tensor,
    schedule: Schedule,
    deadlines: np.ndarray,
) -> list[torch.Tensor]:
    """Expose whatever local work has physically completed by each deadline."""
    states: list[torch.Tensor] = []
    h = h0
    op_index = 0

    for deadline in deadlines:
        while (
            op_index < len(schedule.async_ops)
            and schedule.async_ops[op_index].end_time <= float(deadline) + 1e-12
        ):
            h = apply_one_edge(model, h, tok, schedule.async_ops[op_index])
            op_index += 1
        states.append(h)

    return states


def sync_states_at_deadlines(
    model,
    h0: torch.Tensor,
    tok: torch.Tensor,
    schedule: Schedule,
    deadlines: np.ndarray,
) -> list[torch.Tensor]:
    """Expose only states behind completed global checkerboard barriers."""
    phase_states = canonical_phase_states(model, h0, tok)
    out: list[torch.Tensor] = []

    for deadline in deadlines:
        n_complete = int(
            np.searchsorted(
                np.asarray(schedule.sync_phase_ends),
                float(deadline) + 1e-12,
                side="right",
            )
        )
        out.append(phase_states[n_complete])
    return out


def phase_supervised_logits(model, tokens: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Shared-head logits at every homogeneous phase and ordinary final logits."""
    bsz, length = tokens.shape
    h = model.h0.unsqueeze(0).expand(bsz, -1)
    phase_logits: list[torch.Tensor] = []
    final_logits: list[torch.Tensor] = []

    for t in range(length):
        tok = tokens[:, t]
        for sweep, _phase, edge_ids in phase_edge_lists(model):
            h = model._scatter_phase(h, tok, sweep, edge_ids)
            phase_logits.append(model.readout(h))
        final_logits.append(model.readout(h))

    return phase_logits, torch.stack(final_logits, dim=1)


def train_model(
    *,
    training_mode: str,
    task: str,
    seed: int,
    state_dim: int,
    topology: str,
    train_length: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int,
):
    if training_mode not in {"final", "phase"}:
        raise ValueError("training_mode must be 'final' or 'phase'")

    seed_everything(seed)
    spec = TASKS[task]
    model = build_model(
        "geom_scatter",
        spec.vocab_size,
        spec.n_classes,
        state_dim,
        topology=topology,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    started = time.perf_counter()

    for step in range(1, steps + 1):
        x, y = generate_batch(task, batch_size, train_length, device)

        if training_mode == "final":
            logits = model(x)
            loss = criterion(logits.reshape(-1, spec.n_classes), y.reshape(-1))
            train_acc = float((logits.argmax(dim=-1) == y).float().mean().detach())
        else:
            phase_logits, final_logits = phase_supervised_logits(model, x)

            # phase_logits are ordered token-major: every phase for token 0, then
            # every phase for token 1, ... . Supervise all of them with the running
            # target for that token, using the SAME readout parameters.
            phases_per_token = 2 * model.sweeps
            losses = []
            cursor = 0
            for t in range(train_length):
                target = y[:, t]
                for _ in range(phases_per_token):
                    losses.append(criterion(phase_logits[cursor], target))
                    cursor += 1
            loss = torch.stack(losses).mean()
            train_acc = float(
                (final_logits.argmax(dim=-1) == y).float().mean().detach()
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            print(
                f"[{training_mode:5s}] seed={seed} step={step:4d}/{steps} "
                f"loss={float(loss.detach()):.4f} final-train-acc={train_acc:.3f}",
                flush=True,
            )

    return model, time.perf_counter() - started


@torch.no_grad()
def evaluate_deadline_curve(
    model,
    *,
    task: str,
    eval_seed: int,
    delay_seed: int,
    delay_log_sd: float,
    sequence_length: int,
    batch_size: int,
    batches: int,
    deadline_fractions: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate only the final token, after a completely settled prefix."""
    model.eval()
    spec = TASKS[task]
    edge_durations = make_edge_durations(model, delay_seed, delay_log_sd)
    schedule = build_schedule(model, edge_durations)

    # Fractions are normalized to synchronized full latency. ASYNC may finish before 1.
    deadlines = deadline_fractions * schedule.sync_full_time

    async_correct = np.zeros(len(deadlines), dtype=np.float64)
    sync_correct = np.zeros(len(deadlines), dtype=np.float64)
    total = 0

    wrong_early_right_final = 0
    right_early_wrong_final = 0
    early_index = int(np.argmin(np.abs(deadline_fractions - 0.10)))
    max_final_state_error = 0.0

    seed_everything(eval_seed)

    for _ in range(batches):
        x, y = generate_batch(task, batch_size, sequence_length, device)
        target = y[:, -1]

        if sequence_length > 1:
            h0 = settled_prefix_state(model, x[:, :-1])
        else:
            h0 = model.h0.unsqueeze(0).expand(batch_size, -1)
        tok = x[:, -1]

        canonical_final = canonical_step(model, h0, tok)
        async_states = async_states_at_deadlines(model, h0, tok, schedule, deadlines)
        sync_states = sync_states_at_deadlines(model, h0, tok, schedule, deadlines)

        # Complete ASYNC state independent of the user-specified deadline grid.
        all_async = async_states_at_deadlines(
            model,
            h0,
            tok,
            schedule,
            np.asarray([schedule.async_full_time + 1e-9]),
        )[0]
        all_sync = canonical_phase_states(model, h0, tok)[-1]

        err = max(
            float((all_async - canonical_final).abs().max()),
            float((all_sync - canonical_final).abs().max()),
        )
        max_final_state_error = max(max_final_state_error, err)

        async_pred = []
        sync_pred = []
        for i, state in enumerate(async_states):
            pred = model.readout(state).argmax(dim=-1)
            async_pred.append(pred)
            async_correct[i] += float((pred == target).sum())

        for i, state in enumerate(sync_states):
            pred = model.readout(state).argmax(dim=-1)
            sync_pred.append(pred)
            sync_correct[i] += float((pred == target).sum())

        final_pred = model.readout(canonical_final).argmax(dim=-1)
        early_pred = async_pred[early_index]
        wrong_early_right_final += int(((early_pred != target) & (final_pred == target)).sum())
        right_early_wrong_final += int(((early_pred == target) & (final_pred != target)).sum())

        total += batch_size

    if max_final_state_error > 2e-5:
        raise AssertionError(
            f"async/sync completed state differs from canonical KYY state: {max_final_state_error}"
        )

    async_acc = async_correct / max(1, total)
    sync_acc = sync_correct / max(1, total)

    # Canonical final accuracy is also the accuracy at any deadline after both
    # schedules have fully completed, but record it explicitly because ASYNC may
    # finish before the normalized fraction reaches 1.
    final_index = -1
    final_accuracy = float(async_acc[final_index])

    # numpy.trapz is available in the project's numpy>=1.26 requirement.
    auc_async = float(np.trapz(async_acc, deadline_fractions))
    auc_sync = float(np.trapz(sync_acc, deadline_fractions))

    partial_mask = (deadline_fractions > 0.0) & (deadline_fractions < 1.0)

    def at_fraction(values: np.ndarray, fraction: float) -> float:
        idx = int(np.argmin(np.abs(deadline_fractions - fraction)))
        return float(values[idx])

    return {
        "task": task,
        "n_classes": spec.n_classes,
        "delay_seed": delay_seed,
        "delay_log_sd": delay_log_sd,
        "async_full_over_sync_full": (
            schedule.async_full_time / schedule.sync_full_time
            if schedule.sync_full_time > 0
            else 1.0
        ),
        "deadline_fractions": deadline_fractions.tolist(),
        "async_accuracy": async_acc.tolist(),
        "sync_accuracy": sync_acc.tolist(),
        "auc_async": auc_async,
        "auc_sync": auc_sync,
        "final_accuracy": final_accuracy,
        "async_acc_10pct": at_fraction(async_acc, 0.10),
        "sync_acc_10pct": at_fraction(sync_acc, 0.10),
        "async_acc_20pct": at_fraction(async_acc, 0.20),
        "sync_acc_20pct": at_fraction(sync_acc, 0.20),
        "async_acc_50pct": at_fraction(async_acc, 0.50),
        "sync_acc_50pct": at_fraction(sync_acc, 0.50),
        "min_async_partial_accuracy": (
            float(async_acc[partial_mask].min()) if np.any(partial_mask) else final_accuracy
        ),
        "wrong_10pct_then_right_final": wrong_early_right_final / max(1, total),
        "right_10pct_then_wrong_final": right_early_wrong_final / max(1, total),
        "max_completed_state_abs_error": max_final_state_error,
    }


def print_result(training_mode: str, model_seed: int, row: dict[str, object]) -> None:
    print(
        f"{training_mode:5s} model={model_seed} delay={row['delay_seed']} "
        f"final={row['final_accuracy']:.3f} "
        f"AUC async/sync={row['auc_async']:.3f}/{row['auc_sync']:.3f} "
        f"@10%={row['async_acc_10pct']:.3f}/{row['sync_acc_10pct']:.3f} "
        f"@20%={row['async_acc_20pct']:.3f}/{row['sync_acc_20pct']:.3f} "
        f"min-partial={row['min_async_partial_accuracy']:.3f} "
        f"async/full={row['async_full_over_sync_full']:.3f}"
    )


def aggregate(rows: Iterable[dict[str, object]]) -> dict[str, float]:
    rows = list(rows)
    keys = [
        "final_accuracy",
        "auc_async",
        "auc_sync",
        "async_acc_10pct",
        "sync_acc_10pct",
        "async_acc_20pct",
        "sync_acc_20pct",
        "async_acc_50pct",
        "sync_acc_50pct",
        "min_async_partial_accuracy",
        "wrong_10pct_then_right_final",
        "right_10pct_then_wrong_final",
        "async_full_over_sync_full",
    ]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KYY geom_scatter deadline / mixed-maturity gate"
    )
    parser.add_argument("--training-modes", nargs="+", choices=["final", "phase"], default=["final", "phase"])
    parser.add_argument("--task", choices=sorted(TASKS), default="perm3")
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--topology", choices=["ring", "path", "matching", "disconnected"], default="ring")
    parser.add_argument("--train-length", type=int, default=32)
    parser.add_argument("--eval-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-batches", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--delay-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--delay-log-sd", type=float, default=0.55)
    parser.add_argument("--deadline-points", type=int, default=21)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out", default="results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.state_dim < 4 or args.state_dim % 2:
        parser.error("--state-dim must be even and >= 4")
    if args.deadline_points < 3:
        parser.error("--deadline-points must be >= 3")

    if args.quick:
        args.steps = min(args.steps, 140)
        args.model_seeds = [args.model_seeds[0]]
        args.delay_seeds = args.delay_seeds[:2]
        args.eval_batches = min(args.eval_batches, 2)
        args.eval_batch_size = min(args.eval_batch_size, 256)
        args.log_every = min(args.log_every, 35)

    device = torch.device(args.device)
    deadline_fractions = np.linspace(0.0, 1.0, args.deadline_points)

    config = vars(args).copy()
    config["device"] = str(device)
    print(json.dumps(config, indent=2))

    all_rows: list[dict[str, object]] = []
    train_records: list[dict[str, object]] = []

    for training_mode in args.training_modes:
        for model_seed in args.model_seeds:
            model, train_seconds = train_model(
                training_mode=training_mode,
                task=args.task,
                seed=model_seed,
                state_dim=args.state_dim,
                topology=args.topology,
                train_length=args.train_length,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                log_every=args.log_every,
            )

            train_records.append(
                {
                    "training_mode": training_mode,
                    "model_seed": model_seed,
                    "parameters": parameter_count(model),
                    "train_seconds": train_seconds,
                    "operator": model.operator_summary(),
                }
            )

            for delay_seed in args.delay_seeds:
                row = evaluate_deadline_curve(
                    model,
                    task=args.task,
                    eval_seed=100_000 + 1000 * model_seed + delay_seed,
                    delay_seed=delay_seed,
                    delay_log_sd=args.delay_log_sd,
                    sequence_length=args.eval_length,
                    batch_size=args.eval_batch_size,
                    batches=args.eval_batches,
                    deadline_fractions=deadline_fractions,
                    device=device,
                )
                row["training_mode"] = training_mode
                row["model_seed"] = model_seed
                all_rows.append(row)
                print_result(training_mode, model_seed, row)

    print("\n=== means ===")
    summaries: dict[str, dict[str, float]] = {}
    for training_mode in args.training_modes:
        rows = [row for row in all_rows if row["training_mode"] == training_mode]
        summary = aggregate(rows)
        summaries[training_mode] = summary
        print(f"\n{training_mode}")
        for key, value in summary.items():
            print(f"  {key:34s} {value:.6f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"deadline-scatter-{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "config": config,
                "train": train_records,
                "rows": all_rows,
                "means": summaries,
                "interpretation_guardrail": (
                    "Same local scatter computation and final answer; deadline curves isolate "
                    "synchronization and readout legibility. This is not yet a geometry win."
                ),
            },
            indent=2,
        )
    )
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
