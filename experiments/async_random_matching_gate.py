"""Final cheap control for the KYY mixed-maturity seam.

Question
--------
Does the fixed ring/local-neighbor geometry of ``geom_scatter`` buy an anytime/deadline
advantage once a non-geometric sparse dataflow network gets the same representational
help?

We compare two models with matched high-level budgets:

RING
    the existing KYY ``geom_scatter`` on 32 channels, two sweeps and two checkerboard
    phases per sweep.

RANDOM_MATCH
    four arbitrary perfect matchings over the same channels. Each phase contains the
    same number of disjoint 2-port symmetric orthogonal scatter operations, but the
    pairings are not constrained to a fixed local ring.

For state_dim=32 and vocab=3 both have:

    4 homogeneous phases / token
    16 two-port operations / phase
    64 two-port operations / token
    192 token/phase/edge angle parameters
    one h0 and one shared linear readout

Both are trained with the SAME shared-head phase supervision. Both are then evaluated
under the SAME asynchronous dependency rule: a pair operation may start as soon as
all earlier canonical operations touching either endpoint have finished. Both use IID
lognormal operation durations from the same distribution.

Evaluation is conditioned on non-identity final tokens of ``perm3`` so the old-state
1/3 identity artifact is absent.

This is a software/dataflow control. It contains NO wire-length, placement, energy or
analog-noise model. Therefore a random nonlocal matching is not charged for long wires.
If it matches or beats the ring here, the ring cannot claim the scheduling effect; a
hardware-locality claim would require a different cost model.

Run:

    python experiments/async_random_matching_gate.py --quick

Fuller:

    python experiments/async_random_matching_gate.py \
        --model-seeds 0 1 2 --matching-seeds 101 202 303 --steps 400
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kyy import TASKS, build_model, generate_batch, parameter_count


@dataclass(frozen=True)
class Op:
    phase: int
    slot: int
    src: int
    dst: int
    end_time: float


@dataclass(frozen=True)
class Schedule:
    ops: tuple[Op, ...]
    sync_phase_ends: tuple[float, ...]
    async_full_time: float
    sync_full_time: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RandomMatchingScatter(nn.Module):
    """Four arbitrary perfect-match phases with KYY's same symmetric 2-port cell."""

    name = "random_matching_scatter"

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        state_dim: int,
        *,
        matching_seed: int,
        n_phases: int = 4,
    ):
        super().__init__()
        if state_dim < 4 or state_dim % 2:
            raise ValueError("state_dim must be even and >= 4")
        self.vocab_size = int(vocab_size)
        self.n_classes = int(n_classes)
        self.state_dim = int(state_dim)
        self.n_phases = int(n_phases)
        self.matching_seed = int(matching_seed)

        rng = np.random.default_rng(matching_seed)
        matchings = []
        for _phase in range(n_phases):
            perm = rng.permutation(state_dim)
            matchings.append(np.stack((perm[0::2], perm[1::2]), axis=1))
        self.register_buffer(
            "pairs",
            torch.tensor(np.stack(matchings), dtype=torch.long),
        )

        self.angle_raw = nn.Parameter(
            torch.empty(vocab_size, n_phases, state_dim // 2)
        )
        nn.init.uniform_(self.angle_raw, -0.35, 0.35)
        self.h0 = nn.Parameter(torch.randn(state_dim) * 0.05)
        self.readout = nn.Linear(state_dim, n_classes)

    def apply_phase(self, h: torch.Tensor, tok: torch.Tensor, phase: int) -> torch.Tensor:
        pair = self.pairs[phase]
        src, dst = pair[:, 0], pair[:, 1]
        theta = math.pi * torch.tanh(self.angle_raw[tok, phase])
        c, s = torch.cos(theta), torch.sin(theta)
        a, b = h[:, src], h[:, dst]
        out = h.clone()
        out[:, src] = c * a + s * b
        out[:, dst] = s * a - c * b
        return out

    def apply_slot(
        self,
        h: torch.Tensor,
        tok: torch.Tensor,
        phase: int,
        slot: int,
    ) -> torch.Tensor:
        i = int(self.pairs[phase, slot, 0])
        j = int(self.pairs[phase, slot, 1])
        theta = math.pi * torch.tanh(self.angle_raw[tok, phase, slot])
        c, s = torch.cos(theta), torch.sin(theta)
        a, b = h[:, i], h[:, j]
        out = h.clone()
        out[:, i] = c * a + s * b
        out[:, j] = s * a - c * b
        return out


def ring_phase_slots(model) -> list[list[tuple[int, int, int]]]:
    """Each tuple is (src, dst, slot=edge_id)."""
    phases = []
    for _sweep in range(model.sweeps):
        for edge_ids in (model.phase0, model.phase1):
            phase = []
            for edge in edge_ids.detach().cpu().tolist():
                phase.append((int(model.src[edge]), int(model.dst[edge]), int(edge)))
            phases.append(phase)
    return phases


def random_phase_slots(model: RandomMatchingScatter) -> list[list[tuple[int, int, int]]]:
    phases = []
    for phase in range(model.n_phases):
        row = []
        for slot, pair in enumerate(model.pairs[phase].detach().cpu().tolist()):
            row.append((int(pair[0]), int(pair[1]), slot))
        phases.append(row)
    return phases


def phase_slots(model, kind: str):
    return ring_phase_slots(model) if kind == "ring" else random_phase_slots(model)


def apply_phase(model, kind: str, h: torch.Tensor, tok: torch.Tensor, phase: int) -> torch.Tensor:
    if kind == "ring":
        sweep = phase // 2
        edge_ids = model.phase0 if phase % 2 == 0 else model.phase1
        return model._scatter_phase(h, tok, sweep, edge_ids)
    return model.apply_phase(h, tok, phase)


def apply_slot(
    model,
    kind: str,
    h: torch.Tensor,
    tok: torch.Tensor,
    phase: int,
    slot: int,
) -> torch.Tensor:
    if kind == "ring":
        sweep = phase // 2
        edge_ids = torch.tensor([slot], dtype=torch.long, device=h.device)
        return model._scatter_phase(h, tok, sweep, edge_ids)
    return model.apply_slot(h, tok, phase, slot)


def phase_states(model, kind: str, h: torch.Tensor, tok: torch.Tensor) -> list[torch.Tensor]:
    out = []
    for phase in range(4):
        h = apply_phase(model, kind, h, tok, phase)
        out.append(h)
    return out


def canonical_step(model, kind: str, h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
    return phase_states(model, kind, h, tok)[-1]


def settled_prefix(model, kind: str, tokens: torch.Tensor) -> torch.Tensor:
    h = model.h0.unsqueeze(0).expand(tokens.shape[0], -1)
    for t in range(tokens.shape[1]):
        h = canonical_step(model, kind, h, tokens[:, t])
    return h


def build_schedule(
    model,
    kind: str,
    *,
    delay_seed: int,
    delay_log_sd: float,
) -> Schedule:
    """Matched IID operation durations; only dependency graph differs."""
    slots = phase_slots(model, kind)
    rng = np.random.default_rng(delay_seed)
    durations = [
        np.exp(rng.normal(0.0, delay_log_sd, size=len(phase))).astype(np.float64)
        for phase in slots
    ]

    last_end = np.zeros(model.state_dim, dtype=np.float64)
    ops: list[Op] = []
    for phase_index, phase in enumerate(slots):
        for local_index, (i, j, slot) in enumerate(phase):
            start = max(last_end[i], last_end[j])
            end = start + float(durations[phase_index][local_index])
            ops.append(Op(phase_index, slot, i, j, end))
            last_end[i] = end
            last_end[j] = end

    ops.sort(key=lambda op: op.end_time)
    async_full = max((op.end_time for op in ops), default=0.0)

    sync_phase_ends = []
    sync_time = 0.0
    for duration in durations:
        sync_time += float(duration.max()) if len(duration) else 0.0
        sync_phase_ends.append(sync_time)

    return Schedule(
        ops=tuple(ops),
        sync_phase_ends=tuple(sync_phase_ends),
        async_full_time=async_full,
        sync_full_time=sync_time,
    )


def async_states(
    model,
    kind: str,
    h: torch.Tensor,
    tok: torch.Tensor,
    schedule: Schedule,
    deadlines: np.ndarray,
) -> list[torch.Tensor]:
    out = []
    cursor = 0
    for deadline in deadlines:
        while cursor < len(schedule.ops) and schedule.ops[cursor].end_time <= float(deadline) + 1e-12:
            op = schedule.ops[cursor]
            h = apply_slot(model, kind, h, tok, op.phase, op.slot)
            cursor += 1
        out.append(h)
    return out


def sync_states(
    model,
    kind: str,
    h: torch.Tensor,
    tok: torch.Tensor,
    schedule: Schedule,
    deadlines: np.ndarray,
) -> list[torch.Tensor]:
    states = [h] + phase_states(model, kind, h, tok)
    ends = np.asarray(schedule.sync_phase_ends)
    return [
        states[int(np.searchsorted(ends, float(deadline) + 1e-12, side="right"))]
        for deadline in deadlines
    ]


def train_phase_legible(
    *,
    kind: str,
    task: str,
    model_seed: int,
    matching_seed: int,
    state_dim: int,
    train_length: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
):
    seed_everything(model_seed)
    spec = TASKS[task]
    if kind == "ring":
        model = build_model(
            "geom_scatter",
            spec.vocab_size,
            spec.n_classes,
            state_dim,
            topology="ring",
        ).to(device)
    else:
        model = RandomMatchingScatter(
            spec.vocab_size,
            spec.n_classes,
            state_dim,
            matching_seed=matching_seed,
        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for _step in range(steps):
        x, y = generate_batch(task, batch_size, train_length, device)
        h = model.h0.unsqueeze(0).expand(batch_size, -1)
        losses = []
        for t in range(train_length):
            tok = x[:, t]
            target = y[:, t]
            for phase in range(4):
                h = apply_phase(model, kind, h, tok, phase)
                losses.append(criterion(model.readout(h), target))
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return model


@torch.no_grad()
def evaluate(
    model,
    kind: str,
    *,
    eval_seed: int,
    delay_seed: int,
    delay_log_sd: float,
    sequence_length: int,
    batch_size: int,
    batches: int,
    fractions: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    schedule = build_schedule(
        model,
        kind,
        delay_seed=delay_seed,
        delay_log_sd=delay_log_sd,
    )
    deadlines = fractions * schedule.sync_full_time
    async_correct = np.zeros(len(fractions), dtype=np.float64)
    sync_correct = np.zeros(len(fractions), dtype=np.float64)
    total = 0
    final_correct = 0
    max_error = 0.0

    seed_everything(eval_seed)
    for _ in range(batches):
        x, y = generate_batch("perm3", batch_size, sequence_length, device)
        mask = x[:, -1] != 0
        if not bool(mask.any()):
            continue
        x = x[mask]
        tok = x[:, -1]
        target = y[mask, -1]
        h0 = settled_prefix(model, kind, x[:, :-1])

        canonical = canonical_step(model, kind, h0, tok)
        A = async_states(model, kind, h0, tok, schedule, deadlines)
        S = sync_states(model, kind, h0, tok, schedule, deadlines)
        completed = async_states(
            model,
            kind,
            h0,
            tok,
            schedule,
            np.asarray([schedule.async_full_time + 1e-9]),
        )[0]
        max_error = max(max_error, float((completed - canonical).abs().max()))

        for i, state in enumerate(A):
            async_correct[i] += float((model.readout(state).argmax(-1) == target).sum())
        for i, state in enumerate(S):
            sync_correct[i] += float((model.readout(state).argmax(-1) == target).sum())
        final_correct += int((model.readout(canonical).argmax(-1) == target).sum())
        total += int(target.numel())

    if max_error > 2e-5:
        raise AssertionError(f"async completed state mismatch: {max_error}")

    aa = async_correct / max(1, total)
    ss = sync_correct / max(1, total)

    def at(v, f):
        return float(v[int(np.argmin(np.abs(fractions - f)))])

    return {
        "kind": kind,
        "delay_seed": delay_seed,
        "n_examples": total,
        "final_accuracy": final_correct / max(1, total),
        "auc_async": float(np.trapz(aa, fractions)),
        "auc_sync": float(np.trapz(ss, fractions)),
        "auc_async_minus_sync": float(np.trapz(aa - ss, fractions)),
        "async_acc_10pct": at(aa, 0.10),
        "sync_acc_10pct": at(ss, 0.10),
        "async_acc_20pct": at(aa, 0.20),
        "sync_acc_20pct": at(ss, 0.20),
        "async_acc_50pct": at(aa, 0.50),
        "sync_acc_50pct": at(ss, 0.50),
        "async_full_over_sync_full": schedule.async_full_time / schedule.sync_full_time,
        "max_completed_state_abs_error": max_error,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="KYY ring vs random-matching async deadline gate")
    p.add_argument("--state-dim", type=int, default=32)
    p.add_argument("--train-length", type=int, default=32)
    p.add_argument("--eval-length", type=int, default=32)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--matching-seeds", nargs="+", type=int, default=[101, 202, 303])
    p.add_argument("--delay-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--delay-log-sd", type=float, default=0.55)
    p.add_argument("--deadline-points", type=int, default=21)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if len(args.matching_seeds) < len(args.model_seeds):
        p.error("provide at least as many --matching-seeds as --model-seeds")

    if args.quick:
        args.steps = min(args.steps, 180)
        args.model_seeds = [args.model_seeds[0]]
        args.matching_seeds = [args.matching_seeds[0]]
        args.delay_seeds = args.delay_seeds[:2]
        args.eval_batches = min(args.eval_batches, 4)

    device = torch.device(args.device)
    fractions = np.linspace(0.0, 1.0, args.deadline_points)
    rows = []

    for index, model_seed in enumerate(args.model_seeds):
        matching_seed = args.matching_seeds[index]
        for kind in ("ring", "random"):
            model = train_phase_legible(
                kind=kind,
                task="perm3",
                model_seed=model_seed,
                matching_seed=matching_seed,
                state_dim=args.state_dim,
                train_length=args.train_length,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
            )

            if kind == "ring":
                angle_params = model.vocab_size * model.sweeps * model.n_edges
            else:
                angle_params = model.vocab_size * model.n_phases * (model.state_dim // 2)

            for delay_seed in args.delay_seeds:
                row = evaluate(
                    model,
                    kind,
                    eval_seed=100_000 + model_seed * 1000 + delay_seed,
                    delay_seed=delay_seed,
                    delay_log_sd=args.delay_log_sd,
                    sequence_length=args.eval_length,
                    batch_size=args.eval_batch_size,
                    batches=args.eval_batches,
                    fractions=fractions,
                    device=device,
                )
                row.update(
                    {
                        "model_seed": model_seed,
                        "matching_seed": matching_seed if kind == "random" else None,
                        "parameters": parameter_count(model),
                        "angle_parameters": angle_params,
                    }
                )
                rows.append(row)
                print(
                    f"{kind:6s} model={model_seed} delay={delay_seed} "
                    f"final={row['final_accuracy']:.3f} "
                    f"AUC={row['auc_async']:.3f}/{row['auc_sync']:.3f} "
                    f"delta={row['auc_async_minus_sync']:+.3f} "
                    f"@20={row['async_acc_20pct']:.3f}/{row['sync_acc_20pct']:.3f} "
                    f"async/full={row['async_full_over_sync_full']:.3f}"
                )

    summary = {}
    for kind in ("ring", "random"):
        selected = [r for r in rows if r["kind"] == kind]
        summary[kind] = {
            key: float(np.mean([float(r[key]) for r in selected]))
            for key in (
                "final_accuracy",
                "auc_async",
                "auc_sync",
                "auc_async_minus_sync",
                "async_acc_10pct",
                "sync_acc_10pct",
                "async_acc_20pct",
                "sync_acc_20pct",
                "async_acc_50pct",
                "sync_acc_50pct",
                "async_full_over_sync_full",
            )
        }

    print("\n=== means ===")
    for kind, s in summary.items():
        print(f"\n{kind}")
        for key, value in s.items():
            print(f"  {key:>24s}: {value:.4f}")

    print("\nInterpretation guardrail")
    print("------------------------")
    print("If random matchings obtain the same async deadline benefit, the effect belongs to")
    print("sparse/asynchronous dataflow rather than the fixed ring geometry. This experiment")
    print("does not model physical wire length, placement, energy or analog robustness.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"async-random-matching-{stamp}.json"
    path.write_text(json.dumps({"config": vars(args), "rows": rows, "means": summary}, indent=2, default=str))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
