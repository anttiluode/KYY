"""Diagnose information availability versus shared-readout legibility in geom_scatter.

The deadline gate exposed an important failure mode: an ordinary final-only KYY model
can have excellent final accuracy while its own readout is catastrophically wrong on
partially completed states.

This script asks whether those intermediate states actually *lack the answer*, or
whether the answer is present in a phase-dependent coordinate frame.

Protocol
--------
1. Train ordinary ``geom_scatter`` with the standard final-token-transition objective.
2. Freeze the recurrent model completely.
3. Collect hidden states after each checkerboard phase.
4. Compare three readouts on held-out states:

   native_final_head
       KYY's trained final readout, applied unchanged at every phase.

   phase_specific_probes
       one separately fitted linear probe per phase. Diagnostic only.

   one_shared_posthoc_probe
       one new linear probe fitted jointly to states from every phase.

If phase-specific probes are good while the native/shared probes are poor, then
intermediate computation contains linearly accessible task information but its
coordinate frame changes with maturity. That is an information/legibility distinction,
not evidence for a new architecture.

No sklearn dependency is added; probes are tiny PyTorch linear layers.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kyy import TASKS, build_model, generate_batch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def phase_edge_lists(model):
    for sweep in range(model.sweeps):
        yield sweep, model.phase0
        yield sweep, model.phase1


def collect_states(
    model,
    *,
    task: str,
    batches: int,
    batch_size: int,
    length: int,
    device: torch.device,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return CPU tensors [phase][samples,state] and [phase][samples]."""
    seed_everything(seed)
    model.eval()
    n_phases = 2 * model.sweeps
    states: list[list[torch.Tensor]] = [[] for _ in range(n_phases)]
    labels: list[list[torch.Tensor]] = [[] for _ in range(n_phases)]

    with torch.no_grad():
        for _ in range(batches):
            x, y = generate_batch(task, batch_size, length, device)
            h = model.h0.unsqueeze(0).expand(batch_size, -1)

            for t in range(length):
                tok = x[:, t]
                target = y[:, t]
                for phase_index, (sweep, edge_ids) in enumerate(phase_edge_lists(model)):
                    h = model._scatter_phase(h, tok, sweep, edge_ids)
                    states[phase_index].append(h.detach().cpu())
                    labels[phase_index].append(target.detach().cpu())

    return (
        [torch.cat(parts, dim=0) for parts in states],
        [torch.cat(parts, dim=0) for parts in labels],
    )


def train_base_model(
    *,
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
    seed_everything(seed)
    spec = TASKS[task]
    model = build_model(
        "geom_scatter",
        spec.vocab_size,
        spec.n_classes,
        state_dim,
        topology=topology,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for step in range(1, steps + 1):
        x, y = generate_batch(task, batch_size, train_length, device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, spec.n_classes), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % log_every == 0 or step == steps:
            acc = float((logits.argmax(dim=-1) == y).float().mean().detach())
            print(
                f"seed={seed} base step={step:4d}/{steps} "
                f"loss={float(loss.detach()):.4f} acc={acc:.3f}",
                flush=True,
            )
    return model


def accuracy(head: nn.Linear, X: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    head.eval()
    with torch.no_grad():
        pred = head(X.to(device)).argmax(dim=-1).cpu()
    return float((pred == y).float().mean())


def fit_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    state_dim: int,
    n_classes: int,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> nn.Linear:
    seed_everything(seed)
    head = nn.Linear(state_dim, n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    Xd = X.to(device)
    yd = y.to(device)
    n = len(y)

    for _ in range(steps):
        if batch_size >= n:
            idx = torch.arange(n, device=device)
        else:
            idx = torch.randint(0, n, (batch_size,), device=device)
        logits = head(Xd[idx])
        loss = criterion(logits, yd[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return head


def main() -> None:
    p = argparse.ArgumentParser(description="KYY phase legibility probe diagnostic")
    p.add_argument("--task", choices=sorted(TASKS), default="perm3")
    p.add_argument("--state-dim", type=int, default=32)
    p.add_argument("--topology", choices=["ring", "path", "matching", "disconnected"], default="ring")
    p.add_argument("--train-length", type=int, default=32)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--model-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--probe-train-batches", type=int, default=12)
    p.add_argument("--probe-test-batches", type=int, default=6)
    p.add_argument("--probe-length", type=int, default=24)
    p.add_argument("--probe-steps", type=int, default=350)
    p.add_argument("--probe-batch-size", type=int, default=2048)
    p.add_argument("--probe-lr", type=float, default=1e-2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.steps = min(args.steps, 140)
        args.model_seeds = [args.model_seeds[0]]
        args.probe_train_batches = min(args.probe_train_batches, 5)
        args.probe_test_batches = min(args.probe_test_batches, 3)
        args.probe_length = min(args.probe_length, 16)
        args.probe_steps = min(args.probe_steps, 180)
        args.log_every = min(args.log_every, 35)

    device = torch.device(args.device)
    spec = TASKS[args.task]
    rows: list[dict[str, object]] = []

    for model_seed in args.model_seeds:
        model = train_base_model(
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

        train_states, train_labels = collect_states(
            model,
            task=args.task,
            batches=args.probe_train_batches,
            batch_size=args.batch_size,
            length=args.probe_length,
            device=device,
            seed=10_000 + model_seed,
        )
        test_states, test_labels = collect_states(
            model,
            task=args.task,
            batches=args.probe_test_batches,
            batch_size=args.batch_size,
            length=args.probe_length,
            device=device,
            seed=20_000 + model_seed,
        )

        n_phases = len(train_states)

        native = [
            accuracy(model.readout, test_states[p], test_labels[p], device)
            for p in range(n_phases)
        ]

        phase_probe_acc = []
        for pidx in range(n_phases):
            probe = fit_probe(
                train_states[pidx],
                train_labels[pidx],
                state_dim=args.state_dim,
                n_classes=spec.n_classes,
                seed=30_000 + model_seed * 100 + pidx,
                steps=args.probe_steps,
                batch_size=args.probe_batch_size,
                lr=args.probe_lr,
                device=device,
            )
            phase_probe_acc.append(
                accuracy(probe, test_states[pidx], test_labels[pidx], device)
            )

        shared_X = torch.cat(train_states, dim=0)
        shared_y = torch.cat(train_labels, dim=0)
        shared_probe = fit_probe(
            shared_X,
            shared_y,
            state_dim=args.state_dim,
            n_classes=spec.n_classes,
            seed=40_000 + model_seed,
            steps=args.probe_steps,
            batch_size=args.probe_batch_size,
            lr=args.probe_lr,
            device=device,
        )
        shared = [
            accuracy(shared_probe, test_states[p], test_labels[p], device)
            for p in range(n_phases)
        ]

        row = {
            "model_seed": model_seed,
            "native_final_head_by_phase": native,
            "phase_specific_probe_by_phase": phase_probe_acc,
            "one_shared_posthoc_probe_by_phase": shared,
        }
        rows.append(row)

        print(f"\nmodel seed {model_seed}")
        print("phase   native-final-head   phase-probe   shared-posthoc")
        print("--------------------------------------------------------")
        for pidx in range(n_phases):
            print(
                f"{pidx + 1:5d}   {native[pidx]:17.4f}   "
                f"{phase_probe_acc[pidx]:11.4f}   {shared[pidx]:14.4f}"
            )

    def mean_curve(key: str) -> list[float]:
        return np.asarray([row[key] for row in rows], dtype=float).mean(axis=0).tolist()

    means = {
        "native_final_head_by_phase": mean_curve("native_final_head_by_phase"),
        "phase_specific_probe_by_phase": mean_curve("phase_specific_probe_by_phase"),
        "one_shared_posthoc_probe_by_phase": mean_curve("one_shared_posthoc_probe_by_phase"),
    }

    print("\n=== mean curves ===")
    for key, values in means.items():
        print(f"{key}: " + " ".join(f"{v:.4f}" for v in values))

    print("\nInterpretation")
    print("--------------")
    print("High phase-specific probe accuracy means target information is linearly present.")
    print("Poor native/shared accuracy at the same phase means the issue is coordinate/readout")
    print("legibility, not simple absence of information. This is a diagnostic, not a win.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"phase-legibility-{stamp}.json"
    path.write_text(json.dumps({"config": vars(args), "rows": rows, "means": means}, indent=2, default=str))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
