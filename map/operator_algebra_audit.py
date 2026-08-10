from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from kyy import TASKS, build_model, generate_batch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def linear_affine_transition(model, token: int) -> tuple[np.ndarray, np.ndarray]:
    """Return h' = A h + b for KYY's linear transition families.

    GRU is intentionally excluded: its transition depends nonlinearly on h.
    """
    name = model.name
    with torch.no_grad():
        if name == "diag_signed":
            a = 0.999 * torch.tanh(model.a_raw[token])
            A = torch.diag(a)
            b = model.drive[token]
        elif name == "complex_diag":
            modes = model.state_dim // 2
            r = 0.999 * torch.sigmoid(model.radius_raw[token])
            theta = math.pi * torch.tanh(model.angle_raw[token])
            A = torch.zeros((model.state_dim, model.state_dim), device=r.device, dtype=r.dtype)
            for k in range(modes):
                c, s = torch.cos(theta[k]), torch.sin(theta[k])
                block = r[k] * torch.stack((torch.stack((c, -s)), torch.stack((s, c))))
                A[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = block
            b = model.drive[token].reshape(-1)
        elif hasattr(model, "dense_transition"):
            A = model.dense_transition(token)
            b = torch.zeros(model.state_dim, device=A.device, dtype=A.dtype)
        else:
            raise TypeError(f"{name} has no fixed affine transition")
    return A.detach().cpu().double().numpy(), b.detach().cpu().double().numpy()


def augment_affine(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    out = np.eye(n + 1, dtype=np.float64)
    out[:n, :n] = A
    out[:n, n] = b
    return out


def augmented_transitions(model) -> list[np.ndarray]:
    return [augment_affine(*linear_affine_transition(model, t)) for t in range(model.vocab_size)]


def normalized_commutator(A: np.ndarray, B: np.ndarray) -> float:
    denom = np.linalg.norm(A, "fro") * np.linalg.norm(B, "fro") + 1e-15
    return float(np.linalg.norm(A @ B - B @ A, "fro") / denom)


def pairwise_commutators(mats: list[np.ndarray]) -> dict[str, object]:
    vals: dict[str, float] = {}
    flat = []
    for i in range(len(mats)):
        for j in range(i):
            v = normalized_commutator(mats[i], mats[j])
            vals[f"{j},{i}"] = v
            flat.append(v)
    return {
        "pairs": vals,
        "mean": float(np.mean(flat)) if flat else 0.0,
        "max": float(np.max(flat)) if flat else 0.0,
    }


def normalized_defect(M: np.ndarray) -> float:
    return float(np.linalg.norm(M, "fro") / math.sqrt(M.shape[0]))


def perm3_relation_defects(mats: list[np.ndarray]) -> dict[str, float] | None:
    """Global hidden-space defects for the generators used by KYY perm3.

    token 0 = identity, token 1 = transposition s, token 2 = 3-cycle r.

    IMPORTANT: these relations do *not* have to hold on every hidden direction
    for the classifier to solve the task.  The model may implement a larger
    dynamical extension whose readout quotient tracks S3.  These are therefore
    diagnostics, not correctness conditions.
    """
    if len(mats) != 3:
        return None
    E, S, R = mats
    I = np.eye(E.shape[0])
    try:
        r_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        r_inv = np.linalg.pinv(R)
    return {
        "noop_E_minus_I": normalized_defect(E - I),
        "s_squared_minus_I": normalized_defect(S @ S - I),
        "r_cubed_minus_I": normalized_defect(R @ R @ R - I),
        "srs_minus_r_inverse": normalized_defect(S @ R @ S - r_inv),
    }


def finite_word_span_dimension(mats: list[np.ndarray], max_depth: int = 4, rtol: float = 1e-9) -> int:
    """Dimension of the linear span of transition words up to max_depth.

    This is an associative-algebra diagnostic, not a dynamical-Lie-algebra
    calculation.  It is safe for the discrete/affine transition matrices and
    tells us how rapidly products explore independent operator directions.
    """
    n = mats[0].shape[0]
    cols = [np.eye(n).reshape(-1)]
    frontier = [np.eye(n)]
    for _ in range(max_depth):
        nxt = []
        for P in frontier:
            for A in mats:
                Q = A @ P
                cols.append(Q.reshape(-1))
                nxt.append(Q)
        frontier = nxt
    M = np.stack(cols, axis=1)
    s = np.linalg.svd(M, compute_uv=False)
    threshold = rtol * s[0] if s.size else 0.0
    return int(np.sum(s > threshold))


def train(model, task: str, steps: int, length: int, batch_size: int, lr: float, device: torch.device) -> None:
    spec = TASKS[task]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        x, y = generate_batch(task, batch_size, length, device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, spec.n_classes), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def evaluate(model, task: str, length: int, batch_size: int, batches: int, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = final = total = final_total = 0
    for _ in range(batches):
        x, y = generate_batch(task, batch_size, length, device)
        pred = model(x).argmax(dim=-1)
        correct += int((pred == y).sum())
        total += y.numel()
        final += int((pred[:, -1] == y[:, -1]).sum())
        final_total += batch_size
    return {"accuracy": correct / total, "final_accuracy": final / final_total}


@dataclass
class AuditRow:
    model: str
    seed: int
    state_dim: int
    topology: str | None
    eval: dict[str, dict[str, float]]
    commutator: dict[str, object]
    word_span_dim_depth4: int
    perm3_global_relation_defects: dict[str, float] | None


def main() -> None:
    p = argparse.ArgumentParser(description="Audit the operator algebra of already-existing KYY models")
    p.add_argument("--models", nargs="+", default=["diag_signed", "complex_diag", "householder2", "geom_scatter"])
    p.add_argument("--task", choices=sorted(TASKS), default="perm3")
    p.add_argument("--state-dim", type=int, default=8)
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--test-lengths", nargs="+", type=int, default=[16, 64, 256])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--topology", default="ring")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    spec = TASKS[args.task]
    rows: list[AuditRow] = []

    for name in args.models:
        if name == "gru":
            print("Skipping GRU: no fixed affine token transition to audit.")
            continue
        for seed in args.seeds:
            seed_everything(seed)
            model = build_model(name, spec.vocab_size, spec.n_classes, args.state_dim, topology=args.topology).to(device)
            train(model, args.task, args.steps, args.train_length, args.batch_size, args.lr, device)
            mats = augmented_transitions(model)
            evals = {
                str(L): evaluate(model, args.task, L, args.batch_size, args.eval_batches, device)
                for L in args.test_lengths
            }
            row = AuditRow(
                model=name,
                seed=seed,
                state_dim=args.state_dim,
                topology=args.topology if name in {"geom_wave", "geom_scatter"} else None,
                eval=evals,
                commutator=pairwise_commutators(mats),
                word_span_dim_depth4=finite_word_span_dimension(mats, max_depth=4),
                perm3_global_relation_defects=perm3_relation_defects(mats) if args.task == "perm3" else None,
            )
            rows.append(row)
            print(json.dumps(asdict(row), indent=2))

    payload = {"config": vars(args), "rows": [asdict(r) for r in rows]}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
