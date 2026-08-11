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
MODULE_NAME = "positive_kernel_for_closure_audit"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "cyclic_positive_kernel_port_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
pk = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = pk
SPEC.loader.exec_module(pk)


@dataclass
class ClosureRun:
    n: int
    seed: int
    frequencies: list[int]
    character_gcd: int
    certified: bool
    active_modes: int
    positive_margin: float
    equivariant_margin: float
    clean_accuracy: dict[str, float]
    error_1e4_accuracy: dict[str, float]
    error_1e3_accuracy: dict[str, float]


def runtime_accuracy(
    *,
    n: int,
    projected: np.ndarray,
    q: torch.Tensor,
    h0: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    lengths: list[int],
    batch_size: int,
    max_increment: int,
    random_start: bool,
    angle_error: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    dtype = torch.float64
    angles = torch.as_tensor(projected, dtype=dtype) + float(angle_error)
    Q = q.to(dtype)
    Wd = W.to(dtype)
    bd = b.to(dtype)
    seed = h0.to(dtype)
    for length in lengths:
        x, y = pk.base.base.generate_batch(
            n, batch_size, length, max_increment, random_start=random_start
        )
        h = seed.unsqueeze(0).expand(batch_size, -1, -1).clone()
        correct = 0
        total = 0
        for t in range(length):
            inc = x[:, t].to(dtype).unsqueeze(-1)
            theta = inc * angles.unsqueeze(0)
            c, s = torch.cos(theta), torch.sin(theta)
            hx, hy = h[..., 0], h[..., 1]
            h = torch.stack((c * hx - s * hy, s * hx + c * hy), dim=-1)
            flat = h.reshape(batch_size, -1) @ Q
            logits = flat @ Wd.T + bd
            pred = logits.argmax(dim=-1)
            correct += int((pred == y[:, t]).sum().item())
            total += batch_size
        out[str(length)] = correct / total
    return out


def run_one(
    *,
    n: int,
    modes: int,
    seed: int,
    train_length: int,
    train_steps: int,
    batch_size: int,
    eval_batch_size: int,
    max_increment: int,
    lr: float,
    random_start: bool,
    lengths: list[int],
) -> ClosureRun:
    model = pk.base.train_learned_model(
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
    projected, frequencies = pk.base.base.project_angles_to_characters(n, learned)
    q = pk.base.midpoint_port(n, learned, projected)
    z = pk.base.exact_orbit(n, projected, model.h0.detach().cpu()) @ q
    W_learned = model.readout.weight.detach().cpu().to(torch.float64)
    b_learned = model.readout.bias.detach().cpu().to(torch.float64)
    W_eq, b_eq, _ = pk.base.project_cyclic_equivariant_decoder(
        n, projected, W_learned, b_learned
    )
    W_pos, b_pos, alpha, _ = pk.positive_kernel_projection(
        n, projected, z[0], W_learned, b_learned
    )
    pos = pk.base.readout_metrics(z, W_pos, b_pos)
    eq = pk.base.readout_metrics(z, W_eq, b_eq)
    certified, g = pk.positive_kernel_certificate(n, frequencies, alpha)
    active = int(np.sum(alpha > 1e-12))

    clean = runtime_accuracy(
        n=n, projected=projected, q=q, h0=model.h0.detach().cpu(),
        W=W_pos, b=b_pos, lengths=lengths, batch_size=eval_batch_size,
        max_increment=max_increment, random_start=random_start, angle_error=0.0,
    )
    e4 = runtime_accuracy(
        n=n, projected=projected, q=q, h0=model.h0.detach().cpu(),
        W=W_pos, b=b_pos, lengths=lengths, batch_size=eval_batch_size,
        max_increment=max_increment, random_start=random_start, angle_error=1e-4,
    )
    e3 = runtime_accuracy(
        n=n, projected=projected, q=q, h0=model.h0.detach().cpu(),
        W=W_pos, b=b_pos, lengths=lengths, batch_size=eval_batch_size,
        max_increment=max_increment, random_start=random_start, angle_error=1e-3,
    )
    return ClosureRun(
        n=n,
        seed=seed,
        frequencies=[int(x) for x in frequencies.tolist()],
        character_gcd=int(g),
        certified=bool(certified),
        active_modes=active,
        positive_margin=float(pos[1]),
        equivariant_margin=float(eq[1]),
        clean_accuracy=clean,
        error_1e4_accuracy=e4,
        error_1e3_accuracy=e3,
    )


def certificate_controls() -> list[dict[str, object]]:
    cases = [
        (100, [2, 4, 6, 8], 2),
        (100, [2, 4, 6, 7], 1),
        (105, [3, 6, 9, 12], 3),
        (105, [3, 6, 10, 14], 1),
    ]
    rows = []
    for n, fs, expected_gcd in cases:
        alpha = np.ones(len(fs), dtype=np.float64)
        ok, g = pk.positive_kernel_certificate(n, np.asarray(fs), alpha)
        rows.append({
            "n": n,
            "frequencies": fs,
            "expected_gcd": expected_gcd,
            "measured_gcd": int(g),
            "certified": bool(ok),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Closure audit: composite moduli and systematic phase error")
    p.add_argument("--n", nargs="+", type=int, default=[100, 105])
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=2200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--max-increment", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--lengths", nargs="+", type=int, default=[16, 64, 256, 1024])
    p.add_argument("--random-start", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = []
    for n in args.n:
        for seed in args.seeds:
            rows.append(run_one(
                n=n, modes=args.modes, seed=seed,
                train_length=args.train_length, train_steps=args.train_steps,
                batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                max_increment=args.max_increment, lr=args.lr,
                random_start=args.random_start, lengths=args.lengths,
            ))
    payload = {
        "config": vars(args),
        "certificate_controls": certificate_controls(),
        "results": [asdict(x) for x in rows],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
