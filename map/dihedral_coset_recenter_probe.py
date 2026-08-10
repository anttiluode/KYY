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
MODULE_NAME = "dihedral_legalization_for_coset_recenter"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "dihedral_legalization_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
base = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = base
SPEC.loader.exec_module(base)


def row_rotation_block(phi: float) -> torch.Tensor:
    c, s = math.cos(float(phi)), math.sin(float(phi))
    # Row-vector convention: [x,y] @ Q is active rotation by +phi.
    return torch.tensor([[c, s], [-s, c]], dtype=torch.float64)


def apply_per_mode_phase(z: torch.Tensor, phi: np.ndarray) -> torch.Tensor:
    blocks = [row_rotation_block(p) for p in np.asarray(phi).reshape(-1)]
    return z @ torch.block_diag(*blocks)


def coset_midpoint_recenter(
    n: int,
    z_projected: torch.Tensor,
    learned_angles: np.ndarray,
    projected_angles: np.ndarray,
) -> torch.Tensor:
    """Zero-label D_n port recentering using the exact C2 quotient branch.

    State order is [r^k for k=0..n-1, s r^k for k=0..n-1].
    The rotation-angle snap error accumulates with opposite sign on the two
    cosets because s r s = r^-1.  Therefore the cyclic midpoint correction
    must also flip sign with reflection parity.

    The branch bit is not inferred from the class label: in a compiled D_n
    machine it is the exact one-dimensional quotient/sign representation,
    unchanged by r and toggled by s.
    """
    learned = np.asarray(learned_angles, dtype=np.float64)
    projected = np.asarray(projected_angles, dtype=np.float64)
    phi = -0.5 * (n - 1) * (projected - learned)
    phi = np.arctan2(np.sin(phi), np.cos(phi))
    rot = apply_per_mode_phase(z_projected[:n], phi)
    refl = apply_per_mode_phase(z_projected[n:], -phi)
    return torch.cat((rot, refl), dim=0)


def coset_midpoint_phase_vector(n: int, learned: np.ndarray, projected: np.ndarray) -> np.ndarray:
    phi = -0.5 * (n - 1) * (np.asarray(projected) - np.asarray(learned))
    return np.arctan2(np.sin(phi), np.cos(phi))


@dataclass
class CosetRun:
    seed: int
    pre_relation_defect: float
    projected_frequencies: list[int]
    learned_orbit_accuracy: float
    raw_projected_accuracy: float
    raw_projected_min_margin: float
    global_midpoint_accuracy: float
    global_midpoint_min_margin: float
    coset_midpoint_accuracy: float
    coset_midpoint_min_margin: float
    full_procrustes_accuracy: float
    full_procrustes_min_margin: float
    coset_alignment_error: float
    full_procrustes_alignment_error: float
    phase_vector_norm: float


def train_and_probe(
    *,
    n: int,
    modes: int,
    seed: int,
    train_length: int,
    train_steps: int,
    batch_size: int,
    max_increment: int,
    reflection_probability: float,
    lr: float,
    random_start: bool,
) -> CosetRun:
    base.seed_everything(seed)
    rng = np.random.default_rng(seed + 1009 * n)
    initial = rng.uniform(-math.pi, math.pi, size=modes)
    model = base.DihedralHarmonicTracker(n, initial, learn_angles=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(train_steps):
        x, y = base.generate_batch(
            n,
            batch_size,
            train_length,
            max_increment,
            reflection_probability,
            random_start=random_start,
        )
        logits = model(x)
        loss = criterion(logits.reshape(-1, 2 * n), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    learned = model.angles.detach().cpu().numpy().astype(np.float64)
    projected, frequencies = base.project_angles_to_dn_characters(n, learned)
    defect = base.rotation_relation_defect(n, learned)
    h0 = model.h0.detach().cpu()
    W = model.readout.weight.detach().cpu()
    b = model.readout.bias.detach().cpu()

    z_learned = base.orbit_prototypes(n, learned, h0)
    z_projected = base.orbit_prototypes(n, projected, h0)
    learned_acc, _, _ = base.readout_metrics(z_learned, W, b)
    raw_acc, raw_margin, _ = base.readout_metrics(z_projected, W, b)

    global_q = base.midpoint_mode_port(n, learned, projected)
    z_global = z_projected @ global_q
    global_acc, global_margin, _ = base.readout_metrics(z_global, W, b)

    z_coset = coset_midpoint_recenter(n, z_projected, learned, projected)
    coset_acc, coset_margin, _ = base.readout_metrics(z_coset, W, b)

    q_full = base.orthogonal_procrustes_port(z_projected, z_learned)
    z_full = z_projected @ q_full
    full_acc, full_margin, _ = base.readout_metrics(z_full, W, b)

    denom = max(float(torch.linalg.matrix_norm(z_learned).item()), 1e-12)
    coset_err = float(torch.linalg.matrix_norm(z_coset - z_learned).item()) / denom
    full_err = float(torch.linalg.matrix_norm(z_full - z_learned).item()) / denom
    phi = coset_midpoint_phase_vector(n, learned, projected)

    return CosetRun(
        seed=seed,
        pre_relation_defect=float(defect),
        projected_frequencies=[int(x) for x in frequencies.tolist()],
        learned_orbit_accuracy=float(learned_acc),
        raw_projected_accuracy=float(raw_acc),
        raw_projected_min_margin=float(raw_margin),
        global_midpoint_accuracy=float(global_acc),
        global_midpoint_min_margin=float(global_margin),
        coset_midpoint_accuracy=float(coset_acc),
        coset_midpoint_min_margin=float(coset_margin),
        full_procrustes_accuracy=float(full_acc),
        full_procrustes_min_margin=float(full_margin),
        coset_alignment_error=coset_err,
        full_procrustes_alignment_error=full_err,
        phase_vector_norm=float(np.linalg.norm(phi)),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Zero-label D_n coset-conditioned port recentering")
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=2200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-increment", type=int, default=4)
    p.add_argument("--reflection-probability", type=float, default=0.25)
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
            reflection_probability=args.reflection_probability,
            lr=args.lr,
            random_start=args.random_start,
        )
        for seed in args.seeds
    ]
    payload = {"config": vars(args), "results": [asdict(x) for x in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("seed rel-def learned raw global-mid coset-mid full-proc raw-margin coset-margin")
    for x in rows:
        print(
            f"{x.seed:4d} {x.pre_relation_defect:7.3f} {x.learned_orbit_accuracy:7.3f} "
            f"{x.raw_projected_accuracy:5.3f} {x.global_midpoint_accuracy:10.3f} "
            f"{x.coset_midpoint_accuracy:9.3f} {x.full_procrustes_accuracy:9.3f} "
            f"{x.raw_projected_min_margin:+10.4f} {x.coset_midpoint_min_margin:+12.4f}"
        )


if __name__ == "__main__":
    main()
