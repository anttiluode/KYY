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
MODULE_NAME = "harmonic_training_probe_for_certificate"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "harmonic_training_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
base = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = base
SPEC.loader.exec_module(base)


def phase_prototypes(n: int, angles: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Return the normalized real phase-bank prototype for every state in C_n."""
    a = torch.as_tensor(angles, dtype=torch.float64).reshape(1, -1)
    s = torch.arange(n, dtype=torch.float64).reshape(-1, 1)
    phase = s * a
    scale = 1.0 / math.sqrt(a.shape[1])
    return torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1).reshape(n, -1) * scale


@dataclass(frozen=True)
class PreservationCertificate:
    pre_prototype_accuracy: float
    projected_prototype_accuracy: float
    pre_min_true_margin: float
    projected_min_true_margin: float
    max_snap_distance: float
    mean_snap_distance: float
    cauchy_min_slack: float
    cauchy_certified: bool
    exhaustive_orbit_certified: bool


def decoder_preservation_certificate(
    n: int,
    learned_angles: np.ndarray | torch.Tensor,
    projected_angles: np.ndarray | torch.Tensor,
    readout_weight: np.ndarray | torch.Tensor,
    readout_bias: np.ndarray | torch.Tensor,
) -> PreservationCertificate:
    """Audit whether an unchanged linear decoder survives operator legalization.

    The finite-orbit check is exact for this symbolic C_n model: if all n
    projected prototypes have the correct class with positive pairwise margin,
    the exact projected recurrence cannot encounter another clean symbolic state.

    The Cauchy certificate is stricter.  For every state s and competitor j,

        m_sj > ||w_s-w_j|| ||z*_s-z_s||

    is sufficient to guarantee the old decision survives without evaluating the
    projected logits directly.  Its value is mainly as a compiler-style local
    perturbation certificate; it may reject legalizations that actually work.
    """
    z = phase_prototypes(n, learned_angles)
    zp = phase_prototypes(n, projected_angles)
    W = torch.as_tensor(readout_weight, dtype=torch.float64)
    b = torch.as_tensor(readout_bias, dtype=torch.float64).reshape(-1)
    if W.shape != (n, z.shape[1]):
        raise ValueError(f"expected readout weight {(n, z.shape[1])}, got {tuple(W.shape)}")
    if b.shape != (n,):
        raise ValueError(f"expected readout bias {(n,)}, got {tuple(b.shape)}")

    logits = z @ W.T + b
    logits_p = zp @ W.T + b
    labels = torch.arange(n)
    pred = logits.argmax(dim=-1)
    pred_p = logits_p.argmax(dim=-1)

    rows = torch.arange(n)
    true = logits[rows, labels]
    true_p = logits_p[rows, labels]

    competitor = logits.clone()
    competitor_p = logits_p.clone()
    competitor[rows, labels] = -torch.inf
    competitor_p[rows, labels] = -torch.inf
    pre_margin = true - competitor.max(dim=-1).values
    projected_margin = true_p - competitor_p.max(dim=-1).values

    delta = zp - z
    snap = torch.linalg.vector_norm(delta, dim=-1)

    # Pairwise pre-snap margins m_sj and the Cauchy perturbation bound.
    # Shapes: diff_w[s,j,:], pair_margin[s,j].
    diff_w = W[:, None, :] - W[None, :, :]
    pair_norm = torch.linalg.vector_norm(diff_w, dim=-1)
    pair_margin = true[:, None] - logits
    slack = pair_margin - pair_norm * snap[:, None]
    slack[rows, labels] = torch.inf
    min_slack = float(slack.min().item())

    projected_min_margin = float(projected_margin.min().item())
    return PreservationCertificate(
        pre_prototype_accuracy=float((pred == labels).double().mean().item()),
        projected_prototype_accuracy=float((pred_p == labels).double().mean().item()),
        pre_min_true_margin=float(pre_margin.min().item()),
        projected_min_true_margin=projected_min_margin,
        max_snap_distance=float(snap.max().item()),
        mean_snap_distance=float(snap.mean().item()),
        cauchy_min_slack=min_slack,
        cauchy_certified=bool(min_slack > 0.0),
        exhaustive_orbit_certified=bool(
            torch.equal(pred_p, labels) and projected_min_margin > 0.0
        ),
    )


@dataclass
class CertificateRun:
    seed: int
    n: int
    modes: int
    train_length: int
    steps: int
    state_relation_defect: float
    projected_frequencies: list[int]
    projected_orbit_noise_radius: float
    clean_accuracy: dict[str, float]
    projected_clean_accuracy: dict[str, float]
    certificate: dict[str, float | bool]


def train_and_certify(
    *,
    n: int,
    modes: int,
    seed: int,
    train_length: int,
    test_lengths: list[int],
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    max_increment: int,
    lr: float,
    angle_error: float,
    random_start: bool,
) -> CertificateRun:
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
        test_lengths,
        eval_batch_size,
        max_increment,
        0.0,
        random_start=random_start,
    )

    learned_angles = model.angles.detach().cpu().numpy().astype(np.float64)
    _, state_rel = base.relation_defects(n, learned_angles)
    projected_angles, frequencies = base.project_angles_to_characters(n, learned_angles)
    radius, _ = base.character_margin(n, frequencies)

    cert = decoder_preservation_certificate(
        n,
        learned_angles,
        projected_angles,
        model.readout.weight.detach().cpu(),
        model.readout.bias.detach().cpu(),
    )

    original = model.angles.detach().clone()
    with torch.no_grad():
        model.angles.copy_(
            torch.tensor(
                projected_angles,
                dtype=model.angles.dtype,
                device=model.angles.device,
            )
        )
    projected_clean, _ = base.evaluate(
        model,
        n,
        test_lengths,
        eval_batch_size,
        max_increment,
        0.0,
        random_start=random_start,
    )
    with torch.no_grad():
        model.angles.copy_(original)

    return CertificateRun(
        seed=seed,
        n=n,
        modes=modes,
        train_length=train_length,
        steps=steps,
        state_relation_defect=float(state_rel),
        projected_frequencies=[int(x) for x in frequencies.tolist()],
        projected_orbit_noise_radius=float(radius),
        clean_accuracy=clean,
        projected_clean_accuracy=projected_clean,
        certificate=asdict(cert),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Finite-orbit decoder-preservation certificate for cyclic operator legalization"
    )
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--test-lengths", nargs="+", type=int, default=[16, 64, 256, 1024])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-increment", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--angle-error", type=float, default=1e-3)
    p.add_argument("--random-start", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = [
        train_and_certify(
            n=args.n,
            modes=args.modes,
            seed=seed,
            train_length=args.train_length,
            test_lengths=args.test_lengths,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            max_increment=args.max_increment,
            lr=args.lr,
            angle_error=args.angle_error,
            random_start=args.random_start,
        )
        for seed in args.seeds
    ]
    payload = {"config": vars(args), "results": [asdict(row) for row in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("seed rel-def preL16 postL16 orbit-cert cauchy-cert min-post-margin min-slack")
    for row in rows:
        cert = row.certificate
        print(
            f"{row.seed:4d} {row.state_relation_defect:7.4f} "
            f"{row.clean_accuracy[str(args.train_length)]:7.3f} "
            f"{row.projected_clean_accuracy[str(args.train_length)]:8.3f} "
            f"{str(cert['exhaustive_orbit_certified']):>10s} "
            f"{str(cert['cauchy_certified']):>11s} "
            f"{cert['projected_min_true_margin']:15.5f} "
            f"{cert['cauchy_min_slack']:9.5f}"
        )


if __name__ == "__main__":
    main()
