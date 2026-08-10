from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "dihedral_base_for_joint_legalization"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "dihedral_legalization_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
base = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = base
SPEC.loader.exec_module(base)


def rotation_matrix(theta: torch.Tensor | float, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    t = torch.as_tensor(theta, dtype=dtype)
    c, s = torch.cos(t), torch.sin(t)
    return torch.stack((torch.stack((c, -s)), torch.stack((s, c))))


def nearest_reflection(matrix: np.ndarray | torch.Tensor) -> np.ndarray:
    """Nearest 2x2 orthogonal matrix with determinant -1 in Frobenius norm."""
    m = np.asarray(matrix, dtype=np.float64).reshape(2, 2)
    u, _, vh = np.linalg.svd(m)
    q = u @ vh
    if np.linalg.det(q) > 0.0:
        u[:, -1] *= -1.0
        q = u @ vh
    return q


def project_reflections(matrices: np.ndarray | torch.Tensor) -> np.ndarray:
    a = np.asarray(matrices, dtype=np.float64)
    return np.stack([nearest_reflection(m) for m in a], axis=0)


def relation_defects(
    n: int,
    angles: np.ndarray | torch.Tensor,
    reflections: np.ndarray | torch.Tensor,
) -> tuple[float, float, float, float]:
    """Return rotation-order, reflection-involution, conjugation, orthogonality defects."""
    a = np.asarray(angles, dtype=np.float64).reshape(-1)
    b = np.asarray(reflections, dtype=np.float64).reshape(-1, 2, 2)
    if len(a) != len(b):
        raise ValueError("angle/reflection mode counts must match")
    eye = np.eye(2)
    rot_def = float(np.max(2.0 * np.abs(np.sin(0.5 * n * a))))
    inv_def = 0.0
    conj_def = 0.0
    orth_def = 0.0
    for theta, s in zip(a, b):
        c, si = math.cos(float(theta)), math.sin(float(theta))
        r = np.array([[c, -si], [si, c]], dtype=np.float64)
        rinv = r.T
        inv_def = max(inv_def, float(np.linalg.norm(s @ s - eye, ord="fro")))
        conj_def = max(conj_def, float(np.linalg.norm(s @ r @ s - rinv, ord="fro")))
        orth_def = max(orth_def, float(np.linalg.norm(s.T @ s - eye, ord="fro")))
    return rot_def, inv_def, conj_def, orth_def


class ApproxDihedralTracker(nn.Module):
    """D_n tracker whose reflection generator is learned and only approximately legal."""

    def __init__(
        self,
        n: int,
        angles: np.ndarray,
        *,
        reflection_scale: float,
        seed: int,
    ):
        super().__init__()
        self.n = int(n)
        a = torch.tensor(np.asarray(angles), dtype=torch.float32)
        self.angles = nn.Parameter(a.clone())
        self.modes = int(a.numel())
        self.register_buffer("h0", base.default_h0(self.modes))
        self.reflection_scale = float(reflection_scale)

        g = torch.Generator().manual_seed(seed + 1709)
        raw = 0.35 * torch.randn(self.modes, 2, 2, generator=g)
        self.reflection_raw = nn.Parameter(raw)
        self.readout = nn.Linear(2 * self.modes, 2 * self.n)

    def reflection_matrices(self) -> torch.Tensor:
        f = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=self.reflection_raw.dtype, device=self.reflection_raw.device)
        return f.unsqueeze(0) + self.reflection_scale * torch.tanh(self.reflection_raw)

    def step(self, h: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        ref = token == self.n
        inc = torch.where(ref, torch.zeros_like(token), token).float().unsqueeze(-1)
        theta = inc * self.angles.unsqueeze(0)
        c, s = torch.cos(theta), torch.sin(theta)
        x, y = h[..., 0], h[..., 1]
        rot = torch.stack((c * x - s * y, s * x + c * y), dim=-1)
        mats = self.reflection_matrices()
        refl = torch.einsum("kij,bkj->bki", mats, h)
        return torch.where(ref[:, None, None], refl, rot)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, length = tokens.shape
        h = self.h0.unsqueeze(0).expand(bsz, -1, -1)
        outs: list[torch.Tensor] = []
        for t in range(length):
            h = self.step(h, tokens[:, t])
            outs.append(self.readout(h.reshape(bsz, -1)))
        return torch.stack(outs, dim=1)


def canonical_orbit(
    n: int,
    angles: np.ndarray | torch.Tensor,
    reflections: np.ndarray | torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    """Canonical states [r^k h0, s r^k h0] without assuming exact relations."""
    a = torch.as_tensor(angles, dtype=torch.float64).reshape(1, -1)
    b = torch.as_tensor(reflections, dtype=torch.float64).reshape(-1, 2, 2)
    seed = h0.to(torch.float64).reshape(1, -1, 2)
    k = torch.arange(n, dtype=torch.float64).reshape(-1, 1)
    phase = k * a
    c, s = torch.cos(phase), torch.sin(phase)
    x0, y0 = seed[..., 0], seed[..., 1]
    x = c * x0 - s * y0
    y = s * x0 + c * y0
    rot = torch.stack((x, y), dim=-1)
    refl = torch.einsum("kij,nkj->nki", b, rot)
    return torch.cat((rot, refl), dim=0).reshape(2 * n, -1)


def orthogonal_procrustes(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cross = source.T @ target
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    return u @ vh


def block_procrustes(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape != target.shape or source.shape[1] % 2:
        raise ValueError("source/target must have equal even dimension")
    blocks = []
    for i in range(source.shape[1] // 2):
        blocks.append(orthogonal_procrustes(source[:, 2*i:2*i+2], target[:, 2*i:2*i+2]))
    return torch.block_diag(*blocks)


def quotient_block_port(source: torch.Tensor, target: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate per-mode orthogonal Procrustes maps for C_n and sC_n cosets."""
    return block_procrustes(source[:n], target[:n]), block_procrustes(source[n:], target[n:])


def quotient_full_port(source: torch.Tensor, target: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    return orthogonal_procrustes(source[:n], target[:n]), orthogonal_procrustes(source[n:], target[n:])


def apply_quotient_port(z: torch.Tensor, n: int, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    return torch.cat((z[:n] @ q0, z[n:] @ q1), dim=0)


def metrics(z: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> tuple[float, float, int]:
    return base.readout_metrics(z, W, b)


def train_model(
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
    reflection_scale: float,
) -> ApproxDihedralTracker:
    base.seed_everything(seed)
    rng = np.random.default_rng(seed + 1009 * n)
    initial = rng.uniform(-math.pi, math.pi, size=modes)
    model = ApproxDihedralTracker(
        n,
        initial,
        reflection_scale=reflection_scale,
        seed=seed,
    )
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
        loss = criterion(logits.reshape(-1, 2*n), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


@dataclass
class JointRun:
    seed: int
    n: int
    modes: int
    pre_train_length_accuracy: float
    pre_relation_defects: dict[str, float]
    post_relation_defects: dict[str, float]
    reflection_projection_distance: float
    angle_projection_distance: float
    learned_canonical_accuracy: float
    raw_legalized_accuracy: float
    raw_legalized_min_margin: float
    quotient_block_accuracy: float
    quotient_block_min_margin: float
    quotient_full_accuracy: float
    quotient_full_min_margin: float
    global_full_accuracy: float
    global_full_min_margin: float
    quotient_block_alignment_error: float
    quotient_full_alignment_error: float
    global_full_alignment_error: float


def train_and_probe(
    *,
    n: int,
    modes: int,
    seed: int,
    train_length: int,
    train_steps: int,
    batch_size: int,
    eval_batch_size: int,
    max_increment: int,
    reflection_probability: float,
    lr: float,
    random_start: bool,
    reflection_scale: float,
) -> JointRun:
    model = train_model(
        n=n,
        modes=modes,
        seed=seed,
        train_length=train_length,
        train_steps=train_steps,
        batch_size=batch_size,
        max_increment=max_increment,
        reflection_probability=reflection_probability,
        lr=lr,
        random_start=random_start,
        reflection_scale=reflection_scale,
    )

    model.eval()
    with torch.no_grad():
        x, y = base.generate_batch(
            n,
            eval_batch_size,
            train_length,
            max_increment,
            reflection_probability,
            random_start=random_start,
        )
        pre_acc = float((model(x).argmax(dim=-1) == y).float().mean().item())

    learned_angles = model.angles.detach().cpu().numpy().astype(np.float64)
    learned_ref = model.reflection_matrices().detach().cpu().numpy().astype(np.float64)
    projected_angles, _ = base.project_angles_to_dn_characters(n, learned_angles)
    projected_ref = project_reflections(learned_ref)

    pre_rel = relation_defects(n, learned_angles, learned_ref)
    post_rel = relation_defects(n, projected_angles, projected_ref)
    angle_dist = float(np.linalg.norm(projected_angles - learned_angles))
    ref_dist = float(np.linalg.norm(projected_ref - learned_ref))

    h0 = model.h0.detach().cpu()
    W = model.readout.weight.detach().cpu()
    b = model.readout.bias.detach().cpu()
    z_learned = canonical_orbit(n, learned_angles, learned_ref, h0)
    z_exact = canonical_orbit(n, projected_angles, projected_ref, h0)

    learned_acc, _, _ = metrics(z_learned, W, b)
    raw_acc, raw_margin, _ = metrics(z_exact, W, b)

    qb0, qb1 = quotient_block_port(z_exact, z_learned, n)
    z_qb = apply_quotient_port(z_exact, n, qb0, qb1)
    qb_acc, qb_margin, _ = metrics(z_qb, W, b)

    qf0, qf1 = quotient_full_port(z_exact, z_learned, n)
    z_qf = apply_quotient_port(z_exact, n, qf0, qf1)
    qf_acc, qf_margin, _ = metrics(z_qf, W, b)

    gf = orthogonal_procrustes(z_exact, z_learned)
    z_gf = z_exact @ gf
    gf_acc, gf_margin, _ = metrics(z_gf, W, b)

    denom = max(float(torch.linalg.matrix_norm(z_learned).item()), 1e-12)
    qb_err = float(torch.linalg.matrix_norm(z_qb-z_learned).item()) / denom
    qf_err = float(torch.linalg.matrix_norm(z_qf-z_learned).item()) / denom
    gf_err = float(torch.linalg.matrix_norm(z_gf-z_learned).item()) / denom

    names = ("rotation_order", "reflection_involution", "conjugation", "reflection_orthogonality")
    return JointRun(
        seed=seed,
        n=n,
        modes=modes,
        pre_train_length_accuracy=pre_acc,
        pre_relation_defects=dict(zip(names, map(float, pre_rel))),
        post_relation_defects=dict(zip(names, map(float, post_rel))),
        reflection_projection_distance=ref_dist,
        angle_projection_distance=angle_dist,
        learned_canonical_accuracy=float(learned_acc),
        raw_legalized_accuracy=float(raw_acc),
        raw_legalized_min_margin=float(raw_margin),
        quotient_block_accuracy=float(qb_acc),
        quotient_block_min_margin=float(qb_margin),
        quotient_full_accuracy=float(qf_acc),
        quotient_full_min_margin=float(qf_margin),
        global_full_accuracy=float(gf_acc),
        global_full_min_margin=float(gf_margin),
        quotient_block_alignment_error=qb_err,
        quotient_full_alignment_error=qf_err,
        global_full_alignment_error=gf_err,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Joint legalization of approximate D_n rotation and reflection generators")
    p.add_argument("--n", type=int, default=31)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-increment", type=int, default=4)
    p.add_argument("--reflection-probability", type=float, default=0.25)
    p.add_argument("--reflection-scale", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--random-start", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = [train_and_probe(
        n=args.n,
        modes=args.modes,
        seed=seed,
        train_length=args.train_length,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_increment=args.max_increment,
        reflection_probability=args.reflection_probability,
        lr=args.lr,
        random_start=args.random_start,
        reflection_scale=args.reflection_scale,
    ) for seed in args.seeds]

    payload = {"config": vars(args), "results": [asdict(x) for x in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("seed train rel-r rel-s2 rel-conj learned raw q-block q-full g-full")
    for x in rows:
        d=x.pre_relation_defects
        print(
            f"{x.seed:4d} {x.pre_train_length_accuracy:5.3f} "
            f"{d['rotation_order']:5.3f} {d['reflection_involution']:6.3f} {d['conjugation']:8.3f} "
            f"{x.learned_canonical_accuracy:7.3f} {x.raw_legalized_accuracy:5.3f} "
            f"{x.quotient_block_accuracy:7.3f} {x.quotient_full_accuracy:6.3f} {x.global_full_accuracy:6.3f}"
        )


if __name__ == "__main__":
    main()
