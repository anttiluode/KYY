from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

MERGE = 4
SCALE = 5
N = 4
C4 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=torch.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_orthogonal(seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return torch.tensor(q, dtype=torch.float32)


def merge_q(q: torch.Tensor) -> torch.Tensor:
    return torch.where(q < 2, torch.zeros_like(q), torch.full_like(q, 2))


def generate_batch(batch: int, length: int, merge_probability: float, scale_probability: float):
    tok = torch.randint(0, 4, (batch, length))
    r = torch.rand(batch, length)
    tok[r < merge_probability] = MERGE
    tok[(r >= merge_probability) & (r < merge_probability + scale_probability)] = SCALE

    q0 = torch.randint(0, 4, (batch,))
    a0 = torch.empty(batch).uniform_(-1.5, 1.5)
    q = q0.clone()
    a = a0.clone()
    q_targets: list[torch.Tensor] = []
    a_targets: list[torch.Tensor] = []
    for t in range(length):
        x = tok[:, t]
        q = torch.where(x == MERGE, merge_q(q), torch.where(x == SCALE, q, (q + x) % 4))
        a = torch.where(x == SCALE, 0.9 * a, a)
        q_targets.append(q.clone())
        a_targets.append(a.clone())
    return tok, q0, a0, torch.stack(q_targets, 1), torch.stack(a_targets, 1)


class SoftFiberTracker(nn.Module):
    """Entangled 3D latent: C4 digital base + one continuous analog fiber."""

    def __init__(self, seed: int):
        super().__init__()
        self.register_buffer("P", random_orthogonal(1000 + seed))
        rng = np.random.default_rng(5000 + seed)
        self.theta = nn.Parameter(torch.tensor(math.pi / 2 + rng.uniform(-0.12, 0.12), dtype=torch.float32))
        eps0 = 0.12
        self.eps_logit = nn.Parameter(torch.tensor(math.log(eps0 / (1 - eps0)) + rng.normal(scale=0.15), dtype=torch.float32))
        self.r0 = nn.Parameter(torch.tensor(rng.normal(scale=0.025), dtype=torch.float32))
        self.r1 = nn.Parameter(torch.tensor(rng.normal(scale=0.025), dtype=torch.float32))
        self.gain = nn.Parameter(torch.tensor(0.99 + rng.normal(scale=0.01), dtype=torch.float32))
        self.logscale = nn.Parameter(torch.tensor(math.log(0.91), dtype=torch.float32))
        self.qread = nn.Linear(3, 4)
        self.aread = nn.Linear(3, 1)
        with torch.no_grad():
            prototypes = torch.cat((C4, torch.zeros(4, 1)), 1) @ self.P.T
            self.qread.weight.copy_(3.0 * prototypes)
            self.qread.bias.zero_()
            v = torch.tensor([0.0, 0.0, 1.0]) @ self.P.T
            self.aread.weight.copy_(v.view(1, 3))
            self.aread.bias.zero_()

    def canonical_merge(self) -> torch.Tensor:
        eps = torch.sigmoid(self.eps_logit)
        z = eps * 0.0
        one = z + 1.0
        return torch.stack(
            (
                torch.stack((one, one, z)),
                torch.stack((z, eps, z)),
                torch.stack((self.r0, self.r1, self.gain)),
            )
        )

    def init_hidden(self, q0: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
        canonical = torch.cat((C4[q0], a0[:, None]), 1)
        return canonical @ self.P.T

    def step(self, h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        canonical = h @ self.P
        x, y, a = canonical[:, 0], canonical[:, 1], canonical[:, 2]
        inc = torch.where(tok < 4, tok, torch.zeros_like(tok))
        phase = inc.float() * self.theta
        c, s = torch.cos(phase), torch.sin(phase)
        rotated = torch.stack((c * x - s * y, s * x + c * y, a), 1) @ self.P.T
        merged = (canonical @ self.canonical_merge().T) @ self.P.T
        scaled = torch.stack((x, y, torch.exp(self.logscale) * a), 1) @ self.P.T
        return torch.where((tok == MERGE)[:, None], merged, torch.where((tok == SCALE)[:, None], scaled, rotated))

    def forward(self, tok: torch.Tensor, q0: torch.Tensor, a0: torch.Tensor):
        h = self.init_hidden(q0, a0)
        logits: list[torch.Tensor] = []
        analog: list[torch.Tensor] = []
        for t in range(tok.shape[1]):
            h = self.step(h, tok[:, t])
            logits.append(self.qread(h))
            analog.append(self.aread(h).squeeze(-1))
        return torch.stack(logits, 1), torch.stack(analog, 1)


def train_model(seed: int, steps: int, length: int, batch: int, merge_probability: float, scale_probability: float):
    seed_everything(seed)
    model = SoftFiberTracker(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        tok, q0, a0, qy, ay = generate_batch(batch, length, merge_probability, scale_probability)
        logits, analog = model(tok, q0, a0)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 4), qy.reshape(-1))
        loss = loss + 0.2 * nn.functional.mse_loss(analog, ay)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def fiber_basis(model: SoftFiberTracker):
    p = model.P.detach().to(torch.float64)
    c0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64) @ p.T
    c1 = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64) @ p.T
    v = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64) @ p.T
    return c0, c1, v


def synthesize_exact_operators(model: SoftFiberTracker):
    """Compile exact operators from two digital generators plus the analog tangent."""
    c0, c1, v = fiber_basis(model)
    x = torch.stack((c0, c1, v), 1)
    inv = torch.linalg.inv(x)
    cycle = torch.stack((c1, -c0, v), 1) @ inv
    merge = torch.stack((c0, c0, v), 1) @ inv
    scale = torch.stack((c0, c1, 0.9 * v), 1) @ inv
    return cycle, merge, scale


def exact_ports(model: SoftFiberTracker):
    c0, c1, v = fiber_basis(model)
    prototypes = torch.stack((c0, c1, -c0, -c1))
    wq = 3.0 * prototypes
    bq = torch.zeros(4, dtype=torch.float64)
    x = torch.stack((c0, c1, v), 1)
    wa = torch.linalg.solve(x.T, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    return wq, bq, wa


def nearest_fiber(model: SoftFiberTracker, h: torch.Tensor):
    c0, c1, v = fiber_basis(model)
    centers = torch.stack((c0, c1, -c0, -c1)).to(h)
    _, _, wa = exact_ports(model)
    wa = wa.to(h)
    v = v.to(h)
    a = h @ wa
    recon = centers[None, :, :] + a[:, None, None] * v[None, None, :]
    q = torch.sum((h[:, None, :] - recon) ** 2, dim=-1).argmin(dim=-1)
    return centers[q] + a[:, None] * v, q, a


@torch.no_grad()
def reified_runtime(model: SoftFiberTracker, tok: torch.Tensor, q0: torch.Tensor, a0: torch.Tensor):
    h = model.init_hidden(q0, a0)
    qs: list[torch.Tensor] = []
    aa: list[torch.Tensor] = []
    for t in range(tok.shape[1]):
        h = model.step(h, tok[:, t])
        h, q, a = nearest_fiber(model, h)
        qs.append(q)
        aa.append(a)
    return torch.stack(qs, 1), torch.stack(aa, 1)


@torch.no_grad()
def compiled_runtime(model: SoftFiberTracker, tok: torch.Tensor, q0: torch.Tensor, a0: torch.Tensor):
    cycle, merge, scale = synthesize_exact_operators(model)
    powers = torch.stack([torch.linalg.matrix_power(cycle, k) for k in range(4)])
    h = model.init_hidden(q0, a0).to(torch.float64)
    wq, bq, wa = exact_ports(model)
    logits: list[torch.Tensor] = []
    analog: list[torch.Tensor] = []
    for t in range(tok.shape[1]):
        x = tok[:, t]
        inc = torch.where(x < 4, x, torch.zeros_like(x))
        rotated = torch.einsum("bij,bj->bi", powers[inc], h)
        merged = h @ merge.T
        scaled = h @ scale.T
        h = torch.where((x == MERGE)[:, None], merged, torch.where((x == SCALE)[:, None], scaled, rotated))
        logits.append(h @ wq.T + bq)
        analog.append(h @ wa)
    return torch.stack(logits, 1), torch.stack(analog, 1)


def metrics(q_pred: torch.Tensor, a_pred: torch.Tensor, qy: torch.Tensor, ay: torch.Tensor):
    return {
        "q_accuracy": float((q_pred == qy).to(torch.float64).mean().item()),
        "analog_rmse": float(torch.sqrt(torch.mean((a_pred.to(torch.float64) - ay.to(torch.float64)) ** 2)).item()),
    }


@dataclass
class Run:
    seed: int
    learned_theta_error: float
    learned_merge_epsilon: float
    learned_merge_full_rank: bool
    learned_merge_kernel_residual: float
    learned_fiber_preservation_defect: float
    learned_scale_error: float
    learned: dict[str, dict[str, float]]
    fiber_reified: dict[str, dict[str, float]]
    compiled: dict[str, dict[str, float]]
    compiled_cycle_relation_defect: float
    compiled_merge_kernel_defect: float
    compiled_fiber_defect: float


def run_one(seed: int, args) -> Run:
    model = train_model(seed, args.steps, args.train_length, args.batch_size, args.merge_probability, args.scale_probability)
    c0, c1, v = fiber_basis(model)
    bc = model.canonical_merge().detach().to(torch.float64)
    b_latent = model.P.to(torch.float64) @ bc @ model.P.to(torch.float64).T
    learned_kernel = float(torch.linalg.vector_norm(b_latent @ (c0 - c1)).item())
    learned_fiber = float(torch.linalg.vector_norm(b_latent @ v - v).item())
    cycle, merge, _ = synthesize_exact_operators(model)
    learned: dict[str, dict[str, float]] = {}
    reified: dict[str, dict[str, float]] = {}
    compiled: dict[str, dict[str, float]] = {}
    for length in args.test_lengths:
        tok, q0, a0, qy, ay = generate_batch(args.eval_batch_size, length, args.merge_probability, args.scale_probability)
        with torch.no_grad():
            logits, analog = model(tok, q0, a0)
            rq, ra = reified_runtime(model, tok, q0, a0)
            clogits, ca = compiled_runtime(model, tok, q0, a0)
        learned[str(length)] = metrics(logits.argmax(-1), analog, qy, ay)
        reified[str(length)] = metrics(rq, ra, qy, ay)
        compiled[str(length)] = metrics(clogits.argmax(-1), ca, qy, ay)
    return Run(
        seed=seed,
        learned_theta_error=float(model.theta.detach().item() - math.pi / 2),
        learned_merge_epsilon=float(torch.sigmoid(model.eps_logit).detach().item()),
        learned_merge_full_rank=bool(torch.linalg.matrix_rank(b_latent).item() == 3),
        learned_merge_kernel_residual=learned_kernel,
        learned_fiber_preservation_defect=learned_fiber,
        learned_scale_error=float(torch.exp(model.logscale).detach().item() - 0.9),
        learned=learned,
        fiber_reified=reified,
        compiled=compiled,
        compiled_cycle_relation_defect=float(torch.linalg.matrix_norm(torch.linalg.matrix_power(cycle, 4) - torch.eye(3, dtype=torch.float64)).item()),
        compiled_merge_kernel_defect=float(torch.linalg.vector_norm(merge @ (c0 - c1)).item()),
        compiled_fiber_defect=float(torch.linalg.vector_norm(merge @ v - v).item()),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--test-lengths", nargs="+", type=int, default=[16, 64, 256, 1024])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--merge-probability", type=float, default=0.05)
    p.add_argument("--scale-probability", type=float, default=0.08)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    rows = [run_one(seed, args) for seed in args.seeds]
    payload = {"config": vars(args), "results": [asdict(row) for row in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("seed eps kernel fiber L1024 learned_q reified_q compiled_q learned_a reified_a compiled_a")
        for row in rows:
            k = str(max(args.test_lengths))
            print(f"{row.seed:3d} {row.learned_merge_epsilon:.4f} {row.learned_merge_kernel_residual:.3e} "
                  f"{row.learned_fiber_preservation_defect:.3e} "
                  f"{row.learned[k]['q_accuracy']:.6f} {row.fiber_reified[k]['q_accuracy']:.6f} "
                  f"{row.compiled[k]['q_accuracy']:.6f} {row.learned[k]['analog_rmse']:.3e} "
                  f"{row.fiber_reified[k]['analog_rmse']:.3e} {row.compiled[k]['analog_rmse']:.3e}")


if __name__ == "__main__":
    main()
