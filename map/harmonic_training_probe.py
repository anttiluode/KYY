from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prime_frequencies(n: int, k: int) -> np.ndarray:
    out: list[int] = []
    x = 2
    while len(out) < k:
        if all(x % p for p in range(2, int(math.sqrt(x)) + 1)):
            out.append(x % n)
        x += 1
    return np.asarray(out, dtype=np.int64)


def geometric_character_frequencies(n: int, k: int) -> np.ndarray:
    """RoPE-shaped spacing projected onto exact characters of C_n.

    This is *not* standard RoPE. It keeps the geometric-spacing idea while
    enforcing f in Z_n so the recurrent operator satisfies A^n = I exactly.
    """
    hi = max(1, n // 2)
    return np.rint(np.geomspace(1, hi, k)).astype(np.int64) % n


def character_margin(n: int, frequencies: np.ndarray) -> tuple[float, float]:
    f = np.asarray(frequencies, dtype=np.int64).reshape(-1)
    delta = np.arange(1, n, dtype=np.float64)[:, None]
    corr = np.mean(np.cos(2.0 * math.pi * delta * f[None, :] / n), axis=1)
    mu = float(np.max(corr))
    radius = 0.5 * math.sqrt(max(0.0, 2.0 * (1.0 - mu)))
    return radius, mu


def low_coherence_frequencies(n: int, k: int, trials: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    best_mu = float("inf")
    for _ in range(trials):
        f = rng.integers(1, n, size=k, dtype=np.int64)
        _, mu = character_margin(n, f)
        if mu < best_mu:
            best_mu = mu
            best = f.copy()
    assert best is not None
    return best


def standard_rope_angles(k: int, base: float = 10000.0) -> np.ndarray:
    """Standard RoPE angular increments theta_i=base^(-2i/d), d=2k."""
    i = np.arange(k, dtype=np.float64)
    return base ** (-2.0 * i / (2 * k))


def character_defect(n: int, angles: torch.Tensor | np.ndarray) -> float:
    """Mean distance to the nearest exact C_n character frequency.

    Exact modular recurrence requires n*theta/(2*pi) to be integer (mod n).
    This is an interpretable coordinate diagnostic; `relation_defects` below is
    the representation-invariant algebraic diagnostic used by the compiler view.
    """
    a = np.asarray(angles, dtype=np.float64)
    f = n * a / (2.0 * math.pi)
    return float(np.mean(np.abs(f - np.rint(f))))


def relation_defects(n: int, angles: torch.Tensor | np.ndarray) -> tuple[float, float]:
    """Return (operator, state) defect for the cyclic relation A^n = I.

    For a block-diagonal bank A=diag(R(theta_i)),

        ||A^n-I||_2 = max_i 2 |sin(n theta_i / 2)|.

    For the equal-amplitude normalized phase state used by this probe, the
    distance between histories whose total counts differ by exactly n is

        2 sqrt(mean_i sin^2(n theta_i / 2)).

    Both vanish for exact C_n characters, independent of the character basis.
    """
    a = np.asarray(angles, dtype=np.float64).reshape(-1)
    d = 2.0 * np.abs(np.sin(0.5 * n * a))
    return float(np.max(d)), float(np.sqrt(np.mean(d * d)))


def project_angles_to_characters(n: int, angles: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest per-mode projection onto exact characters of C_n.

    For this diagonal cyclic representation the legal set is simply the n-th
    roots of unity. Rounding n*theta/(2*pi) gives the closest character in
    angular distance for each independent mode.
    """
    a = np.asarray(angles, dtype=np.float64).reshape(-1)
    f = np.rint(n * a / (2.0 * math.pi)).astype(np.int64)
    projected = 2.0 * math.pi * f.astype(np.float64) / n
    return projected, np.mod(f, n)


class RotaryModTracker(nn.Module):
    """Tiny recurrent oscillator bank for modulo state tracking.

    Each input token is a nonnegative integer increment. Every mode rotates by
    increment * theta_i. Only the readout is learned for fixed schedules; the
    `learned` variant also learns theta_i. There is no drive, contraction, or
    hidden MLP that can rescue a bad recurrent orbit.
    """

    def __init__(self, n: int, angles: np.ndarray, learn_angles: bool = False):
        super().__init__()
        self.n = int(n)
        a = torch.tensor(np.asarray(angles), dtype=torch.float32)
        if learn_angles:
            self.angles = nn.Parameter(a.clone())
        else:
            self.register_buffer("angles", a)
        self.k = int(a.numel())
        h0 = torch.zeros(self.k, 2, dtype=torch.float32)
        h0[:, 0] = 1.0 / math.sqrt(self.k)
        self.register_buffer("h0", h0)
        self.readout = nn.Linear(2 * self.k, self.n)

    def forward(self, increments: torch.Tensor, angle_error: float = 0.0) -> torch.Tensor:
        bsz, length = increments.shape
        h = self.h0.unsqueeze(0).expand(bsz, -1, -1)
        outs: list[torch.Tensor] = []
        angles = self.angles + float(angle_error)
        for t in range(length):
            theta = increments[:, t].float().unsqueeze(-1) * angles.unsqueeze(0)
            c, s = torch.cos(theta), torch.sin(theta)
            x, y = h[..., 0], h[..., 1]
            h = torch.stack((c * x - s * y, s * x + c * y), dim=-1)
            outs.append(self.readout(h.reshape(bsz, -1)))
        return torch.stack(outs, dim=1)


def generate_batch(
    n: int,
    batch_size: int,
    length: int,
    max_increment: int,
    random_start: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate modulo-counting sequences.

    With `random_start`, the first increment is sampled uniformly from Z_n and
    subsequent increments remain local in [0,max_increment]. This exposes every
    symbolic class during short training without turning every recurrent step
    into a large jump. It is used for cross-modulus comparisons where a short
    small-increment trajectory would otherwise never visit many classes.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    x = torch.randint(0, max_increment + 1, (batch_size, length))
    if random_start:
        x[:, 0] = torch.randint(0, n, (batch_size,))
    y = torch.cumsum(x, dim=1).remainder(n)
    return x, y


@dataclass
class RunResult:
    variant: str
    seed: int
    n: int
    modes: int
    train_length: int
    steps: int
    random_start: bool
    clean_accuracy: dict[str, float]
    clean_final_accuracy: dict[str, float]
    drift_accuracy: dict[str, float]
    frequencies: list[int] | None
    orbit_noise_radius: float | None
    max_nontrivial_inner_product: float | None
    character_defect_before: float
    character_defect_after: float
    operator_relation_defect_before: float
    operator_relation_defect_after: float
    state_relation_defect_before: float
    state_relation_defect_after: float
    angles_after: list[float]
    projected_frequencies: list[int] | None
    projected_orbit_noise_radius: float | None
    projected_clean_accuracy: dict[str, float] | None
    projected_drift_accuracy: dict[str, float] | None
    projected_operator_relation_defect: float | None
    projected_state_relation_defect: float | None


def make_angles(variant: str, n: int, k: int, trials: int, seed: int) -> tuple[np.ndarray, np.ndarray | None, bool]:
    if variant == "low_coherence":
        f = low_coherence_frequencies(n, k, trials, seed + 1009 * n)
        return 2.0 * math.pi * f / n, f, False
    if variant == "random_characters":
        rng = np.random.default_rng(seed + 1009 * n)
        f = rng.integers(1, n, size=k, dtype=np.int64)
        return 2.0 * math.pi * f / n, f, False
    if variant == "prime_characters":
        f = prime_frequencies(n, k)
        return 2.0 * math.pi * f / n, f, False
    if variant == "geometric_characters":
        f = geometric_character_frequencies(n, k)
        return 2.0 * math.pi * f / n, f, False
    if variant == "rope":
        return standard_rope_angles(k), None, False
    if variant == "learned":
        rng = np.random.default_rng(seed + 1009 * n)
        return rng.uniform(-math.pi, math.pi, size=k), None, True
    raise KeyError(variant)


def evaluate(
    model: RotaryModTracker,
    n: int,
    lengths: list[int],
    batch_size: int,
    max_increment: int,
    angle_error: float,
    random_start: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    all_acc: dict[str, float] = {}
    final_acc: dict[str, float] = {}
    with torch.no_grad():
        for length in lengths:
            x, y = generate_batch(n, batch_size, length, max_increment, random_start=random_start)
            pred = model(x, angle_error=angle_error).argmax(dim=-1)
            all_acc[str(length)] = float((pred == y).float().mean())
            final_acc[str(length)] = float((pred[:, -1] == y[:, -1]).float().mean())
    return all_acc, final_acc


def evaluate_projected(
    model: RotaryModTracker,
    n: int,
    test_lengths: list[int],
    eval_batch_size: int,
    max_increment: int,
    angle_error: float,
    random_start: bool,
) -> tuple[list[int], float, dict[str, float], dict[str, float], float, float]:
    """Snap the trained recurrence to exact characters and evaluate zero-shot.

    The readout is left untouched. This intentionally asks whether algebraic
    legalization alone repairs rollout, not whether a subsequent fine-tune can.
    """
    original = model.angles.detach().clone()
    projected, f = project_angles_to_characters(n, original.cpu().numpy())
    with torch.no_grad():
        model.angles.copy_(torch.tensor(projected, device=model.angles.device, dtype=model.angles.dtype))
    clean, _ = evaluate(
        model, n, test_lengths, eval_batch_size, max_increment, 0.0, random_start=random_start
    )
    drift, _ = evaluate(
        model,
        n,
        test_lengths,
        eval_batch_size,
        max_increment,
        angle_error,
        random_start=random_start,
    )
    final_projected = model.angles.detach().cpu().numpy().astype(np.float64)
    op, state = relation_defects(n, final_projected)
    radius, _ = character_margin(n, f)
    with torch.no_grad():
        model.angles.copy_(original)
    return [int(x) for x in f.tolist()], radius, clean, drift, op, state


def train_one(
    variant: str,
    n: int,
    k: int,
    seed: int,
    train_length: int,
    test_lengths: list[int],
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    max_increment: int,
    lr: float,
    search_trials: int,
    angle_error: float,
    random_start: bool,
) -> RunResult:
    seed_everything(seed)
    angles, f, learn_angles = make_angles(variant, n, k, search_trials, seed)
    char_before = character_defect(n, angles)
    op_before, state_before = relation_defects(n, angles)
    model = RotaryModTracker(n, angles, learn_angles=learn_angles)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(steps):
        x, y = generate_batch(
            n, batch_size, train_length, max_increment, random_start=random_start
        )
        logits = model(x)
        loss = criterion(logits.reshape(-1, n), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    clean_acc, clean_final = evaluate(
        model,
        n,
        test_lengths,
        eval_batch_size,
        max_increment,
        0.0,
        random_start=random_start,
    )
    drift_acc, _ = evaluate(
        model,
        n,
        test_lengths,
        eval_batch_size,
        max_increment,
        angle_error,
        random_start=random_start,
    )
    final_angles = model.angles.detach().cpu().numpy().astype(np.float64)
    char_after = character_defect(n, final_angles)
    op_after, state_after = relation_defects(n, final_angles)

    radius = mu = None
    if f is not None:
        radius, mu = character_margin(n, f)

    projected_frequencies = None
    projected_radius = None
    projected_clean = None
    projected_drift = None
    projected_op = None
    projected_state = None
    if variant in {"learned", "rope"}:
        (
            projected_frequencies,
            projected_radius,
            projected_clean,
            projected_drift,
            projected_op,
            projected_state,
        ) = evaluate_projected(
            model,
            n,
            test_lengths,
            eval_batch_size,
            max_increment,
            angle_error,
            random_start=random_start,
        )

    return RunResult(
        variant=variant,
        seed=seed,
        n=n,
        modes=model.k,
        train_length=train_length,
        steps=steps,
        random_start=random_start,
        clean_accuracy=clean_acc,
        clean_final_accuracy=clean_final,
        drift_accuracy=drift_acc,
        frequencies=None if f is None else [int(x) for x in f.tolist()],
        orbit_noise_radius=radius,
        max_nontrivial_inner_product=mu,
        character_defect_before=char_before,
        character_defect_after=char_after,
        operator_relation_defect_before=op_before,
        operator_relation_defect_after=op_after,
        state_relation_defect_before=state_before,
        state_relation_defect_after=state_after,
        angles_after=[float(x) for x in final_angles.tolist()],
        projected_frequencies=projected_frequencies,
        projected_orbit_noise_radius=projected_radius,
        projected_clean_accuracy=projected_clean,
        projected_drift_accuracy=projected_drift,
        projected_operator_relation_defect=projected_op,
        projected_state_relation_defect=projected_state,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train-short/test-long probe for exact harmonic characters versus RoPE-like oscillator banks")
    p.add_argument("--n", type=int, default=31)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--variants", nargs="+", default=["low_coherence", "random_characters", "prime_characters", "geometric_characters", "rope", "learned"], choices=["low_coherence", "random_characters", "prime_characters", "geometric_characters", "rope", "learned"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--test-lengths", nargs="+", type=int, default=[16, 64, 256])
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--max-increment", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--search-trials", type=int, default=500)
    p.add_argument("--angle-error", type=float, default=1e-3)
    p.add_argument("--random-start", action="store_true", help="first step is a uniform random offset in Z_n; later increments remain local")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = [
        train_one(
            variant=v,
            n=args.n,
            k=args.modes,
            seed=s,
            train_length=args.train_length,
            test_lengths=args.test_lengths,
            steps=args.steps,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            max_increment=args.max_increment,
            lr=args.lr,
            search_trials=args.search_trials,
            angle_error=args.angle_error,
            random_start=args.random_start,
        )
        for v in args.variants
        for s in args.seeds
    ]
    payload = {"config": vars(args), "results": [asdict(r) for r in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("variant              seed  radius   relop    relstate clean@L / projected@L")
    for r in rows:
        rr = "-" if r.orbit_noise_radius is None else f"{r.orbit_noise_radius:.4f}"
        pairs = " ".join(
            f"{L}:{r.clean_accuracy[str(L)]:.3f}"
            + ("/-" if r.projected_clean_accuracy is None else f"/{r.projected_clean_accuracy[str(L)]:.3f}")
            for L in args.test_lengths
        )
        print(
            f"{r.variant:20s} {r.seed:4d} {rr:>7s} "
            f"{r.operator_relation_defect_after:8.4f} {r.state_relation_defect_after:8.4f}  {pairs}"
        )


if __name__ == "__main__":
    main()
