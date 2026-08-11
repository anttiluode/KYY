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
MODULE_NAME = "dihedral_coset_recenter_for_runtime"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "map" / "dihedral_coset_recenter_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
coset = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = coset
SPEC.loader.exec_module(coset)
base = coset.base


def apply_branch_conditioned_phase_batch(
    h: torch.Tensor,
    branch: torch.Tensor,
    phi: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Apply +phi on rotation coset and -phi on reflected coset.

    h: [batch,modes,2]
    branch: [batch], 0 for C_n and 1 for s C_n
    """
    p = torch.as_tensor(phi, dtype=h.dtype, device=h.device).reshape(1, -1)
    sign = torch.where(branch == 0, 1.0, -1.0).to(h.dtype).reshape(-1, 1)
    p = sign * p
    c, s = torch.cos(p), torch.sin(p)
    x, y = h[..., 0], h[..., 1]
    return torch.stack((c * x - s * y, s * x + c * y), dim=-1)


def compiled_forward(
    model: base.DihedralHarmonicTracker,
    tokens: torch.Tensor,
    phi: np.ndarray,
    *,
    angle_error: float = 0.0,
) -> torch.Tensor:
    """Run the legalized tracker with an explicit one-bit C2 quotient sidecar.

    The sidecar is not derived from the target label. It is updated directly
    from the input generator stream:

        rotation token: q <- q
        reflection s:   q <- 1-q

    The bit selects the sign of the readout-only phase correction.  It never
    stores the C_n rotation coordinate.
    """
    bsz, length = tokens.shape
    h = model.h0.unsqueeze(0).expand(bsz, -1, -1)
    branch = torch.zeros(bsz, dtype=torch.long, device=tokens.device)
    outs: list[torch.Tensor] = []
    for t in range(length):
        tok = tokens[:, t]
        h = model.step(h, tok, angle_error=angle_error)
        branch = torch.where(tok == model.n, 1 - branch, branch)
        hp = apply_branch_conditioned_phase_batch(h, branch, phi)
        outs.append(model.readout(hp.reshape(bsz, -1)))
    return torch.stack(outs, dim=1)


def raw_legalized_forward(
    model: base.DihedralHarmonicTracker,
    tokens: torch.Tensor,
    *,
    angle_error: float = 0.0,
) -> torch.Tensor:
    return model(tokens, angle_error=angle_error)


def evaluate_runtime(
    model: base.DihedralHarmonicTracker,
    phi: np.ndarray,
    *,
    n: int,
    lengths: list[int],
    batch_size: int,
    max_increment: int,
    reflection_probability: float,
    random_start: bool,
    angle_error: float,
) -> tuple[dict[str, float], dict[str, float]]:
    raw: dict[str, float] = {}
    compiled: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for length in lengths:
            x, y = base.generate_batch(
                n,
                batch_size,
                length,
                max_increment,
                reflection_probability,
                random_start=random_start,
            )
            p0 = raw_legalized_forward(model, x, angle_error=angle_error).argmax(dim=-1)
            p1 = compiled_forward(model, x, phi, angle_error=angle_error).argmax(dim=-1)
            raw[str(length)] = float((p0 == y).float().mean().item())
            compiled[str(length)] = float((p1 == y).float().mean().item())
    return raw, compiled


@dataclass
class RuntimeRun:
    seed: int
    pre_relation_defect: float
    projected_frequencies: list[int]
    raw_orbit_accuracy: float
    compiled_orbit_accuracy: float
    compiled_orbit_min_margin: float
    raw_clean_accuracy: dict[str, float]
    compiled_clean_accuracy: dict[str, float]
    raw_eta_1e3_accuracy: dict[str, float]
    compiled_eta_1e3_accuracy: dict[str, float]
    phase_vector_norm: float
    sidecar_bits: int


def train_and_runtime_probe(
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
    lengths: list[int],
) -> RuntimeRun:
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
    phi = coset.coset_midpoint_phase_vector(n, learned, projected)

    # Compile the actual recurrent operator. The learned readout remains frozen.
    with torch.no_grad():
        model.angles.copy_(torch.tensor(projected, dtype=model.angles.dtype))

    z_projected = base.orbit_prototypes(n, projected, model.h0.detach().cpu())
    W = model.readout.weight.detach().cpu()
    b = model.readout.bias.detach().cpu()
    raw_orbit, _, _ = base.readout_metrics(z_projected, W, b)
    z_compiled = coset.coset_midpoint_recenter(n, z_projected, learned, projected)
    comp_orbit, comp_margin, _ = base.readout_metrics(z_compiled, W, b)

    raw_clean, compiled_clean = evaluate_runtime(
        model,
        phi,
        n=n,
        lengths=lengths,
        batch_size=eval_batch_size,
        max_increment=max_increment,
        reflection_probability=reflection_probability,
        random_start=random_start,
        angle_error=0.0,
    )
    raw_eta, compiled_eta = evaluate_runtime(
        model,
        phi,
        n=n,
        lengths=lengths,
        batch_size=eval_batch_size,
        max_increment=max_increment,
        reflection_probability=reflection_probability,
        random_start=random_start,
        angle_error=1e-3,
    )

    return RuntimeRun(
        seed=seed,
        pre_relation_defect=float(defect),
        projected_frequencies=[int(x) for x in frequencies.tolist()],
        raw_orbit_accuracy=float(raw_orbit),
        compiled_orbit_accuracy=float(comp_orbit),
        compiled_orbit_min_margin=float(comp_margin),
        raw_clean_accuracy=raw_clean,
        compiled_clean_accuracy=compiled_clean,
        raw_eta_1e3_accuracy=raw_eta,
        compiled_eta_1e3_accuracy=compiled_eta,
        phase_vector_norm=float(np.linalg.norm(phi)),
        sidecar_bits=1,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Live D_n quotient-bit port compiler validation")
    p.add_argument("--n", type=int, default=101)
    p.add_argument("--modes", type=int, default=8)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--train-length", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=2200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-increment", type=int, default=4)
    p.add_argument("--reflection-probability", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--random-start", action="store_true")
    p.add_argument("--lengths", nargs="+", type=int, default=[16,64,256,1024])
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    rows = [
        train_and_runtime_probe(
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
            lengths=args.lengths,
        )
        for seed in args.seeds
    ]
    payload = {"config": vars(args), "results": [asdict(x) for x in rows]}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("seed raw-orbit compiled-orbit raw-L1024 compiled-L1024 compiled-eta1e3-L1024 margin")
    for x in rows:
        print(
            f"{x.seed:4d} {x.raw_orbit_accuracy:9.3f} {x.compiled_orbit_accuracy:14.3f} "
            f"{x.raw_clean_accuracy['1024']:10.3f} {x.compiled_clean_accuracy['1024']:15.3f} "
            f"{x.compiled_eta_1e3_accuracy['1024']:22.3f} {x.compiled_orbit_min_margin:+8.3f}"
        )


if __name__ == "__main__":
    main()
