from __future__ import annotations

import argparse
import json

import torch

from kyy.tasks import _GENERATORS, _PERMS, _PERM_TO_ID, _compose, generate_batch


def permutation_matrix(p: tuple[int, ...], dtype=torch.float64) -> torch.Tensor:
    """P e_i = e_{p(i)} for an image-tuple permutation p."""
    n = len(p)
    P = torch.zeros((n, n), dtype=dtype)
    for i, pi in enumerate(p):
        P[pi, i] = 1.0
    return P


def local_adjacent_swap(n: int, i: int, dtype=torch.float64) -> torch.Tensor:
    """Swap coordinates i and i+1: one nearest-neighbour 2-port reflection."""
    P = torch.eye(n, dtype=dtype)
    P[i, i] = 0.0
    P[i + 1, i + 1] = 0.0
    P[i, i + 1] = 1.0
    P[i + 1, i] = 1.0
    return P


def build_oracle(dtype=torch.float64):
    """Exact 3-channel representation of KYY's S3 transition task.

    This is not a new construction.  It is the natural permutation-matrix
    representation used in state-tracking theory, specialized to KYY's task and
    factored into nearest-neighbour swaps to expose the local implementation cost.
    """
    n = 3
    v0 = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    mats = [permutation_matrix(g, dtype=dtype) for g in _GENERATORS]

    # Explicit local factorizations on the path 0--1--2.
    s01 = local_adjacent_swap(3, 0, dtype=dtype)
    s12 = local_adjacent_swap(3, 1, dtype=dtype)
    local_words = [
        [],          # identity
        [s01],       # (0 1)
        [s12, s01],  # applied in this order: s01 @ s12 = cycle (1,2,0)
    ]

    factored = []
    for word in local_words:
        A = torch.eye(n, dtype=dtype)
        for gate in word:
            A = gate @ A
        factored.append(A)

    # The cycle word above is written in application order.  Check/fix the exact
    # target convention explicitly rather than relying on prose about composition.
    if not torch.equal(factored[2], mats[2]):
        local_words[2] = [s01, s12]
        A = torch.eye(n, dtype=dtype)
        for gate in local_words[2]:
            A = gate @ A
        factored[2] = A

    for A, B in zip(factored, mats):
        assert torch.equal(A, B)

    prototypes = torch.stack([permutation_matrix(p, dtype=dtype) @ v0 for p in _PERMS])
    # Since every prototype is a permutation of the same distinct vector, all
    # have equal norm and dot-product nearest-template decoding is exact.
    readout = prototypes.clone()
    return v0, mats, local_words, prototypes, readout


def run_tokens(tokens: torch.Tensor) -> torch.Tensor:
    v0, mats, _, _, readout = build_oracle()
    bsz, length = tokens.shape
    h = v0.unsqueeze(0).expand(bsz, -1).clone()
    y = torch.empty((bsz, length), dtype=torch.long)
    for t in range(length):
        tok = tokens[:, t]
        next_h = torch.empty_like(h)
        for k, A in enumerate(mats):
            mask = tok == k
            if mask.any():
                next_h[mask] = h[mask] @ A.T
        h = next_h
        y[:, t] = (h @ readout.T).argmax(dim=-1)
    return y


def exhaustive_relation_checks() -> dict[str, bool]:
    _, mats, words, _, _ = build_oracle()
    E, S, R = mats
    I = torch.eye(3, dtype=E.dtype)
    return {
        "identity_exact": bool(torch.equal(E, I)),
        "s_squared_exact": bool(torch.equal(S @ S, I)),
        "r_cubed_exact": bool(torch.equal(R @ R @ R, I)),
        "srs_equals_r_inverse": bool(torch.equal(S @ R @ S, R.T)),
        "all_local_factorizations_exact": all(
            torch.equal(
                (lambda word: _word_matrix(word, 3, E.dtype))(word),
                mats[i],
            )
            for i, word in enumerate(words)
        ),
    }


def _word_matrix(word: list[torch.Tensor], n: int, dtype) -> torch.Tensor:
    A = torch.eye(n, dtype=dtype)
    for gate in word:
        A = gate @ A
    return A


def main() -> None:
    p = argparse.ArgumentParser(description="Exact local oracle for KYY perm3")
    p.add_argument("--length", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    x, target = generate_batch("perm3", args.batch_size, args.length, device="cpu")
    pred = run_tokens(x)
    checks = exhaustive_relation_checks()
    _, _, words, _, _ = build_oracle()
    payload = {
        "state_channels": 3,
        "trainable_recurrence_parameters": 0,
        "primitive": "nearest-neighbour 2-port swap/reflection",
        "path": "0--1--2",
        "token_local_depths": [len(w) for w in words],
        "max_local_depth_per_token": max(len(w) for w in words),
        "max_primitive_wire_span": 1,
        "length": args.length,
        "batch_size": args.batch_size,
        "accuracy": float((pred == target).double().mean()),
        "final_accuracy": float((pred[:, -1] == target[:, -1]).double().mean()),
        "relations": checks,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
