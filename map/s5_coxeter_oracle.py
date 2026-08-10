from __future__ import annotations

import argparse
import json
import random
from collections import Counter, deque
from itertools import permutations
from typing import Iterable

import numpy as np


Permutation = tuple[int, ...]


def identity(n: int) -> Permutation:
    return tuple(range(n))


def compose(left: Permutation, right: Permutation) -> Permutation:
    """Function composition left o right for image-tuple permutations."""
    return tuple(left[right[i]] for i in range(len(left)))


def simple_permutation(n: int, i: int) -> Permutation:
    """Coxeter generator s_i=(i,i+1)."""
    p = list(range(n))
    p[i], p[i + 1] = p[i + 1], p[i]
    return tuple(p)


def permutation_matrix(p: Permutation) -> np.ndarray:
    """P e_i = e_{p(i)}."""
    n = len(p)
    P = np.zeros((n, n), dtype=np.int64)
    for i, pi in enumerate(p):
        P[pi, i] = 1
    return P


def root_basis(n: int) -> np.ndarray:
    """Columns are the A_(n-1) simple roots alpha_i=e_i-e_(i+1)."""
    A = np.zeros((n, n - 1), dtype=np.int64)
    for i in range(n - 1):
        A[i, i] = 1
        A[i + 1, i] = -1
    return A


def cartan_gram(n: int) -> np.ndarray:
    A = root_basis(n)
    return A.T @ A


def simple_root_matrix(n: int, i: int) -> np.ndarray:
    """Action of s_i in the simple-root coefficient basis.

    If v=sum_j c_j alpha_j, then c' = M_i c.  Only row i changes:

        c'_i = -c_i + c_(i-1) + c_(i+1)

    with missing boundary terms omitted.  Thus the minimal (n-1)-dimensional
    standard representation retains radius-1 support on the A_(n-1) Dynkin
    path.  M_i preserves the Cartan/Gram metric rather than the ordinary
    coordinate Euclidean metric.
    """
    r = n - 1
    M = np.eye(r, dtype=np.int64)
    M[i, :] = 0
    M[i, i] = -1
    if i > 0:
        M[i, i - 1] = 1
    if i + 1 < r:
        M[i, i + 1] = 1
    return M


def verify_simple_root_representation(n: int) -> None:
    A = root_basis(n)
    G = cartan_gram(n)
    for i in range(n - 1):
        p = simple_permutation(n, i)
        P = permutation_matrix(p)
        M = simple_root_matrix(n, i)
        assert np.array_equal(A @ M, P @ A)
        assert np.array_equal(M @ M, np.eye(n - 1, dtype=np.int64))
        assert np.array_equal(M.T @ G @ M, G)


def path_matchings(n: int, include_empty: bool = False) -> list[tuple[int, ...]]:
    """Sets of disjoint adjacent Coxeter generators usable in one local layer."""
    out: list[tuple[int, ...]] = []
    for mask in range(1 << (n - 1)):
        chosen = tuple(i for i in range(n - 1) if (mask >> i) & 1)
        if not include_empty and not chosen:
            continue
        if all(b - a > 1 for a, b in zip(chosen, chosen[1:])):
            out.append(chosen)
    return out


def matching_permutation(n: int, matching: Iterable[int]) -> Permutation:
    p = identity(n)
    for i in matching:  # disjoint swaps commute
        p = compose(simple_permutation(n, i), p)
    return p


def bfs_words(n: int, parallel: bool) -> dict[Permutation, list[tuple[int, ...]]]:
    """Exact shortest words from identity.

    If parallel=False, each layer contains one adjacent transposition.
    If parallel=True, a layer may contain any matching of adjacent edges, so
    the word length is exact minimum nearest-neighbour parallel depth.
    """
    if parallel:
        layers = path_matchings(n)
    else:
        layers = [(i,) for i in range(n - 1)]
    layer_perms = [(layer, matching_permutation(n, layer)) for layer in layers]

    e = identity(n)
    words: dict[Permutation, list[tuple[int, ...]]] = {e: []}
    q: deque[Permutation] = deque([e])
    while q:
        cur = q.popleft()
        for layer, g in layer_perms:
            nxt = compose(g, cur)
            if nxt not in words:
                words[nxt] = words[cur] + [layer]
                q.append(nxt)
    assert len(words) == math_factorial(n)
    return words


def math_factorial(n: int) -> int:
    out = 1
    for k in range(2, n + 1):
        out *= k
    return out


def word_permutation(n: int, word: list[tuple[int, ...]]) -> Permutation:
    p = identity(n)
    for layer in word:
        p = compose(matching_permutation(n, layer), p)
    return p


def word_root_matrix(n: int, word: list[tuple[int, ...]]) -> np.ndarray:
    M = np.eye(n - 1, dtype=np.int64)
    for layer in word:
        L = np.eye(n - 1, dtype=np.int64)
        for i in layer:  # disjoint simple reflections commute
            L = simple_root_matrix(n, i) @ L
        M = L @ M
    return M


def direct_root_matrix(p: Permutation) -> np.ndarray:
    """Exact integer action on simple-root coefficients."""
    n = len(p)
    # The sequential BFS word is used as an exact constructive coordinate map;
    # this avoids introducing floating-point pseudoinverses into an integer rep.
    word = bfs_words(n, parallel=False)[p]
    return word_root_matrix(n, word)


def inversion_count(p: Permutation) -> int:
    return sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p)))


def moved_points(p: Permutation) -> int:
    return sum(i != p[i] for i in range(len(p)))


def state_oracle(n: int = 5):
    """Return an exact (n-1)-channel state/readout representation of S_n.

    We use a distinct, zero-sum vector in R^n so the orbit has n! states, then
    express it in the simple-root basis.  The readout uses the Cartan Gram
    matrix so prototype inner products equal Euclidean inner products in R^n.
    """
    if n % 2:
        v = np.arange(-(n // 2), n // 2 + 1, dtype=np.int64)
    else:
        # Distinct integers with zero sum, e.g. n=4 -> [-3,-1,1,3].
        v = np.arange(-(n - 1), n, 2, dtype=np.int64)
    assert len(v) == n and len(set(v.tolist())) == n and int(v.sum()) == 0

    A = root_basis(n)
    # Solve A c = v exactly via cumulative sums: c_i=sum_{j<=i} v_j.
    c0 = np.cumsum(v[:-1]).astype(np.int64)
    assert np.array_equal(A @ c0, v)

    perms = list(permutations(range(n)))
    sequential = bfs_words(n, parallel=False)
    prototypes = np.stack([word_root_matrix(n, sequential[p]) @ c0 for p in perms])
    assert len({tuple(row.tolist()) for row in prototypes}) == math_factorial(n)
    gram = cartan_gram(n)
    readout = prototypes @ gram  # row p gives c_p^T G
    return perms, c0, prototypes, readout


def decode_state(c: np.ndarray, readout: np.ndarray) -> int:
    return int(np.argmax(readout @ c))


def compile_stats(n: int = 5) -> dict[str, object]:
    verify_simple_root_representation(n)
    perms = list(permutations(range(n)))
    seq = bfs_words(n, parallel=False)
    par = bfs_words(n, parallel=True)

    for p in perms:
        assert word_permutation(n, seq[p]) == p
        assert word_permutation(n, par[p]) == p
        assert len(seq[p]) == inversion_count(p)
        assert np.array_equal(word_root_matrix(n, seq[p]), word_root_matrix(n, par[p]))

    seq_depths = [len(seq[p]) for p in perms]
    par_depths = [len(par[p]) for p in perms]

    subsets = {
        "full": perms,
        "identity_or_swap": [p for p in perms if moved_points(p) <= 2],
        "moves_at_most_3": [p for p in perms if moved_points(p) <= 3],
    }

    def substats(ps: list[Permutation]) -> dict[str, object]:
        sd = [len(seq[p]) for p in ps]
        pd = [len(par[p]) for p in ps]
        return {
            "count": len(ps),
            "sequential_mean": float(np.mean(sd)),
            "sequential_max": int(max(sd)),
            "parallel_mean": float(np.mean(pd)),
            "parallel_max": int(max(pd)),
            "parallel_depth_histogram": {str(k): int(v) for k, v in sorted(Counter(pd).items())},
        }

    return {
        "group": f"S{n}",
        "group_size": math_factorial(n),
        "behavioral_state_channels": n - 1,
        "root_system": f"A{n-1}",
        "primitive": "simple reflection / adjacent transposition",
        "primitive_support_radius": 1,
        "primitive_metric": "Cartan Gram matrix of A_(n-1)",
        "subsets": {name: substats(ps) for name, ps in subsets.items()},
        "full_sequential_depth_histogram": {str(k): int(v) for k, v in sorted(Counter(seq_depths).items())},
        "full_parallel_depth_histogram": {str(k): int(v) for k, v in sorted(Counter(par_depths).items())},
    }


def long_horizon_check(n: int, length: int, seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    perms, c, prototypes, readout = state_oracle(n)
    perm_to_id = {p: i for i, p in enumerate(perms)}
    words = bfs_words(n, parallel=True)
    state_perm = identity(n)
    correct = True

    for _ in range(length):
        token = perms[rng.randrange(len(perms))]
        state_perm = compose(token, state_perm)
        c = word_root_matrix(n, words[token]) @ c
        if decode_state(c, readout) != perm_to_id[state_perm]:
            correct = False
            break

    return {"length": length, "seed": seed, "exact": correct}


def main() -> None:
    p = argparse.ArgumentParser(description="Exact Coxeter/Dynkin local compiler oracle for full S_n")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--horizon", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    payload = compile_stats(args.n)
    payload["long_horizon"] = long_horizon_check(args.n, args.horizon, args.seed)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
