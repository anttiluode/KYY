from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass

import numpy as np


Transformation = tuple[int, ...]
Context = tuple[int, ...]


def is_permutation(t: Transformation) -> bool:
    return sorted(t) == list(range(len(t)))


def is_reset(t: Transformation) -> bool:
    return len(set(t)) == 1


def transition_kind(t: Transformation) -> str:
    if is_permutation(t):
        return "permutation"
    if is_reset(t):
        return "reset"
    return "other"


def affine_lowering(t: Transformation) -> tuple[np.ndarray, np.ndarray]:
    """Exact one-hot affine lowering e_q -> e_{t(q)} for permutation/reset maps.

    Permutations are lowered as orthogonal permutation matrices.
    Resets are lowered as A=0, b=e_target so hidden differences are erased.
    """
    n = len(t)
    kind = transition_kind(t)
    if kind == "permutation":
        A = np.zeros((n, n), dtype=np.float64)
        for q, q_next in enumerate(t):
            A[q_next, q] = 1.0
        b = np.zeros(n, dtype=np.float64)
        return A, b
    if kind == "reset":
        A = np.zeros((n, n), dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)
        b[t[0]] = 1.0
        return A, b
    raise ValueError("Sigma affine baseline only accepts permutation/reset transformations")


@dataclass(frozen=True)
class SigmaComponent:
    """One Sigma-chain component.

    Level 0 contexts are `(symbol,)`.
    Later contexts are `(symbol, predecessor_state)`.

    The predecessor state is the PRE-TRANSITION state at the same time step,
    matching the Sigma-chain/cascade product semantics in Borelli et al. 2026.
    """

    n_states: int
    table: dict[Context, Transformation]
    initial_state: int = 0

    def validate(self, alphabet_size: int, predecessor_states: int | None) -> None:
        if self.n_states < 1:
            raise ValueError("component must have at least one state")
        if not 0 <= self.initial_state < self.n_states:
            raise ValueError("initial state out of range")

        expected_contexts: set[Context]
        if predecessor_states is None:
            expected_contexts = {(a,) for a in range(alphabet_size)}
        else:
            expected_contexts = {
                (a, q_prev)
                for a in range(alphabet_size)
                for q_prev in range(predecessor_states)
            }
        if set(self.table) != expected_contexts:
            missing = expected_contexts - set(self.table)
            extra = set(self.table) - expected_contexts
            raise ValueError(f"context table mismatch; missing={sorted(missing)}, extra={sorted(extra)}")

        for context, t in self.table.items():
            if len(t) != self.n_states:
                raise ValueError(f"context {context}: transformation size mismatch")
            if min(t) < 0 or max(t) >= self.n_states:
                raise ValueError(f"context {context}: target state out of range")
            if transition_kind(t) == "other":
                raise ValueError(
                    f"context {context}: Sigma baseline requires permutation/reset, got {t}"
                )


@dataclass(frozen=True)
class SigmaCost:
    height: int
    alphabet_size: int
    state_sizes: tuple[int, ...]
    sigma_transition_entries: int
    explicit_cascade_transition_entries_same_components: int
    sigma_intercomponent_state_edges: int
    all_prefix_intercomponent_state_edges: int


class SigmaChain:
    def __init__(self, alphabet_size: int, components: list[SigmaComponent]):
        if alphabet_size < 1:
            raise ValueError("alphabet_size must be positive")
        if not components:
            raise ValueError("Sigma-chain needs at least one component")
        self.alphabet_size = int(alphabet_size)
        self.components = list(components)

        predecessor_states: int | None = None
        for component in self.components:
            component.validate(self.alphabet_size, predecessor_states)
            predecessor_states = component.n_states

    @property
    def initial_state(self) -> tuple[int, ...]:
        return tuple(component.initial_state for component in self.components)

    def step(self, state: tuple[int, ...], symbol: int) -> tuple[int, ...]:
        """Synchronous discrete Sigma-chain update.

        Every component reads the OLD predecessor state, not its just-updated value.
        """
        if len(state) != len(self.components):
            raise ValueError("state tuple length mismatch")
        if not 0 <= symbol < self.alphabet_size:
            raise ValueError("symbol out of alphabet")

        old = tuple(state)
        new: list[int] = []
        for i, component in enumerate(self.components):
            context = (symbol,) if i == 0 else (symbol, old[i - 1])
            t = component.table[context]
            new.append(t[old[i]])
        return tuple(new)

    def run(self, word: tuple[int, ...] | list[int]) -> list[tuple[int, ...]]:
        state = self.initial_state
        history = [state]
        for symbol in word:
            state = self.step(state, int(symbol))
            history.append(state)
        return history

    def initial_one_hot(self) -> list[np.ndarray]:
        out = []
        for component in self.components:
            x = np.zeros(component.n_states, dtype=np.float64)
            x[component.initial_state] = 1.0
            out.append(x)
        return out

    def affine_step(self, states: list[np.ndarray], symbol: int) -> list[np.ndarray]:
        """Exact one-hot switched-affine lowering of one Sigma-chain step."""
        if len(states) != len(self.components):
            raise ValueError("continuous state list length mismatch")
        old = [np.asarray(x, dtype=np.float64).copy() for x in states]
        decoded = [int(np.argmax(x)) for x in old]
        out: list[np.ndarray] = []
        for i, component in enumerate(self.components):
            context = (symbol,) if i == 0 else (symbol, decoded[i - 1])
            A, b = affine_lowering(component.table[context])
            out.append(A @ old[i] + b)
        return out

    def affine_run(self, word: tuple[int, ...] | list[int]) -> list[list[np.ndarray]]:
        states = self.initial_one_hot()
        history = [[x.copy() for x in states]]
        for symbol in word:
            states = self.affine_step(states, int(symbol))
            history.append([x.copy() for x in states])
        return history

    def cost(self) -> SigmaCost:
        state_sizes = tuple(c.n_states for c in self.components)

        sigma_entries = 0
        for i, component in enumerate(self.components):
            context_count = self.alphabet_size if i == 0 else self.alphabet_size * state_sizes[i - 1]
            sigma_entries += component.n_states * context_count

        cascade_entries = 0
        prefix_product = 1
        for i, component in enumerate(self.components):
            cascade_entries += component.n_states * self.alphabet_size * prefix_product
            prefix_product *= component.n_states

        h = len(self.components)
        return SigmaCost(
            height=h,
            alphabet_size=self.alphabet_size,
            state_sizes=state_sizes,
            sigma_transition_entries=sigma_entries,
            explicit_cascade_transition_entries_same_components=cascade_entries,
            sigma_intercomponent_state_edges=max(0, h - 1),
            all_prefix_intercomponent_state_edges=h * (h - 1) // 2,
        )


def length_threshold_sigma_chain(height: int, alphabet_size: int = 2) -> SigmaChain:
    """Two-state reset Sigma-chain recognizing words of length >= height.

    This is the concrete family used for the Sigma-chain succinctness result
    L_h = Sigma^h Sigma* in Borelli et al. 2026.

    Cell 0 resets to 1 on the first symbol.  Later cell i resets to 1 iff its
    predecessor was already 1 before the current symbol; otherwise it keeps its
    state.  Hence a latched front advances exactly one cell per input symbol.
    """
    if height < 1:
        raise ValueError("height must be >= 1")
    if alphabet_size < 1:
        raise ValueError("alphabet_size must be >= 1")

    components: list[SigmaComponent] = []

    first_table = {(a,): (1, 1) for a in range(alphabet_size)}
    components.append(SigmaComponent(n_states=2, table=first_table, initial_state=0))

    for _ in range(1, height):
        table: dict[Context, Transformation] = {}
        for a in range(alphabet_size):
            table[(a, 0)] = (0, 1)  # identity while predecessor has not latched
            table[(a, 1)] = (1, 1)  # reset/set to 1 once predecessor is active
        components.append(SigmaComponent(n_states=2, table=table, initial_state=0))

    return SigmaChain(alphabet_size=alphabet_size, components=components)


def mixed_demo_sigma_chain() -> SigmaChain:
    """Small nontrivial 3-level permutation/reset chain for compiler tests.

    It is not claimed as a benchmark or literature construction.  It simply
    exercises predecessor-conditioned permutations and resets in one exact chain.
    """
    alphabet_size = 3

    # Level 1: I / C3 / reset-to-0.
    c1 = SigmaComponent(
        n_states=3,
        table={
            (0,): (0, 1, 2),
            (1,): (1, 2, 0),
            (2,): (0, 0, 0),
        },
    )

    # Level 2: predecessor state decides whether symbol 1 toggles; symbol 2
    # resets to predecessor parity.  Every context is permutation or reset.
    t2: dict[Context, Transformation] = {}
    for a in range(alphabet_size):
        for prev in range(3):
            if a == 2:
                target = prev % 2
                t2[(a, prev)] = (target, target)
            elif a == 1 and prev == 1:
                t2[(a, prev)] = (1, 0)
            else:
                t2[(a, prev)] = (0, 1)
    c2 = SigmaComponent(n_states=2, table=t2)

    # Level 3: reset to 0 when predecessor is 0 and symbol 2; otherwise symbol
    # 1 cycles three states iff predecessor is 1.
    t3: dict[Context, Transformation] = {}
    for a in range(alphabet_size):
        for prev in range(2):
            if a == 2 and prev == 0:
                t3[(a, prev)] = (0, 0, 0)
            elif a == 1 and prev == 1:
                t3[(a, prev)] = (1, 2, 0)
            else:
                t3[(a, prev)] = (0, 1, 2)
    c3 = SigmaComponent(n_states=3, table=t3)

    return SigmaChain(alphabet_size=alphabet_size, components=[c1, c2, c3])


def verify_exact_affine(chain: SigmaChain, max_length: int) -> dict[str, object]:
    """Exhaustively verify discrete state == affine one-hot state up to max_length."""
    checked_words = 0
    max_error = 0.0
    for length in range(max_length + 1):
        for word in itertools.product(range(chain.alphabet_size), repeat=length):
            discrete = chain.run(word)
            affine = chain.affine_run(word)
            for ds, xs in zip(discrete, affine):
                for q, x in zip(ds, xs):
                    target = np.zeros_like(x)
                    target[q] = 1.0
                    max_error = max(max_error, float(np.max(np.abs(x - target))))
            checked_words += 1
    return {
        "checked_words": checked_words,
        "max_state_error": max_error,
        "exact": max_error == 0.0,
    }


def threshold_front(chain: SigmaChain, steps: int) -> list[tuple[int, ...]]:
    """Run the length-threshold witness on repeated symbol 0."""
    return chain.run([0] * steps)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Exact Sigma-chain -> switched affine permutation/reset compiler"
    )
    p.add_argument("--height", type=int, default=8)
    p.add_argument("--alphabet-size", type=int, default=2)
    p.add_argument("--verify-length", type=int, default=5)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    threshold = length_threshold_sigma_chain(args.height, args.alphabet_size)
    demo = mixed_demo_sigma_chain()
    payload = {
        "threshold_cost": asdict(threshold.cost()),
        "threshold_front": threshold_front(threshold, args.height + 1),
        "threshold_affine_check": verify_exact_affine(
            threshold, min(args.verify_length, args.height + 1)
        ),
        "mixed_demo_cost": asdict(demo.cost()),
        "mixed_demo_affine_check": verify_exact_affine(demo, args.verify_length),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    cost = payload["threshold_cost"]
    print("Sigma length-threshold witness")
    print(f"height={args.height}, alphabet={args.alphabet_size}")
    print(f"Sigma transition entries: {cost['sigma_transition_entries']}")
    print(
        "Explicit all-prefix cascade entries, same component state sizes: "
        f"{cost['explicit_cascade_transition_entries_same_components']}"
    )
    print(
        f"Intercomponent state edges: Sigma={cost['sigma_intercomponent_state_edges']}, "
        f"all-prefix={cost['all_prefix_intercomponent_state_edges']}"
    )
    print("front:")
    for t, state in enumerate(payload["threshold_front"]):
        print(f"  t={t:2d}: {''.join(str(x) for x in state)}")
    print("affine checks:")
    print("  threshold:", payload["threshold_affine_check"])
    print("  mixed demo:", payload["mixed_demo_affine_check"])


if __name__ == "__main__":
    main()
