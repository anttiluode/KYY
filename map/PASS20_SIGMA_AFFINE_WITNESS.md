# Pass 20 — executable Sigma-chain lowering: the irreversible front walks one neighbour per token

Date: 2026-08-10

Pass 19 stated an immediate architecture corollary of the July-2026 Sigma-chain theorem.

Pass 20 makes that bridge executable.

New code:

- `sigma_affine_compiler.py`
- `tests/test_sigma_affine_compiler.py`

Nothing here changes the novelty status: the finite-state decomposition theorem belongs to Borelli et al. The continuous one-hot lowering is elementary.

The value is that KYY can now **compile and measure** the symbolic object instead of talking about it metaphorically.

---

# 1. Exact Sigma-chain semantics

For a chain

```text
A1, A2, ..., Ah
```

with external symbol `a_t`, component `i>1` receives

```text
(a_t, q_(i-1,t))
```

where `q_(i-1,t)` is the predecessor state **before the current transition**.

All components update synchronously.

This matters physically.

It means information cannot ripple through an arbitrary number of components in one token update. Under the literal chain semantics, a newly changed state of `A1` can influence `A2` on the next token, `A3` one token later, and so on.

The compiler implements this update order explicitly.

Primary source:

- Borelli et al., *The Sigma-Chain Product* (2026), arXiv:2607.16884.

---

# 2. Generic exact affine lowering

For a component with finite states

```text
Q = {0,...,n-1}
```

use one-hot state `e_q`.

Every allowed component input induces either a permutation or reset transformation.

## permutation

If

```text
q -> pi(q)
```

use permutation matrix `P_pi`:

```text
h' = P_pi h.
```

This is orthogonal.

## reset

If every state maps to `q_star`, use

```text
h' = 0 h + e_(q_star).
```

This is a singular affine overwrite.

`SigmaChain.affine_run()` compiles every context this way and `verify_exact_affine()` exhaustively checks the continuous one-hot state against the discrete Sigma-chain state.

The mixed demo combines predecessor-conditioned permutations and resets over three levels and is exact on all exhaustively checked words.

---

# 3. The paper's exponential-succinctness witness becomes a literal travelling front

Borelli et al. use the language family

```text
L_h = Sigma^h Sigma*
```

(words of length at least `h`) to witness the succinctness gap for reset automata.

They prove:

```text
there is a Sigma-chain of h two-state reset automata
with size O(h |Sigma|),
```

while any reset-automata cascade for the same language needs at least `h` components and therefore has exponential explicit size.

KYY's exact chain is:

```text
cell 1:
    every symbol resets 0/1 -> 1

cell i>1:
    predecessor 0 -> identity
    predecessor 1 -> reset to 1
```

Starting from all zeros and feeding any symbols:

```text
t=0   00000000
t=1   10000000
t=2   11000000
t=3   11100000
t=4   11110000
...
```

The last cell becomes one exactly after `h` symbols.

So the symbolic Sigma-chain witness is literally an **irreversible activation front propagating one neighbour per token**.

This resembles the user's chain/oil-slick picture, but the mathematics comes from the Sigma-chain reset construction rather than from the metaphor.

---

# 4. Concrete representation numbers

The compiler reports transition-table entries, counting one finite-state transition per `(state, context)` pair.

For two-state components and alphabet size 2:

```text
Sigma-chain:
    first component      = 2*2 entries
    every later component = 2*2*2 entries

    total = 4 + 8(h-1)
```

The same component state sizes represented as an explicit classical all-prefix cascade have:

```text
component i context alphabet = Sigma x Q1 x ... x Q_(i-1)
```

and therefore

```text
explicit transition entries = 4(2^h - 1).
```

Examples:

| height h | Sigma entries | explicit all-prefix entries | Sigma state wires | all-prefix state wires |
|---:|---:|---:|---:|---:|
| 5 | 36 | 124 | 4 | 10 |
| 10 | 76 | 4,092 | 9 | 45 |
| 20 | 156 | 4,194,300 | 19 | 190 |

These numbers are a direct accounting for this representation.

The **lower-bound/succinctness theorem is the paper's**, not KYY's.

---

# 5. Why this is not yet an AI efficiency result

This witness is intentionally dangerous because it looks too good.

The language

```text
length >= h
```

can also be represented by a single counter with `h+1` discrete levels.

In an unconstrained continuous machine one could even encode those levels in one scalar.

Therefore:

> smaller Sigma-chain controller tables do not imply that an `h`-cell physical chain is the globally cheapest recurrent implementation.

The representation must be optimized too.

This is exactly the one-hot trap from Pass 17 in another form.

---

# 6. But the alternative pays somewhere else

Compare two idealized bounded-range realizations.

## spatial binary chain

```text
h scalar cells
state of each cell in {0,1}
nearest local state separation = 1
neighbour communication only
```

## one normalized scalar counter

```text
c in {0, 1/h, 2/h, ..., 1}
```

with saturating increment.

Nearest state separation is

```text
1/h.
```

So compressing the chain into one bounded scalar trades spatial dimension for precision/noise margin.

This is the irreversible analogue of Pass 18's cyclic orbit result:

```text
C_n phase orbit:
    2 real coordinates
    state separation shrinks with n

length-h latch chain:
    h robust local cells
    or one scalar whose level spacing shrinks with h.
```

No novelty is claimed for this coding trade-off. It is an information/geometry accounting principle.

---

# 7. The emerging common resource triangle

Both the reversible group example and irreversible reset-front example suggest the same three-way exchange:

```text
SPATIAL STATE / DIMENSION
        <->
DYNAMIC RANGE / PRECISION
        <->
COMMUNICATION / UPDATE DEPTH
```

Examples:

```text
more physical cells
    -> larger state separation
    -> local robust updates

fewer analog coordinates
    -> denser state packing
    -> higher precision requirement

scarce write sites
    -> less irreversible hardware
    -> more reversible routing depth
```

This begins to look like a useful KYY cost model even though each individual trade-off is classical.

---

# 8. Relation to MinMax RNC

MinMax RNC is an especially important comparison because identity-reset behavior can be represented by scalar MinMax recurrent units and permutation behavior by units whose state dimension is tied to faithful permutation degree.

Thus KYY must not present the scalar reset/latch cells themselves as new.

The potential Sigma-local difference is structural:

```text
MinMax/RNC classical cascade:
    later layer may depend on the full earlier cascade input/state structure

Sigma-local lowering:
    only immediate predecessor state crosses each component boundary.
```

The July Sigma-chain theorem supplies that dependency restriction after the May MinMax paper.

---

# 9. Search status

The Sigma-chain paper itself contains no occurrences of:

```text
neural
machine learning
physical
recurrent
```

in the full text search performed for this pass.

Targeted web searches also did not locate a post-July neural Sigma-chain lowering or a MinMax/Sigma-chain combination.

This absence is **not a novelty proof**. The paper is only weeks old.

---

# 10. Next target

The next interesting question is no longer whether the affine lowering exists. It does.

It is:

> **At fixed robustness/noise margin, does a Sigma-local decomposition produce a better state-dimension + communication + reset-cost point than an optimized monolithic or classical-cascade recurrent realization?**

That requires explicitly putting precision into the cost model.

The next map pass should therefore formalize the dimension/precision bound before any learned Sigma-KYY architecture is written.