# Pass 19 — Sigma-local recurrent corollary: every regular language has a neighbour-only permutation/reset recurrent realization

Date: 2026-08-10

This pass connects two very recent lines that appear not to have met explicitly yet:

```text
Recurrent Neural Cascades / MinMax RNC          2024 -> May 2026
Sigma-chain automata decomposition             July 2026
```

The mathematical content below is mostly an **immediate lowering/corollary** of established results, not a claim that KYY proved a new universality theorem.

The reason to record it is architectural:

> the all-prefix recurrent cascade dependency can be replaced, at the finite-state symbolic level, by a strict nearest-neighbour chain without losing any regular-language expressivity.

---

# 1. The older neural cascade has all-prefix access

Knorozova & Ronca, *On The Expressivity of Recurrent Neural Cascades* (AAAI 2024 / arXiv 2312.09048), define a cascade so that component `i` can depend on the external input and **the states of all preceding components**.

Their algebraic-automata framework shows that:

```text
flip-flop/reset-capable neurons
    -> star-free / group-free regular behavior

group-capable neurons added
    -> all regular languages in principle.
```

MinMax Recurrent Neural Cascades (Ronca, arXiv 2605.06384, May 2026) make this much more concrete. They provide MinMax recurrent units that realize identity-reset and permutation semiautomata, prove all-regular-language expressivity, relate state dimension to permutation degree, and provide efficient parallel evaluation.

So:

```text
Krohn-Rhodes-like recurrent neural cascade
```

is established modern prior art.

---

# 2. The July-2026 Sigma-chain removes the all-prefix dependency symbolically

Borelli et al., *The Sigma-Chain Product: A Succinct Model of Automata (De)Composition* (arXiv 2607.16884), identify the representation problem in ordinary cascades:

```text
component i may depend on all preceding component states
    -> exponentially large alphabets / descriptions.
```

Their Sigma-chain product restricts the input of component `i` to:

```text
external symbol
    +
state of component i-1 only.
```

They prove:

1. Sigma-chains can have linear-size representations and be exponentially more succinct than ordinary cascades;
2. Sigma-chains and cascades are expressively equivalent for important classes including permutation-reset automata;
3. every regular language is recognized by a Sigma-chain of permutation-reset automata.

This result is only weeks old at the date of this map.

---

# 3. Immediate continuous lowering of one permutation-reset component

Take one finite permutation-reset semiautomaton component

```text
A = (Q, Lambda, delta)
```

with `|Q| = n`.

Represent state `q_j` by one-hot vector

```text
e_j in R^n.
```

Every internal input letter `lambda` induces either:

## permutation

```text
delta_lambda : Q -> Q
```

bijectively.

Use its permutation matrix:

```text
h' = P_lambda h.
```

`P_lambda` is orthogonal, so this operation preserves hidden difference norm.

## reset

If `delta_lambda(q)=q_star` for all `q`, use

```text
h' = 0 h + e_(q_star).
```

The linear difference map is rank zero: all prior state distinctions are removed.

Therefore every finite permutation-reset component has an exact switched affine realization whose token operations are precisely:

```text
orthogonal permutation
or
singular affine reset.
```

Nothing novel is claimed in this one-hot embedding.

---

# 4. Nearest-neighbour recurrent chain

Now take the Sigma-chain promised by Borelli et al.:

```text
A_1, A_2, ..., A_m.
```

Component 1 receives only external token `x_t`.

For `i > 1`, component `i` receives only

```text
(x_t, q_(i-1)).
```

Lower each component with the affine construction above.

The resulting continuous switched recurrent architecture has:

```text
h_1(t+1) = F_1(h_1(t), x_t)

h_i(t+1) = F_i(h_i(t), x_t, decode(h_(i-1)(t)))
            for i > 1.
```

Each `F_i`, on the finite reachable state set, selects either a permutation matrix or reset target determined by `(x_t, q_(i-1))`.

Schematic:

```text
          external token x_t broadcast
        |          |          |          |
        v          v          v          v
      [A1] -----> [A2] -----> [A3] -----> ... -----> [Am]
       ^           ^           ^                       ^
       |           |           |                       |
      h1          h2          h3                      hm

cross-cell dependency: immediate predecessor only
```

Thus the automata theorem gives directly:

> **Every regular language admits a finite-dimensional switched recurrent realization with strict predecessor-only inter-cell state dependence, where every internal finite-state transition is either an orthogonal permutation or a singular reset.**

Treat this as a corollary/translation of the Sigma-chain theorem plus the trivial one-hot affine embedding.

Do **not** currently call it a new theorem.

---

# 5. Why this is closer to the physical KYY picture than MinMax's classical cascade

The architecture has three independently justified structures.

## Locality

Not chosen because local wires look biological.

It is inherited from the Sigma-chain theorem:

```text
only immediate predecessor state is structurally required.
```

## Conservative versus irreversible operation

Not chosen by analogy.

It is inherited from the component transition type:

```text
permutation letter -> distinction-preserving operation
reset letter       -> distinction-destroying operation.
```

## Geometry / representation

The one-hot lowering is only the baseline.

Pass 18 shows that structured permutation factors can often be represented in lower-dimensional geometric group orbits, trading coordinate count for orbit separation / precision.

Therefore the actual compiler problem is:

```text
Sigma-chain symbolic component
        |
        v
choose continuous representation
        |
        +--> one-hot robust/simple
        |
        +--> compact group orbit
        |
        +--> MinMax permutation-degree realization
        |
        +--> another faithful semigroup realization
        v
lower local transitions
        |
        +--> conservative transport
        +--> contraction
        +--> reset/pinch
```

---

# 6. Complexity/resource difference from a classical recurrent cascade

In the 2024 RNC definition, neuron/component `i` can access all preceding recurrent states.

A naive `m`-component implementation therefore exposes `O(m^2)` inter-component dependency edges.

The Sigma-local form exposes only

```text
1 -> 2 -> 3 -> ... -> m,
```

namely `m-1` inter-component state edges, plus external-token broadcast.

This is an architectural wiring reduction from all-prefix to chain-local communication.

**Caution:** this does not automatically prove an `O(m^2) -> O(m)` runtime speedup.

Controller complexity, component state sizes, token broadcast, symbolic decoding, and parallel scheduling all matter. The Sigma-chain paper's succinctness result concerns automata representation size; translating it into wall-clock neural/hardware cost is an additional step.

---

# 7. Relation to MinMax RNC — the closest current baseline

MinMax RNC is the baseline KYY would have to beat or complement, not ignore.

Its May-2026 results include:

```text
all regular functions
parallel scan / logarithmic depth with enough processors
bounded state and activation
bounded gradients
non-vanishing state-gradient construction
state degree governed by permutation degree.
```

A Sigma-local KYY version would need to show something MinMax does not already get for free:

```text
smaller inter-component communication
smaller compiled controller descriptions
better physical locality
or a favorable state-dimension / precision / reset-cost frontier.
```

Simply matching regular-language expressivity is not a result.

---

# 8. Search status

Targeted queries in this pass included:

```text
"Sigma-chain" neural network
"Sigma-chain" recurrent neural cascade
"Sigma-chain" MinMax recurrent
permutation-reset neural cascade group representation
```

The search located:

- Recurrent Neural Cascades / MinMax RNC;
- the Sigma-chain automata paper;
- older Krohn-Rhodes neural constructions;
- no explicit paper, in this pass, lowering the July Sigma-chain product into a neural recurrent architecture or combining it with MinMax RNC.

Because the Sigma-chain paper is only weeks old, this absence is unsurprising.

Status:

**VERY RECENT UNMAPPED CONJUNCTION / NOT YET A NOVELTY CLAIM.**

---

# 9. Cheap next test — still no learned architecture required

Before training anything:

1. obtain or construct small canonical DFAs used in modern state-tracking work;
2. compute / hand-supply both a classical cascade and a Sigma-chain decomposition where feasible;
3. record component count, component state sizes, and controller-alphabet sizes;
4. lower the same factors to the one-hot affine permutation/reset baseline;
5. compare total recurrent state, predecessor bandwidth, controller table size, and number of singular transitions;
6. only if the Sigma-local lowering earns a concrete resource advantage, replace table controllers with learnable continuous controllers.

The first success criterion is therefore not accuracy.

It is:

```text
same exact finite-state behavior
        |
        v
less compiled communication / description cost.
```

---

# Current pin after Pass 19

For the first time, the "universe matrix" picture has landed on an architecture whose three ingredients each have an external mathematical reason:

```text
NEIGHBOUR CHAIN
    <- Sigma-chain decomposition

REVERSIBLE MOTION
    <- permutation/group factors

IRREVERSIBLE WRITE / RESET
    <- reset/aperiodic factors
```

The remaining Geometric-Neuron contribution, if any, would be in **how those factors are physically/continuously realized and compiled**, especially the dimension-versus-precision and locality-versus-reset-site trade-offs.

That is where to dig next.