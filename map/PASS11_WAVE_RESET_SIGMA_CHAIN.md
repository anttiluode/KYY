# Pass 11 — reversible wave + irreversible reset + local Sigma-chain

Date: 2026-08-10

This pass was triggered by a physical picture rather than a matrix construction:

```text
fast wave / chain motion
    -> rearranges information without needing to destroy it

slow write / plastic change
    -> can overwrite, merge, forget, or reset state
```

The question was whether that split says anything useful about recurrent sequence computation.

It does — but almost all of the abstract mathematics is old, and modern sequence-model papers have already rediscovered part of it.

The useful residual is much narrower and more physical/resource-aware.

---

# 1. The exact old theorem hiding under the picture: Krohn–Rhodes

For a finite automaton, every input word induces a transformation of the finite state set. The transformations form a finite transformation semigroup/monoid.

Krohn–Rhodes theory gives a decomposition of arbitrary finite automata into cascades of elementary components built from two broad kinds of behavior:

```text
PERMUTATION / GROUP COMPONENTS
    invertible: distinctions are rearranged but not merged

RESET / APERIODIC COMPONENTS
    irreversible: distinctions may be collapsed / overwritten
```

Useful entry points:

- Zimmermann, *On Krohn-Rhodes theory for semiautomata* (2020), https://arxiv.org/abs/2010.16235
- classic cascade synthesis: https://doi.org/10.1016/S0019-9958(67)90228-8
- Margolis, Rhodes & Schilling, *Decidability of Krohn-Rhodes complexity for all finite semigroups and automata* (2024), https://arxiv.org/abs/2406.18477

Therefore the statement

> reversible dynamics + reset dynamics can build arbitrary finite-state behavior

is not a KYY novelty. It is a classical structural theorem.

---

# 2. Modern sequence models already use this theorem

This is not merely old automata theory sitting disconnected from AI.

## Transformers

Liu et al., *Transformers Learn Shortcuts to Automata* (2022/2023), explicitly use Krohn–Rhodes to construct shallow Transformer simulations of solvable semiautomata:

https://arxiv.org/abs/2210.10749

Their two intuitive atoms are modular/group-like computation and resettable memory, joined by a cascade.

## State-space models

Sarrof, Veitsman & Hahn, *The Expressive Capacity of State Space Models: A Formal Language Perspective* (NeurIPS 2024):

https://arxiv.org/abs/2405.17394

use automata/semigroup structure to characterize regular-language capabilities of SSMs, including set-reset automata for star-free state tracking.

Shakerinava et al., *The Expressive Limits of Diagonal SSMs for State-Tracking* (ICLR 2026):

https://arxiv.org/abs/2603.01959

connect depth of complex diagonal SSMs to solvable-group structure.

## DeltaProduct

Siems et al., *DeltaProduct* (NeurIPS 2025):

https://arxiv.org/abs/2502.10297

use products of generalized Householders for permutation/group state tracking. The gated extension is analyzed using permutation-reset automata to obtain general regular-language expressivity.

### Subtraction

The following claims are occupied:

```text
"group/permutation + reset is enough for finite automata"
"gating supplies reset/forgetting"
"Krohn-Rhodes explains neural state-tracking expressivity"
"Householder-like recurrence handles permutation factors"
```

KYY must add a resource/locality/physical statement, not repackage the theorem.

---

# 3. A tiny algebraic fact that matters physically: rank cannot come back by permutation

Represent a deterministic transition on `n` finite states as a map

```text
f : Q -> Q.
```

Define its transformation rank as

```text
rank(f) = |image(f)|.
```

Then

```text
rank(f o g) <= min(rank(f), rank(g)).
```

A permutation has rank `n`.

A total reset has rank `1`.

Therefore a product of permutation-only transitions can never equal a reset on the same finite state space.

Once a true rank drop occurs, subsequent permutations can move the surviving distinctions around, but cannot recreate distinctions that were destroyed inside that state space.

This gives a clean computational reading of the physical split:

```text
lossless / reversible transport
    -> rank-preserving motion

reset / overwrite / dissipation
    -> rank-reducing motion
```

This is standard semigroup information, not a new theorem.

The script `reversible_reset_probe.py` makes the smallest example explicit.

---

# 4. Important caveat: reversible hidden dynamics can hide the discarded information

Do not overstate the rank argument.

An irreversible *visible* reset can be embedded into a larger reversible machine by retaining the old state in hidden/ancillary degrees of freedom.

For example, if the visible state is `q` and an ancilla starts blank, a reversible swap-like map can send

```text
(q, blank) -> (reset_value, q).
```

The visible state appears reset, but the supposedly erased information is still present in the ancilla.

That is ordinary reversible-computing logic.

So the precise statement relevant to KYY is:

> A bounded, behaviorally minimal finite-state realization cannot obtain a genuinely rank-reducing transition from permutation-only primitives. A larger reversible realization may emulate the visible reset by storing the discarded distinctions somewhere else.

This reconnects directly to Pass 6:

```text
hidden state
    !=
behavioral quotient.
```

If the old information remains forever invisible but physically present, the implementation is paying a hidden-memory/garbage cost.

At finite precision and fixed state capacity, that cost cannot grow without bound for arbitrary irreversible computation; eventually information must be uncomputed, exported, merged, measured, or dissipated.

KYY should therefore price **discarded-information storage** when comparing reversible/local implementations with gated/reset implementations.

---

# 5. Quantum/reversible automata are an explicit prior-art warning

The physical analogy to unitary wave evolution is also old.

Traditional one-way quantum finite automata with unitary evolution recognize only restricted classes of regular languages; generalized quantum finite automata using general trace-preserving operations recover all regular languages.

Useful anchors:

- Brodsky & Pippenger, *Characterizations of 1-Way Quantum Finite Automata*, https://arxiv.org/abs/quant-ph/9903014
- Li et al., *Characterizations of one-way general quantum finite automata* (2012), https://doi.org/10.1016/j.tcs.2011.10.021

This is another reason not to claim

> "reversible waves need an irreversible operation for general finite-state computation"

as new.

It is a known boundary in automata/computation.

---

# 6. The unexpectedly close 2026 hit: the Sigma-chain product

Borelli, Bresolin, Geatti, Montanari & Zavatteri, *The Sigma-Chain Product: A Succinct Model of Automata (De)Composition* (July 2026):

https://arxiv.org/abs/2607.16884

is currently the most important new landmark for KYY.

The ordinary cascade product lets a component depend on the input plus potentially **all previous component states**, which can cause an exponentially large representation.

The Sigma-chain restricts the dependency:

```text
component i sees
    external input
        +
    state of component i-1
```

rather than the entire prefix of the cascade.

Schematic:

```text
             x_t       x_t       x_t
              |         |         |
              v         v         v
            [A1] ----> [A2] ----> [A3] ----> ...
                      local predecessor only
```

The paper proves that Sigma-chains can be exponentially more succinct than ordinary cascades in representation size, and in particular:

> every regular language is recognized by a Sigma-chain of permutation-reset automata.

That is extremely close to the locality constraint KYY kept reaching for.

### Subtraction

We therefore cannot claim:

```text
"a neighbour-only chain of simple reversible/reset state machines is finite-state universal"
```

That statement now has a direct 2026 automata-theory home.

---

# 7. Physical/biological Krohn-Rhodes applications also exist

Even applying Krohn–Rhodes to physical/biological dynamical systems is not a blank region.

Examples include:

- Dini, Nehaniv, Egri-Nagy & Schilstra, *Exploring the concept of interaction computing through the discrete algebraic analysis of the Belousov-Zhabotinsky reaction* (Biosystems 2013), https://doi.org/10.1016/j.biosystems.2013.03.003
- DeDeo, *Effective Theories for Circuits and Automata* (2011), https://arxiv.org/abs/1106.5778
- recent asynchronous-circuit work explicitly using Krohn-Rhodes decomposition/pipeline synthesis.

So the phrase

> "Krohn-Rhodes but physical/organic"

is also too broad to be ours.

---

# 8. What the user's physical lens contributes after subtraction

The interesting residual is not the decomposition theorem.

It is a proposed **mapping from algebraic prime type to physical primitive type under a local resource model**.

Working KYY correspondence:

```text
PERMUTATION / GROUP FACTOR
    -> local nearly-conservative propagation
    -> swaps / phase rotations / reciprocal scatter / reversible control words

RESET / APERIODIC FACTOR
    -> local overwrite / gate / dissipative contraction
    -> explicitly destroys or exports distinctions

SIGMA-CHAIN COUPLING
    -> only token + immediate predecessor state
    -> no all-prefix communication inside the recurrent body
```

This is a design hypothesis, not a biology claim and not a statement that a physical wave is literally a finite permutation.

The potentially useful question is:

> **Can the algebraic decomposition of a task tell us which parts of a recurrent machine should be implemented by cheap reversible/local propagation and where genuinely irreversible write/reset primitives are required?**

That is different from asking one uniform neural matrix family to do everything.

---

# 9. A possible new resource axis: reversibility tax

For a task transition monoid `M`, KYY can distinguish costs that ordinary parameter/FLOP counts hide:

```text
reversible/group work
    local transport depth
    wire span
    phase/rotation precision

irreversible work
    number/support of rank-reducing operations
    overwritten information / hidden garbage retained
    reset/write events

structural work
    Sigma-chain height
    local state sizes
    predecessor communication bandwidth
```

A useful implementation comparison would ask:

```text
same behavioral automaton
        |
        +--> all-dense learned recurrence
        |
        +--> Householder + gate
        |
        +--> exact permutation-reset cascade
        |
        +--> local Sigma-chain permutation-reset realization
        |
        +--> KYY physical primitive cost model
```

and measure a Pareto front rather than one score.

This is not yet a new metric. Cascade height, Krohn-Rhodes complexity, reset-automata hierarchies, routing depth, and reversible-computing costs all have separate literatures. The open question is whether their **joint use as a recurrent hardware/sequence-model cost model** exposes a useful design frontier.

---

# 10. The next exact unit test should mix permutation and reset behavior

`perm3` and full `S5` are pure-group tasks. They test reversible state tracking but say nothing about the reset half of a general automaton.

Before inventing a trainable wave/plastic architecture, add a tiny mixed transformation-semigroup oracle such as:

```text
states: 0,1,2

token I: identity
token C: cycle 0->1->2->0
token R: reset every state -> 0
```

The generated transition monoid contains:

```text
3 rank-3 cyclic permutations
3 rank-1 constant maps
```

so it is the smallest clean demonstration of both reversible and irreversible transition types.

The purpose is not to publish this toy task. It is a regression test for claims such as:

- can a proposed model implement exact reset behavior?
- does it hide erased state in extra dimensions?
- does error accumulate after many alternating cycles/resets?
- how many explicit irreversible primitives are needed?

Only after that diagnostic should KYY consider a learned local wave+write model.

---

# 11. Current novelty status

The broad ingredients are occupied:

```text
Krohn-Rhodes permutation/reset decomposition      OCCUPIED
Krohn-Rhodes in Transformer theory               OCCUPIED
Krohn-Rhodes / reset automata in SSM theory      OCCUPIED
Gated Householder permutation-reset simulation   OCCUPIED
reversible / quantum automata limitations         OCCUPIED
physical / biochemical KR analysis                OCCUPIED
local neighbour Sigma-chain universality          OCCUPIED (July 2026)
```

The current residual is therefore:

> **Use the task's behavioral transition monoid to co-design a bounded-state local recurrent implementation whose reversible factors are mapped to conservative/local propagation primitives and whose irreversible factors are mapped only where necessary to explicit reset/write primitives, with Sigma-chain-style neighbour communication and a measured reversibility/locality/depth/garbage trade-off.**

Status:

**BRIDGE / UNMAPPED IN THIS SEARCH / VERY HIGH PRIOR-ART RISK.**

Do not call it new yet.

But this is a much better coordinate than "wave attention" or "plastic SSM" because the algebra tells us **why two physically different kinds of primitive are needed**.

---

# Current pin after Pass 11

The Geometric-Neuron/KYY question has shifted again:

> **Can a task be factorized into the reversible motion it needs and the irreversible forgetting it needs, then realized as a neighbour-only physical recurrent chain without paying the global communication cost of dense state mixing?**

That is now precise enough to map, measure, and kill.