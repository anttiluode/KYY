# KYY / map

This folder is the **research cartography layer** for KYY.

It exists because the project repeatedly derived something exciting, built it, and only afterwards discovered the nearest established name. The rule is:

> **Search first. Locate the nearest known object. Compute the residual. Build only the residual.**

`UNMAPPED != NEW`.

Snapshot: **2026-08-10**.

## What we are mapping

The starting question was:

> Can useful recurrent computation arise from **local geometry + propagation** rather than unrestricted global state mixing?

The physical version has now become sharper:

```text
BEHAVIOR
 finite automaton / transition monoid / associative memory
        |
        v
REALIZATION
 hidden state / observable quotient / fast-weight medium
        |
        v
PRIMITIVE TYPE
 reversible transport      irreversible write/reset
 swaps / waves / phases    merge / erase / contraction
        \                    /
         \                  /
          v                v
LOCAL GEOMETRY
 graph support / neighbour communication / write sites
                |
                v
SYNTHESIS + COST
 local word / parallel depth / garbage / drift / port error
```

The Geometric Neuron intuition survives only if declared geometry earns something on that whole chain.

## Start here — newest first

1. **[PASS13_TRANSPORT_AND_PINCH.md](PASS13_TRANSPORT_AND_PINCH.md)** — classical transformation-semigroup theorem translated into the exact local resource question: reversible routing versus number/location of irreversible merge sites.
2. **[local_transformation_monoid_oracle.py](local_transformation_monoid_oracle.py)** — exact small-`n` BFS compiler for all `n^n` deterministic transitions using adjacent swaps and local rank-lowering pinches.
3. **[PASS12_LAKE_FAST_WEIGHT_BOUNDARY.md](PASS12_LAKE_FAST_WEIGHT_BOUNDARY.md)** — the lake/read-before-write picture lands directly on fast-weight programmers, DeltaNet, sparse delta memory, and old photorefractive delta-rule hardware.
4. **[PASS11_WAVE_RESET_SIGMA_CHAIN.md](PASS11_WAVE_RESET_SIGMA_CHAIN.md)** — Krohn-Rhodes permutation/reset decomposition plus the July-2026 Sigma-chain result: immediate-neighbour chains of permutation-reset automata still cover all regular languages.
5. **[reversible_reset_probe.py](reversible_reset_probe.py)** — exact rank probe showing true reset versus reversible visible-reset-with-hidden-garbage.
6. **[PASS10_SIMULTANEOUS_REALIZATION.md](PASS10_SIMULTANEOUS_REALIZATION.md)** — common realization / whole-family local synthesis boundary.
7. **[PASS9_LIVE_2026_BOUNDARY.md](PASS9_LIVE_2026_BOUNDARY.md)** — hard group projection, holonomic/gauge ideas, sparse realization, structure-aware automaton compilation.
8. **[PASS8_COXETER_IR.md](PASS8_COXETER_IR.md)** / **[s5_coxeter_oracle.py](s5_coxeter_oracle.py)** — exact full-`S5` local compiler floor in a 4D `A4` representation.
9. **[PASS7_DISCRETE_SYNTHESIS.md](PASS7_DISCRETE_SYNTHESIS.md)** / **[exact_perm3_local_oracle.py](exact_perm3_local_oracle.py)** — catches the finite-group/Lie-algebra mismatch and burns `perm3` as evidence.
10. **[PASS6_BEHAVIORAL_QUOTIENT.md](PASS6_BEHAVIORAL_QUOTIENT.md)** — hidden dynamics are larger than what ports/readout necessarily identify.
11. **[PASS5_CONTROL_ALGEBRA.md](PASS5_CONTROL_ALGEBRA.md)** / **[RING_ALGEBRA_LEMMA.md](RING_ALGEBRA_LEMMA.md)** — geometry/control placement -> reachable Lie algebra.
12. **[OCCUPANCY_MATRIX.md](OCCUPANCY_MATRIX.md)** / **[LANDSCAPE.md](LANDSCAPE.md)** / **[SEARCH_LOG.md](SEARCH_LOG.md)** / **[SOURCES.md](SOURCES.md)** — wider map and search history.

## Hard landmarks already occupied

```text
local 2-port recurrent mesh          -> EUNN / Givens / optical meshes
Householder recurrence               -> oRNN / DeltaProduct
adaptive input-dependent unitary     -> AUSSM
bilinear state-tracking RNN          -> modern bilinear-RNN work
second-order oscillator RNN          -> coRNN
geometry-defined oscillator flow     -> GraphCON
wave physics mapped to an RNN        -> Hughes et al. 2019
input-dependent recurrent Q          -> ISAN / selective SSMs
read-before-write delta memory       -> fast-weight programmers / DeltaNet
sparse delta-rule memory             -> Sparse Delta Memory (2026)
physical delta-rule holographic learn-> photorefractive optical NNs (1989+)
physical self-plastic networks       -> active physical-learning field
Lie algebra explains sequence order  -> 2026 Lie-algebraic sequence theory
few local controls steer networks    -> bilinear / quantum graph controllability
RNN behavior -> automaton            -> WFA / extraction / realization theory
permutation + reset -> arbitrary FSA -> Krohn-Rhodes
neighbour permutation-reset chain    -> Sigma-chain (2026)
S_n + one defect-1 map -> all T_n    -> classical transformation semigroups
graph-local rank-lowering word length-> transformation semigroups on digraphs
operator -> short local word         -> routing / circuit / gate synthesis
choose sparse equivalent realization -> classical control + sparse-realization work
hard projection protects group state -> live non-Abelian state-tracking work
DFA -> compact IR -> local hardware  -> structure-aware compiler work
```

The map is crowded. That is useful.

## Four exact lessons KYY has earned

### 1. Geometry can impose an algebraic ceiling

In the Pass-5 toy, a path plus one local control generated full `so(N)` for tested sizes. An even ring `N=2m` with one controlled edge preserves a hidden complex structure and gives a `u(m)`-type reachable algebra; one extra control breaks the symmetry and restores the larger algebra.

Design lesson:

```text
geometry -> symmetry -> reachable operator family.
```

### 2. Maximum Lie dimension is not the task

Finite state-tracking tasks care about exact discrete relations and the **behavioral quotient**, not about filling a huge continuous Lie algebra.

The original `S3` benchmark has a trivial exact local implementation, so `perm3` is now a regression/optimization test, not evidence for a new architecture.

### 3. Locality has a real, representation-dependent depth price

For full `S5`, KYY's exact 4D `A4` representation keeps adjacent generators radius-1 local. Compiling all 120 permutations into simultaneous nearest-neighbour matchings gives exact parallel depth mean `403/120 ~= 3.3583`, maximum `5`, with zero algebraic drift in exact arithmetic.

The 5D natural representation and 4D quotient solve the same behavior with different substrate primitives. Therefore:

> **minimum behavioral state dimension and minimum physical implementation cost are different optimization problems.**

### 4. Reversible transport plus one pinch is algebraically enough

For finite `n`, classical theory says generators of `S_n` plus one defect-1 singular transformation generate the full transformation monoid `T_n` of all `n^n` deterministic state updates.

On a path this becomes:

```text
adjacent swaps
    +
one fixed local merge 0 -> 1
    ->
all deterministic n-state transformations.
```

The exact KYY BFS oracle confirms the full closure for small `n` and exposes the routing/write-site trade-off.

Measured worst exact depths:

| n | |T_n| | global-3 | path + 1 pinch | path + all pinches | parallel + 1 pinch | parallel + all pinches |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 2 | 2 | 1 | 2 | 1 |
| 3 | 27 | 5 | 6 | 3 | 6 | 3 |
| 4 | 256 | 9 | 11 | 6 | 9 | 4 |
| 5 | 3125 | 16 | 19 | 10 | 13 | 5 |
| 6 | 46656 | 23 | 28 | 15 | 17 | 6 |

These are **computed toy resource points, not claimed formulas**. Word lengths in graph-generated transformation semigroups are established mathematics.

## The lake/bicycle correction

The useful physical translation is now:

```text
chain tooth / wave coupling
    = local transition / transport primitive
    NOT attention by itself

input-dependent modulation
    = routing/control-like operation

medium bending from current retrieval error
    = fast-weight / delta-rule memory
    already known

true overwrite / merge
    = behaviorally irreversible operation
    cannot be manufactured from permutations on the same bounded state set
```

A reversible larger hidden system can make the **visible** state look reset by storing the discarded information in an ancilla/garbage degree of freedom. `reversible_reset_probe.py` demonstrates this exactly.

So KYY must price not only reset operations but also hidden garbage retained when an allegedly reversible implementation emulates forgetting.

## The Sigma-chain landmark

The July-2026 Sigma-chain result is particularly close to the locality intuition.

Instead of a classical cascade component depending on all previous component states, component `i` sees only:

```text
external token x_t
        +
immediately preceding component state
```

and a Sigma-chain of permutation-reset automata is still sufficient for every regular language.

That kills the broad claim that neighbour-only composition is itself a new route to finite-state universality.

It leaves an engineering question:

> can such a symbolic local decomposition be turned into a better physical recurrent cost point than dense/global state mixing?

## The fast-weight landmark

The lake's read-before-write rule is essentially the fast-weight delta rule:

```text
v_bar = W k
W <- W + beta (v - v_bar) outer k
```

and current DeltaNet-family work adds gating, independent erase paths, preconditioning, and sparse memory addressing.

Therefore KYY cannot claim a new memory rule.

The only remaining geometric distinction is that a physical local medium does not receive a free address lookup: reaching a memory interaction requires propagation through the substrate.

## Current pin on the map

The residual is now a **behavior-to-physical-primitive compiler** question:

> **Given a task's behavioral transition monoid and/or online associative-memory requirements, choose a behaviorally sufficient state realization, factor its work into reversible transport and genuinely irreversible write/reset operations, place the irreversible sites on a declared local geometry, and compile the whole input-conditioned family to minimize communication depth, write-site count, hidden garbage, observable error, and long-horizon drift.**

A candidate cost vector is:

```text
behavioral state dimension
local component / Sigma-chain height
runtime control bits / ports
reversible wire span / propagation depth
number and placement of irreversible write sites
hidden garbage / ancilla retained
sequential and parallel local depth
relation defect / behavioral quotient error
long-horizon drift
physical precision / dissipation proxy
```

Status:

**BRIDGE / HIGH PRIOR-ART RISK / EXACTLY TESTABLE / NOT A NOVELTY CLAIM.**

## No new learned architecture yet

The branch now has a mixed benchmark `permreset3`:

```text
I: identity
C: 0 -> 1 -> 2 -> 0
R: every state -> 0
```

It is deliberately tiny. Its purpose is to test whether a supposedly reversible recurrent model really forgets a prefix after `R`, or merely hides the old information and lets it leak back under continuation.

Next gates:

1. run existing KYY models on `permreset3`; no new model yet;
2. pair histories that differ before reset and measure **post-reset continuation divergence at the readout**;
3. keep the exact local transport/pinch oracle as the resource floor;
4. search the joint optimization `behavioral realization + irreversible-site placement + local routing` before coding a trainable version;
5. only then decide whether a geometric/plastic layer earns existence.

## Mandatory build gate

Before another learned architecture is added to `kyy/models.py`, add this to `SEARCH_LOG.md`:

```text
DATE:
PROPOSED OBJECT:
NEAREST KNOWN OBJECTS:
EXACT OVERLAP:
RESIDUAL DIFFERENCE:
WHY THE RESIDUAL MATTERS:
PRIMARY-SOURCE SEARCH QUERIES:
PAPERS INSPECTED:
WHAT WAS NOT LOCATED:
CHEAPEST TEST WITHOUT A NEW ARCHITECTURE:
CHEAPEST FALSIFIER IF CODE IS REQUIRED:
BASELINES THAT CAN KILL IT:
RESULT THAT STOPS THE BRANCH:
```

If `RESIDUAL DIFFERENCE` cannot be written precisely, **keep mapping**.
