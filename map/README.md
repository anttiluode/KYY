# KYY / map

This folder is the **research cartography layer** for KYY.

It exists because the project repeatedly derived something exciting, built it, and only afterwards discovered the nearest established name. The rule is:

> **Search first. Locate the nearest known object. Compute the residual. Build only the residual.**

`UNMAPPED != NEW`.

Snapshot: **2026-08-10**.

## What we are mapping

The starting question was:

> Can useful recurrent computation arise from **local geometry + propagation** rather than unrestricted global state mixing?

After subtracting prior art, the useful axes are now:

```text
BEHAVIOR
 automaton / transition monoid / group
        |
        v
REALIZATION
 hidden dimension / observable quotient / basis freedom
        |
        v
OPERATOR FAMILY
 commuting / noncommuting / input-switched / exact relations
        |
        v
LOCAL SUBSTRATE
 graph support / controls / physical primitive
        |
        v
SYNTHESIS
 reachable? -> local word -> parallel depth -> relation error -> drift
```

The Geometric Neuron intuition survives here only if geometry earns something on that whole chain.

## Start here — newest first

1. **[PASS9_LIVE_2026_BOUNDARY.md](PASS9_LIVE_2026_BOUNDARY.md)** — current live walls: hard group projection, holonomic/gauge-protected reasoning, sparse realization, and structure-aware automaton compilation.
2. **[PASS8_COXETER_IR.md](PASS8_COXETER_IR.md)** — exact full-`S5` compiler floor: 4D `A4` simple-root state, radius-1 generators, exact nearest-neighbour routing costs.
3. **[s5_coxeter_oracle.py](s5_coxeter_oracle.py)** — enumerates all 120 `S5` token operators and computes exact minimum adjacent-swap and parallel-matching depths.
4. **[PASS7_DISCRETE_SYNTHESIS.md](PASS7_DISCRETE_SYNTHESIS.md)** — catches the finite-group/Lie-algebra mismatch and adds an exact local `S3` oracle.
5. **[exact_perm3_local_oracle.py](exact_perm3_local_oracle.py)** — exact 3-channel local oracle for KYY's original `perm3`.
6. **[PASS6_BEHAVIORAL_QUOTIENT.md](PASS6_BEHAVIORAL_QUOTIENT.md)** — hidden dynamics are larger than what the readout necessarily identifies.
7. **[PASS5_CONTROL_ALGEBRA.md](PASS5_CONTROL_ALGEBRA.md)** — geometry/control placement -> dynamical Lie algebra.
8. **[RING_ALGEBRA_LEMMA.md](RING_ALGEBRA_LEMMA.md)** — exact toy result: even ring + one controlled edge gives `u(N/2)` inside `so(N)`.
9. **[control_algebra_probe.py](control_algebra_probe.py)** / **[operator_algebra_audit.py](operator_algebra_audit.py)** — diagnostics, not architectures.
10. **[OCCUPANCY_MATRIX.md](OCCUPANCY_MATRIX.md)** / **[LANDSCAPE.md](LANDSCAPE.md)** / **[VALLEYS.md](VALLEYS.md)** — wider map and older candidate valleys.
11. **[SEARCH_LOG.md](SEARCH_LOG.md)** / **[SOURCES.md](SOURCES.md)** / **[CONTROL_ALGEBRA_SOURCES.md](CONTROL_ALGEBRA_SOURCES.md)** — search history and bibliography.

## Hard landmarks already occupied

```text
local 2-port recurrent mesh        -> EUNN / Givens / optical meshes
Householder recurrence             -> oRNN / DeltaProduct
adaptive input-dependent unitary   -> AUSSM
bilinear state-tracking RNN        -> modern bilinear-RNN work
second-order oscillator RNN        -> coRNN
geometry-defined oscillator flow   -> GraphCON
wave physics mapped to an RNN      -> Hughes et al. 2019
input-dependent recurrent Q        -> ISAN / selective SSMs
input generates current operator   -> Dynamic Filters / HyperNetworks
Lie algebra explains sequence order-> 2026 Lie-algebraic sequence theory
few local controls steer networks  -> bilinear / quantum graph controllability
RNN behavior -> automaton          -> WFA / extraction / realization theory
permutation matrices track groups  -> representation theory + LRNN proofs
operator -> short local word       -> routing / circuit / gate synthesis
choose sparse equivalent realization-> classical control + 2026 sparse-realization work
hard projection protects group state-> live 2026 non-Abelian tracking work
non-Abelian topology/holonomy       -> live 2026 reasoning proposals
DFA -> compact IR -> local hardware -> live structure-aware quantum compilers
```

This is not discouraging. It means the problem finally has coordinates.

## The important exact results inside KYY

### 1. Geometry can impose an algebraic ceiling

For the toy skew-generator system in Pass 5, a path plus one local control generated full `so(N)` for the tested sizes.

For an even ring `N=2m` with only edge `(0,1)` controlled, the apparent numerical sequence

```text
4, 9, 16, 25, 36
```

was proved in [RING_ALGEBRA_LEMMA.md](RING_ALGEBRA_LEMMA.md): the generators preserve an orthogonal complex structure `J`, and

```text
Lie{G_ring, B_01} ~= u(m).
```

This is **not claimed as a new theorem**. Its value here is the design lesson:

```text
geometry -> symmetry -> reachable operator algebra.
```

### 2. A finite-group benchmark wants exact discrete operators, not a huge Lie algebra

KYY's original `S3` task has a trivial exact local implementation. Full `so(N)` controllability is irrelevant overkill.

That led to the rule:

> **Put an exact algebraic resource floor beside every learned state-tracking benchmark.**

### 3. Full S5 gives a nontrivial exact locality floor

Modern LRNN work uses full `S5`: every token may be any one of the 120 permutations.

The exact KYY oracle chooses the 4D standard representation in the `A4` simple-root basis. Each adjacent transposition acts only on its own root coordinate and immediate neighbours:

```text
c'_i = -c_i + c_(i-1) + c_(i+1).
```

So the behaviorally reduced representation stays radius-1 local on the Dynkin path.

For all 120 tokens, exact BFS compilation onto simultaneous nearest-neighbour matchings gives:

```text
behavioral state channels: 4
sequential adjacent-swap depth: mean 5, max 10
parallel local depth:            mean 403/120 ~= 3.3583, max 5
relation error:                  0 in exact arithmetic
long-horizon algebraic drift:    0
```

The parallel-depth histogram is:

```text
0: 1
1: 7
2: 16
3: 35
4: 46
5: 15
```

That is a concrete communication-vs-depth resource point, not a learned-model score.

## The realization/gauge lesson

The same `S5` behavior has at least two attractive forms:

```text
5D natural permutation state
    -> adjacent generator is a literal 2-port swap
    -> contains one invariant/redundant all-ones mode

4D simple-root state
    -> redundant mode removed
    -> generator remains radius-1 local
    -> primitive is now a 3-coordinate stencil
    -> preserves the A4 Cartan metric
```

So **minimum state dimension and minimum substrate cost need not choose the same realization**.

That observation itself is not new: control engineering has long optimized equivalent realizations for numerical/sparsity/implementation properties. The KYY question is whether doing it **jointly for an input-switched behavioral family** gives a useful recurrent compiler.

## Live 2026 warning

Recent preprints already occupy two tempting framings:

- hard projection onto non-Abelian state structure for extreme length extrapolation;
- holonomic/non-Abelian-gauge/topological reasoning.

Treat those as live boundary markers, not settled facts, but do **not** let KYY rediscover "protect the group relation" and call it new.

Likewise, structure-aware automaton -> compact IR -> nearest-neighbour quantum compilation and exact sparse LTI realization are already active compiler/realization research.

## Current pin on the map

The residual has become a compiler question rather than a neural-layer question:

> **Given a task transition monoid/group and a declared local substrate, choose a behaviorally sufficient recurrent realization and compile the whole input-switched transition family into minimum-cost local operator words, preserving the observable algebraic relations and reporting long-horizon drift.**

The key phrase is **the whole family**. Independent matrix approximation is not enough if it breaks relations under repeated composition.

A candidate cost vector is:

```text
behavioral state dimension
runtime control bits / ports
primitive support / wire span
sequential and parallel local depth
global reductions / broadcasts
relation defect
observable quotient error
long-horizon drift
```

Status: **BRIDGE / HIGH PRIOR-ART RISK / TESTABLE. NOT A NOVELTY CLAIM.**

## No new learned architecture yet

Next gates:

1. keep exact oracle floors beside the field-standard group tasks;
2. finish the learned-model operator/continuation audit;
3. search **joint family realization + relation-preserving local synthesis** as an exact phrase/concept, including reversible/quantum/distributed-control literature;
4. if still useful, compare the exact `S5` local compiler point with Householder/PD-style transition costs under an explicit communication model;
5. only then decide whether KYY needs another trainable model at all.

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
