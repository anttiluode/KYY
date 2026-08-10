# KYY / map

This folder is the **research cartography layer** for KYY.

It exists because the project repeatedly derived something exciting, built it, and only afterwards discovered the nearest established name. The rule here is:

> **Search first. Locate the nearest known object. Compute the residual. Build only the residual.**

`UNMAPPED != NEW`.

Snapshot: **2026-08-10**.

## What we are mapping

The cloudy starting proposition was:

> Can useful recurrent computation arise from **local geometry + propagation** rather than unrestricted global state mixing?

That crosses several mature fields, so the map uses several axes at once:

```text
TRANSITION ALGEBRA
 diagonal -> complex -> block/sparse -> Householder/Givens -> dense

COMMUNICATION
 independent -> small block -> graph-local -> sparse long-range -> global

OPERATOR SOURCE
 fixed -> learned -> input-conditioned -> low-control tied -> geometry-generated -> physical

DYNAMICS
 leaky memory -> counting -> noncommutative tracking -> oscillation -> waves/scattering

CONTROL ALGEBRA
 commuting generators -> restricted symmetry algebra -> full reachable algebra

REALIZATION / OBSERVATION
 hidden coordinates -> reachable/observable quotient -> behavioral automaton

SYNTHESIS
 reachable -> exact/approximate token operator -> local word depth -> long-horizon drift
```

The Geometric Neuron intuition now lives in a conjunction:

```text
local geometry
      +
few dynamic controls
      +
propagation / composition
      +
behaviorally sufficient state
      +
local readout / declared ports
      +
resource-audited synthesis
```

## Start here

1. **[PASS7_DISCRETE_SYNTHESIS.md](PASS7_DISCRETE_SYNTHESIS.md)** — catches the finite-group/Lie-algebra mismatch and adds the exact local `S3` resource floor.
2. **[exact_perm3_local_oracle.py](exact_perm3_local_oracle.py)** — exact 3-channel, depth≤2, nearest-neighbour oracle for KYY's `perm3`; known permutation-matrix construction, not a novelty claim.
3. **[PASS5_CONTROL_ALGEBRA.md](PASS5_CONTROL_ALGEBRA.md)** — local geometry/control placement -> dynamical Lie algebra.
4. **[RING_ALGEBRA_LEMMA.md](RING_ALGEBRA_LEMMA.md)** — proves the specific even-ring/one-control toy algebra is `u(N/2)` inside `so(N)`.
5. **[PASS6_BEHAVIORAL_QUOTIENT.md](PASS6_BEHAVIORAL_QUOTIENT.md)** — hidden algebra versus what the task readout can actually identify.
6. **[control_algebra_probe.py](control_algebra_probe.py)** — robust numerical Lie-closure probe for path/ring geometries.
7. **[operator_algebra_audit.py](operator_algebra_audit.py)** — audits the already-existing learned KYY transition families.
8. **[CONTROL_ALGEBRA_SOURCES.md](CONTROL_ALGEBRA_SOURCES.md)** — focused bibliography for the current seam.
9. **[OCCUPANCY_MATRIX.md](OCCUPANCY_MATRIX.md)** / **[LANDSCAPE.md](LANDSCAPE.md)** / **[VALLEYS.md](VALLEYS.md)** — wider map.
10. **[SEARCH_LOG.md](SEARCH_LOG.md)** / **[SOURCES.md](SOURCES.md)** — search history and broad bibliography.

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
rich state mixing for tracking     -> DeltaProduct / PD / BD-LRU / SLiCE / bilinear / fixed-point
Lie algebra explains order         -> 2026 Lie-algebraic sequence-model theory
few local controls steer networks  -> classical bilinear / quantum graph controllability
RNN behavior -> automaton          -> WFA / automata extraction / realization theory
permutation matrices track groups  -> classical representation theory + LRNN state-tracking proofs
operator -> short gate word        -> control / circuit / quantum synthesis literature
geometry generates connectivity    -> HyperNEAT / indirect encodings
input changes scattering operator  -> Wanjura-Marq. / programmable wave systems
abstract matrix -> physical device -> photonic surrogate & inverse-design work
hidden-state gauge freedom         -> realization & NN symmetry literature
```

This is not discouraging. It gives the problem coordinates.

## Pass 5: geometry as a control algebra

Let

```text
X_ij = E_ij - E_ji
G_path = X_01 + X_12 + ... + X_(N-2,N-1)
B = X_01.
```

For every tested `N=3..12`, the numerical Lie closure is full `so(N)`. This is standard controllability territory.

The even ring is more revealing. For `N=2m`, one controlled edge produces dimensions `m^2`. We no longer leave that as a numerical guess: [RING_ALGEBRA_LEMMA.md](RING_ALGEBRA_LEMMA.md) constructs an orthogonal complex structure `J` commuting with the drift and control, giving an upper bound `u(m)`, then generates the whole `u(m)` in `J`-adapted coordinates. Thus for this exact toy construction:

```text
Lie{G_even_ring, B_01} ~= u(m),   m=N/2.
```

This is a derived lemma, **not claimed new to the literature**. Its KYY value is the structural lesson:

```text
geometry -> symmetry -> reachable operator algebra.
```

An optimizer cannot escape a symmetry enforced by the generator set.

## Pass 6: the hidden algebra is not automatically the computation

A model can implement

```text
large hidden dynamics
        |
        v
behavioral / observable quotient
        |
        v
small target automaton
```

and solve a task even when its full hidden matrices do not obey the target group's relations everywhere. Weighted-automata, RNN-realization, and automata-extraction literature already map this distinction.

So KYY must price the **behaviorally visible quotient**, not reward hidden algebra dimension for its own sake.

## Pass 7: a finite group is not a Lie algebra target

`perm3` asks for a discrete `S3` computation. Full `so(N)` controllability is massive overkill.

The exact task already has a tiny local oracle:

```text
state: h0=(1,2,3)
transition: 3x3 permutation matrices
hardware geometry: path 0--1--2
identity depth: 0
swap depth:     1 local 2-port reflection
3-cycle depth:  2 adjacent reflections
relation error: 0
length drift:   0
```

This construction is not ours; the modern LRNN state-tracking literature explicitly uses distinct-vector/permutation-matrix realizations and decomposes permutations into swaps/Householders. KYY now includes the oracle because **learned models should be compared with the exact algebraic resource floor of the benchmark**.

This changes the compiler question. Lie closure answers whether a target is reachable. The next issue is the shortest exact or sufficiently stable local word implementing each behaviorally required token operator.

## Current pin on the map

Three mature theories now meet:

```text
automata / representation / realization
        -> what behavior must be represented?

control / Lie algebra
        -> what can the constrained geometry reach?

operator / circuit synthesis
        -> how cheaply can it reach the required transitions?
```

The exact bridge currently left after subtraction is:

> **Given a task transition monoid/group and a constrained local substrate, find a low-dimensional behaviorally sufficient recurrent realization and compile its token transitions into minimum-cost local operator words, with explicit relation error and long-horizon guarantees.**

Status: **BRIDGE / HIGH PRIOR-ART RISK. NOT A NOVELTY CLAIM.**

A useful resource report now looks like:

```text
behaviorally sufficient state dimension
control ports / local primitive family
wire span
word depth per token
exact/approximate group or monoid relation defects
observable quotient error
length extrapolation / drift
```

The geometry only earns its place if it improves that joint realization+synthesis resource point.

## No new learned architecture yet

The next gates are:

1. keep exact oracle floors beside every algebraic state-tracking task;
2. audit learned KYY models against their defining relations and continuation-equivalence, not just accuracy;
3. search the automata-to-local-operator-synthesis conjunction more deeply;
4. only then attempt a compiler/synthesis experiment on a harder standard group/monoid where the local resource trade-off is nontrivial;
5. close the AI-efficiency branch if standard structured transitions dominate once the correct oracle floor is included.

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
