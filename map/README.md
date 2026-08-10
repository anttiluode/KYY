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
 commuting generators -> small noncommutative algebra -> full reachable algebra

REALIZATION / OBSERVATION
 hidden coordinates -> reachable/observable quotient -> behavioral automaton
```

The Geometric Neuron intuition now lives in a conjunction:

```text
local geometry
      +
few dynamic control ports
      +
propagation / commutators
      +
noncommutative state evolution when required
      +
local readout / declared ports
      +
testable input/output behaviour
```

## Start here

1. **[PASS5_CONTROL_ALGEBRA.md](PASS5_CONTROL_ALGEBRA.md)** — connects modern Lie-algebraic state-tracking theory to graph/local controllability and gives a reproducible local-control calculation.
2. **[PASS6_BEHAVIORAL_QUOTIENT.md](PASS6_BEHAVIORAL_QUOTIENT.md)** — asks which part of the hidden algebra is actually visible/required at the readout; connects automata/minimal realization to the control-algebra question.
3. **[control_algebra_probe.py](control_algebra_probe.py)** — robust numerical Lie-closure probe for path/ring geometries plus one or two local control ports.
4. **[operator_algebra_audit.py](operator_algebra_audit.py)** — retrains existing KYY linear models and reports affine transition commutators, finite word-span dimension, length extrapolation, and diagnostic `S3` relation defects.
5. **[OCCUPANCY_MATRIX.md](OCCUPANCY_MATRIX.md)** — papers/families × mechanisms.
6. **[PASS3_EXACT_RESIDUAL.md](PASS3_EXACT_RESIDUAL.md)** — AUSSM, Dynamic Filters, HyperNetworks, and the geometry-tying boundary.
7. **[PASS4_COMPILER_BOUNDARY.md](PASS4_COMPILER_BOUNDARY.md)** — subtracts photonic surrogate/inverse-design prior art from the KYY↔TWC compiler idea.
8. **[DEEP_PASS_2026-08-10.md](DEEP_PASS_2026-08-10.md)** — wave-as-RNN, fan-in/FSM theory, richer modern state mixing, TCP-SSM.
9. **[LANDSCAPE.md](LANDSCAPE.md)** / **[VALLEYS.md](VALLEYS.md)** — detailed regions and residual hypotheses.
10. **[SEARCH_LOG.md](SEARCH_LOG.md)** / **[SOURCES.md](SOURCES.md)** — search history and bibliography.

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

The corrected numerical Lie closure gives, for every tested `N=3..12`,

```text
dim Lie{G_path, B} = dim so(N) = N(N-1)/2.
```

So one fixed local path drift plus one locally addressable edge can generate the full rotation algebra. That is established controllability mathematics, not a KYY novelty claim.

The ring is more revealing. With one local control, the corrected even-`N` dimensions for `N=4,6,8,10,12` are

```text
4, 9, 16, 25, 36 = (N/2)^2,
```

while the tested odd rings close to full `so(N)`. A second neighboring control restores full closure in every even case in the table. The `(N/2)^2` sequence strongly suggests a `u(N/2)`-type symmetry-restricted algebra; that identification is a hypothesis here, not yet a proved KYY theorem.

The point is structural: **geometry + symmetry + control placement decide the generated algebra before training begins.**

## Pass 6: the hidden algebra is not automatically the computation

Maximizing hidden Lie dimension is the wrong objective if most of it is invisible at the readout.

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

and solve a task even when its full hidden matrices do not obey the target group's relations everywhere. Weighted-automata, RNN-realization, and automata-extraction literature already maps this distinction.

So the emerging KYY target is now:

> **Find the smallest locally controlled geometry whose behaviorally visible quotient implements the required state-transition algebra.**

That is stricter than "generate a huge Lie algebra" and closer to the TWC discipline of reasoning only about what declared ports can identify.

## Current pin on the map

Both sides are known:

```text
modern sequence theory:
    operator algebra / noncommutativity -> state tracking

control theory:
    graph + sparse controls -> reachable dynamical algebra

automata / realization theory:
    hidden dynamics -> minimal observable behavior
```

The exact bridge currently left after subtraction is:

> **Can a sparse local geometry with very few input-conditioned control ports realize the task-relevant behavioral quotient with a better control/communication/depth trade-off than existing efficient recurrent transitions?**

Status: **BRIDGE / UNMAPPED. NOT A NOVELTY CLAIM.**

A working metric is **behavioral control-algebra efficiency**:

```text
task-relevant quotient capability
---------------------------------
control ports × local depth × communication cost
```

with state size, algebra class, word-span/commutator diagnostics, accuracy, and length extrapolation reported alongside it.

## No new architecture yet

The next gates are diagnostics:

1. audit transitions of the models already trained in KYY;
2. map Lie closure versus geometry/control placement before optimization;
3. test equality of task-equivalent words under long continuations rather than demanding identical hidden matrices;
4. only then attempt short-word synthesis of standard state-tracking transitions from local controls;
5. close the AI-efficiency branch if local depth/control count destroys the resource case.

## Mandatory build gate

Before another architecture is added to `kyy/models.py`, add this to `SEARCH_LOG.md`:

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
