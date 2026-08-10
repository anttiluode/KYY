# KYY / map

This folder is the **research cartography layer** for KYY.

It exists because the project repeatedly derived something exciting, built it, and only afterwards discovered the nearest established name. The fix is:

> **Search first. Locate the nearest known object. Compute the residual. Build only the residual.**

`map/` is not a novelty claim. It is a live attempt to draw enough of the known mathematical landscape around the Geometric Neuron / KYY idea that we stop mistaking a known landmark for an unmapped valley.

Snapshot: **2026-08-10**.

## What we are mapping

The cloudy original proposition was:

> Can useful recurrent computation arise from **local geometry + propagation** rather than unrestricted global state mixing?

That crosses several mature fields. We map it on multiple axes:

```text
TRANSITION ALGEBRA
 diagonal -> complex -> block/sparse -> Householder/Givens -> dense

COMMUNICATION
 independent -> small block -> graph-local -> sparse long-range -> global

OPERATOR SOURCE
 fixed -> learned -> input-conditioned -> low-control tied -> geometry-generated -> physical

DYNAMICS
 leaky memory -> counting -> noncommutative tracking -> oscillation -> waves/scattering

REALIZATION
 checkpoint coordinates -> symmetry orbit -> invariant subspace -> I/O equivalence
```

The Geometric Neuron intuition lives in a conjunction:

```text
local geometry
      +
propagation / modes
      +
few dynamic controls
      +
recurrent state
      +
local readout
      +
testable input/output behaviour
```

## Legend

- **OCCUPIED** — exact or very close mechanism already exists.
- **NEIGHBOUR** — a major ingredient is established but an important axis differs.
- **BRIDGE** — both sides exist; exact conjunction not located in the current search.
- **UNMAPPED** — targeted search did not locate the exact object.

`UNMAPPED != NEW`.

## Start here

1. **[OCCUPANCY_MATRIX.md](OCCUPANCY_MATRIX.md)** — compact map: papers/families × mechanisms.
2. **[DEEP_PASS_2026-08-10.md](DEEP_PASS_2026-08-10.md)** — second pass: wave-as-RNN, fan-in/FSM theory, richer modern state mixing, TCP-SSM.
3. **[PASS3_EXACT_RESIDUAL.md](PASS3_EXACT_RESIDUAL.md)** — third pass: AUSSM, Dynamic Filters, HyperNetworks, and the exact geometry-tying residual.
4. **[PASS4_COMPILER_BOUNDARY.md](PASS4_COMPILER_BOUNDARY.md)** — fourth pass: subtracts photonic surrogate/inverse-design prior art from the KYY↔TWC compiler idea.
5. **[LANDSCAPE.md](LANDSCAPE.md)** — detailed regions and borders.
6. **[VALLEYS.md](VALLEYS.md)** — residual hypotheses and kill conditions.
7. **[SEARCH_LOG.md](SEARCH_LOG.md)** — what was searched and what remains unresolved.
8. **[SOURCES.md](SOURCES.md)** — annotated primary-source bibliography.

## The hard landmarks now on the map

```text
local 2-port recurrent mesh        -> EUNN / Givens / optical meshes
Householder recurrence             -> oRNN / DeltaProduct
adaptive input-dependent unitary   -> AUSSM
second-order oscillator RNN        -> coRNN
geometry-defined oscillator flow   -> GraphCON
wave physics mapped to an RNN      -> Hughes et al. 2019
input-dependent recurrent Q        -> ISAN / selective SSMs
input generates current operator   -> Dynamic Filters / HyperNetworks
shared dynamics + token control    -> TCP-SSM / selective SSM family
rich state mixing for tracking     -> DeltaProduct / PD / BD-LRU / SLiCE / bilinear / fixed-point
geometry generates connectivity    -> HyperNEAT / indirect encodings
local fan-in vs FSM complexity     -> Horne & Hush 1993
input changes scattering operator  -> Wanjura-Marq. / programmable wave systems
abstract matrix -> physical device -> photonic surrogate & inverse-design work
hidden-state gauge freedom         -> realization & NN symmetry literature
```

This is not discouraging. It means the map has coordinates.

## Where the map points now

The strongest remaining KYY question is no longer "invent another recurrent architecture." It is a cheaper structural test:

> **Do the freely learned local KYY operators already collapse onto a small geometry-derived basis?**

The current `geom_scatter` has essentially free token × sweep × edge angles. Analyze that tensor before replacing it with a new model.

```text
learned free-edge operator
          |
          +--> generic best low-rank/SVD basis
          |
          +--> Fourier basis over ring geometry
          |
          +--> graph-Laplacian modes
          |
          +--> tiny coordinate/CPPN generator
          |
          v
 replace free angles with reconstruction
          |
          v
 rerun state tracking
```

The central adversarial comparison is:

```text
GEOMETRY BASIS  vs  BEST GENERIC RANK-r BASIS
```

at the same control dimension `r`.

If geometry needs nearly one degree of freedom per edge, or loses badly to a generic rank-`r` basis, the strongest remaining Geometric-Neuron representation story has no evidence here.

If a tiny geometry-derived `r` preserves behavior **and** competes with the best generic compression, then we have earned a reason to search the exact conjunction harder and only then build a trainable geometry-generated operator.

That is the first valley currently visible on the map.

## The compiler is not an escape hatch

The broad idea

```text
learn abstract operator -> realize it in physical wave hardware
```

also has direct photonic prior art. The possible TWC/KYY residual is narrower: compile an **input-conditioned recurrent operator family** into a specific constrained reciprocal transient-wave substrate, preserve only behaviorally load-bearing state, price runtime controls, and report unrealizable directions rather than pretending every target can be mapped.

See [PASS4_COMPILER_BOUNDARY.md](PASS4_COMPILER_BOUNDARY.md).

## Mandatory build gate

Before adding another architecture to `kyy/models.py`, add this to `SEARCH_LOG.md`:

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

That is what this folder is for.