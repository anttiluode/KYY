# KYY / map

This folder is the **research cartography layer** for KYY.

It exists because the project repeatedly derived something exciting, built it, and only afterwards discovered the nearest established name. The fix is simple:

> **Search first. Locate the nearest known object. Compute the residual. Build only the residual.**

`map/` is not a novelty claim. It is an attempt to draw enough of the known mathematical landscape around the Geometric Neuron / KYY idea that we stop mistaking a known landmark for an unmapped valley.

Snapshot: **2026-08-10**.

## What we are mapping

The cloudy original proposition was:

> Can useful recurrent computation arise from **local geometry + propagation** rather than unrestricted global state mixing?

That sentence crosses mature fields. We therefore map it on several axes at once:

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

The Geometric Neuron intuition lives in a **conjunction**, not at one point:

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

1. **[OCCUPANCY_MATRIX.md](OCCUPANCY_MATRIX.md)** — the compact map. Rows are known families; columns are mechanisms. This is the fastest way to see which conjunctions are actually unoccupied in the current search.
2. **[DEEP_PASS_2026-08-10.md](DEEP_PASS_2026-08-10.md)** — second search pass. This added several hard walls that materially changed the frontier.
3. **[LANDSCAPE.md](LANDSCAPE.md)** — detailed regions and borders.
4. **[VALLEYS.md](VALLEYS.md)** — candidate residuals and their kill conditions.
5. **[SEARCH_LOG.md](SEARCH_LOG.md)** — what was searched, what hit, and what remains unresolved.
6. **[SOURCES.md](SOURCES.md)** — annotated primary-source bibliography.

## The hard landmarks now on the map

The broad territory is far more occupied than the first KYY sketch suggested:

```text
local 2-port recurrent mesh       -> EUNN / Givens / optical meshes
Householder recurrence            -> oRNN / DeltaProduct
second-order oscillator RNN       -> coRNN
geometry-defined oscillator flow  -> GraphCON
wave physics mapped to an RNN     -> Hughes et al. 2019
input-dependent recurrent Q       -> ISAN / selective SSMs
shared dynamics + token control   -> modern selective SSMs / TCP-SSM
rich state mixing for tracking    -> DeltaProduct / PD / BD-LRU / SLiCE / bilinear RNNs
geometry generates connectivity   -> HyperNEAT / indirect encodings
local fan-in vs FSM complexity    -> Horne & Hush 1993
input changes scattering operator -> Wanjura-Marq. / programmable wave systems
hidden-state gauge freedom        -> realization & NN symmetry literature
```

This is not discouraging. It means the map has coordinates.

## Where the map points *now*

The strongest remaining KYY question is no longer "invent another recurrent architecture." It is a cheaper structural test:

> **Do the freely learned local KYY operators already collapse onto a small geometry-derived basis?**

The current `geom_scatter` has essentially free token × sweep × edge angles. Before replacing it with another architecture, analyze that learned tensor.

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

The key comparison is **geometry basis versus generic low-rank basis at the same control dimension `r`**.

If geometry needs almost one degree of freedom per edge, the strongest remaining Geometric-Neuron representation claim has no evidence here.

If a tiny geometry-derived `r` preserves the behavior and competes with the best generic rank-`r` compression, then we have earned a reason to search that exact conjunction harder and only then build a trainable geometry-generated operator.

That is the first valley currently visible on the map.

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