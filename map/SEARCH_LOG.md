# Search log

This file exists to fix the process bug that produced repeated **build first -> prior art later** collisions.

Date: **2026-08-10**

Rule:

> A search miss means only **not located under these terms**. It does not establish novelty.

---

## Search pass 1 — local two-port recurrent meshes

### Proposed object

Checkerboard nearest-neighbour 2x2 orthogonal/scattering cells composing into a global recurrent transition.

### Queries / concepts searched

- efficient unitary RNN Givens rotations
- local 2x2 unitary recurrent mesh
- orthogonal RNN Householder
- optical interferometer unitary Givens mesh

### Hits

- Jing et al. 2017, EUNN — very close / direct hit.
- Mhammedi et al. 2017, Householder oRNN — direct control prior art.
- Clements et al. 2016 — physical 2-port rectangular unitary mesh.
- DizzyRNN — Givens recurrent parameterization.

### Status

**OCCUPIED.**

### Residual

Token-selective/local-Givens mesh on the **modern state-tracking benchmark suite** versus DeltaProduct/PD-style models was not immediately located. Search must be repeated with terminology from current LRNN papers before claiming even an incremental gap.

---

## Search pass 2 — second-order wave / oscillator RNN

### Proposed object

Position/momentum state, second-order recurrent dynamics, controlled/damped oscillation, possibly graph local.

### Queries / concepts searched

- coupled oscillator recurrent neural network sequence modeling
- second order recurrent neural network oscillator
- graph coupled oscillator neural network

### Hits

- coRNN (Rusch & Mishra 2020) — direct hit on second-order controlled oscillator recurrence.
- GraphCON (Rusch et al. 2022) — graph-defined second-order controlled/damped oscillator network.
- Neural Oscillators are Universal (2023) — broad theoretical umbrella.

### Status

**OCCUPIED.**

### Residual

None at the level of "wave/oscillator recurrence". Any KYY residual must add a more specific structural/resource constraint.

---

## Search pass 3 — input-dependent / moving recurrent operator

### Proposed object

`Q_t` changes with the current input/token.

### Queries / concepts searched

- input dependent recurrent matrix
- input switched affine recurrent network
- selective state space recurrent transition
- dynamic recurrent operator

### Hits

- Input Switched Affine Networks (Foerster et al. 2016/2017).
- Mamba / Mamba-2 selective SSM line.
- Mamba-3 richer input-dependent state-space dynamics.
- HyperNetworks generate network weights, including recurrent applications.

### Status

**OCCUPIED.**

### Residual

The source/constraint of the operator family, not its input dependence.

---

## Search pass 4 — modern state-tracking transition algebra

### Proposed object

Use richer structured transitions to solve parity, modular counting, permutation composition and related state-tracking tasks while preserving efficient recurrence.

### Queries / concepts searched

- state tracking linear RNN negative eigenvalues
- complex eigenvalues modulo counting LRNN
- Householder products state tracking
- structured sparse transition state-space model finite automata

### Hits

- Grazzi et al., negative eigenvalues / complex eigenvalues.
- DeltaProduct, generalized Householder products.
- PD-SSM, structured sparse permutation-diagonal transition.
- Flash PD-SSM.
- Mamba-3.

### Status

**OCCUPIED / active frontier.**

### Residual

Only resource/support/physical-realization questions that are not captured by existing transition families.

---

## Search pass 5 — geometry generates connectivity

### Proposed object

Large connectivity/operator derived from coordinates, distances or geometry rather than one free parameter per connection.

### Queries / concepts searched

- geometry generated neural connectivity weights
- neural network connectivity as function of coordinates
- geometry substrate recurrent neural network
- low-dimensional generator large neural network weights

### Hits

- HyperNEAT: CPPN-generated connectivity as a function of endpoint geometry.
- ES-HyperNEAT: geometry/topology generation and substrate placement.
- HyperNetworks: compact networks generate weights of target networks.
- low-rank RNN literature: low-dimensional connectivity structure generates low-dimensional recurrent dynamics.

### Status

**OCCUPIED in broad form.**

### Residual searched

> token-by-token state tracking where a deliberately low-dimensional control vector modulates a **fixed geometry-derived transition family**, with control dimension explicitly much smaller than edge count.

### Result

No exact hit located in this pass.

### Status

**BRIDGE / UNMAPPED. Do not call novel.**

This is currently the most important search target before any new KYY architecture.

---

## Search pass 6 — graph / topology defines dynamics

### Proposed object

Graph topology is part of the recurrent computation; local coupling and repeated propagation generate global behavior.

### Queries / concepts searched

- graph Laplacian recurrent neural network
- graph state space model Laplacian
- reservoir topology computation
- ring reservoir neural network

### Hits

- GraphCON.
- GraphSSM.
- Ring Reservoir Neural Networks for Graphs.
- topology/connectome reservoir studies.

### Status

**OCCUPIED.**

### Residual

Must be a specific modern sequence/resource result, not "topology matters."

---

## Search pass 7 — delay geometry as computation

### Proposed object

Delay lines / path length create memory and high-dimensional state.

### Queries / concepts searched

- delay based reservoir computing single dynamical node
- time delay reservoir embedding theory
- photonic delay reservoir

### Hits

- Appeltant et al. 2011 single-node delayed reservoir.
- later delayed-reservoir/embedding theory.
- many photonic/physical reservoir implementations.

### Status

**OCCUPIED.**

---

## Search pass 8 — physical scattering operator as neural computation

### Proposed object

Input changes parameters of a physical wave system; the resulting scattering response performs computation.

### Queries / concepts searched

- linear wave scattering neural computation input encoded physical parameters
- programmable metasurface structural input encoding neural network
- wave scattering physical neural network recurrent

### Hits

- Wanjura & Marquardt 2024: direct hit on input encoded into physical scattering parameters.
- Hammami et al. 2026: structural input encoding / mutual coupling / depth in programmable wave physical NNs.
- physical reservoir and photonic RNN literature.

### Status

**OCCUPIED in feedforward/physical-computing form.**

### Residual searched

> recurrent composition of structurally input-encoded reciprocal scattering operators specifically for modern state tracking.

### Result

No exact hit located in this pass.

### Status

**BRIDGE / UNMAPPED. High prior-art risk.**

---

## Search pass 9 — hidden-state gauge / identifiability audit

### Proposed object

Determine which internal claims survive function-preserving reparameterizations / realization transformations.

### Queries / concepts searched

- neural network gauge symmetry
- parameter symmetry neural identifiability
- automatic symmetry discovery architecture
- recurrent neural network realization theory
- controllability observability neural interpretability
- Mamba controllability observability

### Hits

- RNN realization theory (Defourneau & Petreczky).
- affine-symmetry neural identifiability.
- GENNI equivalence-space visualization.
- gauge-symmetry formulations for neural networks/transformers.
- parameter-symmetry surveys and symmetry-reduced networks.
- automated parameter-symmetry discovery (Zhao et al.).
- control-theoretic NN interpretability.
- Sparse Mamba controllability/observability.
- 2026 SAE work: unstable individual features but reproducible subspaces.

### Status

**Crowded.**

### Residual

Possible only if narrowed to an exact selective-SSM realization group and a concrete invariant/non-invariant classification not already provided by generic symmetry tools.

### Priority

Low until that exact distinction is found.

---

## Search pass 10 — physical/local implementation of recurrent matrices

### Proposed object

Local physical primitives compose into global recurrent transforms; possible hardware advantage from local communication.

### Queries / concepts searched

- photonic unitary recurrent neural network mesh
- local 2x2 recurrent optical matrix
- photonic RNN accelerator
- coupled waveguide unitary inverse design
- optoacoustic recurrent operator

### Hits

- Clements/unitary photonic meshes.
- photonic circuit work explicitly looping a unitary mesh as an RNN.
- RecLight.
- OREO.
- coupled-waveguide unitary inverse design.

### Status

**OCCUPIED as hardware principle.**

### Residual

A KYY/TWC result has to be backend-specific: e.g. compile a behaviorally relevant recurrent operator family into **this constrained reciprocal substrate** and report approximation, routing, depth, and impossible directions.

---

# Current unresolved searches

These are the queries to pursue before the next build.

## U1 — exact A1 conjunction

Search for all variants of:

```text
input-conditioned geometry-generated recurrent transition
coordinate-generated recurrent weights state tracking
hypernetwork geometric recurrent operator state space model
low-dimensional control graph recurrent transition
structured generator token-dependent transition matrix
```

Need to inspect references/citations around HyperNEAT, HyperNetworks, dynamic filter networks, ISAN, GraphCON, structured RNNs, and modern LRNN state tracking.

## U2 — token-dependent EUNN / Givens state tracking

```text
dynamic EUNN
input-dependent unitary RNN
switched orthogonal RNN
conditional Givens recurrent transition
Givens state tracking linear RNN
```

A search miss so far is not enough.

## U3 — communication-aware state-tracking comparisons

```text
state space model communication cost interconnect
linear RNN global reduction hardware cost
nearest-neighbour recurrent transition state tracking hardware
local communication orthogonal RNN
```

Need to distinguish theoretical arithmetic complexity from physical communication/synchronization cost.

## U4 — recurrent structural wave encoding

```text
structural input encoding recurrent wave neural network
input-modulated scattering recurrent neural network
programmable metasurface recurrent state tracking
coupled-mode recurrent state machine
```

Very high risk of prior art in photonics/reservoir literature.

---

# Mandatory pre-build template

Copy this section and fill it before adding a new model to `kyy/models.py`.

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

If `RESIDUAL DIFFERENCE` cannot be written in one precise paragraph, do not code yet.