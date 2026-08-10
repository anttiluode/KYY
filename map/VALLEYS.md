# Candidate valleys after subtraction

This file is deliberately harsher than the root README. Each valley is a **residual** after subtracting known work.

Nothing here is labelled novel. The labels mean only how the current search looks.

Priority scale:

- **A** — small, falsifiable, fits KYY now.
- **B** — interesting but needs more literature work or infrastructure.
- **C** — conceptually related, currently too crowded or too broad.
- **PARK** — do not build in KYY now.

---

# Valley A0 — finish the modern state-tracking comparison properly

**Question**

How does a local Givens/scatter mesh compare with Householder-product and PD-style transitions on **modern state-tracking tasks**, when parameter count, state dimension, transition depth, and communication structure are reported explicitly?

**Nearest known objects**

- EUNN: local 2x2 unitary/Givens mesh, tunable depth/capacity.
- oRNN: Householder products.
- DeltaProduct: token-conditioned generalized Householder products for state tracking.
- PD-SSM: structured sparse transition with FSA guarantees.
- ISAN: input-switched recurrent operators.

**What is already dead**

The local 2-port architecture itself is not new. The Householder adversary is not new. "Depth is a capacity knob" is not new.

**Residual**

The targeted search did not immediately locate an EUNN-style **token-selective adjacent mesh benchmarked in the 2024-2026 state-tracking regime against DeltaProduct/PD-style controls**. This may exist under different terminology; treat this as `UNMAPPED`, not novel.

**Why do it anyway?**

Because KYY already has most of the harness, and it closes the current experiment honestly. It may produce a useful negative result or a compact comparison that reconnects a 2017 architecture family with a modern benchmark family.

**Do not rig the task for geometry.**

Use field-standard tasks: parity, modular counting, permutation composition, flip-flop, MQAR/associative recall variants if feasible.

**Measure**

- accuracy / final-state accuracy;
- length extrapolation;
- state dimension;
- trainable parameters;
- number of local 2-port cells per token;
- sequential mesh depth;
- maximum communication span per primitive;
- number of global reductions/broadcasts;
- wall-clock on CPU/GPU, explicitly labelled as software rather than hardware cost.

**Kill condition**

If matched Householder/PD-style models dominate accuracy, extrapolation, and a reasonable communication proxy across the standard tasks, close the algorithmic-locality branch. Keep only the physical-backend interpretation.

**Priority: A.** This is a finish-the-current-story experiment, not a new architecture project.

---

# Valley A1 — geometry-generated operator, not free edge parameters

This is the residual that most closely matches the original Geometric Neuron intuition after EUNN is subtracted.

## Proposed object

A fixed geometry `G` contains node positions, topology, lengths/delays, perhaps static material/coupling parameters. A small context vector `a_t` changes a few global/local control fields. The full recurrent operator is generated from these quantities:

```text
G = {positions, lengths, topology, static parameters}

context/token x_t
      |
      v
small control a_t  (dimension r << number of edges)
      |
      v
Q_t = F(G, a_t)
      |
      v
state propagation
```

The key restriction is:

> **There is not one independently learned dynamic parameter per edge.**

For example:

```text
w_ij(t) = f(distance_ij, direction_ij, type_ij, a_t)
```

or

```text
phase_e(t) = phase(length_e, global controls a_t)
```

## Prior-art boundary

This idea is surrounded.

- HyperNEAT generates large connectivity patterns as functions of node geometry/endpoint coordinates.
- HyperNetworks generate the weights of another network and have been applied to recurrent models.
- ISAN uses input-dependent recurrent matrices.
- EUNN gives the local unitary mesh.
- GraphCON makes oscillator coupling follow graph structure.
- physical wave neural networks encode inputs into parameters of a scattering system.

So "geometry generates weights", "small network generates big network", and "input changes recurrent weights" all exist separately.

## Residual

The current targeted search did **not** locate the exact conjunction:

> **modern token-by-token state tracking with a recurrent transition generated from fixed spatial geometry by a deliberately low-dimensional control signal, evaluated against modern selective/Householder/sparse SSM baselines.**

That statement is `BRIDGE/UNMAPPED`, not a novelty claim.

## Why it matters

This is different from EUNN in a way the original KYY implementation was not. EUNN learns the transformation parameters. Here geometry plus a small control law is the representation. It tests whether a large useful operator family can be *generated* rather than independently parameterized.

It also gives a real compression hypothesis:

```text
E edge controls per token
        vs
r << E controls per token + fixed geometry
```

## Cheapest falsifier before architecture work

Do **not** invent a new task. Take the current local scatter model and post-fit its learned token-specific edge angles with increasingly low-dimensional geometric generators:

1. constant/shared angle families;
2. Fourier functions of edge coordinate;
3. small basis `theta_e(token) = sum_k alpha_k(token) phi_k(e)`;
4. distance/direction basis if the substrate has coordinates.

Ask how small `r` can become before the already-learned state-tracking behavior collapses.

This is a **representation test**, not yet a new model.

If `r` must be essentially `E`, the geometry-generated story has no compression evidence on this task.

**Priority: A/B.** Search more before building a train-from-scratch version.

---

# Valley B0 — locality as communication cost, not FLOPs

**Question**

Can a transition family with nearest-neighbour primitives compete with a mathematically O(N) global transition while avoiding global reduction/broadcast?

This is the most defensible version of the hardware-locality intuition.

## Why FLOPs are insufficient

A dense Householder reflection can be O(N) arithmetic but requires a global dot product and distribution of the result. A checkerboard nearest-neighbour mesh also uses O(N) arithmetic but only local communication, at the cost of multiple propagation layers before distant coordinates interact.

Those costs look similar in Python and very different on some physical/mesh substrates.

## Prior-art boundary

- EUNN and photonic interferometer literature already exploit local 2-port meshes.
- hardware RNN literature emphasizes memory movement and physical mapping.
- optical unitary meshes already physically implement global matrices from local couplers.

Therefore "local wiring might be cheaper" is not new.

## Residual

A KYY contribution would need a **specific resource accounting framework** connected to the recurrent/state-tracking task:

```text
accuracy
vs
state
vs
parameters
vs
local depth
vs
wire span / reductions / broadcasts
```

and eventually a backend-specific cost model for TW-1A rather than generic claims about energy.

## Kill condition

If the local mesh requires depth proportional to state size just to match a two-reflector global baseline, and the target hardware has no compensating propagation advantage, the locality story is weak.

**Priority: B.** Useful only if tied to a real substrate/cost model.

---

# Valley B1 — structural input encoding + recurrence + state tracking

Wanjura & Marquardt's wave-scattering neural network encodes the input in the physical parameters of a linear scattering system. That is extremely close to a "moving operator":

```text
input x -> H(x, theta) -> scattering response
```

Modern LRNN work asks whether input-dependent transition operators can track finite states.

## Residual question

What happens when **structural input encoding in a reciprocal wave/scattering operator is recurrently composed over a sequence** and tested on state tracking?

```text
state z_t
   |
   v
physical/effective scattering operator S(x_t)
   |
   v
state z_{t+1}
```

## Prior-art risk

High. Physical reservoir computing, photonic RNNs, optoacoustic recurrent operators, EUNN, and modern oscillator-attention work are all nearby. A deeper search is required before code.

## Why it is interesting

This is where KYY and TWC genuinely meet: the same parameterized reciprocal system is both the sequence transition and a possible physical compile target.

## Cheapest falsifier

No hardware. Use an audited coupled-mode/scattering matrix simulator and ask whether a very small structurally modulated reciprocal network can solve one standard state-tracking task at all. Compare against an unconstrained input-switched linear operator of identical state size.

**Priority: B.** Potentially conceptually clean, but not until the literature map is deeper.

---

# Valley B2 — compile an effective recurrent operator into a constrained reciprocal substrate

This is less an architecture question than a compiler question.

## Input

A target family of recurrent transitions `A(x)` obtained from some trained sequence model.

## Output

A local reciprocal realization `Q(x)` or a short sequence of reciprocal/local primitives whose port-level transition approximates `A(x)` on the relevant subspace.

```text
trained abstract operator
        |
        v
factor / approximate / identify
        |
        v
local reciprocal program
        |
        v
TWC / TW-1A-like backend
```

## Prior-art boundary

- Givens/Clements meshes compile arbitrary unitary transforms into local 2-port optical elements.
- photonic inverse design maps target unitary transforms to physical geometries.
- TWC already maps sparse reciprocal operators to its own backend semantics.

## Residual

The residual cannot simply be "matrix -> mesh". That is old.

It would have to be something like:

> compile a **sequence-model transition family** into a constrained reciprocal transient-wave fabric while preserving the behaviorally important state subspace and reporting what cannot be realized.

That brings in approximation, non-unitarity/damping, causal state semantics, and identifiability.

**Priority: B.** This may ultimately be more distinctive than a new GPU architecture, but it depends on TWC rather than KYY alone.

---

# Valley C0 — realization-aware interpretability of selective SSM state

Originally this looked unusually open. The map makes it less so.

## Occupied neighbourhood

- classical realization theory;
- RNN realization/minimality theory;
- generic NN parameter symmetry/gauge work;
- automatic parameter-symmetry discovery;
- GENNI equivalence visualization;
- controllability/observability analyses;
- reproducible-subspace findings in SAEs.

## Possible residual

A very narrow architecture-specific tool might still be useful:

> given a particular selective SSM/LRNN parameterization, derive its **exact allowed realization group**, transform a trained checkpoint along that group, and automatically classify proposed hidden-state observables as invariant/equivariant/basis-dependent.

This is much narrower than "find gauge symmetries in AI."

## Why low priority

The symmetry/identifiability field is moving quickly and already has automatic methods. KYY has no special advantage here unless the selective-SSM realization group yields an unusually simple/useful audit.

**Priority: C.** Literature first, no code.

---

# Valley C1 — operator atlas / low-dimensional family of computations

Question:

> Do prompt/token-conditioned effective operators visited by a trained model live on a much smaller quotient manifold than the parameterization suggests?

This remains conceptually interesting, but it overlaps heavily with:

- low-rank RNN connectivity/dynamics;
- hypernetworks and low-dimensional task embeddings;
- model compression;
- representation similarity;
- symmetry quotienting;
- local linearization/control-theoretic interpretability.

Without a precise invariant and a strong baseline this becomes a very broad interpretability project.

**Priority: C.** Park until KYY has a sharper reason to need it.

---

# PARK — body operator / medical scanning

The living-body operator idea is conceptually connected but is not a KYY software research problem. It requires transducers, calibration, repeatable physical experiments, and eventually domain validation.

Do not let KYY become a proxy for that project.

---

# Current recommended order

```text
1. SEARCH / MAP
      |
      v
2. Close A0: modern state-tracking comparison of known families
      |
      v
3. Analyze A1 without new architecture:
   can existing learned edge controls be compressed into a small geometry basis?
      |
      +---- no --> geometry-generated control lacks evidence here; stop
      |
      `---- yes
             |
             v
4. Search the exact A1 conjunction again
             |
             v
5. Only then build a trainable geometry-generated version
             |
             v
6. If it survives, connect to B0/B2 hardware/compiler costs
```

The important change is step 3. We can test the central Geometric Neuron residual **without inventing another architecture first**.