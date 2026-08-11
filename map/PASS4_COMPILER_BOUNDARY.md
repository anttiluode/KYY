# Pass 4 — compiler boundary

Date: 2026-08-10

The architecture side of KYY became increasingly crowded, which made the TWC/compiler side look comparatively distinctive. This pass checks that instinct before it becomes another build-first collision.

---

## Direct hit: abstract operator first, physical realization second

**Muda & Teğin — Scalable Photonic Neural Networks via Surrogate Scattering-Matrix Inverse Design** (2026)  
https://arxiv.org/abs/2604.21301

Their workflow deliberately separates task learning from electromagnetic realization:

```text
TASK
 |
 v
passive complex matrix surrogate
 |
 | learn target operator cheaply
 v
selected target matrix
 |
 | adjoint physical inverse design
 v
fabrication-aware freeform photonic device
```

That is a major hit on the broad TWC/KYY sentence:

> "Learn an operator in software, then compile/realize it in wave hardware."

That sentence is not ours.

The paper also uses a banded/local routing structure combined with an evanescent-coupling region to realize dense effective operators, further occupying the idea that clever local physical structure can implement an abstract dense transform.

---

## Earlier hard landmarks already around the same region

### Clements / programmable unitary meshes

Arbitrary unitary matrices can be decomposed into local 2-port interferometer operations.

### inverse-designed wave RNN

Hughes et al. 2019 inverse-design a physical wave medium whose temporal dynamics implement RNN-like computation.

### coupled-waveguide unitary inverse design

Recent photonic work inverse-designs coupled-waveguide geometry to approximate target unitary matrices.

### scattering-matrix differentiable/inverse design

A broad photonics literature optimizes physical geometry against target scattering/transmission behavior, including differentiable scattering-matrix methods.

### Wanjura–Marquardt

Physical parameters of a coupled scattering system are trained while input data alter scattering parameters, making the physical operator itself part of computation.

---

# What is therefore occupied

Do **not** frame any future TWC/KYY result as a discovery of:

```text
abstract matrix -> physical wave geometry
train surrogate -> realize physical device
local wave couplers -> global matrix
inverse design physical scattering response
wave medium -> recurrent computation
```

All of those have clear ancestors.

---

# What remains different about TWC, if anything

The possible residual is a conjunction of constraints rather than the compiler pattern itself.

## 1. Recurrent **transition family**, not one static target matrix

A sequence model has

```text
A(x_t)
```

or a family of token/context-conditioned transitions, not just one classification operator.

The compilation target is therefore a **family** of state updates whose controls must remain cheap at runtime.

## 2. Reciprocal transient semantics

TW-1A/TWC is not an arbitrary universal photonic unitary mesh. Its candidate physical semantics are a sparse reciprocal transient-wave body with particular local bond/self/state primitives.

The question is not merely whether a matrix factorization exists. It is whether the desired recurrent behavior is realizable under **this** constrained dynamics.

## 3. Observable/load-bearing subspace rather than full matrix equality

A trained recurrent model may contain state directions that do not matter at the relevant input/output ports.

The compiler question can therefore be:

```text
not:
  reproduce every hidden coordinate exactly

but:
  reproduce the behaviorally observable transition
  on the reachable/load-bearing subspace
```

This connects realization/minimality theory to physical compilation.

## 4. Negative capability as an output

Most inverse design asks for the closest physical realization.

TWC has developed a different habit:

> report which target parameter/direction distinctions the measurements or substrate **cannot support**.

A recurrent compiler could analogously report:

```text
representable exactly
representable approximately
requires extra state/depth
not representable under reciprocity/locality
not identifiable from available ports
```

This is not claimed unique. It is simply a sharper residual to search.

## 5. Dynamic resource cost

For a transition family `A(x)`, a useful physical compiler has to price not only static area/couplers but also:

```text
runtime control channels
control bandwidth
number of reconfigured elements per token
local propagation depth
latency
state retention
noise / quantization
```

A geometry-generated low-control family from KYY would matter here if it reduces runtime physical control complexity.

---

# The surviving compiler question

After subtraction, the strongest formulation is approximately:

> **Given a trained input-conditioned recurrent operator family, find the smallest behaviorally equivalent realization available to a specific reciprocal transient-wave substrate, together with the runtime control map and a certificate/report of the target dynamics that the substrate cannot realize.**

Written as a pipeline:

```text
trained recurrent model
       |
       v
reachable / observable operator family
       |
       v
quotient out irrelevant realization coordinates
       |
       v
approximate with substrate-realizable reciprocal family
       |
       +--> physical program / controls
       |
       +--> error on held-out sequences
       |
       +--> area / latency / control cost
       |
       `--> UNREALIZABLE DIRECTIONS report
```

That is much narrower than "compile AI into waves."

---

# Search status

The current pass found direct prior art for:

- learning in surrogate scattering-matrix space then transferring to physical geometry;
- physical inverse design toward target operators;
- recurrent wave computation;
- local interferometer/coupled-waveguide matrix realization.

It did **not** locate, in this pass, an exact match to the full pipeline above for modern input-conditioned recurrent state-transition families plus explicit realization/negative-capability analysis.

Status: **UNMAPPED HIGH-RISK CONJUNCTION**, not novel.

Before building a KYY->TWC recurrent compiler, this exact conjunction needs its own dedicated literature pass through:

```text
photonic compiler literature
model-order reduction / behavioral realization
hardware-aware RNN compilation
analog recurrent accelerators
physical state-space realization
robust control/model matching
network synthesis
```

---

# Effect on KYY priorities

This pass makes the current order even clearer:

```text
FIRST:
  inspect whether KYY's learned free-edge operators have a
  genuinely geometry-aligned low-dimensional structure

ONLY IF YES:
  ask whether that structure reduces runtime controls / realization
  cost in a constrained physical backend

ONLY THEN:
  build a recurrent operator-family compiler
```

The map is doing its job: every broad claim gets smaller before code is written.