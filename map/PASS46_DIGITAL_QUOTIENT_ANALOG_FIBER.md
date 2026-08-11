# Pass 46 — digital quotient / analog fiber boundary

Date: 2026-08-11

This pass deliberately moves beyond a finite-state-only benchmark.

The geometric object is no longer four legal points. It is four **continuous fibers**:

```text
F_q = { c_q + a v : a in R }
```

where:

- `q` is a finite C4 state carried by the transverse geometry `c_q`;
- `a` is a real-valued analog payload;
- `v` is a common continuous tangent direction.

A token may change the digital base state, evolve the analog coordinate, or identify two digital histories while preserving the analog coordinate.

This is exactly the discrete/continuous seam that motivated the experiment.

## Prior-art boundary first

The seam itself is not new.

### Hybrid automata

Henzinger and the hybrid-systems literature explicitly model systems containing digital modes and analog/continuous variables, including discontinuous reset maps at discrete events.

### Mixed Logical Dynamical systems

Bemporad & Morari (Automatica, 1999), *Control of systems integrating logic, dynamics, and constraints*, give a mixed logical/continuous formalism combining finite-state/logical structure with real-valued dynamics.

### Geometric gluing / hybrifolds

Simić, Johansson, Lygeros & Sastry, *Towards a Geometric Theory of Hybrid Systems*, introduce the hybrifold viewpoint: reset-related pieces of hybrid state space can be glued into a quotient-like geometric object carrying the hybrid flow.

### Learned continuous quotient representations

Teng, Liu, Song & Sreenath, *CHyLL: Learning Continuous Neural Representations of Hybrid Systems* (2025/2026), explicitly use the reset map to glue guard surfaces and learn a continuous representation of the resulting quotient manifold.

### Dimension-reducing resets

Modern hybrid-control work also treats rank/dimension-dropping reset maps (for example Clark, Oprea & Shaw, *Optimal Control of Hybrid Systems with Submersive Resets*, 2024).

Therefore KYY must not claim novelty for:

```text
digital + analog state
hybrid jumps / resets
quotient / gluing interpretation
singular or dimension-dropping reset maps
learning hybrid dynamics.
```

## What this pass actually asks

KYY's remaining question is narrower and reversed in direction.

Instead of starting from an explicit hybrid automaton and learning a continuous representation, start from an **already-trained continuous recurrent realization** whose digital and analog directions are geometrically mixed.

Then ask:

> Can a compiler recover the analog tangent and the digital state generators, replace approximate learned token maps with exact maps that obey both the symbolic quotient and the analog action, and do so without a runtime nearest-state projection?

This is a compiler experiment, not a new hybrid-system formalism.

## The rail geometry

Canonical geometry before entangling coordinates:

```text
q=0 rail: ( 1, 0, a)
q=1 rail: ( 0, 1, a)
q=2 rail: (-1, 0, a)
q=3 rail: ( 0,-1, a)
```

A fixed random orthogonal matrix mixes all three coordinates, so the analog tangent is not an axis in the deployed latent coordinates.

The task alphabet contains:

```text
0..3  C4 increments: change q, preserve a
M     partial merge: {0,1}->0 and {2,3}->2, preserve a
S     analog scale: q unchanged, a -> 0.9 a
```

The output contains both:

```text
exact finite q class
real-valued analog a
```

Because `a` is sampled from a continuum, a finite DFA alone cannot reproduce the full task exactly. An explicit hybrid automaton `Q x R` can, and is therefore the correct conceptual baseline.

## Fiber-generator lowering

Let:

```text
c0 = digital generator for q=0 at a=0
c1 = digital generator for q=1 at a=0
v  = analog tangent
```

These three vectors form a basis of the 3D latent space in the controlled probe.

The exact token maps are synthesized from basis constraints.

Cycle:

```text
C c0 =  c1
C c1 = -c0
C v  =  v
```

Partial merge:

```text
M c0 = c0
M c1 = c0
M v  = v
```

Analog scale:

```text
S c0 = c0
S c1 = c1
S v  = 0.9 v
```

Thus Pass 43's finite-column lowering extends naturally to a continuous fiber by adding tangent generators.

Again, this is elementary linear algebra and compatible with standard hybrid reset-map theory. The value is as a concrete compiler contract.

## Why state reification is now a more interesting baseline

For four legal points, nearest-state projection repaired Pass 44 completely.

For fibers, the legal set is a union of one-dimensional manifolds.

A nearest-fiber projection can correct a **transverse** error:

```text
wrong rail / off-rail displacement
```

while preserving the inferred analog coordinate.

But it cannot in general detect a **tangential semantic error**:

```text
right rail, wrong value of a
```

because that point is still perfectly legal geometrically.

This is the key diagnostic of Pass 46:

```text
manifold membership != correct dynamics along the manifold.
```

A smarter runtime hybrid controller can of course use the declared reset semantics to restore the correct analog coordinate. That is an explicit hybrid baseline, not something KYY should pretend does not exist.

The comparison is therefore:

```text
learned approximate continuous operators
vs
runtime nearest-fiber reification
vs
explicit hybrid Q x R semantics
vs
post-training exact operator legalization in the entangled continuous coordinates.
```

## What would be interesting

The experiment earns something only if it exposes a useful separation such as:

- reification repairs digital/transverse drift but leaves along-fiber semantic error;
- exact operator compilation repairs both without a runtime projection;
- the compiled continuous representation retains the analog payload and exact digital quotient simultaneously;
- the cost/robustness tradeoff differs measurably from an explicit mode + continuous-state hybrid implementation.

The first three are diagnostic results. The last one would begin to justify KYY as a deployment compiler rather than merely another description of hybrid systems.

## Files

- `map/mixed_fiber_compiler_probe.py`
- `tests/test_mixed_fiber_compiler_probe.py`
- `.github/workflows/mixed-fiber-compiler.yml`

No novelty claim is made in this pass.