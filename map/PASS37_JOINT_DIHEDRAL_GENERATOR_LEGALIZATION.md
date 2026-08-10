# Pass 37 — joint dihedral generator legalization

Date: 2026-08-10

Pass 36 left one important loophole.  The `D_n` transition algebra was non-commutative, but the reflection generator was exact throughout training:

```text
s^2 = I
s r s = r^-1
```

held structurally, and only the finite-order rotation relation `r^n = I` needed post-training legalization.

This pass makes the reflection generator approximate too.

The result is still **not** a claim of generic non-Abelian legalization.  It is a controlled dihedral case where the legal operator family has an unusually simple closed-form projection.

## Approximate learned generators

Each mode has

```text
rotation:    R(theta_i)
reflection:  B_i = F + 0.25 tanh(M_i)
```

where `F = diag(1,-1)` and `M_i` is learned.

`B_i` is not constrained to be orthogonal or involutory during training.  Consequently, after a short-horizon fit all of the following can be wrong:

```text
R(theta_i)^n = I
B_i^2 = I
B_i R(theta_i) B_i = R(theta_i)^-1
B_i^T B_i = I
```

The readout remains an ordinary learned linear classifier.

## Joint operator compiler

After training, the compiler performs two independent classical projections.

### Rotation generator

```text
theta_i -> nearest exact n-th-root angle
```

### Reflection generator

```text
B_i -> nearest determinant -1 orthogonal 2x2 matrix
```

using the SVD/polar Procrustes projection.

In two real dimensions an orthogonal matrix with determinant `-1` is a reflection.  Therefore the projected pair obeys the dihedral relations exactly up to floating-point error.

This projection is standard linear algebra and group representation theory.  No novelty claim is made for it.

## D31 first check

With five seeds, training length 16 and 8 complex modes, the learned operators were visibly non-legal:

```text
reflection involution defect   about 0.83–0.85
conjugation defect             about 0.83–0.85
reflection orthogonality       about 0.83–0.85
rotation-order defect          about 0.22–0.88
```

Joint projection drove all measured relation defects to about `1e-14`.

Yet `D31` remained too easy as a port stress test: all five frozen learned readouts still classified all 62 exact legal states correctly after the operator surgery.

That negative result motivated the same experiment on `D101`.

## D101 joint legalization

Configuration:

```text
D101                     202 legal states
modes                     8 complex / 16 real
train length              16
steps                     2200
reflection probability    0.25
reflection perturbation   0.25 tanh(M)
seeds                     0..9
```

Before compilation, the learned reflection relations were substantially illegal:

```text
reflection involution / conjugation / orthogonality defects
approximately 0.846–0.859

rotation-order defect
approximately 0.711–1.809
```

After the joint projection, the maximum measured relation defects are only a few times `1e-14`.

Every learned canonical 202-state orbit is classified correctly before projection.  The question is what happens when the operators are replaced by exact ones while the decoder is frozen.

### Raw frozen port

```text
7 / 10 exhaustive exact certificates
```

Failures:

```text
seed 1   200 / 202   min margin -0.193
seed 2   194 / 202   min margin -0.596
seed 9   200 / 202   min margin -0.281
```

So joint algebraic legalization can damage the observable interface even when the learned canonical states were all decoded correctly.

### One global orthogonal port alignment

A single unrestricted 16x16 orthogonal Procrustes map between the learned and legalized complete canonical orbits gives

```text
9 / 10 exact
```

and still fails seed 1.

### Quotient-conditioned per-mode port alignment

The dihedral quotient `D101/C101 ≅ C2` supplies the same one-bit branch information identified in Pass 36.

Instead of one global map, compute two block-diagonal `O(2)^8` Procrustes transports:

```text
Q_0   for the rotation coset C101
Q_1   for the reflected coset s C101
```

The maps are derived from corresponding pre/post operator-generated states; no task labels or decoder retraining are used.

Result:

```text
10 / 10 exhaustive exact certificates
```

with minimum output margin at least about `+2.04` across all ten seeds.

A fully unrestricted pair of quotient-conditioned 16x16 orthogonal maps is also 10/10, but buys no correctness over the much smaller per-mode family.

Mean minimum margins in this sample are approximately:

```text
raw jointly legalized port        +1.10
one global 16x16 Procrustes        +1.50
quotient block O(2)^8 transport    +2.73
quotient full 16x16 transport      +2.75
```

Mean normalized orbit-alignment errors are approximately:

```text
quotient block transport    0.237
quotient full transport     0.233
one global full transport   0.346
```

The structured smaller family wins because the required transport changes with the quotient branch; a single global linear map cannot express that conditional action.

## Live recurrent validation

`map/dihedral_joint_runtime_probe.py` runs the actual compiled machine rather than merely evaluating the canonical orbit.

At runtime:

1. the learned rotation angles are replaced by exact roots;
2. the learned reflection matrices are replaced by exact reflections;
3. a one-bit quotient sidecar toggles on reflection tokens and is unchanged on rotation tokens;
4. that bit selects `Q_0` or `Q_1` before the original frozen learned decoder.

The quotient bit is updated from the generator stream.  It never sees the target state label and never stores the 101-way rotation coordinate.

Across all ten seeds, the jointly compiled machine gives:

```text
clean random-word accuracy
L16      100%
L64      100%
L256     100%
L1024    100%

for all 10 / 10 seeds.
```

The raw jointly legalized operators reproduce their exhaustive port failures in streaming evaluation.  For example, seed 2 is about 96% at length 1024 with the raw frozen port but 100% with the quotient-conditioned compiled port.

With an additional systematic `1e-3` error added to every legalized rotation angle at runtime, the compiled machines remain 100% through length 1024 for all ten seeds in this probe.

## What is actually earned

The controlled dihedral result now includes **joint operator surgery**, not merely finite-order snapping:

```text
approximate learned non-commuting generators
        |
        v
PROJECT GENERATORS ONTO AN EXACT D_n REPRESENTATION
        |
        v
TRANSPORT THE OBSERVABLE PORT WITH THE MINIMAL KNOWN QUOTIENT STRUCTURE
        |
        v
EXHAUSTIVELY CERTIFY THE LEGAL GROUP ORBIT
        |
        v
RUN THE SAME CONSTRUCTION ONLINE
```

The key distinction is becoming operational:

```text
operator legality != port legality
```

and the resources needed for port legality can depend on the algebraic extension/quotient structure of the state machine.

## Prior-art boundary

None of the individual ingredients are new:

- dihedral and semidirect-product representations;
- nearest orthogonal/reflection projection;
- Procrustes representation alignment;
- quotient and cascade decompositions of finite groups/automata;
- group-equivariant intertwiners and centralizers;
- exact finite-state verification.

Recent diagonal-SSM theory also explicitly connects non-Abelian group tracking depth to subnormal series with Abelian factors.  Therefore KYY does not claim to discover the `C_n` plus `C_2` factorization of `D_n`.

The experimental residue remains the compiler composition and its resource accounting:

```text
which learned relations are illegal?
what exact operator family do we project onto?
what observable port is broken by that projection?
what symmetry/quotient information is sufficient to transport it?
how much additional state is required?
can the resulting finite behavior be certified completely?
```

## Current limitation

The quotient block port in this pass is **zero-label but still orbit-derived**: it is computed by enumerating corresponding learned and exact canonical group states and solving per-coset Procrustes problems.

That is harmless for 202 states but not a scalable compiler principle.

The next falsifier should therefore not be a larger `n` or another neural architecture.

It is:

> can the same quotient-conditioned port transport be computed directly from the pre/post generator matrices and relations, with no group-orbit enumeration?

Pass 38 attacks exactly that limitation.

## Files

- `map/dihedral_joint_legalization_probe.py`
- `map/dihedral_joint_runtime_probe.py`
- `tests/test_dihedral_joint_legalization_probe.py`
- `tests/test_dihedral_joint_runtime_probe.py`
- `.github/workflows/dihedral-joint-legalization.yml`
- `.github/workflows/dihedral-joint-d101.yml`
- `.github/workflows/dihedral-joint-runtime.yml`
- `results/dihedral_d31_joint_legalization.csv`
- `results/dihedral_d101_joint_legalization.csv`
- `results/dihedral_d101_joint_runtime.csv`
