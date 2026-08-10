# Pass 36 — quotient-conditioned port compiler on D101

Date: 2026-08-10

Pass 35 established a clean negative result: the zero-label cyclic midpoint phase from Pass 34 is not a legal generic port gauge once reflection is part of the state algebra.  On `D31`, blindly applying the cyclic correction can break a correct exact machine.

The next question is whether the **reason** it fails tells us what information a non-Abelian port compiler needs.

For the dihedral group

```text
D_n = C_n ⋊ C_2
```

the quotient bit distinguishes the rotation coset `C_n` from the reflected coset `s C_n`.  Conjugation by reflection reverses the rotation coordinate:

```text
s r s = r^-1.
```

Therefore the snap-induced phase error winds with opposite orientation on the two cosets.

That suggests one minimal extension of the cyclic compiler:

```text
use +phi_i on r^k
use -phi_i on s r^k
```

where

```text
phi_i = -(n-1)/2 * (theta*_i - theta_i).
```

The phase constants are still computed analytically from the learned and legalized operators.  The only additional dynamic resource is the exact one-bit `C_2` quotient state.

## Important prior-art subtraction

This decomposition is not new group theory.

`D_n` is solvable, with an Abelian normal subgroup `C_n` and Abelian quotient `C_2`.  Shakerinava et al. (ICLR 2026), *The Expressive Limits of Diagonal SSMs for State-Tracking*, characterize multilayer diagonal complex SSM group tracking in terms of subnormal series with Abelian factors.  Krohn-Rhodes/cascade decompositions and semidirect-product representation theory are older still.

KYY therefore does **not** claim to discover Abelian-factor decomposition of non-Abelian state machines.

The narrower compiler question is:

> after post-training operator legalization, what quotient information must be retained so that a learned observable port can be transported onto the exact representation with minimal additional state?

## D101 stress setup

```text
D101                    202 legal group states
8 complex modes          16 real harmonic coordinates
train length              16
training steps            2200
random initial rotation   yes
later rotation increments {0,1,2,3,4}
reflection probability    0.25
seeds                     0..9
```

The reflection operator is exact by construction:

```text
F^2 = I
F R(theta) F = R(-theta)
```

for every learned angle.  Only the finite-order rotation relation is approximate before compilation:

```text
R(theta)^101 ≈ I.
```

So this is a non-commutative stress test of finite-order operator legalization plus port transport.  It is **not yet** a generic projection of arbitrary approximate non-Abelian generators.

## Raw D101 legalization

The approximate models classify the canonical learned 202-state orbit perfectly, but long sequence evaluation can drift because different words for the same finite group element need not agree before `r^101=I` is enforced.

After nearest-character snapping, the recurrent algebra becomes exact.  The inherited frozen readout then gives:

```text
raw snapped exhaustive certificates: 7 / 10
```

The three failures are:

```text
seed 1   160 / 202    min margin -1.437
seed 2   192 / 202    min margin -0.376
seed 7   200 / 202    min margin -0.116
```

A single global cyclic midpoint correction performs worse:

```text
global midpoint exact: 4 / 10
```

A single unrestricted 16x16 orthogonal Procrustes map reaches 9/10, rescuing seeds 2 and 7 but not seed 1.

## Quotient-conditioned zero-label recentering

Let `q` be the exact quotient/sign coordinate:

```text
q = +1   for r^k
q = -1   for s r^k.
```

Use the branch-conditioned modal port

```text
Q_q = diag_i R(q phi_i).
```

The orientation reversal is the key.  If `delta_i = theta*_i-theta_i`, the rotational coset has phase displacement proportional to `+k delta_i`, while the reflected coset carries the opposite orientation under `F R(alpha) = R(-alpha) F`.  Transporting the midpoint phase by the quotient action centers the error on both cosets instead of fixing one and damaging the other.

This can be read as a tiny semidirect-product port gauge: the quotient acts on the normal-subgroup phase correction by inversion.

Again, that mathematics is standard.  The experiment asks whether this is the right **compiler resource**.

## Exhaustive result

It is.

```text
raw snapped port              7 / 10 exact
one global midpoint phase     4 / 10 exact
full 16x16 Procrustes         9 / 10 exact
C2-conditioned midpoint      10 / 10 exact
```

The quotient-conditioned correction produces 202/202 correct legal states for every seed, with no labeled calibration and no optimization.

The hardest raw case becomes:

```text
seed 1
raw snapped:       160 / 202,  min margin -1.437
C2-conditioned:    202 / 202,  min margin +1.205
full Procrustes:   162 / 202,  min margin -1.358
```

Across all ten seeds, the minimum readout margin averages approximately:

```text
raw snapped              +1.25
global midpoint           -0.80
full global Procrustes    +1.49
C2-conditioned midpoint   +3.05
```

The quotient-conditioned orbit is also closer to the learned canonical orbit than the best single global orthogonal alignment in every run in this sample:

```text
mean normalized alignment error
C2-conditioned midpoint   0.181
full global Procrustes    0.335
```

The reason a smaller-looking structured transform can beat a 16x16 global orthogonal map is that it is **state-family conditioned by the quotient action**.  A single global linear map is not allowed to reverse itself on the reflected coset.

## Live sidecar validation

A possible objection to the exhaustive calculation is that splitting the 202 prototypes into two cosets might secretly use the target state label.

`map/dihedral_coset_runtime_probe.py` removes that ambiguity.

The runtime compiler carries exactly one sidecar bit initialized to zero and updated directly from the input generator stream:

```text
rotation token:   q <- q
reflection token: q <- 1-q
```

The bit never sees the target label and never stores the `C101` rotation coordinate.  It is simply the one-dimensional sign representation of the quotient `D101/C101 ≅ C2`.

At every recurrent step, `q` chooses `+phi` or `-phi` before the unchanged learned linear decoder.

Live random-word evaluation gives:

```text
compiled clean accuracy
length 16       10 / 10 seeds at 100%
length 64       10 / 10 seeds at 100%
length 256      10 / 10 seeds at 100%
length 1024     10 / 10 seeds at 100%
```

The raw legalized model reflects its exhaustive port failures in live rollouts.  For example seed 1 remains around 79% at length 1024, while the one-bit compiled port is 100%.

Under the same systematic `1e-3` rotation-angle implementation error used in the recent harmonic tests:

```text
compiled length-1024 accuracy: 10 / 10 seeds at 100%
```

So the complete-orbit certificate and the actual streaming implementation agree.

## What this earns

For this controlled dihedral family, the compiler has moved from

```text
exact operator + hope the old readout survives
```

to

```text
1. train approximate recurrent dynamics
2. legalize the normal cyclic operator onto exact characters
3. retain the exact quotient/sign coordinate
4. transport the zero-label port recentering by the quotient action
5. exhaustively certify all 2n legal states
6. run the same construction online with one sidecar bit
```

This is substantially more specific than “use group representations.”

The individual ingredients remain classical:

- dihedral/semidirect-product representation theory;
- Abelian normal subgroup + quotient decomposition;
- phase rotations and Procrustes alignment;
- exact finite-state side information;
- group/automata cascade decompositions.

The experimental residue is the **post-training compiler composition and resource accounting**:

```text
which relations are legalized?
which port gauges survive the full algebra?
which quotient state is required to transport them?
how many extra bits/parameters does the compiled interface need?
does complete observable behavior certify afterward?
```

## Relation to current non-Abelian state-tracking work

This should not be conflated with Lee (2026), *A Held-Out Transition-Pair Falsifier for Long-Horizon Non-Abelian State Tracking*.  That work uses hard finite-group projection of recurrent state/prototypes and demonstrates very long S3xS3 tracking in its controlled benchmark.

Here the projection locus is different:

```text
KYY: post-training operator parameters -> exact representation,
     then analytically transport the learned port.
```

The state is not hard-projected back onto a group prototype at every step.

## Files

- `map/dihedral_legalization_probe.py`
- `map/dihedral_coset_recenter_probe.py`
- `map/dihedral_coset_runtime_probe.py`
- `tests/test_dihedral_legalization_probe.py`
- `tests/test_dihedral_coset_recenter_probe.py`
- `tests/test_dihedral_coset_runtime_probe.py`
- `.github/workflows/dihedral-d101-stress.yml`
- `.github/workflows/dihedral-coset-recenter.yml`
- `.github/workflows/dihedral-coset-runtime.yml`
- `results/dihedral_d101_legalization.csv`
- `results/dihedral_d101_coset_recenter.csv`
- `results/dihedral_d101_coset_runtime.csv`

## Current stopping pin

Do **not** call this general non-Abelian operator legalization yet.

In the current `D_n` tracker, the reflection relation is exact throughout training and only the finite order of the rotation generator is legalized post hoc.

The next serious falsifier is therefore not a larger `n`.  It is to let the reflection operator itself become approximate, measure all three relation defects,

```text
r^n = I
s^2 = I
srs = r^-1,
```

then jointly project the learned generators onto an exact dihedral representation and ask whether a quotient-conditioned/intertwiner-constrained port compiler can still preserve observable behavior.
