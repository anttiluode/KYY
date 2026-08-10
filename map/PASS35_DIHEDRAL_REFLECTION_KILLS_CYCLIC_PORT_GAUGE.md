# Pass 35 — reflection kills the cyclic port gauge

Date: 2026-08-10

Pass 34 found a zero-label port recentering for a cyclic `C_n` harmonic tracker.  For each learned generator angle `theta_i` snapped to an exact character `theta*_i`, the readout-only phase

```text
phi_i = -(n-1)/2 * (theta*_i - theta_i)
```

recentered the finite cyclic orbit and rescued all ten C101 stress seeds.

That result could be an Abelian luxury.

This pass inserts the smallest non-commutative obstruction explicitly by moving from a cyclic group to the dihedral group

```text
D_n = < r, s | r^n = 1, s^2 = 1, s r s = r^-1 >.
```

`D_3` is isomorphic to `S_3`, so this is the same relation pattern as KYY's existing non-commutative permutation oracle, but `D_31` gives 62 legal states and room for a harmonic stress test.

## Model

The state is still an 8-mode / 16-real-coordinate harmonic bank.

For each mode:

```text
rotation token r^j:  R(j theta_i)
reflection token s:  F(x,y) = (x,-y)
```

The useful property is that for **every** learned angle,

```text
F^2 = I
F R(theta) F = R(-theta)
```

exactly.

Therefore the non-commutative conjugation relation is structural throughout training.  Only the finite-order relation

```text
R(theta)^n = I
```

is approximate and needs post-training legalization.

This cleanly isolates the same nearest-character snap used in the cyclic experiments while making the legal transition algebra non-commutative.

## D31 training result

Configuration:

```text
n = 31                  -> 62 legal group states
modes = 8               -> 16 real recurrent coordinates
train length = 16
steps = 1800
random initial rotation
later rotation increments in {0,1,2,3,4}
reflection probability = 0.25
10 seeds
```

Every seed reaches 100% at training length 16, but unsnapped long-horizon accuracy can drift badly:

```text
seed 2: L1024 = 0.5723
seed 6: L1024 = 0.6327
seed 7: L1024 = 0.5416
```

After snapping each learned rotation angle to the nearest exact 31st root of unity, the frozen learned readout classifies the **complete 62-state legal orbit** correctly for all ten seeds:

```text
raw exact operator legalization: 10 / 10 exhaustive certificates
```

So at D31 the operator compiler already fixes the long-horizon problem; a port repair is not required.

## The important negative result

Blindly importing the cyclic midpoint phase is **not legal as a generic dihedral port gauge**.

For seed 2:

```text
raw snapped orbit:       62 / 62
minimum readout margin:  +1.2118

cyclic midpoint phase:   57 / 62
minimum readout margin:  -0.5379
```

The Abelian port formula actively breaks a correct non-Abelian machine.

This is the desired boundary result.

## Why the phase freedom disappears

For one 2D dihedral block, let `F = diag(1,-1)` be reflection.  A generic cyclic output phase is a rotation `Q(phi)`.

Although `Q(phi)` commutes with all rotation powers,

```text
Q(phi) R(theta) = R(theta) Q(phi),
```

it does not commute with reflection except at the discrete special cases:

```text
F Q(phi) F = Q(-phi).
```

Equivalently, the continuous `U(1)` centralizer of the cyclic subgroup is cut down by the reflection generator.

There is a second way to see the same thing directly from the finite orbit.  Every rotational state has a reflected partner.  In a complete-orbit Procrustes alignment their cross-covariance contributions occur in pairs

```text
C + F C F,
```

which cancels the off-diagonal rotation component.  In the D31 runs, the independent 2D-block Procrustes alignment therefore leaves the same raw port behavior and does not recreate the cyclic midpoint rotation.

All of this is standard representation/group theory, not a novelty claim.

## What survives from KYY

The operator-legalization part survives the non-Abelian transition:

```text
approximate learned rotations
        -> nearest exact characters
        -> exact r^n = I
```

while

```text
s^2 = I
srs = r^-1
```

were already exact by construction.

What fails is the assumption that an Abelian output gauge can be carried over unchanged.

So the compiler now has to distinguish:

```text
LEGAL OPERATOR ALGEBRA

from

LEGAL PORT TRANSFORMATIONS FOR THAT ALGEBRA.
```

The latter are constrained by the full group, not by one generator at a time.

## Files

- `map/dihedral_legalization_probe.py`
- `tests/test_dihedral_legalization_probe.py`
- `.github/workflows/dihedral-legalization.yml`
- `results/dihedral_d31_legalization.csv`

## Next falsifier

D31 was still easy enough that raw snapping preserved all ten ports.  The next stress test is `D101` with 202 legal states.

There are two distinct questions:

1. does raw operator legalization begin to lose inherited ports at the larger group size?
2. if it does, can a non-Abelian port repair use the **exact quotient/coset structure** rather than a generic neural adapter?

For `D_n`, the quotient by the rotation subgroup is `C_2`: rotation branch versus reflected branch.  Because reflection reverses rotation orientation, any cyclic-style phase recentering should reverse sign between those two cosets.  Pass 36 tests that one-bit, quotient-conditioned compiler sidecar directly.
