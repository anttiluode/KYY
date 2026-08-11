# Pass 34 — zero-label commutant recentering

Date: 2026-08-10

Pass 33 found that one independent output phase per harmonic mode is an unusually effective port repair family after exact cyclic operator legalization.

This pass removes the labels and the optimizer.

## Setup

For one complex mode, let

```text
learned generator angle      theta
legalized generator angle    theta*
snap displacement            delta = theta* - theta
```

The pre-snap learned orbit is

```text
z(s) = exp(i s theta)
```

and the exact legalized orbit is

```text
z*(s) = exp(i s theta*),    s = 0,...,n-1.
```

Without port correction, the phase displacement from the learned orbit grows linearly:

```text
s delta.
```

A constant readout phase `phi` changes this to

```text
s delta + phi.
```

## Midpoint formula

For uniform weighting of the complete finite orbit, choose

```text
phi = -(n-1)/2 * delta.
```

Then

```text
s delta + phi = (s - (n-1)/2) delta.
```

So the snap-induced phase displacement is centered around zero instead of anchored at state 0.

This has two equivalent interpretations.

### Minimax interval centering

Over states `0,...,n-1`, the midpoint choice minimizes the maximum absolute *linear* phase displacement.

Nearest-character snapping gives

```text
|delta| <= pi/n.
```

Therefore the simple worst-case phase-displacement bound changes from

```text
raw snap:       max_s |s delta| < pi
midpoint port:  max_s |(s-(n-1)/2) delta| < pi/2.
```

The experiment verifies the exact factor-of-two reduction in the measured maximum mismatch for every seed.

### Per-mode complex Procrustes alignment

The same phase is also the exact unitary least-squares alignment of the snapped orbit to the learned orbit:

```text
argmin_phi sum_s | exp(i(s theta* + phi)) - exp(i s theta) |^2.
```

The relevant geometric sum is

```text
sum_{s=0}^{n-1} exp(i s delta)
 = exp(i (n-1)delta/2) * sin(n delta/2) / sin(delta/2).
```

For nearest-root snapping, the real scalar factor has the compatible sign over the snap cell, so the optimal compensating phase is exactly

```text
phi = -(n-1)delta/2
```

up to the usual `2 pi` wrapping.

Orthogonal/unitary Procrustes alignment and commutants are old mathematics. No novelty claim is made for either theorem.

## Algebraic legality is untouched

For `k` complex modes, define the readout adapter

```text
D = diag(exp(i phi_1), ..., exp(i phi_k)).
```

The legalized cyclic operator is diagonal in the same modal basis, so

```text
D A* = A* D.
```

Thus this adapter lies in the commutant of the exact representation.

It changes the port phase reference while preserving the exact recurrent algebra.

This is the important compiler property.

## C101 result

The existing ten-seed C101 stress set was rerun.

```text
seed   pre-snap orbit   raw snapped port   zero-label recentered port   min margin after
---------------------------------------------------------------------------------------
0          101/101          101/101                101/101                 +1.715
1           99/101           83/101                101/101                 +0.578
2          101/101          101/101                101/101                 +2.449
3          101/101          101/101                101/101                 +2.463
4          101/101          101/101                101/101                 +2.565
5           99/101           84/101                101/101                 +0.987
6          101/101           99/101                101/101                 +1.987
7           94/101           47/101                101/101                 +0.118
8          101/101          101/101                101/101                 +1.919
9           98/101           76/101                101/101                 +1.118
```

So the deterministic zero-label compiler pass gives

```text
10 / 10 exhaustive legal-orbit certificates.
```

All ten also remain `101/101` under the same `1e-3` systematic angle-error probe used in the recent phase experiments.

The hardest case is seed 7:

```text
pre-snap learned orbit       94/101
raw snapped legal orbit      47/101
midpoint-recentered legal    101/101
minimum output margin        +0.118
```

This is important because the adapter is **not merely reconstructing the pre-snap learned behavior**. The combination of exact operator snapping and legal port recentering can produce a finite machine whose observable behavior is cleaner than the approximate learned orbit itself.

## What changed conceptually

The current cyclic compiler is now:

```text
TRAIN APPROXIMATELY
        |
        v
SNAP OPERATOR TO EXACT CHARACTERS
        |
        v
COMPUTE SYMMETRY-PRESERVING PORT RECENTERING
        |
        v
EXHAUSTIVELY VERIFY THE COMPLETE LEGAL ORBIT
        |
        v
PRICE FINITE-PRECISION / PHYSICAL ERROR
```

No readout retraining is required in this C101 ten-seed test.
No labeled calibration states are required for the recentering itself.

## Relation to Pass 30 relation-defect predictor

This result reduces the importance of relation defect as a *predictor of zero-shot raw-port survival* in this simple cyclic setting, because the raw inherited port is no longer the only allowed compiler interface.

Relation defect still measures distance from the legal operator family and remains potentially useful where:

- legalization is not a nearest-character scalar snap;
- there is no closed-form port alignment;
- the representation is non-Abelian;
- the commutant/intertwiner space is more constrained;
- physical implementation errors dominate.

## Prior-art boundary

The following pieces are standard and occupied:

- Fourier / harmonic cyclic representations;
- unitary phase rotations;
- orthogonal/unitary Procrustes alignment;
- commutants, centralizers and intertwiners in representation theory;
- equivariant linear maps constrained by group representations;
- automata extraction and stable finite-state encoding from recurrent networks.

The KYY residue remains an experimentally demonstrated compiler composition:

```text
approximate learned dynamical operator
        -> exact task-algebra legalization
        -> analytically derived symmetry-preserving port alignment
        -> complete finite behavioral certificate.
```

That composition is the object to keep testing, not the individual mathematical ingredients.

## Files

- `map/midpoint_phase_compensation_probe.py`
- `tests/test_midpoint_phase_compensation_probe.py`
- `.github/workflows/harmonic-midpoint-phase.yml`
- `results/harmonic_n101_midpoint_phase.csv`

## Next falsifier

Do not add another generic neural adapter.

The next useful question is whether this compiler pattern survives beyond an Abelian diagonal representation.

For a non-Abelian legal representation, replace the cyclic phase torus with the appropriate **intertwiner / commutant-constrained port alignment space**, then ask the same questions:

1. can an approximate trained operator be legalized to exact group relations?
2. can a symmetry-preserving interface transform retain the learned observable port?
3. can the complete finite group orbit be certified?

`S3` is already present in KYY and is the natural first target.
