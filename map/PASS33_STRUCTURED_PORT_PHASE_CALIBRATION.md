# Pass 33 — structured port phase calibration beats generic repair

Date: 2026-08-10

Pass 32 showed that a single fractional observation time `tau` can sometimes repair a frozen readout after exact cyclic operator legalization.

This pass asks the mechanical follow-up:

> how much port freedom is actually needed, and does structure matter more than raw parameter count?

The recurrent operator is frozen after snapping in every experiment below.

## Repair hierarchy

For the existing `C_101`, 8-complex-mode setup, the following readout-side families were compared:

```text
1 parameter      common timing tau
8 parameters     one phase offset per complex mode
32 parameters    rank-1 residual hidden adapter
33 parameters    timing tau + rank-1 hidden adapter
256 parameters   full 16x16 hidden adapter
1717 parameters  full linear readout correction
```

Calibration used randomly chosen legal states. Every fitted repair was then evaluated on the complete 101-state legal orbit.

This is a calibration experiment, not a claim about statistical sample complexity. One multiclass labeled state supplies many logit inequalities, and the optimizer/hyperparameters matter for the generic adapters.

## Main result

The 8-parameter per-mode phase family is qualitatively different from the generic adapters.

Across the five seeds that failed raw inherited-port legalization (`1,5,6,7,9`), with 3 random calibration repeats each:

```text
4 calibration states

method                 mean full-orbit acc   exact runs / 15
------------------------------------------------------------
common timing tau            0.8502               4 / 15
per-mode phase               1.0000              15 / 15
rank-1 hidden                0.2647               0 / 15
tau + rank-1 hidden          0.2911               0 / 15
full hidden                  0.4898               0 / 15
full readout                 0.5828               0 / 15
```

The per-mode phase correction also remained 15/15 exact under the `1e-3` systematic angle-error check in this first sweep.

The negative results for the generic adapters are optimizer-dependent and should not be read as impossibility results. Their purpose is only to show that *more free parameters did not automatically make calibration easier* under the same small-data repair protocol.

## One-to-four-state stress

The phase-only family was then repeated 10 times per seed with only 1–4 randomly selected legal calibration states.

```text
calibration states   exact phase repairs / 50   mean exhaustive accuracy
-----------------------------------------------------------------------
1                         29 / 50                     0.9655
2                         39 / 50                     0.9891
3                         41 / 50                     0.9877
4                         46 / 50                     0.9990
```

At four calibration states, all failures were seed 1 near-misses:

```text
99/101
99/101
99/101
98/101
```

Seeds 5, 6, 7 and 9 were 10/10 exact with four random calibration states.

This is already enough to reject the naive story that the port needs a large generic retraining operation.

## Why this family is special

Write the exact legalized cyclic recurrence in complex coordinates as

```text
A* = diag(exp(i theta*_1), ..., exp(i theta*_k)).
```

A per-mode readout phase adapter is

```text
D(phi) = diag(exp(i phi_1), ..., exp(i phi_k)).
```

Then

```text
D(phi) A* = A* D(phi).
```

In real coordinates this is simply a block-diagonal bank of 2D rotations.

So the useful repair family is not an arbitrary neural adapter. It lies in the **commutant / centralizer** of the legalized representation.

This mathematics is standard representation theory and is not a KYY novelty claim.

The KYY compiler implication is narrower:

> after operator legalization, search the symmetry-preserving interface freedoms before retraining a generic port.

The one-scalar timing parameter from Pass 32 is a subfamily:

```text
phi_i = tau * theta*_i.
```

The full phase torus allows the phase reference of each harmonic channel to be calibrated independently while leaving the legal recurrent algebra untouched.

## Result files

- `map/port_calibration_cost_probe.py`
- `tests/test_port_calibration_cost_probe.py`
- `.github/workflows/harmonic-port-calibration.yml`
- `.github/workflows/harmonic-mode-phase-fewshot.yml`
- `results/harmonic_n101_port_calibration_summary.csv`
- `results/harmonic_n101_mode_phase_fewshot_summary.csv`

## Stopping point

The supervised 8-phase fit is already useful, but it raises a sharper question:

> because the compiler knows both the learned angles and the snapped angles, can the required port phases be computed directly with **zero labels**?

Pass 34 answers that question.
