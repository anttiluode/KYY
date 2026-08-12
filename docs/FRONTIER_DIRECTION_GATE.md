# Frontier direction gate: order survives weakly, orientation does not

Date: 2026-08-12

This note closes one loophole in the KYY partial-maturity branch.

The existing random coordinate splice showed that a shared readout can remain useful on
nonphysical mixtures of homogeneous update phases. That control destroyed any spatial
ordering of maturity, so it did **not** test whether a smooth/ordered maturity frontier
carries something special.

`experiments/frontier_direction_control.py` attacks that remaining variable while holding
the maturity histogram and maximum span fixed.

## Construction

The phase-supervised `geom_scatter` model is trained exactly as in the earlier splice
control. At evaluation, homogeneous states are collected at

```text
pre, phase 1, phase 2, phase 3, final
```

and coordinate-wise mixtures are built with the same multiset of maturity labels:

```text
forward   monotone maturity order around the coordinate ring
reverse   the exact same labels in the opposite order
shuffled  the exact same labels randomly permuted
```

For ring topology, forward/reverse are averaged over every cyclic origin so that
coordinate zero cannot create the effect. Shuffled labels are redrawn per origin.

Two gates are reported:

```text
phase-only  use only phase1..final, which become individually readable after training
with-pre    include the truly immature pre-update state
```

The replicated GitHub Actions run used five model seeds, 300 training steps per seed,
eight evaluation batches of 256 examples, and eight shuffled draws per origin.

Workflow run: `31589406988` (`replicated-direction-gate`).

## Replicated result

All five models reached essentially perfect readout accuracy on the trained post-update
phases.

Across seeds:

```text
PHASE-ONLY
forward   0.9660
reverse   0.9696
shuffled  0.9461

mean ordered advantage over shuffled = +0.0217
forward - reverse                     = -0.0037
```

Per-seed ordered advantages `((forward+reverse)/2 - shuffled)` were approximately:

```text
+0.0164, +0.0210, +0.0429, +0.0182, +0.0104
```

So spatially smooth/block-ordered mixtures are modestly easier for the inherited
readout than fully coordinate-shuffled mixtures **when every constituent phase is
already individually legible**.

But there is no consistent orientation effect. Forward and reverse swap which one is
slightly better across seeds and their aggregate difference is essentially zero.

When the genuinely immature `pre` state is included:

```text
WITH-PRE
forward   0.7816
reverse   0.7771
shuffled  0.8034

mean ordered advantage over shuffled = -0.0240
forward - reverse                     = +0.0044
```

Per-seed ordered advantages were approximately:

```text
-0.0311, -0.0076, +0.0263, -0.0457, -0.0620
```

Thus the more literal immature-to-mature frontier does **not** gain a robust advantage
from being spatially ordered in this gate. If anything, the fully shuffled splice is
slightly more readable on average.

## Verdict

This closes the proposed rescue:

> **KYY does not show a generic directional-frontier advantage.**

There is a smaller residual observation:

> When all source phases already encode the target correctly, keeping phase blocks
> spatially coherent gives a small readout advantage over scattering those coordinates
> randomly.

That is an order/smoothness effect, not a direction effect. Forward and reverse are
indistinguishable, so it does not support the claim that temporal orientation itself is
special.

A still stricter control could compare monotone blocks with contiguous blocks whose phase
labels occur in a non-monotone permutation. That would distinguish `smooth block support`
from `monotone maturity order`. It is not required to close the directional claim.

## Cross-repo implication

Recent human neuroscience can still show functionally relevant wave direction. This KYY
null simply says we should not infer that direction is generically valuable because a
partial computation is spatially ordered.

Brain directionality must earn itself from brain data and interventions.

That separation is useful:

```text
KYY toy:
    orientation does not survive the matched gate

brain data:
    direction/phase may still matter for biological reasons specific to its circuits
```

Do not use KYY's partial-maturity branch as evidence for the biological direction claim.
