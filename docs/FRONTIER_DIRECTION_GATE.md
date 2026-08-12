# Frontier direction gate: temporal order does not survive the matched controls

Date: 2026-08-12

This note closes one loophole in the KYY partial-maturity branch.

The existing random coordinate splice showed that a shared readout can remain useful on
nonphysical mixtures of homogeneous update phases. That control destroyed any spatial
ordering of maturity, so it did **not** test whether a smooth/ordered maturity frontier
carries something special.

Two controls were therefore added:

```text
experiments/frontier_direction_control.py
experiments/frontier_order_vs_contiguity.py
```

Together they separate three different properties that were previously conflated:

```text
orientation / direction
monotone temporal maturity order
spatial contiguity of phase blocks
```

---

## 1. Direction control

The phase-supervised `geom_scatter` model is trained exactly as in the earlier splice
control. At evaluation, homogeneous states are collected at

```text
pre, phase 1, phase 2, phase 3, final
```

and coordinate-wise mixtures are built with the same multiset of maturity labels:

```text
forward   monotone phase labels along physical coordinates
reverse   the exact same labels in the opposite order
shuffled  the exact same label multiset randomly permuted
```

For ring topology, forward/reverse are averaged over every cyclic origin so coordinate
zero cannot create the effect.

Two gates are reported:

```text
phase-only  use only phase1..final, which become individually readable after training
with-pre    include the genuinely immature pre-update state
```

The replicated GitHub Actions run used five model seeds, 300 training steps per seed,
eight evaluation batches of 256 examples, and eight shuffled draws per origin.

Workflow run: `31589406988` (`replicated-direction-gate`).

### Result

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

There is no consistent orientation effect. Forward and reverse swap which is slightly
better across seeds and the aggregate gap is effectively zero.

When the genuinely immature `pre` state is included:

```text
WITH-PRE
forward   0.7816
reverse   0.7771
shuffled  0.8034

mean ordered advantage over shuffled = -0.0240
forward - reverse                     = +0.0044
```

So the more literal immature-to-mature frontier gains no directional or generic ordered
advantage. If anything, coordinate shuffle is slightly easier for the inherited readout.

The remaining question after this gate was whether the `+0.0217` phase-only residual
came from **temporal order** or merely from the fact that the forward/reverse arms kept
maturity phases in spatially coherent blocks.

---

## 2. Order versus contiguity control

`frontier_order_vs_contiguity.py` holds the phase histogram fixed and compares:

```text
monotone
    contiguous blocks in maturity order

block-permuted
    every maturity phase remains one contiguous block,
    but the order of the blocks is non-monotone

coordinate-shuffled
    the same labels scattered coordinate-wise
```

All non-dihedral block permutations are averaged, and all ring origins are averaged.

The replicated five-seed GitHub Actions run is:

```text
workflow run 31590228430
job replicated-contiguity-gate
```

### Phase-only result

```text
monotone mean       0.9671
block-permuted      0.9677
coordinate-shuffled 0.9450

monotone - block    -0.0006
block - shuffle     +0.0227
monotone - shuffle  +0.0221
```

This cleanly identifies the earlier residual.

The small ~2.2 point advantage is preserved when the phase blocks are contiguous but
placed in a non-monotone temporal order. Monotone order itself contributes essentially
nothing:

```text
0.9671 versus 0.9677
```

So the residual is a **spatial block-contiguity / coherence effect**, not a temporal
maturity-order effect.

### With genuinely immature pre-state

```text
monotone mean       0.7773
block-permuted      0.7985
coordinate-shuffled 0.8030

monotone - block    -0.0213
block - shuffle     -0.0045
monotone - shuffle  -0.0258
```

Here even spatial contiguity does not rescue the frontier. The monotone immature-to-mature
arrangement is the worst of the three aggregate conditions.

---

## Final verdict

The loophole is now closed:

> **KYY does not show a generic advantage for temporal frontier direction or maturity
> ordering.**

What survives is much smaller:

> When every constituent phase is already individually legible, keeping coordinates from
> the same phase spatially contiguous is modestly easier for this inherited local/readout
> geometry than scattering them salt-and-pepper across coordinates.

That is a spatial-coherence result, not a temporal one.

The hierarchy of controls now says:

```text
forward vs reverse
    -> no orientation effect

monotone blocks vs non-monotone contiguous blocks
    -> no temporal-order effect

contiguous blocks vs coordinate shuffle
    -> small spatial-coherence effect when all phases are already readable

include genuinely immature pre-state
    -> no frontier advantage; monotone ordering is slightly worse
```

This means the proposed KYY rescue through `ordered partial maturity` fails.

---

## Cross-repo implication

Recent human neuroscience can still show functionally relevant travelling-wave direction,
phase gradients or receiver order. This KYY null simply says there is no generic
computational virtue of temporal orientation in this toy that can be exported to the
brain story.

Brain directionality must earn itself from brain data and interventions.

Keep the separation explicit:

```text
KYY toy
    temporal direction/order fails matched controls

brain data
    direction/phase may still matter for circuit-specific biological reasons
```

Do not use KYY's partial-maturity branch as evidence for the biological direction claim.
