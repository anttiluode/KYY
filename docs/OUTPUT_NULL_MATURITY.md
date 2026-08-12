# Output-null maturity — the machine keeps computing without thrashing its public readout

Date: 2026-08-12

This note follows `DEADLINE_SCATTER_GATE.md`.

The deadline gate separated two properties:

```text
partial computation is physically available
            !=
partial computation is legible to one fixed downstream readout
```

Ordinary final-only KYY strongly exhibited the failure: intermediate
`geom_scatter` states often contained enough information for a separately fitted
linear probe to recover the current `perm3` state, while the model's own final readout
mapped those same states to the wrong answer.

That raised a more precise question:

> **When phase supervision makes one shared readout useful throughout the transition,
> does the recurrent state stop moving, or does ongoing computation move into directions
> that are largely null to the public readout?**

`experiments/output_null_motion.py` now measures exactly that.

---

## 1. Readout geometry

KYY's public classifier is linear:

```text
logits = W h + b
```

For a hidden-state change `delta_h`, decompose it into the row space of `W` and its
orthogonal complement:

```text
delta_h = delta_potent + delta_null
```

where

```text
delta_potent in row(W)
W delta_null = 0
```

Only the output-potent component changes the current logits under that readout.

This gives a direct diagnostic:

```text
potent motion fraction = ||delta_potent||^2 / ||delta_h||^2
null motion fraction   = ||delta_null||^2   / ||delta_h||^2
```

The terminology is deliberately borrowed from the established neural population
literature; it is not a KYY novelty claim.

---

## 2. Biological prior art

Kaufman, Churchland, Ryu & Shenoy (2014), *Cortical activity in the null space:
permitting preparation without movement* (Nature Neuroscience 17, 440-448,
DOI 10.1038/nn.3643), showed that substantial preparatory motor-cortical activity can
occupy population patterns that cancel at the muscle readout.  Activity can therefore
evolve internally without prematurely driving movement.

Semedo et al. (2019), *Cortical areas interact through a communication subspace*
(Neuron 102, 249-259.e4, DOI 10.1016/j.neuron.2019.01.026), found that interactions
between visual cortical areas are concentrated in a low-dimensional communication
subspace rather than reflecting all dominant within-area activity.

So the broad computational strategy is known:

```text
large private/internal population dynamics
               +
small receiver-relevant public projection
```

The question here is only whether the KYY partial-maturity fix happens to organize
itself in the same mathematical *kind* of way.

---

## 3. Important dimensional baseline

Do not be impressed merely by a large null fraction.

For `perm3` with state dimension 32 and a six-class linear readout, the readout row
space has rank at most 6.  An isotropically random state-motion vector would therefore
place, in expectation,

```text
1 - 6/32 = 0.8125
```

of its squared energy in the null space simply because the null space is much larger.

This baseline is essential.

The interesting signal is therefore not:

> "94% of motion is null."

It is whether motion is **more or less output-potent than the dimensional baseline**,
and how that changes with training.

---

## 4. Compact three-seed scratch result

Before committing the repo-native diagnostic, the exact KYY reconstruction was run on
three independent model seeds at a deliberately compact training budget (160 updates
per condition).  These numbers are exploratory and should be reproduced with the
committed script.

### Ordinary final-only training

Mean shared-head accuracy after the four checkerboard phases:

```text
phase 1   .230
phase 2   .012
phase 3   .087
phase 4  1.000
```

Mean output-null fraction of phase-to-phase hidden motion:

```text
phase 1   .510
phase 2   .647
phase 3   .654
phase 4   .495
```

Relative to the isotropic `.8125` reference, ordinary KYY motion is unusually
**output-potent throughout the transition**.  The final receiver is therefore being
strongly driven through changing coordinate semantics before the transition ends.

### Same architecture, shared phase supervision

The same six-logit readout is supervised after each homogeneous checkerboard phase.
There are no additional heads.

Mean accuracy:

```text
phase 1   .887
phase 2   .855
phase 3   .955
phase 4  1.000
```

Mean null-motion fraction:

```text
phase 1   .691
phase 2   .909
phase 3   .856
phase 4   .937
```

After the first phase, continuing state motion has moved to roughly the isotropic/null
side of the readout geometry rather than repeatedly sweeping through strongly potent
directions.

---

## 5. It did not simply stop computing

A critical control is total hidden-state motion.

Mean squared hidden-state displacement per phase in the same compact sweep:

```text
ordinary final-only    13.38   15.22   13.54   14.78
phase-supervised       14.44   14.05   10.64   13.95
```

The state continues to move substantially.

What changes is the decomposition of that motion.

Mean output-potent motion energy:

```text
ordinary final-only     6.56    5.45    4.71    7.51
phase-supervised        4.40    1.30    1.54    0.90
```

Mean output-null/private motion energy:

```text
ordinary final-only     6.81    9.76    8.83    7.27
phase-supervised       10.04   12.76    9.10   13.05
```

So the phase-legible model has **not** merely learned to become stationary after an
early answer.

It continues to transform its hidden state while relocating much of the later motion
away from the public readout.

---

## 6. The public logits stabilize while the private state moves

Mean squared change in the six public logits per sample:

```text
ordinary final-only    57.24   43.69   36.76   69.69
phase-supervised       49.86    5.01   10.09    4.08
```

The first phase must still establish/update the public answer, so substantial potent
motion there is expected.

After that, the phase-supervised model's public channel changes far less even though
the hidden state continues moving by a similar amount.

A useful schematic is therefore:

```text
phase 1
    update public answer strongly

phases 2..4
    continue internal transformation
    mostly in directions weakly seen by the public head
```

This is a much more specific mechanism than "deep supervision gives early exits."

---

## 7. Connection to asynchronous mixed maturity

The companion deadline experiment found that the phase-supervised model generalized
well to **asynchronous mixed-maturity states that were never shown during training**.

This output-null result suggests one reason why.

If different local parts of the state are at different later phases, but much of the
phase-to-phase difference lives in directions null to the shared readout, then mixing
those maturities perturbs the public logits much less than it would in the final-only
model.

That yields a concrete hypothesis:

> **Maturity robustness can emerge when the differences between valid computational
> stages are concentrated in receiver-null dimensions.**

This should be attacked directly rather than assumed.

---

## 8. Stronger tests now available

### A. Predict async robustness from null geometry

Across model seeds, tasks, state dimensions and training strengths, ask whether

```text
more null-aligned inter-phase motion
```

predicts

```text
higher accuracy under unseen mixed-maturity schedules.
```

A correlation would connect the geometry to the deadline behavior.
A null correlation would weaken the interpretation.

### B. Vary public-subspace rank

Change state size and class/readout bottleneck so that the trivial isotropic null
fraction changes.

Report excess null alignment relative to

```text
1 - rank(W)/state_dim
```

rather than raw null percentage.

### C. Delay-warp OOD

The null-space organization is useful only if it survives edge-delay distributions not
used to motivate the experiment.

### D. Premature-action task

Sometimes partial computation should remain private.

Construct a task where an early local cue is systematically misleading until another
route arrives.  Reward abstention/stability rather than immediate output.

A useful system should learn both:

```text
make safe partial variables public
keep dangerous unfinished variables null
```

This is closer to the motor-preparation interpretation than simply maximizing early
accuracy.

### E. Strong non-geometric controls

Deep supervision and output-null geometry are not local-geometry inventions.
The same diagnostics must be run on appropriate anytime/recurrent/asynchronous
baselines before KYY claims an advantage.

---

## 9. Current ledger

### Survives

- ordinary KYY intermediate states strongly thrash the final public readout;
- shared phase supervision makes intermediate task meaning much more stable;
- the hidden state keeps moving substantially after the public answer becomes useful;
- in the compact scratch sweep, later motion shifts strongly toward the readout-null
  space;
- this provides a plausible mechanism for the unseen mixed-maturity robustness in the
  deadline gate.

### Guardrails

- a large null space is expected from dimension alone; `.8125` is the relevant
  isotropic baseline for 6-of-32;
- output-null/output-potent computation is established neuroscience;
- deep supervision / anytime readouts are established ML;
- these are exploratory scratch numbers until the committed multi-seed script
  reproduces them.

### Open

The strongest next question is no longer:

> can KYY answer early?

It is:

> **Can a small receiver-relevant public subspace remain useful under arbitrary local
> computational maturity while the larger private state continues to compute — and
> does local physical scheduling buy anything once strong controls receive the same
> opportunity?**
