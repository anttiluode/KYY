# Pass 29 — cross-modulus replication finds a legalization radius, not a miracle

Date: 2026-08-10

Pass 28 found that snapping a freely trained `C_31` rotary recurrence to the nearest exact cyclic characters repaired length extrapolation in 3/3 seeds without retraining the readout.

The obvious confound was class coverage: with length 16 and increments in `{0,...,4}`, `C_31` is small enough that training trajectories can visit the whole symbolic state set, while `C_101` is not.

Pass 29 fixes that before changing modulus.

The result survives through `C_61`, and mostly survives `C_101`, where one seed finally breaks the zero-shot projection assumption.

That failure is useful. It turns "projection works" into a sharper question:

> **How close must a learned approximate representation be to the legal representation manifold for post-training algebraic legalization to preserve the trained decoder?**

---

## 1. Random-start protocol

For every training and evaluation sequence:

```text
x_0 ~ Uniform(Z_n)
x_t in {0,1,2,3,4}, t > 0
state_t = cumulative sum mod n.
```

The first token exposes a uniformly random symbolic starting state. Subsequent updates remain local small increments.

This means a length-16 run can train a decoder over all symbolic classes even when `n > 64`.

Shared configuration:

```text
modes            = 8 complex = 16 real coordinates
train length      = 16
steps             = 1500
seeds             = 0,1,2
test lengths      = 16,64,256,1024
```

Models:

```text
learned          unconstrained recurrent angles + learned linear readout
low_coherence    exact C_n character bank + learned linear readout
```

After training the learned model, the same nearest-character legalization from Pass 28 is applied once. The linear readout is unchanged.

Archived results:

- `results/harmonic_cross_modulus_random_start.csv`

Workflow:

- `.github/workflows/harmonic-cross-modulus.yml`

---

## 2. Exact controls

Every exact low-coherence control reaches

```text
100% at L16, L64, L256, and L1024
```

for all three seeds at

```text
C_31, C_61, C_101.
```

So the 8-mode state size and the random-start protocol themselves do not prevent exact tracking at these moduli.

---

## 3. Unconstrained learned models still drift

The freely learned models all solve, or nearly solve, the training horizon and then degrade badly with length.

Mean clean L1024 accuracy before legalization:

```text
C_31    0.091
C_61    0.081
C_101   0.083
```

So random starts do not remove the original failure mode.

---

## 4. C31 replication under random starts

```text
seed   pre L16   pre L1024   post L16   post L1024
0      1.000     0.090       1.000      1.000
1      1.000     0.088       1.000      1.000
2      1.000     0.095       1.000      1.000
```

Pass 28 therefore was not an artifact of every trajectory starting at the identity.

---

## 5. C61 replication

```text
seed   pre L16   pre L1024   post L16   post L1024
0      1.000     0.078       1.000      1.000
1      1.000     0.071       1.000      1.000
2      1.000     0.096       1.000      1.000
```

Again:

```text
3 / 3 zero-shot operator legalizations
preserve the decoder and restore clean L1024 accuracy to 100%.
```

---

## 6. C101 finally breaks one seed

```text
seed   pre L16   pre L1024   post L16   post L1024
0      1.000     0.088       1.000      1.000
1      0.996     0.074       0.833      0.820
2      1.000     0.088       1.000      1.000
```

So the clean zero-shot result across the cross-modulus sweep is:

```text
C31     3/3 exact after legalization
C61     3/3 exact after legalization
C101    2/3 exact after legalization
```

or

```text
8/9 learned runs repaired perfectly through L1024.
```

The ninth run is not a small long-horizon degradation. Its **short-horizon** accuracy falls from 0.996 before legalization to 0.833 afterward.

That tells us what failed:

> the snap moved the recurrent representation far enough that the old decoder no longer matches it.

This is a behavior-preservation failure at the representation/readout interface, not a failure of the exact cyclic representation itself.

The exact `C_101` controls remain 100%.

---

## 7. Relation defect tracks the boundary qualitatively

Pre-legalization one-cycle state relation defects for the `C_101` learned runs:

```text
seed 0    0.726   -> zero-shot projection succeeds
seed 1    0.909   -> zero-shot projection fails
seed 2    0.692   -> zero-shot projection succeeds
```

The failed seed has the largest relation defect in the whole 9-run learned cross-modulus set.

Its projected character code also has the smallest orbit radius among the three `C_101` projections:

```text
seed 0   r = 0.468
seed 1   r = 0.407   <-- failed decoder preservation
seed 2   r = 0.462
```

This is not enough data to define a threshold.

But it suggests the compiler needs **two** checks, not one:

```text
1. relation legality:
       is the compiled operator on the exact algebra?

2. behavior-preservation / decoder margin:
       did the projection move the representation farther than the trained
       readout can tolerate?
```

The first can be exact while the second fails.

---

## 8. Geometry of the snap

For a learned phase bank `z(s)` and its legalized version `z*(s)`, define

```text
d_snap(s) = ||z(s) - z*(s)||.
```

Across the finite orbit, the largest snap displacements in the `C_101` seeds are approximately

```text
seed 0   0.719   succeeds
seed 1   0.902   fails
seed 2   0.687   succeeds
```

Again, only three points.

But this is the geometric quantity the physical/compiler story actually wants:

> **How far is the trained orbit from the nearest legal orbit, relative to the decision margin of the readout?**

That is a much more concrete object than "does the model generalize?"

---

## 9. A very recent neighboring result sharpens the warning

Janati et al. (arXiv:2608.07436, 7 Aug 2026), *Post-Grokking Collapse at the Representation–Readout Interface in Muon-Trained Transformers*, study modular arithmetic in a Transformer and show that a learned Fourier computation can fail when representation and readout drift out of alignment even while spectral support looks nearly unchanged.

This is a different architecture and a different failure mechanism, but the conceptual warning is directly relevant:

```text
representation and readout are a coupled object.
```

Pass 29 demonstrates that same practical constraint in the legalization setting: modifying the operator onto a mathematically better exact representation is only useful zero-shot if the existing decoder remains compatible.

KYY should therefore stop treating operator legality and readout compatibility as one criterion.

---

## 10. The compiler contract gets one more field

The current cost/acceptance vector should now include something like

```text
relation defect
orbit separation
implementation drift budget
snap distance to legal manifold
readout preservation margin
wiring / communication cost
```

A candidate compiled representation is accepted only if it is both:

```text
ALGEBRAICALLY LEGAL
and
BEHAVIORALLY COMPATIBLE.
```

That is exactly the TWC identifiability lesson reappearing in a different form: an internal representation is not correct merely because it satisfies a preferred internal model; it must preserve the observable behavior at the ports/readout.

---

## 11. Next falsifier

Do not add more architecture yet.

Stress the boundary at `C_101` with more seeds.

Preregistered questions:

1. Among models that reach at least 99% at L16 before projection, how often does zero-shot legalization preserve at least 99% at L16?
2. Does pre-projection relation defect predict preservation failure?
3. Does orbit snap distance predict it better?
4. Does projected orbit radius add information beyond snap distance?
5. If zero-shot preservation fails, can a **readout-only calibration** repair the legalized operator cheaply?

The fifth test matters for the compiler interpretation. A compiler is allowed to regenerate or calibrate a port map after changing the internal operator; it is not allowed to pretend that a broken port map is fine.

---

# Current pin

Pass 28 survives a real falsifier, but with a boundary:

> **Across `C_31`, `C_61`, and `C_101`, 8 of 9 freely learned short-horizon cyclic trackers could be snapped onto their nearest exact recurrent group representation after training while keeping the decoder fixed, yielding 100% clean accuracy through length 1024. The one failure was the learned run with the largest relation defect; the snap itself broke short-horizon decoder compatibility.**

So the emerging compiler hypothesis is no longer simply

```text
train -> snap -> exact.
```

It is

```text
train
  -> audit distance to legal operator family
  -> snap only if behavioral margin permits
  -> otherwise recalibrate ports/readout
  -> deploy exact operator.
```

That is less magical and more useful.
