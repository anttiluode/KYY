# Pass 30 — ten-seed C101 stress: relation defect predicts whether zero-shot legalization preserves the decoder

Date: 2026-08-10

Pass 29 found one failed zero-shot legalization at `C_101` and suggested a possible distance-to-legal-manifold boundary.

Pass 30 increases the `C_101` sample to ten seeds under the same random-start protocol.

The boundary survives.

More specifically:

> **Among the nine learned runs that reach at least 99% clean accuracy at the training horizon before projection, pre-projection cyclic relation defect is strongly negatively associated with post-projection decoder accuracy.**

The projected code's orbit radius is not.

This gives KYY two distinct geometric quantities with different jobs.

---

## 1. Configuration

```text
group             C_101
modes             8 complex / 16 real
train length      16
steps             1500
seeds             0..9
random start      yes: x_0 ~ Uniform(Z_101)
later increments  {0,1,2,3,4}
readout            learned linear 101-way classifier
legalization       nearest exact C_101 character per mode
post-snap training none
```

Archived data:

- `results/harmonic_n101_ten_seed_stress.csv`

Workflow:

- `.github/workflows/harmonic-n101-stress.yml`

---

## 2. Raw result

```text
seed  pre-L16  pre-L1024  relation defect  post-L16  post-L1024
0      1.000      0.091       0.729          1.000      1.000
1      0.996      0.065       0.917          0.833      0.820
2      1.000      0.089       0.691          1.000      1.000
3      1.000      0.106       0.443          1.000      1.000
4      1.000      0.089       0.669          1.000      1.000
5      0.991      0.098       0.774          0.867      0.831
6      1.000      0.137       0.703          0.983      0.980
7      0.981      0.053       0.956          0.522      0.465
8      1.000      0.094       0.505          1.000      1.000
9      0.996      0.083       0.884          0.718      0.751
```

Seed 7 misses the preregistered `>=0.99` pre-projection training-horizon filter and is excluded from the preservation correlation below.

Among the remaining nine:

```text
post-snap L16 >= 0.99 in 5 / 9 runs.
```

The point of the stress test is not that five succeeded.

It is that the failures are structured rather than random.

---

## 3. Relation defect predicts preservation

For the nine eligible runs, correlate pre-projection one-cycle state relation defect

```text
D_rel = ||z(s+101)-z(s)||
```

with clean L16 accuracy after zero-shot legalization.

Result:

```text
Pearson r   ~= -0.757    p ~= 0.0183
Spearman rho ~= -0.840   p ~= 0.0046
```

The five `>=0.99` post-snap successes have mean pre-snap relation defect approximately

```text
0.607.
```

The four preservation failures have mean relation defect approximately

```text
0.820.
```

This is still a tiny synthetic experiment. But unlike the earlier three-point pattern, it is now enough to say that relation defect is carrying predictive information in this protocol.

---

## 4. Direct snap distance says almost the same thing

Because this is a diagonal cyclic bank, relation defect is closely related to how far the finite orbit moves when angles are rounded to exact characters.

For every symbolic state `s`, define

```text
d_snap(s) = ||z_learned(s) - z_legal(s)||.
```

Use the maximum over the 101-state orbit.

Across the same nine eligible runs:

```text
max snap distance vs post-snap L16 accuracy

Pearson r   ~= -0.756    p ~= 0.0184
Spearman rho ~= -0.840   p ~= 0.0046.
```

So in this special representation, relation defect is functioning as a cheap algebraic proxy for representation displacement.

That will not remain this trivial for general noncommuting generators.

---

## 5. Projected orbit radius does NOT predict preservation

Correlate the legalized exact code's orbit noise radius with post-snap L16 accuracy:

```text
Pearson r   ~= +0.151    p ~= 0.70
Spearman rho ~= -0.183   p ~= 0.64.
```

Essentially nothing in this sample.

That is important because earlier KYY passes found orbit radius useful for a *different* question: how much implementation phase error an already-legal code can tolerate before neighboring symbolic states collide.

Pass 30 separates the roles:

```text
DISTANCE TO LEGAL MANIFOLD / RELATION DEFECT
    predicts whether zero-shot compilation preserves the old decoder.

MARGIN OF THE LEGAL CODE
    can contribute to robustness after a compatible decoder is in place.
```

Those are not the same resource.

---

## 6. Decoder geometry is now unavoidable

A further warning appears in the noisy post-legalization runs.

For these projected learned `C_101` codes, projected orbit radius by itself does not cleanly predict L1024 accuracy under the shared phase-error perturbation either.

That is not surprising in retrospect.

The decoder was learned on the approximate orbit and then retained. Two exact character codes with similar geometric separation can have very different **classifier margins** under their inherited readouts.

So the compiler state should now be thought of as

```text
(OPERATOR, PORT/READOUT)
```

rather than operator alone.

This echoes both TWC's port-equivalence discipline and the broader warning from recent modular-arithmetic representation/readout work: a representation and the map that reads it are jointly responsible for behavior.

---

## 7. A finite-state zero-shot preservation certificate

For this toy cyclic model, a simple rigorous compiler certificate is available.

Let the learned linear readout have class weights `w_j` and biases `b_j`.

For symbolic state `s`, let `z_s` be its pre-snap prototype and `z*_s` its legalized prototype.

For every competing class `j != s`, define the pre-snap pairwise margin

```text
m_sj = (w_s - w_j)^T z_s + (b_s - b_j).
```

After the snap, the pairwise logit difference changes by

```text
(w_s - w_j)^T (z*_s - z_s).
```

Cauchy-Schwarz gives

```text
change >= - ||w_s-w_j|| ||z*_s-z_s||.
```

Therefore a sufficient condition for class `s` to survive is

```text
m_sj > ||w_s-w_j|| ||z*_s-z_s||
```

for every `j != s`.

If this holds for all 101 symbolic states, the post-snap readout is certified correct on the entire exact finite orbit.

Since the legalized recurrence satisfies `A^101=I`, that finite-orbit certificate implies arbitrary clean rollout length in exact arithmetic.

Nothing deep is being claimed mathematically: this is ordinary linear-classifier margin plus Cauchy-Schwarz.

What matters is that it becomes a **compiler acceptance test**:

```text
TRAIN
  -> find nearest legal operator
  -> compute finite orbit displacement
  -> certify readout margin survives
  -> only then commit the snap.
```

No 1,000-token validation rollout is required for this special finite-state case.

---

## 8. Prior-art posture remains conservative

The individual ingredients continue to have owners:

- finite-group / roots-of-unity recurrence: known;
- diagonal complex SSM group tracking: known;
- approximate representations and relation defect: known mathematics;
- neural-network repair by constrained weight changes: known as a broad methodology;
- automata extraction from trained RNNs: known;
- representation/readout alignment as a failure mechanism: known in several settings.

The residual KYY question is a compiler question:

> **Can a trained approximate recurrent computation be mapped onto a cheaper / exact / physically legal operator family, with an explicit algebraic defect and a behavior-preservation certificate, and does that improve long-horizon deployment?**

Passes 28--30 show a small cyclic instance where the answer is often yes and where the failures are measurable before deployment.

That is the right scope.

---

# Current pin

The strongest new empirical sentence after the ten-seed stress is:

> **For short-trained `C_101` rotary trackers, the amount by which the learned recurrent operator violates the cyclic relation predicts whether zero-shot projection onto the nearest exact cyclic representation will preserve the trained decoder; the geometric separation of the target exact code does not predict that preservation.**

This suggests a two-stage compiler objective:

```text
1. LEGALIZABILITY
   distance / relation defect / decoder-preservation margin

2. DEPLOYMENT ROBUSTNESS
   exact-code margin / implementation precision / wiring cost
```

That split did not exist when KYY started.
