# Pass 31 — exhaustive finite-orbit port verification beats the pretty norm certificate

Date: 2026-08-10

Pass 30 ended with a proposed compiler acceptance test for post-training operator legalization.

For a trained linear readout, a Cauchy-Schwarz perturbation bound can certify that the snap from the learned orbit to the legalized orbit is too small to change any class decision.

That bound is mathematically valid.

In the `C_101` stress experiment it is also **completely useless**.

It certifies zero out of ten runs.

The more direct finite-state compiler pass is much better:

> **enumerate every state in the legalized finite orbit and verify the actual observable port/readout on all of them.**

For `C_101` that is only 101 states.

The exhaustive certificate accepts exactly the five clean zero-shot legalizations and rejects all five incompatible ones, including a 98%-looking near miss that sampled rollout did not expose sharply.

Code:

- `map/legalization_certificate_probe.py`
- `tests/test_legalization_certificate_probe.py`

Archived data:

- `results/harmonic_n101_certificate.csv`

Workflow:

- `.github/workflows/harmonic-n101-certificate.yml`

CI tests are green.

---

## 1. What is being certified

For every symbolic state

```text
s in {0,...,100},
```

construct two normalized hidden prototypes:

```text
z_s     = learned phase-bank state
z*_s    = legalized exact-character state.
```

The readout is the trained ordinary linear classifier

```text
logits(z) = W z + b.
```

No readout parameter is changed.

The compiler asks two questions.

### A. Exhaustive finite-orbit check

Evaluate

```text
argmax(W z*_s + b)
```

for all 101 legal states.

If every state has the correct label with positive pairwise margin, then the legalized **ideal** recurrence has no unseen clean symbolic state left to test.

Because the exact cyclic operator satisfies

```text
A^101 = I,
```

every clean integer-input history maps onto one of those same 101 states.

So in exact arithmetic, verifying the whole finite orbit replaces arbitrary-length clean rollout testing.

Implementation precision is a separate deployment question; floating/hardware phase errors can still move the realized trajectory away from the ideal finite orbit.

### B. Cauchy perturbation certificate

For each state `s` and competing class `j`, let the pre-snap pairwise margin be

```text
m_sj = (w_s-w_j)^T z_s + (b_s-b_j).
```

The snap changes the pairwise logit by

```text
(w_s-w_j)^T (z*_s-z_s).
```

A sufficient condition for preserving the decision is

```text
m_sj > ||w_s-w_j|| ||z*_s-z_s||
```

for every `s,j`.

This is ordinary Cauchy-Schwarz, not new mathematics.

---

## 2. Ten-seed result

```text
seed  pre-L16  post-L16  post-L1024  projected prototype acc  exhaustive cert
0      1.000     1.000      1.000              1.000               PASS
1      0.996     0.816      0.822              0.822               FAIL
2      1.000     1.000      1.000              1.000               PASS
3      1.000     1.000      1.000              1.000               PASS
4      1.000     1.000      1.000              1.000               PASS
5      0.991     0.890      0.832              0.832               FAIL
6      1.000     0.980      0.980              0.980               FAIL
7      0.981     0.495      0.467              0.465               FAIL
8      1.000     1.000      1.000              1.000               PASS
9      0.996     0.714      0.752              0.752               FAIL
```

The exhaustive legalized-orbit certificate therefore gives:

```text
PASS:  5 / 10
FAIL:  5 / 10
```

and the pass set is exactly

```text
{0,2,3,4,8}.
```

Those are also the five runs whose sampled post-snap clean rollout is perfect through length 1024.

---

## 3. The useful near miss: seed 6

Seed 6 looked almost good if we only sampled rollout:

```text
post-snap L16     0.9805
post-snap L1024   0.9803.
```

But exhaustive legal-orbit checking finds

```text
projected prototype accuracy = 0.980198...
```

which is exactly

```text
99 / 101 states.
```

Two legal symbolic states are misread.

Its minimum projected true-class margin is slightly negative:

```text
-0.0874.
```

So the compiler can reject seed 6 deterministically without pretending that 98% sampled accuracy is close enough to an exact finite-state implementation.

This is the kind of distinction the TWC-style port discipline was supposed to enforce.

---

## 4. The Cauchy certificate fails as an engineering tool here

The conservative perturbation slack is negative in **all ten runs**, including every perfect legalization.

Examples:

```text
seed 0   exhaustive PASS   Cauchy slack -2.784
seed 3   exhaustive PASS   Cauchy slack -0.705
seed 8   exhaustive PASS   Cauchy slack -0.996
```

Thus:

```text
Cauchy certified = 0 / 10.
```

No false positives; also no useful positives.

The bound throws away too much directional information by replacing the actual readout displacement with

```text
||w_s-w_j|| ||delta_s||.
```

KYY should keep the derivation as a mathematically clean sufficient condition but **not make it the main compiler pass** for small finite machines.

The negative result is preferable to adding a tuned constant until it looks useful.

---

## 5. Exhaustive port checking is not the same as another long rollout

A length-1024 rollout samples trajectories through the state machine.

The exhaustive pass evaluates the complete legal state set directly.

For `C_101`:

```text
101 prototypes
```

replace arbitrarily many clean time steps.

This is possible only because the compiler already knows the symbolic algebra and has projected the recurrence onto an exact finite representation.

The order of operations matters:

```text
learn approximate operator
        |
        v
legalize to exact finite algebra
        |
        v
enumerate complete legal orbit
        |
        v
verify observable readout/ports
        |
        v
accept or reject compile
```

Before legalization, the freely learned recurrence does not factor through exactly 101 hidden states, so finite-orbit enumeration is not a complete long-horizon proof of that original model.

---

## 6. This makes the TWC connection unusually literal

TWC's recurring lesson was:

> an internal operator is not validated merely because its hidden parameters look physically plausible; correctness is defined at observable ports and by identifiability/equivalence.

KYY now has the same structure in a symbolic recurrent setting.

The compiler cannot say:

```text
"A^101 = I, therefore compile succeeded."
```

Seed 6 disproves that.

The correct contract is:

```text
INTERNAL LEGALITY
    exact task relations hold

PLUS

PORT EQUIVALENCE
    the legal finite orbit is read as the intended symbolic behavior.
```

That is a cleaner reunion of the TWC and KYY lines than the earlier wave metaphors were.

---

## 7. Stronger interpretation of the stress data

Pass 30 found that pre-snap relation defect predicts whether zero-shot legalization preserves the readout.

Pass 31 says what that predictor is for:

```text
relation defect / snap distance
    = cheap triage before compilation

exhaustive legal-orbit port verification
    = deterministic acceptance test after compilation.
```

Those are different roles.

A compiler could therefore do:

```text
1. Estimate nearest legal operator.
2. Reject obviously distant candidates cheaply.
3. Apply legalization.
4. Exhaustively verify the complete legal orbit when the quotient is small.
5. Only then price implementation precision / physical realization.
```

For larger groups where enumeration is expensive, stronger structural certificates or factorized verification become interesting. But `C_101` does not need them.

---

## 8. Next question: failed ports, not failed operators

Once a snap produces an exact legal operator but the inherited readout fails, there are two logically different outcomes:

```text
A. reject the compile;

or

B. keep the exact legal operator and recalibrate only the readout/ports.
```

The second is now the natural next falsifier.

It must be cheap enough that it still deserves to be called compilation rather than retraining the model from scratch.

For this cyclic toy, a readout-only recalibration is intentionally easy because all 101 legal prototypes are enumerable. That makes it a useful compiler mechanics test but **not evidence of learning power**.

The question to measure is resource cost:

```text
How little port calibration is required after operator legalization?
```

If the answer is "a tiny linear recalibration always restores exact behavior," then the deployment pipeline becomes:

```text
train approximate recurrence
-> legalize recurrence
-> verify ports
-> recalibrate ports if needed
-> verify again
-> deploy exact operator.
```

That would be a concrete compiler pipeline.

---

# Current pin

The compiler result is now cleaner than Pass 28:

> **In the `C_101` legalization stress test, exhaustive verification of the complete legalized finite orbit exactly separates the five zero-shot-compatible operator projections from the five incompatible ones. A generic norm perturbation certificate is safe but certifies none of them, so direct port verification is the practical compiler acceptance test for small finite machines.**

The next experiment should repair **ports only**, not touch the legalized recurrent operator.
