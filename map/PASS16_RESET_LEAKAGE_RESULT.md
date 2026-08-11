# Pass 16 — reset leakage result: a lossless local body can hide reset history, then leak it back

Date: 2026-08-10

This is the first empirical result produced specifically by the wave/reset split introduced in Passes 11–15.

It is deliberately small.

It does **not** establish a new architecture or a general theorem.

It does give a clean experimental example of the distinction between:

```text
looks reset at the current port
        !=
is behaviorally reset under every future continuation.
```

---

# 1. Task

`permreset3` is a 3-state deterministic machine with three tokens:

```text
I : identity
C : cycle 0 -> 1 -> 2 -> 0
R : reset every prior state -> 0
```

Training sequences have length 16.

The audit also constructs pairs of sequences that:

1. have different I/C-only prefixes and therefore different behavioral states;
2. receive the same reset `R`;
3. receive the exact same continuation afterward.

The correct task behavior is identical from the reset onward.

The audit compares the model outputs for the paired histories at every lag after reset.

Metrics:

```text
logit L2
probability total variation (TV)
argmax prediction mismatch rate
ordinary task accuracy
```

---

# 2. Models in the strengthened run

All use hidden/state dimension 8.

Four existing KYY models were used; **no new architecture was added**.

## `geom_scatter`

Token-dependent products of local orthogonal two-port scatterers on a ring.

All hidden linear transitions are norm-preserving.

## `householder2`

Token-dependent product of two dense Householder reflections.

Also orthogonal/norm-preserving.

## `complex_diag`

Independent rotary 2D blocks with learned radii strictly between zero and one.

Each finite-step linear map is technically invertible, but history differences contract.

## `gru`

Dense nonlinear gated reference with direct contraction/overwrite mechanisms.

---

# 3. Run configuration

GitHub Actions workflow `reset-leakage`, run 3:

```text
seeds:                 0,1,2
training steps:        1000
train length:          16
test lengths:          16,64,256
state dimension:       8
batch size:            64
paired prefix length:  16
shared continuation:   64
pair batches:          5
```

The workflow completed successfully and stored the raw JSON as the `reset-leakage` artifact.

---

# 4. Accuracy result

Mean ordinary sequence accuracy over the three seeds:

| model | L=16 | L=64 | L=256 |
|---|---:|---:|---:|
| complex_diag | **1.0000** | **1.0000** | **1.0000** |
| householder2 | 0.9204 | 0.6766 | 0.4714 |
| geom_scatter | **0.9946** | 0.6096 | 0.4213 |
| GRU | **1.0000** | **1.0000** | **1.0000** |

Population standard deviation across the 3 seeds at `L=256`:

```text
complex_diag: 0
householder2: ~0.00773
geom_scatter: ~0.00374
GRU:          0
```

The crucial point is `geom_scatter`.

Unlike the earlier 250-step smoke result, it now fits the training-length problem almost perfectly and consistently:

```text
L=16 accuracy = 0.9946 +/- ~0.00018
```

but loses the reset behavior as sequence length grows.

So its long-horizon failure is no longer well described as simple failure to learn the short task.

`householder2` still underfits at `L=16`, so its leakage result is less diagnostic and should not be used as strong evidence by itself.

---

# 5. The geometric scatterer hides the old history at reset, then leaks it back

Average paired-history leakage over the three `geom_scatter` seeds:

| lag after R | mean probability TV | prediction mismatch rate |
|---:|---:|---:|
| 0 | **0.000169** | **0.0000** |
| 1 | 0.000208 | 0.0000 |
| 2 | 0.000511 | 0.0000 |
| 4 | 0.02548 | 0.0250 |
| 8 | 0.11713 | 0.11875 |
| 16 | 0.14298 | 0.15313 |
| 32 | 0.15955 | 0.16354 |
| 64 | 0.15494 | 0.16250 |

Across the full curve:

```text
mean TV across lags:       ~0.13806
worst observed max TV:     ~0.91795
max mismatch rate:         ~0.20417
```

This is the important qualitative shape:

```text
different histories
        |
        v
      reset R
        |
        v
ports nearly identical          lag 0
        |
        v
same future orthogonal tokens
        |
        v
old difference reappears        lag 4..64
```

The current readout therefore behaves as if reset succeeded immediately, while the future continuation reveals that the hidden distinction was not removed.

That is exactly the failure mode Pass 15 calls **non-future-proof reset**.

---

# 6. Why this is structurally expected for `geom_scatter`

For two trajectories receiving the same token, hidden difference obeys

```text
d' = A_x d.
```

Every `geom_scatter` `A_x` is orthogonal.

Therefore

```text
||d'||_2 = ||d||_2.
```

The reset token cannot shrink the hidden-history difference at all.

Training can rotate the difference into a direction to which the current linear readout is insensitive.

But later token matrices rotate that same persistent difference again.

Unless the difference lies in a common future-unobservable invariant subspace, it can return to the port.

The empirical leakage curve is exactly consistent with that mechanism.

This is an interpretation supported jointly by the architecture and the observed port behavior; it is not a proof that every orthogonal recurrent model must show the same optimization trajectory.

---

# 7. Contractive complex recurrence behaves very differently

`complex_diag` reached exact sampled accuracy at all three tested lengths in every seed:

```text
L=16  = 1.0
L=64  = 1.0
L=256 = 1.0
```

and had zero paired-history prediction mismatches after reset.

Average probability TV curve:

| lag | mean TV |
|---:|---:|
| 0 | 0.000217 |
| 1 | 0.000232 |
| 2 | 0.000285 |
| 4 | 0.000412 |
| 8 | 0.000465 |
| 16 | 0.000576 |
| 32 | 0.0000821 |
| 64 | 0.00000154 |

Overall:

```text
mean TV:          ~0.0001724
max observed TV:  ~0.01742
max mismatch:     0
```

Its transition blocks are technically invertible because their learned radii are nonzero.

Therefore **invertibility alone is not the useful empirical boundary**.

The material distinction is that the radii are less than one, so old differences can contract toward zero and become numerically/behaviorally irrelevant.

---

# 8. GRU nearly deletes the port-visible history immediately

GRU also achieved sampled accuracy 1.0 at all tested lengths and all seeds.

Average leakage:

| lag | mean probability TV |
|---:|---:|
| 0 | 0.000947 |
| 1 | 0.000165 |
| 2 | 0.0000503 |
| 4 | 0.0000118 |
| 8 | 0.000000965 |
| 16 | ~1.58e-8 |
| 32 | ~1.95e-12 |
| 64 | 0 |

Across seeds:

```text
mean TV across lags: ~1.86e-5
max mismatch:        0
```

This is consistent with direct gated contraction/overwrite.

---

# 9. Householder result is suggestive but not yet clean evidence

`householder2` has the same norm-preserving hidden-difference issue in principle.

Its three-seed result:

```text
L=16  ~0.9204
L=64  ~0.6766
L=256 ~0.4714
mean leakage TV ~0.1414
max mismatch    ~0.2021
```

But because it did not fit the training-length task well even after 1000 steps, do **not** use it as evidence that trained Householder recurrence necessarily fails reset extrapolation.

It remains a control showing that generic dense orthogonal mixing does not automatically solve the problem.

---

# 10. What this result does establish

At this scale and in these implementations:

1. The original local orthogonal `geom_scatter` can almost perfectly fit short mixed permutation/reset sequences.
2. Its immediate reset output can be nearly history-independent.
3. Nevertheless, the prior history becomes visible again under identical future inputs.
4. The failure is highly reproducible across the three tested seeds.
5. A contractive complex recurrence and a GRU do not show this failure and extrapolate perfectly to the tested length 256.

The cleanest sentence is:

> **KYY's lossless local scatterer learned to hide reset history, not to forget it.**

That is a result about this benchmark/model pair, not a universal theorem about waves.

---

# 11. What this does NOT establish

Do not claim:

- all orthogonal/unitary RNNs cannot implement reset-like classification;
- contraction is always better than norm preservation;
- the Geometric Neuron needs dissipation everywhere;
- this diagnostic is novel without a dedicated literature search;
- the toy `permreset3` task predicts language-model quality;
- probability-TV differences imply thermodynamic energy loss.

Gated orthogonal/unitary RNNs, reset automata, quantum/reversible automata, and switched-system observability already cover much of the conceptual territory.

---

# 12. What changed in the physical picture

The useful three-way split is now:

```text
CONSERVATIVE PROPAGATION
    difference norm persists
    good for reversible/group-like work

CONTRACTION
    history difference fades continuously
    can approximate forgetting without an exact singular event

RESET / PINCH
    distinction is removed in the effective realization
    appropriate for genuinely irreversible behavioral transitions
```

This is better than the earlier binary split "invertible versus reset."

The KYY compiler should therefore ask where each of the three behaviors is useful, rather than trying to force one primitive type to do every token transition.

---

# 13. Next falsifier

Do **not** build the hybrid yet.

Next, use exact/symbolic task structure to assign transition types and ask whether a mixed primitive compiler can beat uniform recurrence under a declared cost model:

```text
permutation/group token
    -> conservative local word

contractive but not exact-reset token
    -> controlled damping primitive

true behavioral reset / singular token
    -> explicit pinch/write
```

Then compare against existing gated/DeltaNet/SSM baselines after another prior-art pass.

The new thing worth preserving from this experiment is not an architecture.

It is the **future-continuation reset leakage test**.