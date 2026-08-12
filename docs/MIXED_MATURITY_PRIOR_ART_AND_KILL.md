# Mixed maturity — prior-art mapping and another kill

Date: 2026-08-12

The partial-maturity branch has now collided with direct Transformer prior art.
That is useful because it separates three effects that were initially being bundled
together.

---

## 1. Phase-dependent coordinate frames are not special to KYY

The `phase_legibility_probes.py` diagnostic asks whether an intermediate KYY state lacks
task information or merely presents it in coordinates that the final readout does not
understand.

In a compact scratch run of an ordinary final-only `perm3` model:

```text
phase                         1      2      3      4
native final head           .288   .036   .191  1.000
phase-specific linear probe .986  1.000  1.000  1.000
one shared post-hoc probe   .368   .371   .936   .969
```

The answer is therefore almost perfectly linearly decodable very early, while the
final head can map the same state to nearly always-wrong labels.

This has a close Transformer analogue.

Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens*
(arXiv:2303.08112), fit an affine translator for each frozen Transformer block so that
intermediate hidden states can be decoded into vocabulary predictions more reliably
than with the raw final unembedding/logit lens.

So:

```text
intermediate answer present
+
final output map misaligned
+
layer/phase-specific affine probe repairs it
```

is already an established kind of phenomenon.

KYY does not own this.

---

## 2. One shared output head across depths is directly occupied too

The phase-supervised KYY condition uses one shared linear head after each homogeneous
checkerboard phase.

That also has direct Transformer prior art.

Elhoushi et al., *LayerSkip: Enabling Early Exit Inference and Self-Speculative
Decoding* (ACL 2024, DOI 10.18653/v1/2024.acl-long.681), train LLMs with an early-exit
loss in which **all Transformer layers share the same exit**.  LayerSkip adds no
auxiliary output heads for those exits and uses the training recipe to make earlier
layers decodable by the shared LM head.

That is extremely close to the abstract operation performed by KYY phase supervision.

Therefore the following are not novelty claims:

```text
shared public decoder across computation depth
training intermediate states to use that decoder
early prediction without per-depth heads
```

---

## 3. Output-null geometry is an interpretation, not yet an architecture claim

The KYY output-null diagnostic remains useful.

After shared phase supervision, later hidden-state motion becomes far less aligned with
the shared readout's row space while total hidden-state motion stays large.  This
provides a geometric explanation for why the public logits remain stable.

But because LayerSkip and other early-exit training already deliberately make a shared
output head useful across depth, KYY cannot assume that this null-space reorganization
is unique.

A direct next control would be to run the same row-space/null-space analysis on a
LayerSkip-style Transformer or another shared-exit network.

Until then:

> **output-null maturity is a diagnostic description of what KYY learned, not a KYY
> invention.**

---

## 4. What was still apparently different: heterogeneous mixed maturity

LayerSkip exits at whole Transformer layers.

Its supervised states look like:

```text
all representation coordinates after layer L
```

The asynchronous KYY schedule creates states of a different form:

```text
some local coordinates / gates have advanced farther
some are still earlier
```

These mixed-maturity states are not among the homogeneous phase states used during
KYY phase supervision.

The initial result was therefore:

> supervision on a handful of homogeneous maturity surfaces generalized to unseen
> heterogeneous maturity mixtures.

That looked like the residual worth carrying.

---

## 5. Then the nonphysical splice control cut that down too

A stronger control asks whether the shared phase-trained readout needs the *causal
local schedule* at all.

Take the homogeneous phase states for one example and build a Frankenstein state:

```text
coordinate 0  <- random phase
coordinate 1  <- random phase
coordinate 2  <- random phase
...
```

This state need not correspond to any legal sequence of local scatter operations.

In a compact three-model scratch check:

```text
random coordinate splice from phases 1..4
    accuracy ~ .78 - .85

random coordinate splice allowing pre-token state too
    accuracy ~ .66 - .72

random convex mixture of whole phase states
    accuracy ~ .92 - 1.00
```

The real asynchronous 10%-deadline accuracy in the earlier three-seed sweep was about
`.68` on average.

So arbitrary nonphysical mixtures can be similarly readable.

`experiments/mixed_state_splice_control.py` commits this attack for reproduction.

---

## 6. Why convex mixtures are almost guaranteed to work

The readout is linear:

```text
f(h) = W h + b
```

Suppose phase supervision makes

```text
W h_1 ~ z
W h_2 ~ z
...
W h_k ~ z
```

for approximately the same target logits `z`.

Then for convex coefficients `alpha_i` summing to one:

```text
W (sum alpha_i h_i) + b
```

inherits approximately the same public prediction.

So high convex-mixture accuracy is not evidence for special recurrent structure.  It
is an algebraic consequence of aligned shared readout semantics.

Coordinate-wise splicing is less trivial, but its strong performance still shows that
the public representation has become broadly robust to noncausal hidden-state mixing.

---

## 7. Current attribution ledger

### Shared/public semantic robustness

Credit primarily to:

```text
shared intermediate supervision
+ learned representation geometry
```

This is closely neighboring LayerSkip / early-exit prior art.

### Availability of heterogeneous partial work at physical deadlines

Credit to:

```text
local dependency graph
+ heterogeneous operation durations
+ asynchronous scheduling
```

This is neighboring asynchronous/event-driven computing and delay-reservoir prior art.

### KYY/local geometry advantage

**Not yet earned.**

The splice control specifically prevents us from saying:

> local causal geometry makes mixed states semantically robust.

It does not appear necessary for that part.

---

## 8. What remains worth testing

The residue is now almost entirely a systems/hardware question:

> **Does a local dependency graph expose useful completed subcomputation earlier, more
> cheaply, or more robustly than synchronized/global alternatives once everybody gets
> equally good shared-output/anytime training?**

That requires controls which receive the same representational help.

Useful next comparisons:

```text
KYY local async + shared-output training
vs
asynchronous/event-driven non-geometric network + same training
vs
synchronized KYY + same training
vs
LayerSkip/early-exit style synchronized network
vs
simple-cycle / delay reservoir
```

Measure actual:

```text
accuracy vs deadline
operations/messages completed by deadline
critical-path latency
energy / communication proxy
straggler robustness
delay-warp OOD
calibration and premature-error curves
```

The remaining KYY hypothesis is therefore no longer about representational magic.

It is the old narrow one in a new setting:

> **Does actual local physical support buy anything when the algebra and training
> controls are already strong?**

That is exactly where KYY was supposed to end up.
