# Deadline-control scratch ledger — 2026-08-12

These are **scratch reconstruction results**, not canonical benchmark numbers. They were
produced while checking the logic of the committed controls and should be reproduced by
running the repo-native scripts before being promoted into the README or a claim.

The purpose of this file is to prevent conversational numbers from turning into
untraceable folklore.

---

## 1. Identity-token correction survives

The original `perm3` deadline gate contained token 0, the identity generator. Before the
final token transition begins, retaining the previous state is therefore exactly correct
on roughly one third of examples. The old synchronized plateau near `.33` was partly a
task artifact.

Scratch reconstruction:

- 3 phase-legible ring models (model seeds 0,1,2)
- 3 unseen delay fabrics each (delay seeds 0,1,2)
- 6000 evaluation episodes / row before conditioning
- final token conditioned on `token != 0`
- 21 deadline points from 0 to synchronized full latency

Mean over 9 model/delay rows:

```text
non-identity perm3

                         async      sync
AUC                      0.781      0.616
accuracy @ 10% latency   0.237      ~0.0002
accuracy @ 20% latency   0.731      ~0.0002
accuracy @ 50% latency   0.884      0.816
```

The 10% async value varied strongly by delay fabric (from near zero to roughly `.70`).
That variability is expected for a critical-path phenomenon: some physical schedules
complete useful operations earlier than others.

Interpretation:

> Removing the identity mass kills the misleading `.33` synchronized baseline but does
> **not** kill the existence of an async deadline advantage in this reconstruction.

This remains a scheduling result, not a ring-geometry result.

Repo control: `experiments/deadline_nonidentity_control.py`.

---

## 2. Output-null stabilization is receiver-specific

A second scratch control compared hidden-state motion against:

```text
W   = the trained six-output public readout
W'  = a random, never-trained rank-matched receiver
```

For three ordinary final-only models, mean null fraction versus the random receiver was
roughly:

```text
0.832  0.815  0.798  0.824
```

For three phase-supervised models it was roughly:

```text
0.807  0.829  0.793  0.806
```

Both sit near the rank-6-in-32 isotropic null baseline:

```text
1 - 6/32 = 0.8125
```

By contrast, the trained receiver W showed the large previously documented change.

Interpretation:

> phase supervision does not appear to make hidden motion generically null. It aligns
> continuing computation specifically relative to the trained receiver.

That is exactly what should be expected from the objective and prevents over-reading the
output-null result as an intrinsic property of the state space.

Repo control: `experiments/output_null_transfer_control.py`.

---

## 3. Held-out homogeneous phase fails

A more direct circularity attack omitted phase index 2 (the third homogeneous
checkerboard phase) from shared-head supervision.

Scratch reconstruction with three model seeds, compact 140-step training:

```text
held-out phase accuracy
seed 0   0.0170
seed 1   0.0002
seed 2   0.0008
```

The supervised neighboring phases remained useful and the final phase was roughly
`.91-.99` in the same compact runs.

Interpretation:

> semantic alignment did not spontaneously transfer to an omitted homogeneous maturity
> surface. Direct intermediate supervision is doing the semantic work.

This substantially weakens any attempt to treat the KYY phase-supervision result as
biological evidence. It remains an engineering mechanism and an analysis analogy.

Repo control: `experiments/output_null_transfer_control.py`.

---

## 4. Non-geometric random matchings also benefit from async scheduling

The final cheap scratch control replaced the fixed ring with four arbitrary perfect
matchings over the same 32 state channels.

Matched high-level budget:

```text
4 homogeneous phases / token
16 disjoint 2-port operations / phase
64 operations / token
192 token/phase/edge angle parameters
one h0
one shared six-output readout
same phase-legibility objective
same IID operation-duration distribution
same asynchronous dependency rule
```

This software control does **not** charge nonlocal pairings for wire length, placement,
energy or analog noise.

A first random matching (topology seed 123) showed a positive async deadline benefit.
Two additional rewires (topology seeds 456 and 789), both converged to final accuracy
~1.0 on the checked model seed, produced mean async AUCs across three delay fabrics of
approximately:

```text
random matching seed 456   async AUC 0.863   sync AUC 0.787
random matching seed 789   async AUC 0.821   sync AUC 0.790
```

For comparison, the checked ring model seed had mean async AUC around `.795` in the
non-identity reconstruction.

This is not a matched canonical result yet: training budgets/topology sampling should be
standardized by the committed script before quoting the ordering.

But the qualitative kill is already clear enough to motivate the gate:

> **asynchronous anytime availability does not require the fixed ring geometry.**

A scrambled sparse dependency graph can also obtain it and, in some scratch rewires,
matched or exceeded the ring.

Repo gate: `experiments/async_random_matching_gate.py`.

---

## 5. Current attribution ledger

What is currently supported by the scratch work:

```text
shared phase supervision
    -> receiver-specific semantic alignment

sparse dependency graph + async scheduling
    -> heterogeneous partial work can become physically available before global barriers

fixed ring/local-neighbor geometry
    -> no software anytime advantage earned yet
```

What is **not** tested by this software gate:

```text
wire length
placement/routing cost
analog locality
energy
noise/quantization robustness
physical propagation delay tied to distance
```

Those are the only places a genuinely physical locality advantage should now be sought.

---

## 6. Standing rule

The repeated lesson across GeometricNeuron_V20, WidePresent and KYY is now recorded in
`docs/PHYSICS_DOES_NOT_SUPPLY_SEMANTICS.md`:

> **A substrate supplies trajectories and constraints. A receiver/training relation
> supplies task semantics.**

The deadline branch should be considered finished as an architecture-semantics claim
unless the repo-native random-matching gate reverses the scratch result.
