# Closure result — state reification baseline for Pass 44

Date: 2026-08-11

This is a nearest-known-baseline check, not a new numbered pass.

Pass 44 showed that a learned full-rank partial merge can make two required-to-merge histories almost output-indistinguishable at the merge while retaining a hidden distinction that a common future rotation later exposes. Replacing the learned merge with the exact singular lowering removes the distinction and restores exact long-horizon behavior.

The question here is whether that repair is specific to KYY's operator surgery.

It is not.

## Prior-art baseline

Lamb et al., *State-Reification Networks* (ICML 2019), explicitly project hidden states observed at test time back toward the learned/familiar hidden-state distribution/manifold to keep recurrent computation on track.

The present baseline is deliberately simpler and more favorable than their learned density model because this toy already declares its four legal state prototypes:

```text
( 1, 0)
( 0, 1)
(-1, 0)
( 0,-1)
```

After a learned update, snap the hidden vector to the nearest legal prototype.

Two schedules were tested:

```text
MERGE ONLY    project only immediately after the partial-merge token
EVERY STEP    project after every recurrent update
```

The learned recurrent parameters and learned output decoder are otherwise left intact.

## Ten-seed result

Same Pass-44 family, ten seeds, training length 16, evaluation through 1024.

Mean sequence accuracy:

```text
method                 L16        L64        L256       L1024
learned soft           1.0000     0.99985    0.99829    0.73405
merge-only reify       1.0000     1.00000    0.99975    0.99984
every-step reify       1.0000     1.00000    1.00000    1.00000
exact operator compile 1.0000     1.00000    1.00000    1.00000
```

At L1024:

```text
learned soft:       0 / 10 exact seeds
merge-only reify:   7 / 10 exact seeds, range 0.998516 .. 1.000000
every-step reify:  10 / 10 exact seeds
exact compile:     10 / 10 exact seeds
```

## Forgetting audit

For the paired histories used in Pass 44:

```text
0 -> merge -> common rotation-only suffix
1 -> merge -> same common rotation-only suffix
```

merge-only and every-step reification both give

```text
hidden difference at merge = 0
max future hidden difference = 0
max future probability TV = 0
max future prediction mismatch = 0
```

So nearest-state projection is a complete repair of the Pass-44 leakage witness in this controlled toy.

The small non-perfect L1024 accuracy of `merge-only` reification is therefore **not** zombie-memory leakage from the merge. It comes from leaving the learned rotation operator approximate between merges.

Every-step reification repairs both kinds of accumulated state drift.

## What this kills

KYY should not claim:

> exact singular operator surgery is uniquely required to make the partial-merge machine truly forget.

It is not.

A known state-reification/quantization-style runtime projection also makes the required histories literally coincide and gives perfect long-horizon behavior when applied every step.

Likewise, because the benchmark's symbolic automaton is declared, a pure software deployment could simply replace the continuous machine with the exact finite automaton.

That is an even stronger baseline.

## What distinction remains

The two repairs have different deployment contracts.

### state reification

```text
learned approximate operator
        ↓ every step
nearest-state / manifold projection
        ↓
continue learned dynamics
```

This introduces a nonlinear projection/lookup into runtime.

### KYY exact lowering

```text
learned approximate operator
        ↓ once, post training
replace with exact legal operator
        ↓
run continuous/geometric dynamics directly
```

No runtime nearest-prototype decision is needed for this toy.

This is a real architectural difference, but it is **not yet a demonstrated advantage**.

For ordinary software finite-state tasks, exact DFA extraction/replacement may be simpler than either approach.

The operator-compiler story becomes more meaningful only if there is a reason the deployed system must retain a continuous/geometric representation—for example:

- a physical/analog substrate;
- continuous outputs or dynamics between symbolic events;
- a larger mixed continuous/discrete state in which only some token actions have exact symbolic contracts;
- locality/wiring constraints that make a direct DFA/table implementation the wrong cost model.

Those are hypotheses to test, not current results.

## Consequence for novelty

This baseline substantially weakens a standalone software-AI novelty claim around Pass 44.

The empirical audit remains a useful clean demonstration:

> immediate output agreement at a merge does not establish future equivalence; a common suffix can expose residual hidden state.

But the broad mechanism and repair sit inside established automata/observability/reification territory.

Do not promote Pass 44 to `main` as a new forgetting theory.

The more defensible residual is narrower:

> post-training replacement of approximate learned operators by exact algebraic/semigroup lowerings, together with port transport/canonicalization and exact certificates, may be useful when the deployment representation itself must remain continuous/geometric or satisfy substrate constraints.

That is where the next comparison should be made if KYY continues.

## Files

- `map/partial_merge_reification_baseline.py`
- `tests/test_partial_merge_reification_baseline.py`
- `.github/workflows/partial-merge-reification-baseline.yml`
- workflow artifact: `partial_merge_reification_baseline.json`
