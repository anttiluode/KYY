# Pass 15 — observable reset lemma: when invertible hidden dynamics cannot truly forget

Date: 2026-08-10

This pass formalizes the distinction introduced by the `permreset3` leakage audit:

```text
"old history still exists in hidden state"
        !=
"old history can ever affect the declared ports again."
```

The result is elementary switched-linear observability mathematics. It is **not claimed as a new theorem**.

Its value is that it gives KYY an exact criterion for the phrase "the model only hid the reset history in garbage."

---

# 1. Switched affine recurrence

Consider a finite token alphabet `Sigma` and a fixed finite-dimensional affine recurrent realization

```text
h_(t+1) = A_x h_t + b_x
y_t     = C h_t
```

for token `x`.

Take two histories that reach hidden states `h` and `h'` before receiving the same future input.

Their difference

```text
d = h - h'
```

evolves linearly because the affine drives cancel:

```text
d_(t+1) = A_x d_t.
```

For a continuation word `w = x_1 ... x_k`, write `A_w` for the corresponding product.

The future logit/readout difference is

```text
Delta y(w) = C A_w d.
```

---

# 2. Future-unobservable subspace

Define

```text
N = intersection over all continuation words w of ker(C A_w).
```

A difference `d` lies in `N` exactly when **no future token sequence can ever make it visible at the linear readout**.

This is the natural switched-system analogue of an unobservable subspace.

It is a linear subspace.

It is also forward invariant under every token transition:

```text
A_x N subseteq N.
```

Proof: if `d in N`, then for every continuation `w`,

```text
C A_w A_x d = 0
```

because `x` followed by `w` is just another possible future word.

---

# 3. Exact behavioral reset criterion

Let `R` be a reset token.

Let `D` be the linear span of all hidden-state differences between histories that the reset is supposed to make behaviorally identical.

Immediately after the reset the difference is

```text
A_R d.
```

The two histories remain indistinguishable at the readout under **every possible future continuation** iff

```text
C A_w A_R d = 0
```

for every `d in D` and every word `w`.

By definition of `N`, this is equivalent to

```text
A_R D subseteq N.
```

So:

> **Exact future-proof linear-readout reset iff the reset transition maps every reset-relevant history difference into the future-unobservable subspace.**

---

# 4. Invertible reset corollary

Suppose `A_R` is invertible.

Because `N` is forward invariant,

```text
A_R N subseteq N.
```

But an invertible linear map preserves dimension, hence

```text
dim(A_R N) = dim(N).
```

Therefore

```text
A_R N = N
```

and consequently

```text
A_R^(-1) N = N.
```

If exact reset requires

```text
A_R D subseteq N,
```

then

```text
D subseteq A_R^(-1) N = N.
```

Thus:

> **A fixed invertible reset transition cannot make a previously future-observable linear difference become permanently future-unobservable.**

If the histories were behaviorally distinguishable before reset (`D` contains directions outside `N`), an exact future-proof reset requires a non-invertible action somewhere in the effective linear realization.

---

# 5. Minimal observable realization

In a behaviorally minimal observable linear realization,

```text
N = {0}.
```

Then exact reset requires

```text
A_R D = {0}.
```

If `D` is nonzero, `A_R` must be singular on `D`.

If reset must erase arbitrary state differences (`D` is the full state space), then the reset's linear part must annihilate the whole state difference space.

This is the linear-realization version of Pass 13's finite transformation-semigroup "pinch".

---

# 6. Why the ancilla loophole is subtler than it first looked

Pass 11 showed a one-shot reversible embedding such as

```text
(q, blank) -> (reset_value, q).
```

The visible coordinate looks reset while the old state is retained in an ancilla.

That does **not** contradict the lemma.

After one application, the old distinction is hidden from the chosen projection, but it is not necessarily in the **future-unobservable** subspace. If later allowed operations can mix the ancilla back into the visible coordinates — for example by reusing the same invertible swap — the distinction leaks back.

To keep arbitrary erased histories permanently invisible with a purely reversible implementation, one needs additional structure such as:

```text
fresh ancillas / expanding history tape
an environment that never couples back
explicit uncomputation
measurement / discard / dissipation
or a genuinely non-invertible effective map.
```

That is the resource cost that a one-shot visible reset hides.

---

# 7. Important limitation: logits versus class labels

The lemma uses the linear readout `y = C h` and exact equality for all future continuations.

KYY's benchmark ultimately scores an `argmax` class label.

Two distinct logits can stay in the same decision region, so exact **classification** reset is weaker than exact **logit** reset.

Therefore an invertible recurrence can potentially keep hidden/logit differences while still producing the correct class indefinitely over the tested language.

That is why `reset_leakage_audit.py` reports several levels separately:

```text
logit L2 divergence
probability total variation
argmax mismatch rate
ordinary task accuracy
```

Do not collapse those into one notion of forgetting.

---

# 8. What this predicts for the existing KYY models

## `householder2`

Each token linear part is a product of Householder reflections and is orthogonal/invertible.

Hidden difference norm is preserved.

An exact linear-readout reset cannot newly bury an observable difference in a permanent common unobservable subspace.

## `geom_scatter`

Each token linear part is a product of local orthogonal scatterers and is likewise orthogonal/invertible.

The same prediction applies, despite the local geometry.

## `complex_diag`

Each token block has nonzero radius and is technically invertible at every finite step, but the radius is less than one, so history differences can **contract exponentially**.

The lemma says they cannot become exactly zero/future-unobservable solely by an invertible step, but they can become arbitrarily small at finite precision and invisible to the classifier.

This is a materially different resource from a norm-preserving wave body.

## `GRU`

The nonlinear gated recurrence can contract/overwrite hidden differences directly.

---

# 9. Prior-art boundary

Do not claim the ingredients as new.

Nearby established areas include:

- observability/unobservable subspaces in switched linear systems;
- switched systems with reset/state-jump maps;
- minimal linear/weighted-automata realizations;
- synchronizing/reset automata and low-rank transition words;
- reversible/quantum finite automata limitations;
- gated orthogonal/unitary RNNs introduced specifically because strict norm preservation makes forgetting difficult.

Relevant examples:

- Tanwani, Shim & Liberzon, *Observer design for switched linear systems with state jumps* (2015), https://doi.org/10.1007/978-3-319-10795-0_7
- Berlinkov & Szykuła, *Algebraic synchronization criterion and computing reset words* (2016), https://doi.org/10.1016/j.ins.2016.07.049
- Jing et al., *Gated Orthogonal Recurrent Units: On Learning to Forget* (2017), https://arxiv.org/abs/1706.02761

The exact KYY statement is best treated as an **elementary bridge lemma**, not a novelty theorem.

---

# 10. Why this matters for the physical picture

The physical story can now be stated without metaphor inflation:

```text
lossless local propagation
    preserves hidden distinctions

behavioral reset
    requires those distinctions to become permanently irrelevant at the ports

minimal observable implementation
    therefore needs a singular/contractive/discarding mechanism
    for genuinely irreversible task transitions
```

This does not mean every wave computer needs dissipation on every step.

It suggests the opposite design question:

> **Use conservative propagation where the task transition is reversible; pay for contraction/reset only where the behavioral quotient actually requires forgetting.**

That is the next KYY compiler question.