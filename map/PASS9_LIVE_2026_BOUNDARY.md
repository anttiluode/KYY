# Pass 9 — live 2026 boundary: hard group projection, holonomy, and structure-aware compilation

Date: 2026-08-10

This pass exists to stop KYY from escaping prior art by saying:

> "Fine — instead of learning approximate token matrices, explicitly project the recurrent state back onto the group / make the recurrence topological."

That direction is already live in 2026.

These are **preprints / very recent results**. They are boundary markers, not settled benchmarks, and KYY should not rely on their strongest claims without independent reproduction.

---

## 1. Hard projected non-Abelian recurrence

Jeonghoon Lee, *A Held-Out Transition-Pair Falsifier for Long-Horizon Non-Abelian State Tracking* (2026):

https://arxiv.org/abs/2606.07254

The paper introduces a split that holds out ordered generator pairs rather than only longer words. In its controlled `S3 x S3` benchmark, a projected recurrent state model trained on length-8 sequences is reported to maintain perfect final-state accuracy through horizons up to 1,048,576 tokens across five seeds.

The mechanism diagnostics explicitly include:

- homomorphism error;
- state-consistency drift;
- commutator separation;
- hard versus softened projection.

The reported hard projection is strongly associated with stable long-horizon behavior, while softening the projection causes collapse.

### KYY subtraction

The following is **not** an open KYY claim by itself:

```text
explicitly enforce/projection-correct finite non-Abelian group structure
    -> stable long-horizon state tracking.
```

That is now a directly occupied live research direction.

KYY's exact oracle baselines are still useful because they make relation error literally zero, but "relation preservation prevents drift" is not a novelty claim.

---

## 2. Holonomic / gauge-protected reasoning

Ilmo Sung, *Robust Reasoning as a Symmetry-Protected Topological Phase* (2026):

https://arxiv.org/abs/2601.05240

The preprint frames robust logical state evolution in terms of non-Abelian gauge symmetry / holonomy and reports an `S10` variable-binding experiment with large length extrapolation.

The terminology and claims are ambitious and should be independently stress-tested before being treated as established. But for KYY's map the important point is simpler:

> **"Use non-Abelian geometric/topological structure to protect sequential reasoning" is already an explicit 2026 proposal.**

### KYY subtraction

Do not rename Coxeter/group recurrence "topological reasoning" and treat the language as a contribution. If KYY uses topology, it must be tied to a concrete substrate/resource/realization statement.

---

## 3. Structure-aware automaton compilation is also live

Bellante et al., *Compiling Quantum Regular Language States* (2026):

https://arxiv.org/abs/2602.02698

This work accepts regular-language specifications, converts/minimizes a DFA, maps it to an optimal matrix-product-state intermediate representation, and then emits hardware-aware quantum state-preparation circuits including a linear-nearest-neighbour backend with resource guarantees.

This is **state preparation**, not a recurrent transition-family compiler. Still, its pipeline is close enough to constrain KYY's language:

```text
regular/automaton specification
    -> minimize structure
    -> compact IR
    -> hardware-aware local backend
```

is not an untouched compiler idea.

### KYY residual

If KYY pursues a compiler, it must distinguish:

- persistent recurrent runtime state, not one-shot state preparation;
- a family of input-switched transition operators, not one target state;
- exact/observable transition relations under indefinite composition;
- physical/local execution cost per token;
- unrealizable directions / negative-capability reporting.

---

## 4. Sparse realization selection is active too

Du & Li, *Sparse State-Space Realizations of Linear Controllers* (2026):

https://arxiv.org/abs/2603.28754

Given an LTI transfer behavior and a desired sparsity pattern, they solve for an equivalent sparse state-space realization by finding a similarity transformation, using algebraic-geometry tools for the resulting polynomial system.

That directly occupies the broad statement:

```text
choose an equivalent state realization to match hardware sparsity.
```

### KYY residual

KYY's possible distinction is the **joint input-switched family / automaton** case, where one realization must make *all token transitions* cheap while preserving the behavioral quotient under arbitrary composition.

Even that needs deeper search.

---

# 5. What is still not killed

After Passes 5-9, the current candidate is almost entirely a compiler/resource statement:

```text
behavioral transition system
        |
        v
minimal / sufficient recurrent realization
        |
        | choose representation / gauge jointly for all tokens
        v
family {A_x}
        |
        | exact / relation-aware compilation
        v
local substrate words {w_x}
        |
        v
certified long-horizon observable behavior
```

The central cost is not one matrix norm. It is something like

```text
state dimension
+ runtime control bits/ports
+ local primitive count
+ parallel depth
+ wire span / reductions
+ relation defect
+ long-horizon behavioral drift.
```

The compiler is allowed to choose a different but behaviorally equivalent realization when that reduces substrate cost.

---

# 6. Current novelty status

The exact conjunction has **not been established as novel**.

Every ingredient has mature or rapidly developing prior art:

- automata minimization / realization;
- representation theory;
- exact group/state tracking;
- hard algebraic projection;
- Lie/control theory;
- circuit/gate synthesis;
- sparse state-space realizations;
- hardware-aware automata/state compilers.

The surviving KYY question is therefore a **bridge hypothesis**:

> **Can joint behavioral-realization selection plus relation-preserving compilation of an input-switched recurrent operator family produce a measurable locality/communication advantage over modern LRNN transition parameterizations on the same state-tracking tasks?**

Status: **BRIDGE / HIGH PRIOR-ART RISK / TESTABLE.**

That is narrow enough to search and benchmark without inventing a new neural layer.
