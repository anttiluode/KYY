# Pass 10 — simultaneous realization: the family gauge is old too

Date: 2026-08-10

Pass 8 noticed that equivalent recurrent realizations can trade state dimension against primitive locality. That suggested a tempting compiler objective:

> choose **one** basis / gauge that makes the whole input-switched token family cheap on the hardware.

This pass searches that exact structural idea before we turn it into code.

The verdict: **the common-basis problem is old.** The remaining residual, if any, is in the objective and in the behavioral/algebraic constraints.

---

## 1. Simultaneous matrix reduction is mature linear algebra

There is a long literature on applying one similarity transform to an entire set of matrices:

- Shapiro, *Simultaneous block triangularization and block diagonalization of sets of matrices* (1979).
- Laffey, *Simultaneous reduction of sets of matrices under similarity* (1986).
- Kaczorek, *Similarity transformation of matrices to one common canonical form...* (2010).

Representative sources:

- https://doi.org/10.1016/0024-3795(79)90012-0
- https://doi.org/10.1016/0024-3795(86)90311-3
- https://eudml.org/doc/208003

So

```text
find one basis that simplifies all A_x
```

is not a KYY idea.

---

## 2. Switched systems already need a common realization

A switched linear system has finitely many modes sharing one persistent state. That is structurally close to an input-switched recurrent model.

Monshizadeh, Trentelman & Camlibel, *A Simultaneous Balanced Truncation Approach to Model Reduction of Switched Linear Systems* (IEEE TAC 2012), seek conditions for a **single state-space transformation** that simultaneously balances all modes and give a cost-based fallback when exact simultaneous balancing is unavailable.

https://doi.org/10.1109/TAC.2012.2202031

Bencherki, Türkay & Akçay, *Realization of multi-input/multi-output switched linear systems from Markov parameters* (2023), recover switched submodels only up to separate similarities and then explicitly bring them into a **common basis** before prediction.

https://arxiv.org/abs/2106.10942

Therefore

> "input-switched modes have basis freedom, so align/choose one common gauge"

is firmly occupied control theory.

---

## 3. Sparse realization is also a direct objective

Du & Li, *Sparse State-Space Realizations of Linear Controllers* (2026), ask whether an LTI behavior can be represented with a desired state-space sparsity pattern and solve for the necessary similarity transform via polynomial equations / Gröbner bases.

https://arxiv.org/abs/2603.28754

Older finite-word-length design work optimizes realizations over similarity transformations while explicitly considering sparse Schur forms.

So

```text
behavior-equivalent realization
    -> choose basis for sparsity / implementation quality
```

is already an engineering discipline.

---

## 4. What the S5 example adds to our question

KYY's exact S5 oracle is not an LTI system. It is a **family** of 120 token transitions with exact multiplication relations.

The compiler is allowed to change realization, but a candidate common basis has to be judged jointly on:

```text
all token matrices
all products / defining relations
behavioral readout
substrate-local synthesis cost.
```

For example:

```text
5D natural S5 realization
    -> perfect 2-port local swaps
    -> one redundant invariant channel

4D A4 simple-root realization
    -> removes redundant channel
    -> radius-1 local stencil
    -> preserves Cartan metric
```

Neither basis is universally "better." The answer depends on the physical primitive/cost model.

That is the useful toy demonstration, not a novelty claim.

---

## 5. The residual is an optimization objective, not a new equivalence relation

The current search has not located one standard formulation whose objective is exactly:

> **Among behaviorally equivalent realizations of an input-switched transition monoid, choose one common realization and one exact/approximate local synthesis for every token so as to minimize a declared hardware communication/depth cost while preserving observable monoid relations under arbitrary composition.**

But nearly every phrase in that sentence has prior art:

- behaviorally equivalent/minimal realizations;
- simultaneous similarity/common basis;
- switched-system balancing/reduction;
- sparse realization;
- automata minimization;
- representation theory;
- gate/circuit/routing synthesis;
- relation-preserving group representations;
- hardware-aware compilers.

Status: **BRIDGE / VERY HIGH PRIOR-ART RISK.**

A search miss is not a novelty claim.

---

## 6. What would actually distinguish a KYY compiler experiment

A useful first experiment would have to combine the pieces in one benchmark and expose a trade-off not visible when they are optimized separately.

For a standard transition family such as full `S5`, compare realization choices under the *same* substrate:

```text
representation / state dimension
        x
common-basis choice
        x
local primitive library
        x
parallel routing/synthesis depth
        x
relation exactness / drift.
```

Then compare that exact compiler floor against learned LRNN transitions.

The output should be a Pareto front, not one scalar score:

```text
(state dimension,
 local memory,
 wire span,
 reductions,
 parallel depth,
 arithmetic,
 relation defect,
 long-horizon behavioral error).
```

If standard simultaneous-realization + routing tools already produce the entire front, KYY has no compiler contribution. If the **joint behavioral-relation constraint** changes the optimum in a useful way, that is the residual worth studying.

---

# Current pin after Pass 10

Not:

> choose a good basis.

Not:

> sparsify a state-space model.

Not:

> simultaneously reduce token matrices.

The remaining question is:

> **Does optimizing a common recurrent realization and its local token syntheses *jointly for long-horizon behavioral relations* produce a useful resource frontier that ordinary matrix-wise or LTI realization objectives miss?**

That is the narrowest statement the map currently supports.
