# Closure prior-art audit — 2026-08-11

This note deliberately narrows the claims around Passes 39–45 before anything is promoted to `main`.

## Decision

Do **not** currently advertise Pass 44 as a new theory of forgetting, and do **not** advertise the broad KYY compiler slogan (`train soft -> deploy exact`) as novel.

Both collide with mature and recent work.

The branch remains a research notebook until the residual below survives stronger comparison.

## Pass 44: what is occupied

### Reversible dynamics cannot truly forget

MacKay, Vicol, Ba & Grosse, *Reversible Recurrent Neural Networks* (NeurIPS 2018), explicitly show that perfectly reversible RNNs are fundamentally limited because they cannot forget information from hidden state.

So KYY must not claim the general fact that invertible / reversible / norm-preserving state evolution preserves hidden distinctions.

### Immediate output agreement is not proof that information is gone

Recent machine-unlearning work directly studies the gap between apparent output-level forgetting and residual internal information.

Especially close is Gao et al., *An Illusion of Unlearning? Assessing Machine Unlearning Through Internal Representations* (2026): many unlearning methods appear successful because of feature-classifier misalignment while forgotten information remains discriminative in hidden representations.

Goel, Ritter & Gurevych, *Auditing Language Model Unlearning via Information Decomposition* (EACL 2026), likewise reports forgotten information remaining linearly decodable internally despite apparent unlearning.

Therefore the broad statement

> "the output says forgotten while the representation still remembers"

is occupied.

### A future suffix that distinguishes two histories is classical automata theory

The correct behavioral notion is future equivalence: two histories are equivalent only if no common future word can distinguish their behavior. This is classical Myhill–Nerode / automata minimization language.

Synchronizing-automata theory likewise treats reset words as maps that place different starting states into one state, after which every common future remains identical.

So Pass 44's exposing suffix is a concrete continuous-state witness of a classical distinction, not a new definition.

### Continuous RNN -> finite automaton is old

Relevant boundaries include:

- Watrous & Kuhn, *Induction of Finite-State Automata Using Second-Order Recurrent Networks* (NeurIPS 1991).
- Das & Mozer, *A Unified Gradient-Descent/Clustering Architecture for Finite State Machine Induction* (DOLCE, NeurIPS 1993): continuous recurrent state is treated as noisy finite state and quantized toward discrete states.
- Omlin & Giles, *Extraction of rules from discrete-time recurrent neural networks* (Neural Networks, 1996).
- Weiss, Goldberg & Yahav, *Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples* (ICML 2018): exact-learning/abstraction extraction of a DFA from a trained RNN.
- Wang & Niepert, *State-Regularized Recurrent Neural Networks* (ICML 2019).
- Lamb et al., *State-Reification Networks* (ICML 2019): project test-time hidden states toward a learned hidden-state manifold.
- Xie et al., *RNNRepair* (ICML 2021): post-hoc repair of RNN errors by model-based analysis.
- Merrill & Tsilivis, *Extracting Finite Automata from RNNs Using State Merging* (2022).
- Dankowiakowski & Ronca, *Metric Automata Theory: A Unifying Theory of RNNs* (NeurIPS 2025): imports classical automata structure and robustness into continuous RNN/SSM analysis.

### `train soft -> exact symbolic deploy` is also occupied

Umili & Capobianco, *DeepDFA: Automata Learning through Neural Probabilistic Relaxations* (2024), is a particularly close philosophical boundary. It trains a differentiable probabilistic automaton and drives it toward a discrete DFA, explicitly noting that naive end-of-training discretization can change behavior.

Thus KYY cannot distinguish itself merely by saying:

> learn in a soft continuous space, then deploy an exact algebraic machine.

## Important correction to the "pinch" story

Hidden-state equality is **sufficient** for task forgetting, but it is not generally necessary.

The exact behavioral condition is future observational equivalence.

A residual hidden difference can be harmless if it lies permanently in a future-unobservable invariant subspace for every allowed suffix.

Pass 15 already gave this language:

```text
N = intersection_w ker(C A_w).
```

Pass 44 is useful because the learned merge residual is **not** in such a subspace. A common suffix made only of rotations preserves the residual norm and rotates it back into decoder visibility.

Therefore the precise audit result is:

> immediate output agreement at a declared merge is insufficient; test future equivalence, or certify that the residual lies in the permanent future-unobservable subspace. In the Pass-44 learned machines it did not, and a common suffix exposed it.

The exact singular pinch removes the distinction entirely, which is a strong sufficient repair, not the only theoretically valid repair.

## What KYY may still have as a residual

No direct owner was found in this audit for the exact composition:

```text
already-trained continuous recurrent realization
        ↓
audit declared symbolic relations / kernel partitions
        ↓
post-training projection or replacement of token operators
onto exact legal group/semigroup actions
        ↓
algebra-aware transport or canonicalization of the learned output port
        ↓
relation + kernel + observable-behavior certificate
```

In Pass 44 specifically:

```text
learned full-rank approximate partial merge
        ↓
identify violation of the declared symbolic kernel
        ↓
replace it with an exact singular lowering derived from state code + transition
        ↓
preserve/canonicalize output port
        ↓
certify that the merged histories cannot be separated by any deterministic common future
```

This is currently a **search residual**, not a novelty claim.

It is surrounded by:

- automata extraction,
- differentiable automata induction,
- state quantization/reification,
- neural-network repair,
- Koopman/finite-state realization,
- post-training projection/quantization,
- synchronizing automata,
- modern representation-level unlearning audits.

A paper would need to show that the composition provides a measurable capability those nearest baselines do not.

## What would earn promotion to `main`

At least one of the following should happen before a novelty-facing main-page claim:

1. **General compiler theorem or algorithm** beyond the hand-tractable cyclic/dihedral/C4 cases, with a clear resource/correctness guarantee not inherited directly from known realization theory.
2. **Nontrivial learned-model result** where exact operator/kernel legalization solves a failure that state quantization, DFA extraction, state reification, constrained training, or ordinary post-training projection does not solve at comparable cost.
3. **New audit guarantee** for continuous recurrent systems that is stronger or cheaper than classical future-equivalence / observability checks in a clearly specified regime.
4. **Physical/backend result** where the algebraic compiler chooses exact implementable operators under a substrate constraint and the compiler tradeoff itself is the contribution.

Until then the branch should retain the evidence and negative results without implying that the ingredients or the broad story are new.

## Immediate closure work

Before deciding whether the cyclic port/certificate story is ready even as a bounded result:

- test composite moduli (`C100`, `C105`) rather than only prime `C101`;
- include explicit bad-gcd controls so the algebraic certificate is observed to reject a non-faithful code;
- report positive-kernel port margins alongside parameter count and certificate cost;
- test the 9-parameter port under systematic phase error;
- keep exactness and robustness as separate columns.
