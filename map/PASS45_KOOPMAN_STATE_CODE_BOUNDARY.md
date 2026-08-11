# Pass 45 — Koopman state-code boundary: the invariant-observable idea is occupied

Date: 2026-08-10

Pass 43 wrote exact task lowering as

```text
A_x Z = Z T_x,
```

where:

- columns of `Z` are the geometric code vectors for finite behavioral states;
- `T_x` is the exact finite-state transition for input token `x`;
- `A_x` is the desired low-dimensional linear lowering.

Looking at the **rows** of `Z` immediately gives another interpretation:

> they are observable functions on the finite state set, and their span must be invariant under the token-induced transition operators.

That is Koopman language.

The collision is direct enough that it must be subtracted before state-code search is presented as a new direction.

---

## 1. Prior art is explicit

### Boolean-network Koopman representations

Qi, Valcher & Shi, *Koopman Representation for Boolean Networks* (IFAC World Congress / IFAC-PapersOnLine, 2023), explicitly constructs Koopman operators for logical dynamics and states that a finite-dimensional linear representation follows from a Koopman-invariant family of logical functions.

Kong, Qi, Valcher & Shi subsequently develop a fuller *Koopman Theory for Boolean Networks* and controlled Boolean networks, including feedback shaping in the Koopman representation (IEEE Transactions on Automatic Control; online 2025 / volume publication 2026).

### finite-state systems

Anantharaman & Sule, *Koopman operator approach for computing structure of solutions and Observability of non-linear finite state system* (2020), construct a Koopman linear system for finite-state dynamics and reduce it while retaining structural/observability information.

### automata/computation directly

Caravelli & Delvenne, *Analog and symbolic computation through the Koopman framework* (Journal of Physics: Complexity, published May 28, 2026), explicitly treats finite automata and symbolic computation through Koopman operators.  The paper connects finite transition structure, invariant observables, absorbing states, cycles, reachability/halting, and spectral constraints.

So the broad idea

```text
finite automaton
   -> observables
   -> finite Koopman-invariant subspace
   -> linear operator representation
```

is occupied.

KYY should not claim it.

---

## 2. Why Pass 43 looked Koopman-like

Let a row `f` of `Z` be viewed as a scalar function on the finite behavioral states:

```text
f : Q -> R.
```

For token `x`, composition with the symbolic transition gives

```text
(K_x f)(q) = f(delta(q,x)).
```

The collection of code-coordinate functions closes linearly iff their span is invariant under every relevant `K_x`.

In matrix form, that closure is exactly the Pass-43 lowering equation, up to row/column convention:

```text
A_x Z = Z T_x.
```

So Pass 43's dependency-preservation test can be read as a concrete finite-dimensional invariant-observable closure test for a *chosen* code.

This is an interpretation/translation, not a new Koopman theorem.

---

## 3. The previous harmonic results fit this language automatically

For a cyclic translation machine, group characters are standard Koopman eigenfunctions.

That explains why the harmonic code is so natural:

```text
character observable
   -> token translation
   -> phase multiplication / planar rotation.
```

The multiple-frequency C_n code from Passes 23 onward is therefore a small separating collection of exact eigen-observables for the finite translation action.

Again, this is classical harmonic/Koopman structure.

---

## 4. Reset explains the constant observable

The centered C4 square in Pass 43 is closed linearly under the cycle and the partial merge, but not under a total reset.

The affine augmentation

```text
[z
 1]
```

adds the constant observable.

That is exactly the kind of closure enlargement Koopman language suggests: the chosen observable subspace was not invariant under the reset transition until the constant direction was included.

So even the linear-versus-affine split has a clean invariant-observable interpretation.

---

## 5. What remains after the collision

The following are **not** residual claims:

```text
finite machines have Koopman representations
invariant observables give finite linear representations
characters diagonalize cyclic translations
state-code design can be seen as observable selection
reduced finite-state Koopman systems can preserve structural information.
```

All are occupied in one form or another.

The part KYY has actually tested is more specific:

```text
TRAIN an approximate continuous recurrent realization
        ↓
observe that short-horizon behavior can be nearly/perfectly correct
while exact task relations or required transition kernels are wrong
        ↓
identify a declared finite behavioral machine / operator family
        ↓
post-training LEGALIZE or SYNTHESIZE exact operators
        ↓
repair / canonicalize the observable output port
        ↓
certify exact finite behavior and kernel semantics.
```

That is a **compiler procedure around a learned approximate realization**, not a proposal to use Koopman theory as the representation in the first place.

The current empirical evidence includes:

- cyclic character snapping repairing long-horizon drift;
- non-Abelian dihedral generator legalization;
- port transport versus port canonicalization;
- exact reset surgery replacing residual soft forgetting;
- exact partial-kernel surgery replacing a full-rank learned map whose hidden distinction later leaked back through a lossless continuation.

The learned machine and the exact Koopman/automaton lowering are playing different roles:

```text
learned dynamics = optimization scaffold
exact operator representation = deployment contract.
```

---

## 6. Pass 44 makes the residual particularly visible

In the partial-merge experiment, the trained model is perfect at length 16 and makes the two required-to-merge histories almost indistinguishable at the current output port.

But the hidden vectors are not equal.

The subsequent rotation preserves the forbidden hidden distinction and later reveals it again.

So an approximate observable match is not the same thing as having compiled the correct invariant/kernel structure.

The exact singular lowering fixes the transition itself, not merely the current prediction.

That distinction is where KYY currently has empirical content beyond simply writing the finite machine as a Koopman operator.

---

## 7. Revised next question

The phrase

> "search for a Koopman-invariant state code"

is too broad and too occupied.

A sharper next compiler question is:

> Given a trained approximate hidden representation and transitions, can we infer/project a **nearby exact separating invariant observable subspace** together with exact token operators and a compatible port, while explicitly pricing how much of the learned representation had to be discarded?

That adds the piece KYY has so far held fixed by hand: the state code itself.

The cost vector would need to include at least:

```text
code dimension
state separation / margin
operator lowering class (linear / affine / singular / nonlinear)
operator locality / physical cost
projection distance from learned representation
port preservation cost
finite-precision drift
required kernel exactness.
```

This is still adjacent to extensive realization/model-reduction/Koopman-observable-selection literature, so it must be mapped carefully before implementation.

---

## 8. Why this is a useful collision rather than a dead end

The collision tells us that the old KYY/Geometric-Neuron/Koopman threads were not unrelated.

But it also prevents a false novelty claim.

The right interpretation is now:

```text
Koopman / automata theory
    owns the exact operator language

state assignment / realization theory
    owns much of compact-code construction

semigroup / group theory
    owns transition algebra and kernels

KYY experiments
    ask whether a learned approximate dynamical machine can be
    compiled post hoc into those exact structures with a small,
    robust, behaviorally equivalent interface.
```

That is the residual worth testing.
