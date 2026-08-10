# Pass 24 — harmonic drift runway: larger geometric code margin buys a modulus-independent phase-error budget

Date: 2026-08-10

Pass 23 established a harmonic-frame state code for cyclic counting:

```text
exact group action
norm-preserving phase updates
O(log n) real dimensions
constant symbolic-state separation.
```

The obvious objection is the one emphasized by current error-control theory:

> an affine norm-preserving tracker does not correct implementation drift.

Pass 24 therefore asks a narrower question:

> **Does the larger harmonic-frame state margin at least buy a quantitatively larger finite-horizon tolerance to phase error?**

Yes, in the simplest common phase-defect model.

This is a robustness-budget calculation, not an error-correcting architecture.

Code:

- `cyclic_harmonic_drift_probe.py`
- `tests/test_cyclic_harmonic_drift_probe.py`

---

# 1. Error model

For mode `l`, the intended increment rotation is

```text
theta_l = 2*pi*f_l/n.
```

Assume the implemented block has the same small additive phase defect `eta` per token:

```text
theta_l_bad = theta_l + eta.
```

This deliberately simple model corresponds to a coherent calibration/systematic phase error rather than independent roundoff.

After `t` repeated increments, every mode has accumulated extra phase

```text
t * eta.
```

---

# 2. Exact ideal-versus-defective state distance

The harmonic code gives equal weight `1/sqrt(k)` to every 2D block.

For one unit 2D vector, rotating by an additional phase `phi` changes it by Euclidean distance

```text
2 |sin(phi/2)|.
```

Every mode has the same extra `phi=t*eta`, so after normalization across `k` modes the full-state error is still exactly

```text
E(t) = 2 |sin(t*eta/2)|.
```

Importantly, this expression does **not** grow with `k`.

Adding more harmonic modes can therefore increase symbolic-state separation without increasing the state error caused by this common phase defect.

---

# 3. Guaranteed decoder runway

Let `r` be the nearest-prototype robustness radius of the symbolic state code.

Nearest-state decoding is guaranteed correct while

```text
E(t) < r.
```

On the first monotone lobe this gives roughly

```text
t_safe ~= r / |eta|
```

for small `eta`.

The exact conservative bound used by the probe is derived from

```text
2 sin(t |eta|/2) < r.
```

---

# 4. Single phase versus harmonic code

## one complex phase

For the regular `n`-gon counter,

```text
r_single = sin(pi/n) ~ pi/n.
```

Therefore

```text
t_safe_single = O(1 / (n |eta|)).
```

At fixed implementation phase error, increasing the modulus directly consumes the finite-horizon error budget.

## constant-margin harmonic frame

Pass 23 gives harmonic codes with

```text
r_harmonic >= r_0 > 0
```

using `O(log n)` modes/coordinates.

Then

```text
t_safe_harmonic = O(1 / |eta|)
```

with no modulus in the leading scaling.

This is the precise robustness benefit of spending logarithmically more resonant coordinates.

---

# 5. Concrete phase-error example

Take a common systematic defect

```text
eta = 1e-4 radians per token per mode.
```

For the **single-phase** code, the conservative geometric runway is approximately:

| cyclic task | real dim | guaranteed safe steps |
|---:|---:|---:|
| `C_31` | 2 | ~1,012 |
| `C_101` | 2 | ~311 |
| `C_1009` | 2 | ~31 |

The exact first nearest-prototype errors in the direct rollout are about the same scale (`~1014`, `~312`, `~32`).

For a 32-mode (`d=64`) random harmonic code found by a short frequency search, representative first decoder failures under the **same** common phase defect were roughly:

```text
C_31    ~13,500 tokens
C_101   ~11,400 tokens
C_1009  ~12,000 tokens
```

The precise numbers depend on the selected frequency set and are **not claimed bounds**.

The important shape is that the harmonic failure horizon remains around the same scale while the modulus changes by more than 30x, whereas the single-phase horizon collapses approximately inversely with modulus.

---

# 6. This does not contradict affine error-control theory

Chung, Choi & Kim (2026) show that exact affine tracking does not provide selective restoring contraction along state-separating directions.

The harmonic tracker is still affine and norm-preserving.

It has **no attractor** pulling a perturbed state back onto its symbolic orbit.

Pass 24 only changes the denominator in the current error-control picture:

```text
within-state drift
------------------
between-state margin.
```

A larger margin means more accumulated drift can be tolerated before the decoder becomes ambiguous.

It does not make the drift disappear.

---

# 7. Error-model dependence

The common additive phase defect is chosen because it admits an exact calculation.

Other errors can scale differently:

```text
independent phase noise per mode
relative frequency calibration error
amplitude damping
cross-mode coupling
readout noise
finite-precision matrix multiplication
systematic relation defect A^n != I.
```

A physical implementation comparison must report its own error model.

In particular, if adding modes introduces independent hardware noise or expensive synchronization, the harmonic advantage can shrink.

---

# 8. Relation defect remains important

For an exact cyclic representation,

```text
A^n = I.
```

Under a systematic phase defect,

```text
A_bad^n != I.
```

so a complete cycle no longer returns exactly to the same continuous state.

This is the same finite-group lesson KYY learned in Pass 7:

> small per-step operator error can accumulate indefinitely when the defining group relations are not enforced exactly.

The harmonic frame gives a larger geometric tolerance to that relation defect; it does not remove the defect.

A later compiler may therefore need to jointly optimize:

```text
state-code separation
operator relation defect
precision required to represent phase angles
periodic projection / correction cost.
```

---

# 9. Why this matters for the Geometric Neuron lens

The original resonance idea becomes much more precise here.

A single resonant phase mode is an extremely compressed state coordinate, but its symbolic states crowd together as the counter grows.

A **bank of differently tuned resonant modes** can represent the same symbolic state by a joint phase fingerprint:

```text
mode 1 phase
mode 2 phase
mode 3 phase
...
        |
        v
one robust discrete state codeword.
```

The extra modes are not being used to hold independent memories.

They act like an **error-separating geometric code** for one shared group state.

That is a better interpretation of why multiple resonances might matter computationally than simply saying "more modes = more memory."

---

# 10. Prior-art status

Occupied ingredients:

```text
harmonic/group frames
low-coherence Fourier state codes
complex cyclic recurrent transitions
state separation as a robustness quantity
finite-horizon affine drift.
```

Pass 24 is therefore best viewed as a KYY-specific consequence/diagnostic:

> **Use frame coherence to price the finite-horizon error budget of an exact group-state recurrence.**

A targeted search has not yet located that exact recurrent formulation, but no novelty claim is made.

---

# Current pin after Pass 24

The strongest current mathematical object in KYY is now:

```text
finite group state
      |
      v
harmonic/group-frame code
      |
      +--> small recurrent dimension
      +--> explicit state margin
      +--> exact norm-preserving group action
      |
      v
error-control budget
      |
      v
Sigma-local chain with reset factors only where behavior requires forgetting
```

The next step is to close the finite-abelian-group generalization and then ask whether the full Sigma-local compiler gets a resource advantage on a nontrivial regular automaton.