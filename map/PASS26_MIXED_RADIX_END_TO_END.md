# Pass 26 — first end-to-end comparison: local factorization does not get a free geometric win

Date: 2026-08-10

Passes 19–25 built several pieces separately:

```text
Sigma-local predecessor-only decomposition
harmonic/group-frame state code
explicit reset
state packing / precision accounting.
```

Pass 26 combines them in the smallest structured machine where locality actually does something:

> a two-digit mixed-radix counter with local carry and reset.

This is still an exact oracle, not a trained architecture.

Code:

- `mixed_radix_sigma_oracle.py`
- `tests/test_mixed_radix_sigma_oracle.py`

---

# 1. Behavior

Choose radices `p` and `q`.

State is

```text
(low, high) in C_p x C_q.
```

External alphabet:

```text
I      identity
INC    increment
RESET  reset both digits to zero.
```

The synchronous update is

```text
low' = low + 1 mod p                         on INC

high' = high + 1 mod q                       on INC and OLD low == p-1
        high                                 on INC otherwise

(low', high') = (0,0)                        on RESET.
```

Thus the high component needs only:

```text
external token
    +
old state of immediate predecessor low.
```

This is exactly the local dependency form KYY wants.

Repeated `INC` visits

```text
0,1,2,...,pq-1
```

in order and returns to zero after `pq` steps.

So the two local factors realize the same cyclic behavior as one monolithic `C_(pq)` state plus reset.

---

# 2. Four exact realizations compared

## A. monolithic single phase

One 2D phase orbit for `C_(pq)`:

```text
real dimension = 2
margin = sin(pi/(pq)).
```

Tiny state, poor margin at large `pq`.

## B. monolithic harmonic frame

Use the full harmonic-mode budget in one `C_(pq)` code.

Exact block rotations, constant-ish margin with enough modes.

No inter-cell communication.

## C. Sigma-factor harmonic

Use one harmonic code for `C_p` and one for `C_q`.

The high cell receives the low predecessor state to decide carry.

If the carry predicate

```text
old_low == p-1
```

is decoded at the boundary, only **one carry bit** needs to cross the cell boundary.

If not, the full low harmonic vector crosses the boundary.

This distinction is reported explicitly.

## D. Sigma-factor one-hot

Robust local baseline: one-hot code for each digit.

Large state dimension but simple large-margin states.

---

# 3. Fair-normalization trap

The factorized harmonic state contains two unit-norm cell states.

Therefore its concatenated state norm is

```text
sqrt(2).
```

A monolithic harmonic code has norm 1.

Comparing their raw Euclidean state margins would therefore silently give the factorized model more dynamic range/energy.

Pass 26 reports both:

```text
native per-cell margin
```

and

```text
margin after scaling the complete factorized state to the same total norm as the monolith.
```

For two factors, equal-total-norm scaling divides distances/margins by `sqrt(2)`.

This correction changes the conclusion of the first experiment.

---

# 4. Concrete p=31, q=29 comparison

There are

```text
31 * 29 = 899
```

behavioral counter states.

Give the harmonic alternatives the **same total mode budget**:

```text
monolithic:
    16 complex modes = 32 real coordinates

factorized:
    8 modes in C_31 + 8 modes in C_29
    = 32 real coordinates total.
```

A short deterministic frequency search gives representative margins near:

```text
monolithic C_899 harmonic:
    native/equal-norm radius ~0.55

C_31 factor:
    local radius ~0.65

C_29 factor:
    local radius ~0.65

factorized joint state at equal total norm:
    worst radius ~0.65/sqrt(2) ~0.46.
```

So in this example:

> **the monolithic harmonic code is geometrically better packed at the same total real dimension and total state norm.**

The local factorization does not win merely because it is local.

This is an intentionally useful negative result.

---

# 5. What the factorization actually buys

The factorized implementation gives a different resource profile, not a universally better one.

It buys:

```text
smaller algebraic factors
local carry structure
only one inter-cell dependency edge
potentially one-bit carry communication
separate reset/control locality.
```

It may cost:

```text
more total state norm if cells are not rescaled
worse global packing at equal total norm
boundary decoding/control logic
more recurrent components
carry latency in deeper/ripple generalizations.
```

That is exactly the sort of Pareto trade KYY should report.

---

# 6. Controller-table trap

If represented as a generic Sigma component table, the high digit has one context for every

```text
(external token, old low state).
```

An explicit one-hot transition table therefore scales with `p`.

But the actual mixed-radix controller has a constant-size rule:

```text
if RESET: zero
elif INC and low==p-1: increment
else: identity.
```

So explicit transition-table entries are **not** the right controller complexity for structured factors.

Pass 26 reports the table count but also records the compact rule and the sufficient one-bit carry message.

This is another warning against measuring only matrix/table size.

---

# 7. Why this is a better benchmark than the Sigma length-front witness

The `length >= h` witness is designed to expose Sigma-chain description succinctness, but a monolithic scalar counter can solve the behavior compactly if precision is allowed to shrink.

The mixed-radix counter forces a more interesting comparison:

```text
monolithic group representation
        vs
explicitly factored local group representation
```

under the same exact behavior.

The first result says the global harmonic code can be better geometrically packed.

That is important: locality needs to earn its cost on communication/control/hardware axes, not on a benchmark chosen to make factorization look good.

---

# 8. Relation to established mathematics/hardware

None of the ingredients are claimed new:

```text
mixed-radix / ripple counters
finite abelian group representations
harmonic frames
FSM decomposition/state assignment
local carry logic
Sigma-chain products.
```

The KYY contribution at this stage is an **accounting framework plus exact cross-representation compiler tests**.

---

# 9. What the first end-to-end result tells us

A useful design rule emerges:

> **Decomposition and representation are independent compiler decisions.**

A symbolic factorization can reduce controller/communication structure while making the continuous state code less packing-efficient.

Conversely, a monolithic representation can have excellent global state geometry while demanding a less local controller for a harder automaton.

So the compiler objective must jointly choose:

```text
behavioral decomposition
continuous state code
state scaling/dynamic range
controller representation
communication topology
reset placement
error-control mechanism.
```

Optimizing one layer first can give the wrong answer.

---

# Current pin after Pass 26

The next benchmark should **not** be another pure counter, because the monolithic cyclic representation is unusually simple.

We now need a regular automaton where:

1. the transition monoid genuinely has more than one nontrivial factor;
2. the Sigma/local decomposition reduces controller communication or description in a way a monolithic harmonic group orbit cannot trivially absorb;
3. both reversible group behavior and irreversible reset behavior are present;
4. all representations are compared at fixed total state norm and explicit state-separation margin.

That is the first place a local algebraically typed KYY compiler could earn a real resource win.