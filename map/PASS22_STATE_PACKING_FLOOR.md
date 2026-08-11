# Pass 22 — bounded-state packing floor: dimension, dynamic range, and precision cannot all be cheap

Date: 2026-08-10

Passes 18–21 exposed a recurring representation trick:

```text
many symbolic states
    ->
small continuous recurrent state
```

Examples:

- a cyclic `C_n` state machine fits on one complex phase / two real coordinates;
- a length threshold can be stored in one analog scalar rather than `h` binary latches.

Those representations are exact in ideal real arithmetic.

But a bounded continuous state cannot pack arbitrarily many **robustly distinguishable** states into fixed dimension for free.

This pass records the elementary geometric floor.

Code:

- `state_packing_floor.py`
- `tests/test_state_packing_floor.py`

Nothing in this pass is novel sphere-packing mathematics.

---

# 1. Elementary volume bound

Suppose `N` symbolic states are represented by points

```text
c_1, ..., c_N in R^d
```

inside a Euclidean ball of radius `R`.

Assume the representation must tolerate perturbations of norm `< epsilon` under nearest-state decoding. Then the open `epsilon`-balls centered on the code points must be disjoint.

Every such small ball lies inside the concentric radius-`R+epsilon` ball.

Comparing Euclidean volumes gives the necessary condition

```text
N * epsilon^d <= (R + epsilon)^d.
```

Therefore

```text
N <= (1 + R/epsilon)^d
```

and equivalently

```text
epsilon <= R / (N^(1/d) - 1).
```

For a target robustness radius `epsilon`, a necessary dimension is

```text
d >= log N / log(1 + R/epsilon).
```

This is only a coarse packing bound. It is not generally achievable and ignores transition structure.

But it prevents a recurrent architecture from claiming that arbitrary continuous-state compression has made memory free.

---

# 2. A useful bookkeeping quantity

Define

```text
B_pack = d log2(1 + R/epsilon).
```

The volume bound says

```text
log2 N <= B_pack.
```

`B_pack` is a geometric resolution budget.

It is **not**:

- literal ADC bits;
- Shannon channel capacity;
- thermodynamic information;
- Landauer cost.

Those require additional noise/distribution/physical assumptions.

KYY should use it only as a representation-independent sanity floor.

---

# 3. Three representations of N states

## A. one normalized scalar

Put `N` equally spaced states in `[-R,R]`.

The nearest-state noise radius is

```text
epsilon = R/(N-1).
```

In one dimension this saturates the elementary volume bound exactly.

So constant dynamic range + one coordinate forces resolution to shrink like `1/N`.

## B. two-dimensional cyclic phase orbit

Put `N` states on the unit circle:

```text
v_k=(cos(2*pi*k/N), sin(2*pi*k/N)).
```

The nearest-state noise radius is

```text
epsilon = sin(pi/N) ~ pi/N.
```

Again the dimension stays constant and robustness shrinks with the number of states.

## C. regular simplex

A regular `N`-vertex simplex lives in `R^(N-1)` with all vertices on the unit sphere.

Its nearest-state radius is

```text
epsilon = sqrt(N / (2(N-1))) -> 1/sqrt(2).
```

So linear-in-`N` dimension buys constant separation.

These are endpoints, not an optimal frontier.

---

# 4. The real compiler axis

The relevant recurrent resource is therefore not just

```text
state dimension d.
```

It is at least

```text
(dimension d, dynamic range R, robustness epsilon).
```

Then transition cost must be added:

```text
(d, R, epsilon,
 transition operator complexity,
 communication depth,
 reset/write cost,
 relation defect,
 long-horizon drift).
```

This is why the smallest algebraic representation is not automatically the best physical representation.

---

# 5. Why this matters to the recent state-tracking literature

Chung, Choi & Kim, *Rethinking State Tracking in Recurrent Models Through Error Control Dynamics* (2026), make the same **between-state separation** operationally important from a different direction.

They measure

```text
q(t) = within-class spread / minimum between-class separation
```

and show that affine recurrent trackers fail when accumulated state-relevant error consumes the available readability margin.

Their theorem also shows that an exact affine return map preserving the symbolic states cannot selectively contract errors along the state-separating subspace.

So KYY's packing radius is not an error-correction mechanism.

It is the initial geometric error budget available before drift makes states unreadable.

This distinction is critical:

```text
large margin
    -> longer runway

state-dependent contraction
    -> actual restoring dynamics.
```

---

# 6. Prior-art wall

The mathematics around this is enormous:

- sphere packing and spherical codes;
- error-correcting codes;
- vector quantization;
- frame theory;
- Johnson-Lindenstrauss embeddings;
- analog coding and finite-precision computation.

Therefore KYY must not claim the dimension/precision trade-off itself as new.

The useful role of the packing floor is to keep comparisons honest when different recurrent architectures use very different continuous state encodings.

---

# 7. The next seam

The 2D phase orbit is not the only way to represent a cyclic group.

The packing floor suggests an intermediate point:

```text
2 dimensions
    -> margin ~1/N

N-1 dimensions
    -> constant margin

QUESTION:
    can O(log N) dimensions already give constant margin
    while preserving the exact cyclic rotation update?
```

The answer is yes using a small collection of Fourier characters / a cyclic harmonic frame.

That is Pass 23.

---

# Current pin after Pass 22

Whenever KYY says one realization uses a smaller recurrent state, it must now report at least:

```text
state dimension
state dynamic range
minimum symbolic-state separation
noise/decoder margin
```

Otherwise "compression" may simply mean hiding the memory bill in precision.