# Pass 23 — harmonic counter: exact cyclic state tracking with logarithmic dimension and constant geometric margin

Date: 2026-08-10

This pass connects three mature subjects:

```text
cyclic state tracking / complex rotations
        +
harmonic frames / Fourier characters
        +
state-separation / error-control margin.
```

Each ingredient is known.

The useful KYY result is a resource statement and executable construction:

> **An `n`-state cyclic counter has an exact norm-preserving linear recurrent realization in `O(log n)` real dimensions whose symbolic states have a constant Euclidean nearest-neighbour margin.**

The logarithmic existence bound below is an elementary probabilistic corollary of random character sampling. It is not claimed as new frame theory.

Code:

- `cyclic_harmonic_state_oracle.py`
- `tests/test_cyclic_harmonic_state_oracle.py`

---

# 1. Construction

Let the cyclic group be

```text
C_n = Z / nZ.
```

Choose `k` frequencies

```text
f_1, ..., f_k in Z_n.
```

Represent state `j` by

```text
v_j = 1/sqrt(k) * [
    cos(2*pi*f_1*j/n), sin(2*pi*f_1*j/n),
    ...,
    cos(2*pi*f_k*j/n), sin(2*pi*f_k*j/n)
].
```

Thus

```text
v_j in R^(2k)
||v_j||_2 = 1.
```

The increment token `+1 mod n` is one block-diagonal orthogonal matrix:

```text
A = diag(
    R(2*pi*f_1/n),
    ...,
    R(2*pi*f_k/n)
).
```

Then, in exact real arithmetic,

```text
A v_j = v_(j+1 mod n).
```

More generally token `a in C_n` uses the same blocks with angles

```text
2*pi*f_l*a/n.
```

So the entire cyclic state tracker is an exact diagonal-complex / block-rotary affine recurrence.

---

# 2. This is a cyclic harmonic frame

The vectors above are not a new object.

Selecting characters of a finite cyclic group and taking the resulting Fourier columns gives a **cyclic harmonic frame**, also described in the frame/coding literature as a geometrically uniform group frame.

Harmonic frames and low-coherence partial Fourier constructions are mature subjects.

Useful references include:

- Hirn, *The number of harmonic frames of prime order* / cyclic harmonic-frame literature;
- broader finite-group harmonic-frame work;
- random/low-coherence partial Fourier frames in compressed sensing and coding.

Therefore KYY does not claim the vector construction itself.

---

# 3. Pair separation is controlled by character averages

For two states whose cyclic difference is `Delta != 0 mod n`, their real inner product is

```text
<v_j, v_(j+Delta)>
    = (1/k) sum_l cos(2*pi*f_l*Delta/n).
```

Let

```text
mu_plus = max_(Delta != 0) <v_0, v_Delta>.
```

Then

```text
||v_j - v_m||^2 >= 2(1 - mu_plus)
```

for all distinct states.

Therefore nearest-prototype decoding has guaranteed Euclidean perturbation radius

```text
epsilon >= sqrt((1 - mu_plus)/2).
```

This is the direct bridge from harmonic-frame coherence to the state-tracking margin that Pass 22 says must be reported.

---

# 4. Elementary logarithmic existence bound

Choose every frequency `f_l` independently and uniformly from `Z_n`.

For a fixed nonzero `Delta`, define

```text
X_l = cos(2*pi*f_l*Delta/n).
```

Character orthogonality gives

```text
E[X_l] = 0.
```

Also

```text
-1 <= X_l <= 1.
```

By Hoeffding's inequality,

```text
P[(1/k) sum_l X_l >= alpha]
    <= exp(-k alpha^2 / 2).
```

There are only `n-1` nonzero shifts. A union bound gives

```text
P[mu_plus >= alpha]
    <= (n-1) exp(-k alpha^2 / 2).
```

Hence if

```text
k > 2 log(n-1) / alpha^2,
```

that upper bound is below one, so **some** frequency multiset exists with

```text
mu_plus < alpha.
```

For that realization,

```text
real recurrent dimension d = 2k = O(log n / alpha^2)
```

and

```text
nearest-prototype noise radius
    >= sqrt((1-alpha)/2).
```

For any fixed `alpha<1`, the margin is therefore constant while dimension grows only logarithmically with the number of cyclic states.

---

# 5. A concrete conservative specialization

Take

```text
alpha = 1/2.
```

Then it suffices that

```text
k > 8 log(n-1)
```

complex modes, i.e. roughly

```text
d > 16 log n
```

real coordinates.

The guaranteed nearest-prototype perturbation radius is

```text
sqrt((1-1/2)/2) = 1/2.
```

The constants are deliberately loose. The bound only proves existence; good deterministic harmonic frames or a short search can use far fewer modes.

Do not market `16 log n` as an optimal dimension theorem.

---

# 6. Small numerical search shows the constants are loose

`cyclic_harmonic_state_oracle.py` includes deterministic random search over frequency multisets.

Typical 1000-trial examples found during this pass:

```text
C_31,   k=8   (real d=16):  radius ~0.65 in good searches
C_101,  k=16  (real d=32):  radius ~0.62-0.63
C_1009, k=32  (real d=64):  radius ~0.60
```

Compare the single-phase `d=2` encoding:

```text
C_31:   sin(pi/31)   ~0.101
C_101:  sin(pi/101)  ~0.0311
C_1009: sin(pi/1009) ~0.00311.
```

The harmonic representation spends logarithmically more state coordinates to stop the geometric margin collapsing with the modulus.

These random-search numbers are demonstrations, not optimized records.

---

# 7. Relation to the newest recurrent state-tracking papers

## Complex State Propagator — August 4, 2026

Li & Lu, *State Propagation Also Satisfies: A Complex-Valued State-Space Model for Deterministic State Tracking* (arXiv:2608.03425), use input-dependent complex rotations and argue that cyclic state transitions such as parity and mod-3 can be represented exactly in phase.

Their experiments use hidden dimension 64 and evaluate parity, mod-3, and parenthesis matching.

The paper does **not**, in the current reading, derive a modulus-dependent state-dimension/robustness trade-off or a logarithmic harmonic-frame encoding for large `C_n`.

Thus:

```text
"complex rotations implement modular counting"
```

is occupied.

The harmonic result addresses a different resource question.

## Error Control Dynamics — May 2026

Chung, Choi & Kim, *Rethinking State Tracking in Recurrent Models Through Error Control Dynamics* (arXiv:2605.07755), show that expressivity alone does not make affine recurrent trackers robust.

They measure a distinguishability ratio

```text
q(t) = within-class spread / minimum between-class separation
```

and find that long-horizon accuracy fails when accumulated state-relevant error consumes the separation margin.

This makes the harmonic-frame improvement directly relevant: it increases the **between-state separation budget**.

But it does not supply state-dependent error correction.

---

# 8. Constant margin is not an attractor

This is the most important caveat.

The harmonic transition is affine and norm preserving.

It does not selectively pull a perturbed state back to the nearest symbolic orbit point.

If the implemented rotation has a systematic error, the error can accumulate over repeated recurrence.

Therefore:

```text
constant state margin
    -> larger finite-precision runway

NOT

constant state margin
    -> indefinite error-free tracking.
```

This distinction is exactly the one emphasized by the 2026 error-control analysis.

A complete robust recurrent tracker may need one of:

```text
exact/hard group operators
periodic projection / decoding
state-dependent contraction
error-correcting nonlinear dynamics
or sufficient numerical precision for the required horizon.
```

Those add costs and must be counted.

---

# 9. Comparison with MinMax permutation degree

MinMax Recurrent Neural Cascades use a different representation resource for permutation factors: **faithful permutation degree**.

A permutation semiautomaton of degree `m` is realized by rearranging `m` recurrent coordinates.

For a cyclic group of prime order `p`, a nontrivial faithful permutation action requires an orbit of size `p`, so its faithful permutation degree is `p`.

The harmonic construction represents the same abstract cyclic group in

```text
O(log p)
```

real continuous coordinates at constant Euclidean state margin.

This does **not** contradict MinMax.

It changes the representation class:

```text
MinMax:
    robust coordinate-permutation state

harmonic KYY baseline:
    continuous orthogonal group orbit
    with explicit precision/margin accounting.
```

The correct comparison must include numerical precision, readout, and error-control cost, not only dimension.

---

# 10. Prior-art status

The following are occupied:

```text
complex phase for cyclic state tracking
harmonic frames / group frames
low-coherence partial Fourier embeddings
random character concentration
state separation as an error-control resource.
```

Targeted searches in this pass did **not** locate a modern recurrent-state paper whose central result is stated as:

> exact `C_n` state tracking by a norm-preserving recurrent harmonic frame with constant Euclidean margin in `O(log n)` state dimension.

This is a **search miss, not a novelty proof**.

The mathematical derivation is simple enough that it may well exist under coding/frame terminology rather than RNN terminology.

Status:

**KNOWN MATHEMATICS + CURRENTLY UNMAPPED RECURRENT RESOURCE INTERPRETATION.**

---

# 11. Why this is a better Geometric-Neuron result than the original mesh

The first KYY mesh said:

```text
local geometry can express a recurrent operator.
```

EUNN already owned most of that.

The harmonic counter says something more specifically geometric:

```text
choose several physical/resonant phase coordinates
        |
        v
one symbolic cyclic state = joint phase pattern
        |
        v
one token = simultaneous local phase advances
        |
        v
many discrete states remain far apart geometrically.
```

The computation is not in one freely learned dense matrix.

It is in the **geometry of a group orbit across resonant modes**.

That does not make it new, but it is much closer to the original Geometric Neuron intuition.

---

# 12. The immediate generalization — do not build it yet

For a finite abelian group `G`, one can likewise choose characters

```text
chi_1, ..., chi_k
```

and map group element `g` to

```text
v_g = 1/sqrt(k) [chi_1(g), ..., chi_k(g)]
```

in complex coordinates.

Group multiplication acts diagonally by phase:

```text
v_(gh) = diag(chi_1(h),...,chi_k(h)) v_g.
```

Random characters should give the same concentration logic over the `|G|-1` nonidentity differences, suggesting `O(log |G|)` complex modes for constant pair separation.

This is essentially an abelian harmonic-frame / random-character embedding, not a new construction.

Do not jump to non-Abelian groups until the cyclic/abelian prior-art boundary is closed.

---

# 13. Connection back to Sigma-chain

If the result survives deeper prior-art search, the emerging compiler becomes:

```text
DFA / regular behavior
        |
        v
Sigma-chain permutation/reset decomposition
        |
        +--> abelian/group factor
        |       -> compact harmonic orbit state
        |       -> norm-preserving phase update
        |
        +--> reset factor
                -> explicit singular/contractive update
        |
        v
strict predecessor-only recurrent chain
```

Then the reported resources are not only parameter count:

```text
state dimension per factor
minimum orbit separation
operator relation defect
error-control mechanism
reset count
inter-factor bandwidth
physical/local update cost.
```

That is the current KYY frontier.