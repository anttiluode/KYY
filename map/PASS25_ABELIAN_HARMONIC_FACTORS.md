# Pass 25 — finite abelian factors: random characters give logarithmic-dimensional robust recurrent group state

Date: 2026-08-10

Pass 23 treated one cyclic group `C_n`.

The same construction extends immediately to every finite abelian group.

Again, this is standard finite-group harmonic/frame mathematics. The KYY value is the recurrent/compiler interpretation and explicit resource accounting.

Code:

- `abelian_harmonic_state_oracle.py`
- `tests/test_abelian_harmonic_state_oracle.py`

---

# 1. Finite abelian group

Write

```text
G = C_(n_1) x ... x C_(n_m).
```

A group element is

```text
g = (g_1,...,g_m).
```

The dual group has the same cardinality. A character can be indexed by a frequency tuple

```text
f = (f_1,...,f_m)
```

and written

```text
chi_f(g)
  = exp(2*pi*i * sum_j f_j g_j / n_j).
```

Choose `k` characters `chi_(f_1),...,chi_(f_k)`.

Represent `g` by

```text
v_g = 1/sqrt(k) [chi_(f_1)(g), ..., chi_(f_k)(g)] in C^k.
```

In real coordinates this is `2k` scalar channels.

---

# 2. Group update is diagonal phase propagation

For token/group increment `h`, character multiplicativity gives

```text
chi_f(g+h) = chi_f(g) chi_f(h).
```

Therefore

```text
v_(g+h)
  = diag(chi_(f_1)(h), ..., chi_(f_k)(h)) v_g.
```

In real coordinates this is a block-diagonal bank of `k` independent 2D rotations.

So every abelian group token is an exact norm-preserving linear recurrent update in ideal arithmetic.

No dense state mixing is needed.

---

# 3. The same concentration proof depends only on |G|

For distinct states `g` and `h`, let

```text
delta = h-g != 0.
```

Their real inner product is

```text
Re <v_g,v_h>
  = (1/k) sum_l Re chi_(f_l)(delta).
```

Choose each character independently and uniformly from the dual group.

Character orthogonality gives, for every nonidentity `delta`,

```text
E[chi_f(delta)] = 0
```

and hence

```text
E[Re chi_f(delta)] = 0.
```

Each real part lies in `[-1,1]`.

Hoeffding + union bound over `|G|-1` nonidentity differences gives

```text
P[max_(delta != e) Re <v_e,v_delta> >= alpha]
    <= (|G|-1) exp(-k alpha^2/2).
```

Therefore some character multiset exists with constant pair separation whenever

```text
k > 2 log(|G|-1) / alpha^2.
```

Thus, for every fixed `alpha<1`,

```text
complex recurrent modes = O(log |G|)
real recurrent dimension = O(log |G|)
nearest-prototype geometric margin = Omega(1).
```

The bound depends on group order, not on whether `G` is cyclic or a product of many cyclic factors.

---

# 4. This is standard harmonic-frame mathematics

Do not claim the group-frame construction as new.

For a finite abelian group, selecting characters and using the corresponding columns of the character table is exactly the classical harmonic-frame construction.

References include:

- Chien & Waldron, *A classification of the harmonic frames up to unitary equivalence* (2011);
- *On the number of harmonic frames* and related finite-group frame theory;
- geometrically uniform frames and harmonic coding literature.

The selected characters are the recurrent coordinates; the character table supplies the state codebook.

The probabilistic existence calculation is a simple random-character/coherence bound, not a new harmonic-analysis theorem.

---

# 5. Exact executable examples

The oracle handles groups specified as products such as

```text
C31
C4 x C5
C3 x C3 x C3.
```

For every candidate frequency set it verifies exhaustively, on small groups, that

```text
A_h v_g = v_(g+h)
```

for every state `g` and every group token `h`.

A short deterministic random search for `C4 x C5` with

```text
|G| = 20
k = 8 complex modes
real dimension = 16
```

finds codes with nearest-prototype radius well above `0.6` in typical runs.

This is a numerical illustration, not an optimal code claim.

---

# 6. Why this is relevant to Sigma-chain rather than just another counter

The Sigma-chain theorem decomposes regular behavior into permutation-reset components.

A permutation component may have transition group `G`.

For an **abelian** group factor, KYY now has two very different exact continuous lowerings:

```text
PERMUTATION / MinMax-LIKE
    faithful permutation action on coordinates
    coordinate count tied to permutation degree

HARMONIC
    characters of G as phase coordinates
    O(log |G|) modes sufficient for constant geometric margin
    diagonal norm-preserving update.
```

The second representation does not automatically dominate the first.

It may require:

```text
higher analog precision
phase calibration
continuous nearest-state decoding
control logic for arbitrary predecessor-conditioned token actions
error management over long horizons.
```

Those costs are exactly why Pass 22 exists.

---

# 7. The important compiler distinction: state code versus controller code

Compressing the **state representation** does not necessarily compress the transition controller.

A generic Sigma-chain component chooses its internal permutation/reset transition from

```text
(external token, predecessor state).
```

If that context-to-group-element map is an arbitrary table, then even a tiny harmonic state can sit behind a large controller table.

Therefore the harmonic construction only earns an end-to-end efficiency result when at least one of these is true:

```text
controller map is itself algebraically structured;
controller description is compressed separately;
physical predecessor signal selects phases cheaply;
or state/communication savings dominate controller cost.
```

This is a major caveat.

---

# 8. Relation to the 2026 state-tracking papers

Recent papers already establish that complex/negative eigenvalues and input-dependent rotations matter for modular/group state tracking.

The new KYY lens is more resource-specific:

```text
not merely:
    does a complex recurrence contain the right group action?

but:
    how many phase coordinates are needed
    so that all symbolic states remain geometrically well separated?
```

The harmonic-frame answer for finite abelian groups is logarithmic in group order at fixed separation margin.

A targeted search in this pass did not locate this exact statement as a modern recurrent-state result.

Status:

**KNOWN HARMONIC-FRAME MATHEMATICS + UNMAPPED RECURRENT RESOURCE INTERPRETATION.**

---

# 9. Do not generalize to all finite groups yet

For nonabelian groups, irreducible representations are matrix-valued and minimum faithful representation dimensions can behave very differently.

There is no justification here for saying that every finite group has an `O(log |G|)` robust recurrent unitary representation.

The abelian theorem is clean precisely because all irreducible characters are one-dimensional.

Keep the claim abelian until the nonabelian representation/packing problem is mapped properly.

---

# Current pin after Pass 25

We now have an exact possible lowering for one major class of Sigma-chain factors:

```text
ABELIAN PERMUTATION FACTOR
        |
        v
sample/select characters
        |
        v
harmonic-frame state code
        |
        +--> O(log |G|) recurrent dimension
        +--> constant state margin
        +--> diagonal phase updates
        |
        v
explicit error-control / precision budget
```

The other component type remains:

```text
RESET / APERIODIC FACTOR
    -> contraction or singular reset only where the behavioral decomposition requires it.
```

The next meaningful test is no longer another toy group in isolation.

It is a **small regular automaton whose Sigma-chain contains both an abelian permutation factor and a reset factor**, compiled end-to-end with all controller, state, margin, communication, and reset costs reported.