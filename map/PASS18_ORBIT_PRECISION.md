# Pass 18 — group orbits: geometry trades state dimension for precision

Date: 2026-08-10

Pass 17 optimized local write sites in a **one-hot automaton-state realization**.

That immediately exposed a limitation:

> the geometry of one-hot state labels is not automatically the geometry of the cheapest recurrent realization.

A structured task can have exponentially/factorially many discrete behavioral states while admitting a much smaller continuous representation.

This pass records the smallest example and the prior-art boundary.

---

# 1. Exact cyclic-reset machine in two real dimensions

Consider `n` behavioral states arranged cyclically:

```text
0 -> 1 -> ... -> n-1 -> 0
```

plus a reset token sending every state to 0.

Represent behavioral state `k` by the unit-circle point

```text
v_k = (cos(2*pi*k/n), sin(2*pi*k/n)).
```

Then the cycle token is the fixed orthogonal rotation

```text
C = R(2*pi/n)
```

and

```text
C v_k = v_(k+1 mod n).
```

The reset token is the affine map

```text
z' = 0*z + v_0.
```

Thus:

```text
reversible token -> 2D norm-preserving phase rotation
reset token      -> rank-zero affine overwrite
```

This realizes the behavior exactly in real arithmetic for every `n >= 3`.

Script:

- `cyclic_reset_orbit_oracle.py`

This is elementary group representation / affine-dynamics mathematics, not a novelty theorem.

---

# 2. The hidden cost is precision

Use equal-norm linear prototypes `v_k` as the decoder:

```text
logit_k(z) = v_k dot z.
```

At the exact state `v_k`, the correct class has inner product 1.

The nearest competing state differs by angle `2*pi/n`, so the unit-scale logit margin is

```text
margin(n) = 1 - cos(2*pi/n).
```

The Euclidean distance between neighboring state points is

```text
2 sin(pi/n).
```

The distance from `v_k` to its nearest prototype decision boundary is therefore

```text
r_noise(n) = sin(pi/n).
```

As `n` grows,

```text
r_noise(n) ~ pi/n
margin(n)  ~ 2*pi^2/n^2.
```

So the recurrent state dimension stays fixed at 2 while the noise/precision burden tightens.

That is the important resource conversion:

```text
fewer state coordinates
        ->
smaller geometric separation
        ->
more analog precision / lower noise / sharper decoding.
```

No free memory has appeared.

---

# 3. Exact examples

The oracle gives:

| n states | recurrent real dim | nearest-state distance | guaranteed nearest-prototype noise radius | unit logit margin |
|---:|---:|---:|---:|---:|
| 3 | 2 | 1.73205 | 0.866025 | 1.5 |
| 5 | 2 | 1.17557 | 0.587785 | 0.690983 |
| 10 | 2 | 0.618034 | 0.309017 | 0.190983 |
| 100 | 2 | 0.0628215 | 0.0314108 | 0.00197327 |
| 1000 | 2 | 0.00628317 | 0.00314159 | 1.97391e-5 |

The state dimension column looks spectacular if reported alone.

The last two columns explain why it is not magic.

---

# 4. General finite-group orbit construction

The cyclic example is a special case of a general observation.

Let a finite group `G` have a faithful real linear representation

```text
rho : G -> GL(V).
```

Because `G` is finite, one can average an inner product over the group and choose coordinates in which the representation is orthogonal:

```text
rho(G) subset O(d).
```

Choose a vector `v` with trivial stabilizer.

For a faithful finite-group representation, such vectors exist generically: for each nonidentity `g`, the fixed-point set of `rho(g)` is a proper subspace, and a finite union of proper subspaces cannot cover `V` over `R`.

Then the orbit

```text
{rho(g) v : g in G}
```

contains exactly `|G|` distinct points.

A group-state machine can therefore be represented by this orbit, with group tokens acting by orthogonal matrices.

A reset to a chosen group state can be added affinely:

```text
z -> v_reset.
```

Again:

```text
reversible group work -> orthogonal orbit motion
irreversible reset    -> singular affine event.
```

This is standard representation theory plus an elementary affine reset, not a KYY theorem.

---

# 5. Why this corrects the one-hot path compiler

The full transformation-monoid path oracle in Passes 13/17 uses one physical coordinate per behavioral state.

That is a legitimate representation for a worst-case transformation monoid.

But it can grossly overstate the recurrent dimension needed for a structured task.

Example already in KYY:

```text
S5 behavioral group states: 120
natural one-hot action:       5 coordinates for point permutation
A4 standard representation:   4 real coordinates
```

Pass 8 compiled the whole `S5` action exactly in the 4D representation.

The cyclic example is even more dramatic:

```text
C_p behavioral states for prime p
one-hot/permutation degree: grows with p
real rotary orbit state:    2 dimensions
```

The representation changes the hardware problem.

Therefore the compiler cannot optimize write-site placement until it has decided **which realization space** the behavior will inhabit.

---

# 6. Prior-art wall: minimal representation dimension is an established field

Do not claim:

```text
"finite semigroups/groups can have smaller faithful representations"
```

as new.

There are multiple mature notions:

## Minimum transformation degree

For a finite semigroup `S`, the minimum transformation degree is the least `n` such that `S` embeds faithfully in the full transformation monoid `T_n`.

Recent example:

- Cirpons, East & Mitchell, *Minimum Transformation Representations of Diagram Monoids* (IMRN 2026), https://doi.org/10.1093/imrn/rnag041

## Minimum faithful linear dimension / effective dimension

The least dimension of a faithful matrix representation of a finite semigroup is likewise established.

- Mazorchuk & Steinberg, *Effective dimension of finite semigroups* (2012), https://doi.org/10.1016/j.jpaa.2012.04.014

## Finite-group representation dimension

Minimum faithful real/complex group representation dimension is classical representation theory.

So KYY does not own the compression mechanism.

---

# 7. Very close modern neural prior art: Recurrent Neural Cascades

Knorozova & Ronca, *On The Expressivity of Recurrent Neural Cascades* (AAAI 2024 / arXiv 2312.09048), explicitly connect recurrent neural cascades to algebraic automata theory. They show that group-capable recurrent neurons are sufficient to extend recurrent cascades to all regular languages.

MinMax Recurrent Neural Cascades (Ronca, May 2026, arXiv 2605.06384) go substantially further. They give a practical recurrent architecture with all-regular-language formal expressivity, stable state/gradient properties, parallel evaluation, and explicitly relate recurrent state dimension to the permutation degree of automata factors.

Therefore:

```text
"use group/reset automata decomposition to design recurrent neural cascades"
```

is occupied.

---

# 8. The July-2026 opening that appeared after MinMax RNC

Borelli et al., *The Sigma-Chain Product* (arXiv 2607.16884), appeared after the May MinMax RNC paper.

Classical cascades allow component `i` to depend on the external input plus all earlier component states.

The new Sigma-chain restricts this to:

```text
external input
    +
immediately preceding component state only
```

and still proves that Sigma-chains of permutation-reset automata recognize exactly all regular languages. The representation can be exponentially more succinct than the corresponding cascade representation.

A targeted search in this pass did **not** locate a paper combining:

```text
Sigma-chain automata decomposition
        +
neural recurrent cascade implementation
        +
group-capable low-dimensional continuous factors.
```

This may simply be because the Sigma-chain paper is only weeks old.

Status:

**UNMAPPED CONJUNCTION / NOT A NOVELTY CLAIM.**

---

# 9. A candidate KYY architecture principle, not yet code

The emerging object is no longer one giant geometric recurrent matrix.

It is closer to a chain of small algebraically typed cells:

```text
external token x_t
     |          |          |
     v          v          v
 [cell 1] --> [cell 2] --> [cell 3] --> ...
                 local predecessor only
```

Each cell is typed by the automata decomposition:

```text
GROUP CELL
    state lives on a low-dimensional geometric orbit
    token action is norm-preserving / orthogonal where possible

RESET / APERIODIC CELL
    state has an explicit singular/contractive set-reset operation
```

The symbolic decomposition tells us **where forgetting is needed**.

Representation theory tells us **how small the reversible orbit can be**.

Geometry tells us **how expensive local communication between factors is**.

Precision analysis tells us **whether the compressed orbit is physically robust**.

This is sufficiently different from KYY's original single-mesh architecture that it should not be implemented until the Sigma-chain/RNC bridge has been searched much harder.

---

# 10. New cost vector

The KYY compiler objective now needs:

```text
behavioral correctness
Sigma-chain height
state dimension per component
minimum orbit separation / decoder margin
local predecessor bandwidth
number of singular/reset components
contraction strength / reset events
parallel recurrent work
hardware wire span / locality
precision / noise tolerance
```

The phrase "2D state stores n states" is meaningless without the precision columns.

---

# Current pin after Pass 18

The best current question is:

> **Can the new Sigma-chain decomposition be lowered into a local recurrent neural/physical chain whose reversible factors use compact geometric group orbits and whose reset factors alone pay for contraction, while explicitly trading state dimension against precision?**

That is a much narrower and more interesting target than "geometry replaces attention."

No architecture code yet.