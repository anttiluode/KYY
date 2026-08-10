# Even-ring one-control lemma

Date: 2026-08-10

This note turns the numerical `(N/2)^2` pattern in `PASS5_CONTROL_ALGEBRA.md` into a small exact statement.

It is **not a novelty claim**. The result is very likely a special case of standard controllability / matrix-Lie-algebra structure. The point of proving it here is to know exactly what KYY's toy geometry is doing rather than treating a dimension sequence as magic.

---

## Statement

Let `N=2m`, and on `R^N` define

```text
X_ij = E_ij - E_ji.
```

Let the oriented cycle drift be

```text
G = X_01 + X_12 + ... + X_(N-2,N-1) + X_(N-1,0)
```

and let the only controlled edge be

```text
B = X_01.
```

Then

```text
Lie{G,B}  ~=  u(m)
```

under the standard real embedding `u(m) subset so(2m)`.

Consequently

```text
dim Lie{G,B} = m^2 = (N/2)^2.
```

This exactly matches the corrected numerical closures

```text
N = 4, 6, 8, 10, 12
m^2 = 4, 9, 16, 25, 36.
```

---

# Proof

## 1. Build the hidden complex structure

Let `S` be the cyclic shift

```text
S e_j = e_(j+1 mod N).
```

With the matrix convention above,

```text
G = S^-1 - S.
```

Let `R` be reflection through the controlled edge:

```text
R e_j = e_(1-j mod N).
```

Let

```text
D e_j = (-1)^j e_j.
```

Because `N` is even, shifting flips the sign pattern everywhere, including across the periodic boundary:

```text
D S = -S D.
```

Reflection reverses the shift:

```text
R S R = S^-1.
```

Define

```text
J = -D R.
```

The reflection sends even sites to odd sites and vice versa, so

```text
R D R = -D.
```

Hence

```text
J^2 = D R D R = -I.
```

Also `J` is orthogonal, so `J^-1=-J` and therefore `J^T=-J`: it is an orthogonal complex structure on `R^(2m)`.

Using `DS=-SD` and `RSR=S^-1`, a direct substitution gives

```text
J G = G J.
```

The reflection pairs the controlled vertices `0 <-> 1`. On that two-dimensional plane, `J` is just `-X_01`; on its orthogonal complement `B=X_01` is zero. Therefore

```text
J B = B J.
```

Every Lie word formed from `G` and `B` also commutes with `J`.

Thus

```text
Lie{G,B} subset {K in so(2m) : KJ=JK}.
```

But the centralizer of an orthogonal complex structure inside `so(2m)` is the real form of the skew-Hermitian algebra:

```text
{K in so(2m) : KJ=JK} ~= u(m).
```

Therefore

```text
Lie{G,B} subset u(m)
```

and in particular its dimension is at most `m^2`.

---

## 2. Rewrite the generators in complex coordinates

Choose real pairs adapted to `J`, so each pair `(f_k, J f_k)` becomes one complex coordinate.

One convenient zig-zag ordering starts

```text
(0,1), (2,N-1), (N-2,3), (4,N-3), ...
```

with orientations chosen so every 2D block of `J` is the same standard complex structure.

In these complex coordinates, `B` becomes, up to an irrelevant sign,

```text
i E_11.
```

The cycle drift `G` becomes

```text
i H
```

where `H` is a real symmetric tridiagonal `m x m` matrix whose every nearest-neighbour off-diagonal entry is non-zero. In the convention produced by the zig-zag basis the off-diagonal signs alternate, but their signs do not matter for the argument.

So the problem has reduced to showing

```text
Lie{iH, iE_11} = u(m)
```

for a connected real symmetric tridiagonal `H`.

---

## 3. Generate the first local u(2) block

Write

```text
P_1 = i E_11.
```

Because `H_12 != 0`,

```text
[P_1, iH]
```

isolates, up to a non-zero scalar, the real skew generator

```text
K_12 = E_12 - E_21.
```

Then

```text
[P_1, K_12]
```

gives the imaginary symmetric generator

```text
i(E_12 + E_21).
```

Commuting those two off-diagonal generators gives

```text
i(E_11 - E_22).
```

Since `iE_11` is already available, we obtain

```text
iE_22.
```

Thus the full `u(2)` algebra on the first two complex coordinates is available.

---

## 4. Walk down the path

Assume we have generated `iE_jj` and the adjacent generators up through edge `(j-1,j)`.

Because `H` is tridiagonal,

```text
[iE_jj, iH]
```

contains only the two neighboring skew directions

```text
K_(j-1,j)  and  K_(j,j+1).
```

The first is already known, so subtracting it isolates the new edge

```text
K_(j,j+1).
```

Repeating the previous two-dimensional commutators yields

```text
i(E_(j,j+1)+E_(j+1,j))
```

and then

```text
iE_(j+1,j+1).
```

Induction generates every adjacent skew, symmetric-off-diagonal, and diagonal skew-Hermitian generator along the complex path.

Finally, commutators of adjacent matrix units generate all longer-range pairs. These are a basis of `u(m)`.

Therefore

```text
u(m) subset Lie{iH, iE_11}.
```

Combined with the centralizer upper bound,

```text
Lie{G,B} ~= u(m).
```

QED.

---

# What this means for KYY

The even ring is not merely "less controllable" than the path.

It carries an exact symmetry that secretly turns the `2m` real state channels into `m` complex channels. One local control respects that same complex structure, so no amount of training or repeated composition can leave `u(m)`.

This is a concrete example of a design principle:

```text
geometry
   -> symmetry / conserved structure
   -> reachable operator algebra
   -> computational capability
```

A second suitably placed control can break the symmetry and enlarge the generated algebra. The numerical probe shows full `so(2m)` after adding the neighboring control `X_12` for the tested sizes, but that stronger all-`m` statement is not proved in this note.

This is exactly why KYY should inspect the algebra **before** optimization: an optimizer cannot learn its way out of a symmetry enforced by the generator set.

---

# Prior-art status

Do not cite this as a new theorem without a dedicated literature check.

The result sits directly inside mature areas:

- bilinear controllability on `SO(N)`;
- graph-theoretic controllability;
- quantum/network controllability;
- classification of dynamical Lie algebras;
- symmetry-restricted control systems.

Especially relevant is Wang et al. (2020), whose `SO(N)` model uses exactly the same basis generators `B_ij=E_ij-E_ji` and separates drift and controlled interaction graphs:

https://arxiv.org/abs/2007.11929

The KYY-specific value of the lemma is not ownership of the algebra. It is that the map can now distinguish **parameter count**, **control count**, and **symmetry-limited generated algebra** with an exact toy example.
