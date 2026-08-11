# Backend boundary — exact cyclic characters lowered to a second-order resonator

Date: 2026-08-11

This note is the first KYY backend calculation tied directly to a recent physical resonant-RNN architecture.

It is **not** a novelty claim for harmonic oscillators, exact difference equations, analog recurrent networks, resonator hardware, or digital-to-physical model mapping.

## External physical target

Zhou et al., *Fully Analog Resonant Recurrent Neural Network via Metacircuit* (arXiv:2604.17277, 2026), implement a recurrent network as coupled electrical local resonators. Their discrete hidden update has the second-order companion structure

```text
u[t+1] = (2I - dt^2 D^-1 Y) u[t] - u[t-1] + input,
```

with `D` representing FDNR/mass-like elements and `Y` the trainable admittance/stiffness-like matrix.

That paper already establishes trained resonant RNN -> coupled electrical-resonator hardware. KYY must not claim that direction as new.

Exact discretizations and difference-equation representations of the harmonic oscillator are also classical.

The KYY question is narrower:

> Given an exact cyclic character already selected by the symbolic/operator compiler, what does that character cost in this concrete second-order resonator coordinate system, and how should its port be transported?

---

## 1. One exact character

For a `C_n` character frequency `f`, the exact phase increment is

```text
theta = 2*pi*f/n.
```

The central-difference resonator block

```text
A(theta) = [[2 cos(theta), -1],
            [1,             0]]
```

has characteristic roots

```text
exp(+i theta), exp(-i theta)
```

whenever `sin(theta) != 0`.

Therefore an exact physical lowering in the metacircuit recurrence family requires

```text
dt^2 D^-1 Y = 2(1 - cos(theta)).
```

Equivalently,

```text
D^-1 Y = 2(1 - cos(2*pi*f/n)) / dt^2.
```

For nondegenerate roots this companion block is similar to the ordinary KYY planar rotation, so

```text
A(theta)^n = I
```

up to floating-point error.

---

## 2. The port transport is analytic

Write the physical companion state as

```text
h = [u_t, u_{t-1}]^T.
```

For an ideal oscillator trajectory `u_t = cos(psi)`, phase/quadrature coordinates are

```text
p = [cos(psi), sin(psi)]^T = T(theta) h
```

with

```text
T(theta) = [[1, 0],
            [-cos(theta)/sin(theta), 1/sin(theta)]].
```

Then

```text
T A T^-1 = R(theta),
```

where `R(theta)` is the ordinary planar rotation.

So if the symbolic/KYY phase-space port is

```text
logits = W_phase p,
```

the physical resonator port is simply

```text
W_companion = W_phase T(theta).
```

This is exact interface transport rather than readout retraining.

---

## 3. Digital equivalence does not imply physical equivalence

For a single faithful character of `C_n`, every frequency `f` with

```text
gcd(f,n)=1
```

has the same matched/prototype minimum symbolic score gap.

Reason: multiplication by `f` permutes the nonzero residues modulo `n`.

So for one-character exact state tracking, the digital algebra does not prefer one faithful frequency over another.

The resonator backend very strongly does.

For `C101`, `dt=1`:

```text
f     one-mode symbolic margin    D^-1Y       cond(T)     ||T||
1       0.0019344                 0.00387       32.14      22.74
4       0.0019344                 0.06160        8.00       5.70
25      0.0019344                 1.96890        1.016      1.008
49      0.0019344                 3.99130       21.42      15.16
50      0.0019344                 3.99903       64.29      45.47
```

Thus `f=25` and `f=50` are equally valid exact digital representations with equal one-mode symbolic margin, but their physical companion-coordinate interface conditioning differs by roughly 63x.

The required physical parameter ratio also moves from the middle of the stable interval to extremely near its upper edge.

For this backend, the best-conditioned faithful `C101` character is `f=25` (or its conjugate), close to a quarter-turn per symbolic increment.

---

## 4. The learned C101 banks already expose the mismatch

Pass 39's ten learned-and-snapped C101 frequency banks were audited without retraining.

All ten are exact/certified cyclic state representations in the software algebra.

But the worst mode in each bank has very different physical companion conditioning:

```text
best seed worst-mode cond(T):   2.44
worst seed worst-mode cond(T): 32.14
```

Likewise the largest required port-transform norm per bank ranges from about

```text
1.87 .. 22.74.
```

So two exact, equally acceptable software deployments can differ by more than an order of magnitude in a concrete analog-interface conditioning measure.

This is exactly the kind of cost that was missing from the earlier abstract state-code search.

---

## 5. The Nyquist boundary is physical, not merely symbolic

At

```text
theta = 0 or pi
```

`sin(theta)=0`, so `T(theta)` is singular.

In the central-difference companion form the repeated eigenvalue sits in a Jordan block rather than giving an ordinary diagonalizable rotation block.

For even `n`, the exact Nyquist character

```text
f = n/2
```

lands at

```text
dt^2 D^-1Y = 4,
```

the edge of the central-difference oscillator stability interval.

The unit test uses `C100, f=50` and confirms that the resulting companion matrix is not a clean finite-order `A^100=I` realization.

Thus a character that is perfectly ordinary in finite-group algebra can be a pathological choice for this specific resonator backend.

This is classical numerical/oscillator structure, not a new Nyquist theorem.

---

## 6. What this changes in KYY

Earlier state-code search looked dangerously unconstrained:

```text
minimize dimension + margin + locality + precision + projection distance + ...
```

A concrete backend gives a natural objective.

For this resonator family the compiler can price, per exact character:

```text
symbolic faithfulness / margin
resonator parameter ratio D^-1Y
stability-edge distance
phase-coordinate / port condition number
required physical readout gain
component tolerance sensitivity
locking / relocking burden.
```

Then representation choice becomes meaningful:

> among algebraically equivalent exact codes, choose the one whose geometry is cheapest for the actual substrate.

That statement is much closer to the original Geometric Neuron instinct than “waves implement a group.”

The geometry matters because the substrate breaks algebraic equivalences that software treats as free.

---

## 7. Prior-art boundary

Do not claim as new:

- harmonic oscillator recurrence mathematics;
- exact/difference discretization of oscillators;
- companion matrices and their similarity to phase coordinates;
- physical or analog recurrent neural networks;
- resonator/metamaterial RNNs;
- mapping trained network parameters to analog hardware;
- hardware-aware neural-network compilation in general.

The bounded KYY result here is simply:

1. take the exact cyclic representations produced earlier in KYY;
2. lower them into the recurrence family of a current fully analog resonator RNN;
3. transport the port analytically;
4. expose a large physical-conditioning difference among digitally equivalent exact characters.

Whether using this extra algebraic knowledge improves a real metacircuit/oscillator deployment remains untested.

## Files

- `map/metacircuit_cyclic_backend.py`
- `tests/test_metacircuit_cyclic_backend.py`
- `.github/workflows/metacircuit-cyclic-backend.yml`
- `results/metacircuit_cyclic_backend_summary.json`
