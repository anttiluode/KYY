# Pass 40 — algebraic cyclic port certificate

Date: 2026-08-10

Pass 39 compressed the explicit C101 output layer from 1717 learned scalars to 17 exact-equivariant parameters and reduced the behavioral check to one relative score function on the cyclic group.

This pass asks a narrower question:

> Can the port itself be projected into a family whose correctness has a closed-form algebraic certificate, so that neither exhaustive state enumeration nor an SDP is needed?

For the controlled cyclic harmonic family, yes.

---

## 1. Synthesized prototype port

Let the exact legalized cyclic state be

```text
z_k = rho(k) z_0,
```

where `rho` is an orthogonal block-diagonal harmonic representation.

Choose class prototypes directly from the legal orbit:

```text
w_j = z_j.
```

With equal-norm nearest-prototype / matched-filter logits,

```text
score(k,k) - score(k,k+d)
  = sum_i a_i [1 - cos(2*pi*f_i*d/n)],
```

where `a_i >= 0` is the squared amplitude of mode i.

Every term is nonnegative.

The margin vanishes exactly when every active character is blind to the same displacement:

```text
f_i d = 0 mod n   for all active i.
```

For C_n this common kernel is trivial iff

```text
gcd(n, active f_1, ..., active f_m) = 1.
```

Therefore:

```text
prototype decoder is uniquely correct on every C_n state
iff
gcd(n, active character frequencies) = 1.
```

The gcd is also the size of the common kernel; the encoded orbit then has size `n/gcd`.

This is an O(m)-sized exact certificate. Enumeration is used in the tests only to check the implementation against the theorem.

Code:

- `map/cyclic_prototype_certificate.py`
- `tests/test_cyclic_prototype_certificate.py`

---

## 2. Preserve more of the learned port: positive correlation kernel

A pure prototype decoder discards the trained output geometry. Pass 39 showed that the trained decoder survives projection into the exact equivariant subspace, so we next projected that equivariant decoder into a smaller cone.

For each harmonic mode, constrain the equivariant base decoder vector to lie on the nonnegative ray through the compiled state-0 vector:

```text
w0_i = alpha_i u_i,
alpha_i >= 0.
```

All class rows still come from symmetry:

```text
w_j = rho(j) w_0.
```

The relative score becomes

```text
q(d) = b + sum_i alpha_i ||u_i||^2 cos(2*pi*f_i*d/n),
```

and therefore

```text
q(0) - q(d)
 = sum_i c_i [1 - cos(2*pi*f_i*d/n)],
c_i >= 0.
```

Again, correctness is automatic once the active characters have trivial common kernel.

The port now has only

```text
m nonnegative modal weights + 1 shared bias
```

free scalars.

For C101,m=8:

```text
raw learned decoder       1717
exact equivariant port      17
positive-kernel port          9

raw -> positive kernel: 190.78x fewer free scalars.
```

---

## Ten-seed C101 result

All ten recent learned C101 models were recompiled through the positive-kernel projection.

Result:

```text
10 / 10 exact on all 101 states
10 / 10 algebraically certified by positivity + gcd
10 / 10 retain all 8 modes with positive weights
10 / 10 positive-kernel margins exceed the 17-parameter equivariant margins
```

Summary:

| seed | equivariant min margin | positive-kernel min margin | active modes | gcd | certified |
|---:|---:|---:|---:|---:|:---:|
| 0 | +4.232 | +4.249 | 8 | 1 | yes |
| 1 | +3.178 | +3.278 | 8 | 1 | yes |
| 2 | +4.476 | +4.576 | 8 | 1 | yes |
| 3 | +4.351 | +4.423 | 8 | 1 | yes |
| 4 | +5.052 | +5.278 | 8 | 1 | yes |
| 5 | +4.197 | +4.670 | 8 | 1 | yes |
| 6 | +5.361 | +5.918 | 8 | 1 | yes |
| 7 | +2.245 | +2.917 | 8 | 1 | yes |
| 8 | +3.890 | +3.982 | 8 | 1 | yes |
| 9 | +3.756 | +4.307 | 8 | 1 | yes |

The weakest learned modal coefficient across the ten runs is still positive by a wide margin (about 6.47 in the probe's decoder scale), so these are not certificates balanced on numerical clipping at zero.

Result summary:

- `results/cyclic_c101_positive_kernel_port_summary.json`

Code:

- `map/cyclic_positive_kernel_port_probe.py`
- `tests/test_cyclic_positive_kernel_port_probe.py`
- `.github/workflows/cyclic-positive-kernel-port.yml`

---

## What happened to the proposed Fejer–Riesz / SOS pass?

It was a good search direction but it is no longer necessary for this cyclic subfamily.

There are now three nested interface families:

```text
A. arbitrary learned decoder
   1717 parameters for C101,m8

B. exact C_n-equivariant decoder
   17 parameters
   one sparse relative-score function q(d)
   generic finite-group Fourier positivity methods are applicable

C. positive correlation-kernel decoder
   9 parameters
   q(0)-q(d) is already an explicit sum of nonnegative 1-cos terms
   correctness reduces to a gcd / faithfulness test
```

So SOS remains relevant for family B when we want to preserve a more general learned equivariant interface.

Family C has a stronger built-in certificate and does not need an SDP.

This is also why it would have been a mistake to jump directly from the original 1717-parameter decoder to a continuous-circle degree-8 Fejer–Riesz claim. The algebra had one more compression step available first.

---

## A more general geometric statement

The prototype argument is not inherently cyclic.

Let a finite group G act through an orthogonal representation `rho`, and let

```text
z_g = rho(g) v.
```

Use the legal orbit itself as prototypes:

```text
w_g = z_g.
```

Then for a competitor h,

```text
score(g,g) - score(g,h)
 = ||v||^2 - <v, rho(g^-1 h) v>
 = 1/2 ||v - rho(g^-1 h) v||^2.
```

Therefore the prototype classifier is uniquely correct on the full group orbit iff the seed vector has trivial stabilizer:

```text
Stab_G(v) = {e}.
```

For C_n the gcd certificate above is exactly the character-coordinate form of this trivial-stabilizer condition.

This theorem is elementary representation/Euclidean geometry, not a novelty claim. Its compiler relevance is that **behavioral certification can sometimes be reduced from checking every group state to proving faithfulness / trivial stabilizer of the chosen legal representation and port seed.**

Whether that produces a cheap non-Abelian certificate depends on the group and representation; the cyclic case is unusually tractable.

---

## Correctness is not robustness

The gcd/stabilizer test is Boolean. It says the legal states are distinguishable by the compiled port.

It does **not** say they are well separated.

A faithful code can have arbitrarily small minimum margin. This keeps the earlier harmonic-code work relevant:

```text
algebra / gcd / stabilizer   -> exact symbolic correctness
harmonic geometry / margin   -> tolerance to phase error, finite precision, noise
```

The ten positive-kernel models happen to have healthy empirical margins, but those margins are deployment quantities and still need to be priced separately.

---

## Prior-art boundary

The ingredients here are standard:

- group characters and faithful representations;
- matched-filter / nearest-prototype classification;
- group-equivariant linear maps;
- Fourier positive functions and positive-definite kernels;
- sum-of-squares / Fourier-SOS certificates on finite abelian groups.

No novelty claim is made for the gcd theorem, the trivial-stabilizer identity, or positive correlation kernels in isolation.

The KYY residual is again the compiler composition and resource accounting:

```text
learn an approximate operator/port
    -> legalize the operator
    -> project the port into symmetry
    -> optionally project further into a self-certifying cone
    -> certify exact behavior algebraically
    -> price robustness separately.
```

---

## New stopping point

For the controlled cyclic family, full-state enumeration is no longer the exposed bottleneck.

The next hard questions are now sharper:

1. **Non-Abelian certification:** can the D_n / later group interfaces be reduced to similarly cheap stabilizer/intertwiner conditions without hiding enumeration in the test?
2. **Learned-interface preservation:** when the positive-kernel cone is too restrictive, can sparse finite-group Fourier/SOS certify the more general equivariant port efficiently?
3. **Robust certification:** can minimum margin under bounded phase/operator error be lower-bounded without enumerating the full orbit?
4. **Beyond group machines:** what replaces these certificates when the task algebra contains irreversible reset/semigroup structure rather than only invertible group actions?

Those are now more useful than another larger cyclic counter.
