# Pass 41 — non-Abelian port compression and stabilizer certificate

Date: 2026-08-10

Pass 40 removed full-state enumeration from the controlled cyclic compiler when the output interface is projected into a positive orbit/correlation kernel.

The immediate question was whether that was only an Abelian/Fourier accident.

It was not.

This pass tests the same idea on the already-established jointly legalized harmonic `D101` family.

---

## Prior-art subtraction first: these are group-orbit codes

The geometric theorem underneath the new certificate is classical.

Let a finite group `G` act orthogonally through `rho`, choose a seed `v`, and form the orbit

```text
C = { rho(g) v : g in G }.
```

This is exactly the old group-orbit / Slepian group-code construction. Orbit size is controlled by the stabilizer of the seed.

Slepian's 1968 group-code work and the later spherical/group-code literature therefore own the mathematical object. KYY makes no novelty claim for orbit codes, orbit-stabilizer, matched filtering, or group-equivariant parameter sharing.

The KYY question is post-training/compiler-specific:

> after training an unconstrained classifier around an approximate recurrent machine, can operator legalization expose a tiny exact group-orbit interface that replaces the large trained classifier while preserving the behavioral machine?

---

## General prototype identity

For legal states

```text
z_g = rho(g) v
```

use the orbit itself as class prototypes:

```text
w_g = z_g.
```

For a correct class `g` and competitor `h`, orthogonality gives

```text
score(g,g) - score(g,h)
  = ||v||^2 - <v, rho(g^-1 h) v>
  = 1/2 ||v - rho(g^-1 h) v||^2.
```

Therefore the prototype decoder is uniquely correct iff

```text
Stab_G(v) = {e}.
```

This is standard orbit geometry, but it gives the compiler a useful target: certifying a seed stabilizer can replace enumerating every class.

---

## D_n stabilizer certificate without a 2n-state scan

The current `D_n` representation is a direct sum of planar blocks.

For mode `i`:

```text
r -> R(2*pi*f_i/n)
s -> S_i
```

with `S_i` an exact projected 2D reflection satisfying

```text
S_i^2 = I
S_i R(theta_i) S_i = R(-theta_i).
```

### Rotation stabilizer

A rotation `r^k` fixes every active mode iff

```text
f_i k = 0 mod n
```

for all active characters. The rotation-kernel size is therefore

```text
gcd(n, active f_1, ..., active f_m).
```

### Reflected stabilizer

Write `beta_i` for the axis angle of reflection `S_i` and `gamma_i` for the phase of the seed vector in that mode.

A reflected element `s r^k` fixes the seed iff

```text
f_i k = n (beta_i - gamma_i)/pi   mod n
```

for every active mode.

The left side lies on the integer lattice modulo `n`.

So:

1. if any active mode's right-hand side is off that lattice, no reflected stabilizer exists;
2. otherwise each mode gives a linear modular congruence in `k`;
3. reduce and intersect those congruences with generalized CRT.

This costs `O(m log n)` arithmetic rather than testing `2n` group states.

For the current learned/projected reflection axes the off-lattice part is necessarily a floating-point numerical certificate rather than a symbolic rational proof. The rotation-kernel part is exact integer arithmetic. The code therefore should be read as a generator/seed certificate for the compiled numerical operator, not as a formal proof assistant artifact.

Code:

- `map/dihedral_stabilizer_certificate.py`
- `tests/test_dihedral_stabilizer_certificate.py`
- `.github/workflows/dihedral-stabilizer-certificate.yml`

---

## Full D_n-equivariant projection of the learned decoder

The bigger surprise was not the prototype decoder. It was the learned port.

The raw trained `D101` classifier has

```text
2n output rows x (2m weights + 1 bias)
= 202 * 17
= 3434 explicit learned scalars.
```

For an exactly equivariant linear classifier, every class row is generated from one base template:

```text
w_g = rho(g) w_e
b_g = constant.
```

The least-squares projection of the arbitrary learned classifier into this exact `D_n`-equivariant space is the group average

```text
w_e = (1/|D_n|) sum_g rho(g)^T w_g.
```

The deployed free parameter count becomes only

```text
2m + 1 = 17.
```

This is standard equivariant parameter sharing / Reynolds projection. Again, the projection itself is not a novelty claim.

What matters here is whether it preserves the already-trained behavioral interface **after recurrent-generator legalization**.

---

## Ten-seed D101 result

Setup:

```text
D101 states            202
complex modes            8
train length             16
train steps            2200
seeds                  0..9
reflection probability 0.25
joint rotation + reflection legalization
```

### Raw inherited learned port

The jointly legalized exact operator breaks three inherited ports:

```text
seed 1   200/202, min margin -0.191
seed 2   194/202, min margin -0.597
seed 9   200/202, min margin -0.279
```

So the baseline remains the same phenomenon as Passes 31/37:

> exact internal algebra does not imply observable correctness.

Overall:

```text
raw inherited port      7 / 10 exact
```

### Exact D101-equivariant decoder projection

Project the learned 3434-scalar decoder into the 17-parameter exact equivariant space.

Result:

```text
10 / 10 exact
minimum margin range: +1.731 .. +3.737
```

The projection changes the learned decoder substantially:

```text
relative Frobenius move: about 0.217 .. 0.284
```

yet all required finite behavior survives and the three broken inherited ports are repaired.

Parameter compression:

```text
3434 -> 17
202x fewer free decoder scalars.
```

---

## Positive orbit-kernel compression: 17 -> 9

As in Pass 40, constrain each mode of the equivariant base template to be a nonnegative multiple of that mode's seed component:

```text
w_e,i = alpha_i v_i
alpha_i >= 0.
```

Then every class row is still generated by the full non-Abelian group action:

```text
w_g = rho(g) w_e.
```

For any relative group element `x = g^-1 h`, the class margin is

```text
Delta(x)
 = sum_i alpha_i [ ||v_i||^2 - <v_i, rho_i(x) v_i> ].
```

Every term is nonnegative because `rho_i(x)` is orthogonal.

Equality holds exactly when `x` fixes every **active** seed component.

So a positive orbit-kernel decoder is uniquely correct whenever the intersection of active stabilizers is trivial.

The free port now has only

```text
m positive weights + 1 bias = 9 parameters.
```

Ten-seed result:

```text
10 / 10 exact
10 / 10 stabilizer certificates pass
all 10 keep all 8 modal weights positive
minimum margin range: +2.064 .. +4.081
```

It improves the 17-parameter equivariant margin on 9/10 seeds. Seed 0 is the only exception:

```text
17-param equivariant   +3.084
9-param positive       +2.941
```

which is still comfortably correct.

The hardest positive-kernel seed has margin about `+2.064`.

Parameter compression:

```text
3434 -> 17 -> 9
raw -> positive kernel: about 381.6x fewer free decoder scalars.
```

Result summary:

- `results/dihedral_d101_equivariant_port_summary.json`

Code:

- `map/dihedral_equivariant_port_probe.py`
- `tests/test_dihedral_equivariant_port_probe.py`
- `.github/workflows/dihedral-equivariant-port.yml`

---

## This does not make Passes 36–38 useless

The compiler now has two genuinely different contracts.

### Contract A — preserve a particular learned interface

Passes 36–38 keep the existing large learned decoder and transport the hidden state into the coordinates that decoder expects.

Advantages:

- preserves the trained output geometry much more directly;
- generator-derived transport can be constructed without enumerating the group orbit;
- useful when exact learned logits / port semantics matter.

Cost:

- the deployed `2n x 2m` learned decoder remains large.

### Contract B — canonicalize/compress the interface

Passes 39–41 instead project the decoder itself into the legal symmetry.

Advantages:

- removes most learned port parameters;
- can repair an inherited port that operator legalization broke;
- positive orbit-kernel version comes with a stabilizer-based correctness certificate.

Cost:

- changes the learned logit geometry;
- only the required finite class behavior is being preserved here, not arbitrary off-orbit logits.

So Pass 38 is not superseded. The design choice has become explicit:

```text
preserve learned interface
        versus
compile a canonical legal interface.
```

That is a very TWC-like distinction: preserve a chosen port realization, or replace it by another realization that is behaviorally equivalent for the declared contract.

---

## The biology/timing detour in retrospect

The old `tau` / phase-gated readout result is now properly demoted.

It was a useful one-dimensional search restriction. It found that **observation coordinates matter independently of internal dynamics**.

The later algebra then expanded that slice into the full equivariant port space and finally into a positive orbit-kernel cone.

This is the good outcome for the foreign-field heuristic:

```text
foreign analogy gives one tiny variable
    -> test it
    -> discover the actual symmetry
    -> discard the biological story
    -> keep the mathematical compiler variable.
```

D31 breaking the naive cyclic phase correction remains the important falsifier showing that the biology was never evidence.

---

## Prior-art boundary

Established owners include:

- Slepian/group-orbit codes and spherical group codes;
- orbit-stabilizer theory;
- group representations and irreducible planar representations of `D_n`;
- matched-filter / nearest-prototype decoding;
- group-equivariant parameter sharing and group averaging;
- positive-definite/group correlation kernels.

Relevant anchors:

- David Slepian, *Group Codes for the Gaussian Channel*, Bell System Technical Journal, 1968.
- Cohen & Welling, *Group Equivariant Convolutional Networks*, 2016.
- Ravanbakhsh, Schneider & Poczos, *Equivariance Through Parameter-Sharing*, 2017.

The residual KYY claim remains procedural and empirical:

> in these controlled trained harmonic group trackers, post-training operator legalization can be followed by exact symmetry projection of the learned classifier, drastically reducing the deployed port while repairing observable errors; a further positive orbit-kernel projection admits a small stabilizer-based correctness certificate.

---

## New stopping pin

For the controlled **group-only** machines, finite-state enumeration is no longer the main conceptual bottleneck.

The next hard boundary is the part groups cannot express:

```text
irreversible forgetting / reset / semigroup structure.
```

That is now an earned return to the reset benchmark rather than another arbitrary harder oracle.

A group orbit is easy to canonicalize because every state is symmetry-related to every other legal state. A reset transition deliberately **collapses distinct histories into one behavioral state**. Orbit-stabilizer cannot certify that operation because it is non-invertible.

So the next compiler question should be:

> can the machine factor into a reversible group-coded component plus a small irreversible quotient/reset component, and can both the transition relations and the observable port be certified without enumerating the full product state space?

That is where the conservative-wave failure from the reset experiment, Krohn–Rhodes/semigroup structure, and the new port compiler finally meet in one problem.
