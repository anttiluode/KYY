# Pass 38 — generator-derived port transport

Date: 2026-08-10

Pass 37 removed the assumption that the non-commuting reflection generator was already exact, but its quotient-conditioned port compiler still used one operation that does not scale conceptually:

```text
enumerate the complete learned canonical group orbit
enumerate the complete legalized canonical group orbit
fit two per-coset Procrustes maps
```

This pass removes orbit enumeration from **port compilation**.

The full finite orbit is still enumerated afterward for the behavioral certificate. Compilation and certification are now separate costs.

## Rotation-coset transport from generators

For one 2D mode, let

```text
A  = R(theta)       learned rotation generator
A* = R(theta*)      legalized exact rotation generator
v                   canonical seed vector
```

The learned and legalized rotation-coset states are

```text
x_k  = A^k v
x*_k = A*^k v
```

for `k=0,...,n-1`.

The orthogonal Procrustes transport depends only on the 2x2 cross covariance

```text
C = sum_k x*_k x_k^T
  = sum_k A*^k (v v^T) (A^k)^T.
```

There is no need to instantiate the `n` states.

Define the 4x4 linear superoperator

```text
K : M -> A* M A^T.
```

Then

```text
vec(C) = (I + K + K^2 + ... + K^(n-1)) vec(v v^T).
```

The finite matrix geometric sum is computed by exponentiation-by-squaring in `O(log n)` 4x4 matrix products.

The port block is the orthogonal polar/Procrustes factor of `C`.

This is classical linear algebra; no novelty claim is made for the finite geometric sum or Procrustes step.

## Reflected coset without enumeration

Let

```text
B   learned reflection generator
S*  legalized exact reflection generator.
```

The reflected states are

```text
y_k  = B x_k
y*_k = S* x*_k.
```

Therefore their cross covariance follows directly from the rotation-coset covariance:

```text
C_ref
  = sum_k y*_k y_k^T
  = S* C B^T.
```

So the second quotient-conditioned Procrustes block also comes directly from the pre/post generator matrices.

For `m` independent 2D modes the compiler repeats this calculation mode by mode and assembles two block-diagonal `O(2)^m` maps:

```text
Q_0   rotation coset
Q_1   reflected coset.
```

The only dynamic side information remains the one exact `C2` quotient bit that selects `Q_0` or `Q_1`.

## What information the compiler now needs

For the controlled dihedral family, port transport requires only:

```text
n
learned rotation generators
legalized rotation generators
learned reflection generators
legalized reflection generators
canonical seed vector h0
```

It does **not** require:

```text
task labels
decoder retraining
group-state labels
explicit enumeration of the n or 2n states
```

The learned decoder is used only after the port is compiled, to certify observable behavior.

## Exact equivalence checks

`tests/test_dihedral_generator_port_transport.py` verifies that:

1. the `O(log n)` matrix-power sum equals explicit summation;
2. the generator-derived rotation cross covariance equals the explicitly enumerated orbit covariance;
3. the resulting quotient-conditioned block Procrustes maps agree numerically with those obtained from complete orbit enumeration on synthetic dihedral examples.

The full repository test suite passes with these checks.

A trained `D101` comparison then reran seeds `0..4`, including the two raw-port failures in that subset (seeds 1 and 2). The generator-derived and orbit-derived quotient maps agree to numerical precision:

```text
seed   raw exact port   generator port   max ||Qgen-Qorbit||
0          202/202          202/202          5.97e-15
1          200/202          202/202          6.27e-15
2          194/202          202/202          1.11e-14
3          202/202          202/202          8.36e-15
4          202/202          202/202          7.28e-15
```

The normalized representation-alignment errors are equal to displayed precision for the generator-derived and orbit-derived implementations, and the output margins are identical to the Pass 37 quotient-block compiler.

Thus the orbit-based Pass 37 port compiler and the generator-derived Pass 38 compiler are not merely similar heuristics in this model: they compute the same finite Procrustes objective, but Pass 38 evaluates its sufficient statistics directly from the generators.

## Why this matters

Pass 37 had conflated two finite-state costs:

```text
PORT COMPILATION
and
BEHAVIORAL CERTIFICATION.
```

They are now separated.

For this dihedral representation:

```text
port compilation       O(m log n) tiny matrix algebra
runtime side state     1 quotient bit
behavioral certificate 2n legal states (still exhaustive)
```

The remaining scaling bottleneck is therefore no longer the port transform. It is the complete behavioral certificate.

## Scope

This derivation uses the specific semidirect-product/modal structure of the controlled `D_n` representation. It is not a generic algorithm for arbitrary finite groups or arbitrary learned recurrent matrices.

The more general pattern to test later is whether port transport can be obtained from generator/relation equations—e.g. Sylvester/intertwiner constraints or finite superoperator sums—rather than from state enumeration.

## Files

- `map/dihedral_generator_port_transport.py`
- `tests/test_dihedral_generator_port_transport.py`
- `.github/workflows/dihedral-generator-port-transport.yml`
- `results/dihedral_d101_generator_port_transport.csv`

## Current stopping pin

Do **not** immediately add another group.

The next real unsolved cost is:

> can observable correctness be certified without enumerating the entire finite state space?

For the current small groups exhaustive checking is a strength, not a weakness. But for a compiler meant to scale, the certificate eventually needs a relation-local or margin/geometry argument that is useful rather than merely valid like the failed generic Cauchy certificate from Pass 31.
