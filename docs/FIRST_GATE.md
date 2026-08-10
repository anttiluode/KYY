# First gate — 2026-08-10

This note records the first local CPU experiments before any large sweep. They are **smoke results**, not qualification results.

## 1. Naive Laplacian wave candidate failed the easiest gate

Command family: parity, state dimension 16, train length 16.

After 1000 updates:

- `diag_signed`: 100% training-length running accuracy;
- `complex_diag`: 100%;
- `GRU`: reached 100% rapidly in the shorter smoke;
- `geom_wave`: stayed near chance (~0.54 running accuracy).

Interpretation: “second-order wave dynamics” by itself is not enough. The first physically tidy kick-drift parameterization is a poor state-tracking inductive bias for parity under this optimizer/budget.

This result is preserved instead of retuning until it passes.

## 2. Local reciprocal scattering changed the result

`geom_scatter` replaces the Laplacian step with products of token-selective local symmetric orthogonal 2-port cells on a ring.

At state dimension 16, train length 16, 300 updates, seed 0:

| task | train-length final acc | 4x final acc | 16x final acc |
|---|---:|---:|---:|
| parity | 1.000 | 0.797 | 0.539 |
| mod3 | 1.000 | 0.008 | 0.070 |
| perm3 | 1.000 | 0.992 | 0.641 |

The lossless local scatterer clearly has enough algebra to learn the non-commutative `S3` task and extrapolates substantially beyond training length in that seed.

Across seeds 0/1/2 on `perm3` (same state and training length, 300 updates), all three reached 1.000 at the training length. At length 64 final-token accuracy was 0.992 / 0.453 / 1.000. This is promising but not stable enough to claim robust length extrapolation.

## 3. Dense Householder control removes the easy interpretation

A two-reflector dense Householder baseline (`householder2`) was added because this algebra is established state-tracking prior art.

On `perm3`, seed 0, state dimension 16, train length 16, 300 updates:

- train-length final accuracy: 1.000
- length 64: 1.000
- length 256: 1.000

It also solved length 64 with state dimensions 4, 6, 8, 10, and 12 in a small seed-0 size sweep. Therefore the first geometric-scatter win over diagonal/complex-diagonal baselines is **not a geometry result**. A generic Householder-product transition is stronger on this gate.

## 4. What remains worth testing

The surviving KYY hypothesis is narrower:

- local geometry may reduce implementation/routing cost;
- overlapping local couplings may offer a useful inductive bias when task structure itself is local/spatial;
- locality may improve perturbation, quantization, or hardware robustness;
- a local scatterer may compile naturally to a reciprocal wave fabric where dense Householders do not;
- topology controls (`ring`, `path`, `matching`, `disconnected`) can test whether connectivity itself matters.

The next important negative control is `matching`: independent 2-state pairs with no overlapping path. If ring/path and matching behave the same, the “geometry” part is doing little.

## 5. Prior art that constrains interpretation

The broad space is crowded. Especially relevant:

- Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues* (ICLR 2025 / arXiv:2411.12537).
- Siems et al., *DeltaProduct: Increasing the Expressivity of DeltaNet Through Products of Householders* (arXiv:2502.10297).
- Terzić et al., *Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models* (arXiv:2509.22284).
- Terzić et al., *Flash PD-SSM* (arXiv:2605.19150).
- Lahoti et al., *Mamba-3* (arXiv:2603.15569).

KYY should be read as an experiment about **local geometric support**, not as a discovery of structured/orthogonal state transitions.
