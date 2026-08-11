# Result — bounded resonator tolerance becomes a relock certificate

Date: 2026-08-11

This is a backend certificate for the normalized phase-mode abstraction. It is not a transistor-level interval proof and not a novelty claim for robust control or interval analysis.

## Question

The earlier noisy-fiber experiment found a diffusion-like correction frontier:

```text
dynamic zero-mean step noise -> state error ~ sigma sqrt(T)
                            -> safe free-run T roughly ~ 1/sigma^2
```

Static resonator fabrication error is different. A fixed error in the resonator ratio changes the phase increment itself, so phase error accumulates coherently:

```text
static ratio bias -> phase error ~ eta T
                  -> expected safe free-run T roughly ~ 1/eta.
```

Can the exact C101 character geometry turn a bounded component tolerance into a conservative symbolic-separation/relock certificate?

Yes, in the normalized independent phase-mode model.

---

## Certificate

For one canonical character with ideal

```text
theta = 2*pi*f/n
lambda = D^-1 Y = 2(1-cos(theta)),
```

assume bounded multiplicative hardware mismatch

```text
lambda_actual = lambda (1 + eps),
|eps| <= eta.
```

Because canonical characters use `0 < theta < pi`, the actual physical phase increment is

```text
phi(eps) = acos(1 - lambda(1+eps)/2)
```

and the largest one-step phase defect occurs at a tolerance endpoint.

For a history containing at most `cycles` complete symbolic windings,

```text
|e_i| <= (n*cycles - 1) * max|phi_i-theta_i|.
```

For competitor displacement `d`, the one-mode positive-kernel score gap under accumulated phase error `e` is

```text
cos(e) - cos(e-Delta)
= -2 sin(Delta/2) sin(e-Delta/2),

Delta = 2*pi*f*d/n.
```

The script minimizes that scalar expression exactly over each allowed phase-error interval, sums the independent worst cases across modes, then minimizes over every nonzero competitor displacement.

A positive result certifies that the ideal phase-domain positive-kernel port still separates all C101 states for **every** static ratio-error vector in the specified box and every history up to that winding horizon.

The certificate is conservative because it allows each modal error to choose its worst sign independently for every competitor.

---

## Result

Compare the same two exact C101 eight-character banks.

```text
ratio tolerance eta    unconstrained certified cycles    conditioned certified cycles    gain
2e-6                         1309                              3281               2.506x
5e-6                          523                              1312               2.509x
1e-5                          261                               656               2.513x
2e-5                          130                               328               2.523x
5e-5                           52                               131               2.519x
1e-4                           26                                65               2.500x
```

The backend-conditioned exact representation therefore buys about **2.5x more certified analog free-running time** across this tolerance range.

The worst one-step phase-error bound at `eta=1e-5` is

```text
unconstrained: 7.098e-5 rad
conditioned:   1.911e-5 rad
```

which is about 3.71x smaller, matching the earlier relative phase-sensitivity metric.

The multi-mode robust symbolic horizon improves by about 2.5x rather than the full 3.71x because the certificate also depends on how the chosen character set distributes symbolic score margin across competitors.

---

## The scaling law is the important part

Doubling static tolerance almost exactly halves the certified free-running horizon:

```text
conditioned:
eta=1e-5 -> 656 cycles
eta=2e-5 -> 328 cycles

unconstrained:
eta=1e-5 -> 261 cycles
eta=2e-5 -> 130 cycles
```

This is the expected coherent-drift law:

```text
T_static ~ 1/eta.
```

That contrasts with the earlier additive per-step noise experiment, whose empirical correction frontier was diffusion-like:

```text
T_dynamic ~ 1/sigma^2.
```

So a physical compiler should not have one generic "robustness" number.

It should distinguish at least:

```text
static relation bias      -> coherent drift clock
zero-mean dynamic noise   -> diffusion clock
interface/basis mismatch  -> port calibration problem
unrelated loss/Q errors   -> separate body objective
```

Each mechanism implies a different correction strategy and a different relock budget.

---

## A useful example at eta=5e-5

At 64 winding cycles, the conservative robust positive-kernel margin is

```text
unconstrained: -0.926   -> no guarantee
conditioned:   +2.837   -> still certified
```

This is not a Monte-Carlo average. It is an interval worst-case statement inside the phase abstraction.

At `eta=1e-4`, 64 cycles sits almost exactly at the conditioned boundary:

```text
unconstrained margin at 64: -3.429
conditioned margin at 64:   +0.135

certified horizons:
26 vs 65 cycles.
```

---

## Why this is useful even if generic hardware optimization wins

Shem and hardware-aware physical-neural-network training already establish that analog systems can be optimized around nonidealities.

KYY does not need to replace those optimizers.

The possible role is now clearer:

```text
behavioral algebra
    -> exact representation family
    -> backend physical sensitivities
    -> bounded tolerance certificate
    -> required relock / trim / calibration contract
```

A generic optimizer can then work **inside** that contract.

The algebra tells it what must remain invariant; the physical backend tells it what costs money; the certificate tells it when a nominally exact analog realization is no longer guaranteed to behave as the same digital machine.

That is a more defensible role than claiming a new analog optimizer.

---

## What is still missing

The certificate currently assumes:

- independent character modes;
- normalized phase coordinates after invertible coordinate calibration;
- bounded static `D^-1Y` mismatch;
- the positive-kernel port;
- no cross-mode parasitic coupling;
- no nonlinear saturation or transistor-level effects.

A real metacircuit backend would need to extend the uncertainty model to measured component/coupling distributions and parasitics.

The next physical step should therefore be **not another abstract automaton test**. It should be a SPICE- or circuit-level small resonator block, or a faithful reduced circuit model, and ask whether the algebraic tolerance prediction remains useful after real parasitics enter.

## Files

- `map/metacircuit_tolerance_certificate.py`
- `tests/test_metacircuit_tolerance_certificate.py`
- `.github/workflows/metacircuit-tolerance-certificate.yml`
- `results/metacircuit_tolerance_certificate_summary.json`

Workflow evidence: Actions run `31458004674`, focused tests green.
