# Correction — discrete companion surrogate versus continuous metacircuit physics

Date: 2026-08-11

This note corrects the physical interpretation of the first KYY metacircuit backend passes.

## The correction

The 2026 *Fully Analog Resonant Recurrent Neural Network via Metacircuit* paper contains two related but different dynamical descriptions.

Its electrical resonator obeys a **continuous-time** second-order law of the form

```text
D u'' + Y u = input.
```

The paper then obtains a recurrent neural-network update by finite-difference discretization:

```text
u[t+1] = (2I - dt^2 D^-1 Y) u[t] - u[t-1] + ...
```

The first KYY metacircuit pass treated that discrete companion recurrence as though its finite-difference parameter relation were the exact free-running physical law of the analog circuit.

That was too strong.

For the **discrete central-difference recurrence**, an exact sampled character with phase step `theta` satisfies

```text
D^-1 Y = 2(1-cos(theta))/dt^2.
```

For the **continuous physical resonator**

```text
u'' + (D^-1Y) u = 0,
```

the natural angular frequency is

```text
omega = sqrt(D^-1Y),
```

so exact phase advance `theta` over sampling interval `dt` requires

```text
D^-1 Y = (theta/dt)^2.
```

These are equal only in the small-angle limit.

---

## What must be withdrawn as physical-circuit claims

The following earlier observations remain valid for the **discrete finite-difference companion backend**, but must not be presented as properties of the free-running continuous analog metacircuit:

- the apparent numerical stability ceiling at `dt^2 D^-1Y = 4`;
- the extremely small positive "component-error headroom" for characters near Nyquist derived from that ceiling;
- the ~13.8x headroom improvement of the first `cond(T)<=2` character bank;
- the ~12x relation-defect reduction under ratio error in the discrete surrogate;
- the ~2.5x bounded static-tolerance free-running certificate in that discrete surrogate.

Those numbers describe the central-difference recurrence as a numerical implementation. They are **not analog-circuit tolerance results**.

The old files are retained because the discrete recurrence remains a legitimate backend in its own right, but they should now be read through this correction.

---

## What survives physically

### 1. Exact sampled cyclic modes still lower to continuous resonators

For canonical character frequency `f` of `C_n`, let

```text
theta = 2*pi*f/n.
```

A continuous free resonator can realize that exact sampled phase with

```text
D^-1Y = (theta/dt)^2.
```

Sampling its displacement at interval `dt` produces the same exact second-order companion relation

```text
u[t+1] = 2 cos(theta) u[t] - u[t-1].
```

So the algebra-to-resonator connection survives; the physical parameter map changes.

### 2. Port-coordinate conditioning still depends strongly on character frequency

The displacement/lag state `[u_t,u_{t-1}]` is related to phase/quadrature coordinates through a transform containing `1/sin(theta)`.

Therefore characters near sampled phase `0` or `pi` remain badly conditioned for port transport even though the continuous resonator itself has no finite-difference `lambda=4` stability cliff.

This is a genuine sampled-coordinate/interface effect.

### 3. Continuous component-to-phase sensitivity is modest and simple

Because

```text
theta = dt sqrt(lambda),
lambda = D^-1Y,
```

the relative phase sensitivity is

```text
lambda * dtheta/dlambda = theta/2.
```

Higher sampled phase characters are therefore more sensitive to relative resonator-ratio error, but the divergence predicted by the discrete stability edge disappears.

---

## Corrected continuous results for the first two banks

The earlier banks were re-evaluated under the continuous physical law with a fresh linear readout calibrated separately for every perturbed device at 16 winding cycles.

### `sigma = 1e-5` relative `D^-1Y` spread, L1024

```text
                         mean accuracy   worst device   mean max relation defect
unconstrained                0.96545        0.76660          0.0025100
conditioned                  0.99008        0.94531          0.0020833
```

The conditioned bank still helps, but the relation-defect improvement is only about `1.20x`, not `12x`.

At larger mismatch:

```text
sigma        unconstrained mean      conditioned mean
2e-5              0.65024               0.68127
5e-5              0.27098               0.28710
1e-4              0.13947               0.16619
```

### bounded static tolerance certificate

```text
eta          unconstrained cycles      conditioned cycles
5e-6                1418                    1800
1e-5                 709                     900
2e-5                 354                     450
5e-5                 141                     180
1e-4                  70                      90
```

This is about a `1.27x` improvement for these two banks.

Static mismatch still behaves as coherent phase drift, with certified horizon approximately proportional to `1/eta`.

See:

- `map/metacircuit_continuous_backend.py`
- `tests/test_metacircuit_continuous_backend.py`
- `results/metacircuit_continuous_backend_summary.json`
- Actions run `31458208189`

---

## Stronger exact-bank baseline removes the remaining easy robustness win

A second corrected search compared:

### physically constrained exact optimum

Exhaustive optimum inside the declared candidate set

```text
cond(T) <= 2
frequency <= 31
8 characters
```

is

```text
[16,18,19,20,25,28,30,31]

symbolic margin            5.712879
max cond(T)                1.84063
max port-transform norm    1.48120
max relative sensitivity   0.96425
```

### strong digital-only exact heuristic

A deterministic `100000`-sample + `100` local-restart search found

```text
[3,5,12,21,22,23,37,50]

symbolic margin            5.888437
max cond(T)               64.2934
max port-transform norm   45.4678
max relative sensitivity   1.55524
```

The physically constrained bank gives up only `2.98%` symbolic margin while improving sampled port conditioning enormously.

But that does **not** translate into a robust-state-tracking win against this stronger digital exact baseline.

At bounded `eta=1e-5`:

```text
certified cycles:
physical constrained = 990
digital-only heuristic = 990
```

After per-device 16-cycle calibration:

```text
sigma     physical constrained    digital-only heuristic
1e-5           0.97705                  0.97477
2e-5           0.69088                  0.73256
5e-5           0.30353                  0.42783
```

The physically constrained bank has lower relation defect at every tested mismatch, but at moderate/high mismatch the stronger symbolic margin of the digital-only bank wins on classification.

Therefore:

> **KYY has not shown that simple backend conditioning improves robust state tracking once the exact digital representation baseline is optimized strongly.**

What it has shown is more limited and cleaner:

> **Exact digital representations with nearly equal symbolic quality can have radically different sampled port/interface conditioning and physical parameter ranges.**

Whether that difference is worth optimizing depends on the actual hardware/interface cost model.

See:

- `map/metacircuit_continuous_design.py`
- `tests/test_metacircuit_continuous_design.py`
- `results/metacircuit_continuous_design_summary.json`
- Actions run `31458392064`

---

## Current interpretation

The metacircuit work no longer supports the simple story

```text
physical conditioning constraint -> universally more robust exact machine.
```

It supports the narrower statement

```text
finite behavioral algebra
    leaves many exact representations available

sampled analog interface/backend
    prices those representations differently

but
    symbolic margin and physical conditioning trade against one another,
    and a strong digital exact representation can be just as or more robust.
```

That is a useful compiler problem, but not yet a KYY win.

The next physical step should use measured or circuit-level costs that cannot be replaced by a hand-chosen `cond(T)` cap: ADC/amplifier gain range, resistor/FDNR implementation range, parasitic coupling, saturation, power, or measured relock/calibration cost.

Until then, keep the metacircuit line on the research branch and do not promote the earlier surrogate numbers to `main`.
