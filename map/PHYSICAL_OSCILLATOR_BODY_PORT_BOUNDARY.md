# Physical oscillator boundary — port calibration cannot repair a broken body relation

Date: 2026-08-11

This note connects the recent KYY compiler work to an actual analog-oscillator hardware result without claiming either oscillator computing or hardware transfer as new.

## External boundary: HORN in analog electronics

Carvalho, Ulmann, Singer & Effenberger, *Analog-electronic implementation of a harmonic oscillator recurrent neural network*, Physical Review Applied 24, 064055 (2025), transfer a digitally trained four-node harmonic oscillator recurrent network to an analog computer.

Their analog dynamics reproduce the digital dynamics reasonably well, but the digital readout used directly on the analog traces produces only 28.39% agreement with the digital model's predictions. Retraining the linear readout on the analog dynamics recovers the classification performance, showing that much of the useful information remains present while the digital/physical interface has shifted.

This is a very close real hardware example of a distinction KYY reached abstractly:

```text
body transfer
is not the same problem as
port transfer.
```

KYY must not claim that observation as new.

The stricter KYY question is when a port-only repair is *mathematically incapable* of restoring the intended recurrent machine because the deployed physical body violates a required algebraic relation.

## Minimal C4 oscillator

Let the symbolic machine be

```text
q -> q+1 mod 4.
```

The exact planar physical generator is a quarter turn

```text
theta = pi/2.
```

Now let the physical oscillator have a tiny static phase error

```text
theta_tilde = pi/2 + delta.
```

After one nominal four-step cycle the symbolic machine has returned to the same state, but the physical body has advanced by

```text
4 delta.
```

Thus the body violates

```text
C^4 = I.
```

## The winding-history sets

The same symbolic state `q` can be reached after

```text
t = q + 4k
```

increments.

Its physical phase is

```text
(q+4k)(pi/2+delta)
= q(pi/2+delta) + 4k delta   mod 2pi.
```

So each symbolic class is not one physical point anymore. It is an orbit under the residual rotation `4 delta`.

If

```text
4 delta / 2pi
```

is irrational, that residual orbit is dense on the circle. Therefore the closure of **every** symbolic class is the full circle, merely with a phase shift.

Consequently no continuous static readout can maintain a strictly positive uniform separation margin among the four classes over arbitrary winding histories.

This is standard irrational-rotation dynamics, not a KYY theorem claim.

The compiler interpretation is important:

> a port can compensate a coordinate/interface mismatch over a bounded operating region, but it cannot make a recurrent body satisfy a relation the body itself violates.

## Finite-horizon experiment

`map/physical_cycle_port_boundary.py` makes the point with the smallest possible model.

Use

```text
n = 4
delta = 0.001 rad / nominal increment
```

so the residual phase after one symbolic C4 cycle is only

```text
0.004 rad.
```

Fit an affine linear readout by least squares on the first 16 winding cycles.

Results:

```text
cycles   calibrated accuracy   min margin    min distance between different classes
16          1.0000              +0.4837       1.3690
64          1.0000              +0.3765       1.2229
256         0.7979              -0.1446       0.5410
1024        0.1995              -1.0003       2.04e-4
4096        0.2416              -1.0003       2.04e-4
```

The calibrated port genuinely works on its calibration horizon and for a while beyond it.

It then fails because the physical sets associated with different symbolic states themselves become interleaved.

The exact legalized quarter-turn body remains

```text
accuracy = 1.0
minimum margin ~= 1.0
```

at every tested horizon.

## What this says about hardware compilation

There are now three distinct deployment repairs:

### 1. port calibration

```text
physical body approximately preserves useful dynamics
        +
retrain / transport readout
```

This is sufficient when the task only needs the readout to work over the encountered physical trajectory distribution.

The analog HORN result demonstrates that this can be highly effective in practice.

### 2. body legalization

```text
physical/recurrent operator violates required relation
        ->
change/tune/synthesize body operator so the relation itself is restored.
```

This is required when the task contract depends on exact recurrent identities such as finite order, exact reset kernels, or other group/semigroup relations over arbitrary histories.

### 3. periodic relocking / runtime correction

If physical noise prevents the body from remaining on the exact legal geometry even after nominal legalization, periodically project/relock the physical state.

That adds a digital correction bandwidth/cost which the compiler should price separately.

## Why this makes the physical backend more plausible

The KYY compiler residual has struggled in pure software because a DFA or an explicit hybrid state often provides a simpler exact implementation.

Analog oscillator hardware changes the cost model.

The deployed object may naturally already be:

- a set of oscillators;
- a resonant network;
- a wave/phase state;
- a continuous transient physical computation.

Then replacing the whole thing by a DFA is no longer the same deployment problem.

A useful compiler could instead ask:

```text
which trained oscillator relations must be hardware-tuned exactly?
which discrepancies can be absorbed by a transported port?
which residual physical noise requires periodic digital relocking?
what margin / precision budget follows from each choice?
```

That is a concrete hardware/software co-design question.

It is not yet a demonstrated KYY advantage on real hardware.

## Files

- `map/physical_cycle_port_boundary.py`
- `tests/test_physical_cycle_port_boundary.py`
- `.github/workflows/physical-cycle-port-boundary.yml`
- `results/physical_cycle_port_boundary_summary.json`
