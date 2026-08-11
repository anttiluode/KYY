# Result — backend-aware exact character design for a resonator substrate

Date: 2026-08-11

This result is a bounded compiler experiment, not a novelty claim for hardware-aware optimization.

## Question

Once an exact finite behavior has many algebraically equivalent cyclic representations, can a concrete analog-resonator backend supply a useful reason to prefer one exact representation over another?

For the second-order resonator companion form audited in `METACIRCUIT_CYCLIC_BACKEND_BOUNDARY.md`, yes.

## Setup

Use `C101` and an eight-character equal-weight positive correlation kernel.

Every selected bank is required to remain faithful:

```text
gcd(101, f1, ..., f8) = 1.
```

Symbolic robustness is measured by the complete-orbit equal-weight minimum score gap

```text
min_{d != 0} sum_i [1-cos(2*pi*f_i*d/101)].
```

The physical backend prices each character by the companion-coordinate quantities derived earlier:

- `cond(T)` for phase/readout transport;
- `||T||` for port gain amplification;
- relative phase sensitivity to `D^-1Y` error;
- positive component-error headroom before the central-difference stability edge.

A simple greedy heuristic chooses characters one at a time to maximize symbolic margin, optionally rejecting candidates whose `cond(T)` exceeds a cap.

This is a reproducible heuristic only; no global-optimum claim is made.

## Result

### unconstrained greedy bank

```text
frequencies = [35,18,25,46,22,16,45,24]
margin = 5.38965
max cond(T) = 7.0976
max ||T|| = 5.0683
max relative phase sensitivity = 7.0976
minimum positive stability headroom = 1.985%
gcd = 1
```

### require every character to satisfy `cond(T) <= 2`

```text
frequencies = [35,18,25,19,15,27,30,29]
margin = 5.21618
max cond(T) = 1.9855
max ||T|| = 1.5719
max relative phase sensitivity = 1.9109
minimum positive stability headroom = 27.386%
gcd = 1
```

Relative tradeoff:

```text
symbolic margin loss:                    3.22%
worst conditioning improvement:          3.57x
worst port-transform norm improvement:   3.22x
relative phase-sensitivity improvement:  3.71x
stability-headroom improvement:         13.80x
```

The composite-modulus unit test also runs the same design logic on `C100` under the condition cap and confirms the resulting bank remains faithful (`gcd=1`).

## Interpretation

This is the first KYY state-code search with a non-arbitrary objective.

Earlier, "optimize the geometry" threatened to become an unconstrained list of unrelated costs.

Here the target substrate itself breaks an equivalence that the finite task leaves free:

```text
finite algebra:
    many faithful character banks are exact legal implementations

resonator backend:
    those same banks require very different physical parameter ranges,
    readout transforms and tolerance budgets.
```

Therefore the compiler can choose among **exactly correct** representations using a physical cost while retaining almost all of the symbolic margin.

In geometric language:

> the quotient/digital behavior fixes the topology of the computation more weakly than the analog substrate fixes its metric realization.

Or more operationally:

> digital equivalence classes contain physically inequivalent embeddings.

That statement is not presented as a new mathematical theorem. Hardware-aware representation selection is a broad established idea. The bounded KYY contribution is the explicit calculation for its exact cyclic character code and this current resonator recurrence family.

## Why this is closer to the original Geometric Neuron question

The interesting issue is no longer whether wave/oscillator geometry *can* compute a finite state machine.

It can, and many theories already say so.

The useful question is:

```text
Given behavior B and physical substrate H,
which exact geometric realization of B is cheapest / most stable in H?
```

That creates a concrete role for a compiler between learned dynamics and physical geometry.

## Files

- `map/metacircuit_frequency_design.py`
- `tests/test_metacircuit_frequency_design.py`
- `.github/workflows/metacircuit-frequency-design.yml`
- `results/metacircuit_frequency_design_summary.json`
