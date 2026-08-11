# Boundary — the metacircuit energy readout erases exact cyclic phase state

Date: 2026-08-11

This is a compatibility boundary, not a novelty claim. Full-period energy invariance under time/phase translation is elementary.

## Why this check was necessary

After correcting the metacircuit from the finite-difference surrogate to the continuous physical resonator law, one apparent KYY cost still looked strong: the conditioning of the sampled displacement/lag state `[u_t,u_{t-1}]` when transported to phase coordinates.

The actual 2026 metacircuit hardware, however, does not deploy a generic KYY-style linear terminal-state decoder over that sampled pair.

Its demonstrated hardware measures voltage responses from selected output oscillators, rectifies them, accumulates output energy over the signal duration, and chooses the output channel with the largest integrated energy.

So the relevant question is not merely whether the body can host cyclic phase.

It is:

> does the physical output contract preserve the phase variable that KYY uses as its exact finite state?

For a complete cyclic observation window, no.

---

## Exact phase-shift invariance

Let the exact cyclic state `q` be represented by a phase-shifted `n`-periodic trajectory

```text
h_q(t) = h_0(t+q).
```

For any fixed physical output projection `W`, define complete-period energy

```text
E(q) = sum_{t=0}^{n-1} |W h_q(t)|^2.
```

Then

```text
E(q)
= sum_t |W h_0(t+q)|^2
= sum_t |W h_0(t)|^2,
```

because addition by `q` only permutes the complete set of period indices.

Therefore

```text
E(q) = constant for all q.
```

This does not depend on the chosen projection `W`.

The same argument applies to a multi-character bank as long as the combined state is `n`-periodic and `q` acts by common time/phase translation.

---

## Numerical check on the corrected C101 bank

Use

```text
frequencies = [16,18,19,20,25,28,30,31].
```

An exact current-state prototype decoder separates all 101 states:

```text
instantaneous prototype accuracy = 1.000
```

For a random fixed scalar output projection, complete-period integrated energy is

```text
minimum = 465.0243183618523
maximum = 465.0243183618543
spread  = 1.99e-12
```

The residual is floating-point noise.

A truncated window remains phase-sensitive:

```text
window 1 spread:   25.15
window 8 spread:   89.13
window 16 spread: 100.82
window 101 spread: ~2e-12
```

So energy can carry phase information over a chosen incomplete observation window, but then the output depends on observation timing/window length rather than representing a time-translation-invariant terminal finite state.

---

## Consequence for KYY

The 2026 metacircuit remains important evidence that coupled electrical resonators can implement useful physical recurrent computation.

But its demonstrated **output contract** is mismatched to the current KYY cyclic-state task.

In KYY language:

```text
body:
    potentially capable of hosting phase / recurrent geometry

physical demonstrated port:
    integrated energy
    -> intentionally quotients away complete-cycle phase translation
```

Therefore it should no longer be called a direct KYY target backend without qualification.

A KYY-compatible oscillator backend would need one of:

- phase-sensitive instantaneous voltage readout;
- quadrature/IQ readout;
- voltage + conjugate/current/derivative state access;
- a phase comparator / phase detector;
- incomplete-window timing explicitly made part of the task contract;
- another physical port that preserves the required finite phase label.

This also weakens the earlier sampled `cond(T)` story as a direct metacircuit hardware cost: `cond(T)` prices the discrete displacement/lag coordinate interface, while the fabricated metacircuit uses a different voltage-energy output path.

The conditioning calculation remains mathematically correct for that sampled interface; it is not established as a cost paid by the paper's physical classifier.

---

## Current verdict on this backend

```text
resonant body relevance:       yes
exact direct C_n backend:      not established
physical energy port match:    no for full-period phase state
useful hardware neighbor:      yes
main-branch promotion:         no
```

The next backend search should require **phase-sensitive state access** from the beginning rather than modifying an energy classifier after the fact.

## Files

- `map/metacircuit_energy_readout_boundary.py`
- `tests/test_metacircuit_energy_readout_boundary.py`
- `.github/workflows/metacircuit-energy-readout-boundary.yml`
- `results/metacircuit_energy_readout_boundary_summary.json`

Workflow evidence: Actions run `31458661433`, focused tests green.
