# Pass 32 — phase-timed ports from the spiral/chandelier detour

Date: 2026-08-10

This pass began from two biology papers supplied as a deliberately speculative architectural prompt:

1. Z. Ye et al., *Brain-wide topographic coordination of traveling spiral waves* (bioRxiv; supplied version posted 2025-03-11).
2. Y. Qi et al., *Specific and Plastic: Chandelier Cell-to-Axon Initial Segment Connections in Shaping Functional Cortical Network*, Neuroscience Bulletin 40, 1774–1788 (2024).

The biology is **not** evidence that the brain implements KYY.

The useful conjunction is narrower:

```text
distributed oscillatory phase structure
        +
specialized output boundary
        +
phase-dependent output gating
        +
output-boundary plasticity
```

KYY had just independently reached a compiler failure mode in which the recurrent operator could be made algebraically exact while the inherited observable port/readout remained wrong.

That made one tiny transfer from the biology worth testing:

> **after operator legalization, can changing only when the harmonic state is read repair port compatibility?**

The answer on the existing C_101 stress setup is: **sometimes, substantially, and not universally.**

---

## 1. What the supplied papers actually support

### Spiral-wave paper

The supplied paper reports 2–8 Hz cortical traveling waves with spiral organization. The spirals have approximately constant angular speed across radius (reported mean equivalent to about 5.1 Hz) while linear speed rises with radius. Local sensory-cortex axonal terminal orientation matches the circular propagation geometry, and a circularly biased coupled-oscillator simulation strengthens spiral dynamics relative to isotropic connectivity.

In discussion, the authors explicitly propose that spiral waves *may* act as spatiotemporal clocks coordinating sequential sensation/action, and note prior evidence that momentary oscillatory phase modulates perceptual detectability.

These are hypotheses about brain function, not established clock circuitry.

### Chandelier/AIS review

The review emphasizes that the axon initial segment (AIS) is the spike-initiation/output boundary of projection neurons and receives highly specific GABAergic input from chandelier cells (ChCs). Individual ChCs can innervate hundreds of projection-neuron AISs. The ChC-AIS system and AIS itself are plastic.

The review also summarizes in-vivo observations that CA3 ChCs fire strongly and rhythmically around theta peaks, while a medial-septal inhibitory population can rhythmically silence them around theta troughs, disinhibiting projection neurons. ChCs can therefore function as phase-dependent output gatekeepers in those hippocampal conditions.

### What is **not** established

The two papers do **not** show that cortical spiral phase drives chandelier cells, that ChCs decode spiral waves, or that the AIS performs a KYY-like compiler operation.

That bridge is only an architectural suggestion.

---

## 2. Prior-art subtraction

The general ideas are heavily occupied.

- Nanda et al. (2023), *Progress measures for grokking via mechanistic interpretability*, reverse-engineered a modular-addition Transformer that uses Fourier components and circle rotations (the "clock" algorithm).
- Zhong et al. (2023), *The Clock and the Pizza*, showed that modular-addition networks need not converge on one unique clock algorithm.
- Oscillatory/phase-code literature long predates KYY. Jensen (2001), for example, explicitly proposed that changing decoder theta phase can select qualitatively different information from a phase-coded hippocampal signal.
- The 1990s Omlin/Giles line already studied extraction, construction and stability of finite-state behavior in recurrent neural networks.

Therefore KYY does **not** claim:

```text
phase codes
phase-dependent decoding
Fourier/clock algorithms for modular arithmetic
finite-state extraction from RNNs
oscillations as clocks
```

The residual experiment is a compiler question:

```text
learn approximate recurrent operator
        -> snap operator to exact task algebra
        -> keep decoder weights frozen
        -> change only one physically meaningful timing coordinate
        -> exhaustively verify all legal observable states
```

---

## 3. One scalar phase/timing compiler knob

New code:

- `map/phase_readout_legalization_probe.py`
- `tests/test_phase_readout_legalization_probe.py`
- `.github/workflows/harmonic-phase-readout.yml`

Archived summary:

- `results/harmonic_n101_phase_readout_summary.csv`

The trained setup is unchanged from Passes 30–31:

```text
C_101
8 complex modes / 16 real coordinates
train length 16
1500 steps
random initial symbolic state
small later increments {0,1,2,3,4}
10 seeds
```

After training, learned angles are snapped to the nearest exact C_101 characters.

No recurrent parameter is subsequently changed.
No readout weight or bias is subsequently changed.

The only new degree of freedom is a scalar fractional observation time `tau`.

For a legalized generator angle `theta_i`, sampling at `s + tau` gives

```text
z_i(s,tau) = [cos((s+tau) theta_i), sin((s+tau) theta_i)].
```

Because the modes are harmonics of the same cyclic generator, one common time shift produces the appropriate frequency-dependent phase shift automatically. This is just the Fourier/harmonic time-shift rule, not new mathematics.

The search was restricted to

```text
tau in [-0.5, 0.5]
```

on a 2001-point grid.

At every candidate `tau`, all 101 legal symbolic states were checked directly with the frozen linear readout.

Selection criterion:

1. maximize exhaustive legal-orbit accuracy;
2. among ties, maximize minimum true-class margin.

---

## 4. Result

```text
seed  baseline correct  best correct  best tau   base min margin  best min margin
0        101 / 101       101 / 101    -0.5000       +0.0478          +0.1880
1         83 / 101        85 / 101    -0.0350       -1.2665          -1.2647
2        101 / 101       101 / 101    +0.3660       +0.2767          +1.2967
3        101 / 101       101 / 101    -0.1490       +1.3740          +1.8624
4        101 / 101       101 / 101    -0.0695       +0.8143          +1.4584
5         84 / 101       101 / 101    +0.2685       -1.0148          +0.0193
6         99 / 101       101 / 101    -0.2625       -0.0874          +1.1136
7         47 / 101        52 / 101    -0.2675       -2.6713          -2.2505
8        101 / 101       101 / 101    -0.1800       +0.5575          +0.6363
9         76 / 101       100 / 101    +0.2890       -1.7833          -0.0695
```

Thus the number of exact exhaustive port-compatible legalizations changes from

```text
5 / 10 at tau = 0
```

to

```text
7 / 10 after one-scalar phase search.
```

The new fully rescued seeds are 5 and 6.

Seed 9 is a useful near miss: a timing shift repairs 24 of its 25 incorrect legal states, reaching 100/101, but does not make the machine exact.

Seeds 1 and 7 remain substantially incompatible.

---

## 5. Timing also buys margin when no repair is needed

All five already-correct baseline legalizations retain 101/101 correctness at their selected timing phase.

Their worst-state readout margins increase:

```text
seed 0   +0.0478 -> +0.1880
seed 2   +0.2767 -> +1.2967
seed 3   +1.3740 -> +1.8624
seed 4   +0.8143 -> +1.4584
seed 8   +0.5575 -> +0.6363
```

So the timing parameter is not merely a rescue trick.

Within this toy representation it can be used as a **post-legalization port-margin optimization** while leaving the exact recurrent algebra untouched.

That could matter before finite-precision/hardware deployment, where Pass 27 showed that geometric margin and phase error interact.

---

## 6. What this earns

It does **not** earn:

> chandelier cells are the biological KYY readout.

It does earn a compiler separation that is now experimentally useful:

```text
OPERATOR LEGALIZATION
    enforce task algebra

PORT LEGALIZATION
    verify observable behavior

PORT TIMING
    choose when to observe the legal harmonic orbit

DEPLOYMENT MARGIN
    price phase/precision error after the port is correct
```

One scalar timing coordinate repaired 2/5 previously incompatible legalizations and nearly repaired a third.

That is large enough to retain as a compiler knob and small enough not to pretend it solves port repair generally.

---

## 7. Link back to the supplied biology, carefully

The useful analogy is now structural rather than decorative:

```text
spiral / distributed phase field
    -> timing scaffold

AIS
    -> specialized output boundary

ChC rhythmic gating
    -> output can depend on phase without changing the upstream state body

AIS / ChC-AIS plasticity
    -> output boundary can adapt on a different timescale from the propagating body
```

The papers separately support those biological ingredients. Their conjunction as one KYY-like computation is **not demonstrated by the papers**.

The correct use of the biology here was therefore not to claim a neural mechanism. It was to suggest a constrained parameter that survived a direct falsification test.

---

## 8. The more surprising conceptual inversion

The motivating intuition used the phrase "high-frequency thinking" for difficult/stressful thought.

The supplied spiral paper actually studies a relatively slow 2–8 Hz phase organization, and the chandelier timing example is theta-scale. The architectural lesson is almost the inverse of "compute faster":

> **a slower phase scaffold may coordinate when faster local events are allowed to matter.**

KYY translation:

```text
not more state transitions
but fewer / better-timed observations of a structured state
```

This is a design intuition only. It is not a claim about subjective cognitive stress or literal neural frequency of hard thought.

---

## 9. Next falsifier

Do not add a more elaborate biological gate yet.

The correct comparison is now port-repair cost on the five baseline failures:

```text
A. zero-parameter inherited port
B. one scalar timing phase
C. timing phase + minimal bias/low-rank port correction
D. full linear-readout recalibration
```

Measure:

```text
number of corrected legal states
minimum legal-state margin
number of calibrated parameters
number of labeled legal states required for calibration
robustness to implementation phase error
```

If a one- or few-parameter port repair consistently approaches full readout recalibration, that is a meaningful compiler result.

If full readout retraining is always required, keep the phase result as the small special case it is.

Only after that should KYY move the legalization machinery to a genuinely non-Abelian representation where exhaustive state enumeration starts to become expensive and relation-defect diagnostics earn their scaling role.
