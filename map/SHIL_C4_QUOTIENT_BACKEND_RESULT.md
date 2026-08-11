# Result — a C4 irreversible quotient lowered to a multi-phase SHIL potential

Date: 2026-08-11

This is a physical-backend lowering experiment, **not** a novelty claim for phase-based computing, oscillator memories, Potts machines, SHIL, or phase-shifted SHIL stages.

The 2025 multi-stage CMOS ring-oscillator Potts-machine literature already uses phase-shifted subharmonic injection locking to make oscillator phase act as multivalued memory/computation and to move between staged phase representations.

The KYY question here is narrower:

> Given a declared irreversible symbolic transition, can the compiler solve for a known phase-locking primitive and choose its geometry so the transition composes robustly with the next physical stage?

For the Pass-44 C4 merge, yes.

---

## 1. Physical state space

Use the phase-reduced locking dynamics

```text
d phi = -K sin(N(phi-alpha)) dt + sqrt(2D) dW.
```

For `N=4, alpha=0`, the continuous oscillator circle contains four attracting wells

```text
0, pi/2, pi, 3pi/2.
```

Their basin labels implement a four-valued digital state.

So this hardware primitive realizes the digital/analog border literally:

```text
analog substrate: continuous phase phi

digital state: which attracting phase basin contains phi
```

The locking potential continuously corrects small phase errors rather than requiring a software nearest-state snap.

---

## 2. Desired irreversible transition

Compile the Pass-44 partial merge

```text
0 -> 0
1 -> 0
2 -> 2
3 -> 2.
```

A natural physical operation is:

1. temporarily replace the four-well landscape by a two-well landscape;
2. let phase relax so each desired pair flows into one common attractor;
3. restore four-well locking.

Let the temporary `N=2` attractors be

```text
alpha
alpha + pi.
```

For `0 < alpha < pi/4`, states `0,1` lie in the first two-well basin and states `2,3` lie in the second.

So many `alpha` values implement the **two-well quotient itself**.

But that is not enough.

The merged attractors must also lie safely inside the intended basins after four-well locking is restored.

---

## 3. Why the obvious midpoint is wrong

The most symmetric two-well choice is

```text
alpha = pi/4.
```

It places the temporary attractors at the exact midpoints of each source pair.

That gives excellent pair capture.

But `pi/4` and `5pi/4` are exactly the **separatrices** between the destination C4 wells.

Therefore the physical pipeline

```text
C4 state
  -> N=2 midpoint merge
  -> restore C4
```

has zero re-entry margin.

Infinitesimal noise chooses which fine C4 basin receives the merged state.

In the simulation, even at zero diffusion the finite relaxation lands symmetrically and the midpoint construction produces only

```text
50% correct final quotient state.
```

With phase diffusion it stays approximately 50%.

So:

> A physical stage can implement the local quotient correctly while placing its output at a geometrically invalid interface for the next stage.

That is exactly the kind of composition failure a backend compiler must audit.

---

## 4. Compile the temporary phase shift

For the first source pair, the relevant angular margins are:

```text
capture q=1 into the alpha basin:       alpha
return merged alpha to C4 state 0:      pi/4 - alpha
```

The other source/input margin is larger over the feasible interval and does not limit the optimum.

Therefore the worst physical composition margin is

```text
m(alpha) = min(alpha, pi/4-alpha).
```

Maximizing it gives

```text
alpha* = pi/8 = 22.5 degrees
m*     = pi/8.
```

The grid search independently returns

```text
alpha = 0.3926990816987242
margin = 0.3926990816987241,
```

i.e. numerical `pi/8`.

This is not a new optimization theorem. It is the exact backend parameter implied by this symbolic transition and this known SHIL geometry.

---

## 5. Noisy phase simulation

The probe uses Euler-Maruyama dynamics with

```text
K = 4
dt = .002
N=2 merge relaxation = .5
N=4 re-lock relaxation = .5
4000 trajectories per source state
```

for 16000 phase trajectories per point.

Compare four temporary attractor offsets:

```text
alpha = pi/16
alpha = pi/8      <- compiled optimum
alpha = 3pi/16
alpha = pi/4      <- symmetric midpoint / zero re-entry margin
```

Results:

```text
diffusion D      pi/16       pi/8       3pi/16      pi/4
0                1.000000    1.000000    1.000000    0.500000
.002             1.000000    1.000000    1.000000    0.498688
.005             1.000000    1.000000    1.000000    0.499875
.010             1.000000    1.000000    0.999938    0.498000
.020             0.999875    1.000000    0.998375    0.497875
```

At the strongest tested diffusion, the compiler-selected `pi/8` phase remains `100%` correct in this simulation while the locally symmetric midpoint remains a coin flip.

This should not be overread: the phase SDE is a reduced model, not transistor-level CMOS simulation.

---

## 6. What is actually interesting

The result is **not** that phase locking discretizes oscillators. That is prior art.

The result is **not** that phase-shifted SHIL can create multiple staged phase representations. That is also prior art.

The useful KYY-shaped statement is:

```text
symbolic transition contract
       +
known physical phase-locking primitive
       ->
solve feasible attractor landscape
       ->
audit interface to following physical stage
       ->
maximize basin composition margin.
```

The transition is irreversible because two distinct fine phase basins relax into one coarse basin.

In geometric language:

> **a digital quotient is implemented by temporarily changing the topology of the analog attractor landscape.**

And the compiler cannot optimize the quotient stage in isolation; it must also price where that coarse attractor sits relative to the next fine-state separatrix.

This is a much better physical match to Pass 44 than the metacircuit energy-classifier backend.

---

## 7. Prior-art boundary

The closest current collision is the 2025 *A Multi-Stage Potts Machine based on Coupled CMOS Ring Oscillators*.

That work already:

- uses oscillator phase as multivalued state;
- uses phase-shifted SHIL signals;
- alternates SHIL stages;
- treats locked oscillator phase as compute and memory;
- implements four-phase state using staged binary phase locks;
- provides explicit phase-readout circuitry.

Therefore KYY must not advertise the physical primitive or staged-phase idea as new.

The residual, if any, would be in a compiler that starts from an arbitrary **behavioral transition algebra**—including irreversible maps—and automatically synthesizes/audits the corresponding phase-locking stages and their robustness margins.

That is not established as novel here.

---

## 8. Next falsifier

The C4 merge is unusually friendly.

The next useful test is to generalize the compiler to equal-block cyclic quotients

```text
C_n -> C_m,
where m divides n,
```

and ask:

- which quotient partitions are realizable by one `m`-well SHIL stage plus return to `n` wells;
- where the temporary attractors must be placed;
- how the optimum margin scales with block size;
- when a desired symbolic quotient requires multiple physical stages rather than one;
- whether the resulting rule is already explicitly present in the Potts/phase-logic literature.

Only after that should this be taken toward transistor-level ring-oscillator simulation.

## Files

- `map/shil_c4_quotient_backend.py`
- `tests/test_shil_c4_quotient_backend.py`
- `.github/workflows/shil-c4-quotient-backend.yml`
- `results/shil_c4_quotient_backend_summary.json`

Workflow evidence: Actions run `31458851560`, focused tests green.
