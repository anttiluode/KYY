# KYY current frontier

Updated: 2026-08-11

This is the short map of the research branch after the metacircuit correction and the phase-locking backend passes.

## One-sentence status

KYY has **not** discovered new automata theory, hybrid systems, oscillator computing, SHIL/Potts hardware, harmonic generation, or generic analog optimization.

The surviving compiler question is now much more concrete:

> **Given an exact behavioral transition, what distinctions must its physical body preserve or destroy, and which target-substrate instruction has the right kernel geometry to realize that transition?**

That remains a research direction, not a novelty declaration.

`main` remains untouched.

---

# 1. What the long subtraction left behind

The branch has already collided with mature work in:

- finite automata, state equivalence, synchronization and transformation semigroups;
- RNN automata extraction, state quantization/reification and repair;
- observability, reversible-RNN forgetting limits and unlearning audits;
- group representations, equivariant ports and Koopman realizations;
- hybrid automata, quotient/gluing resets and tangent/nullspace geometry;
- hardware-aware compilation and analog optimization;
- physical wave/oscillator RNNs;
- multi-phase SHIL/Potts oscillator computing.

The rule is unchanged:

> **Do not rename prior art and call it ours.**

---

# 2. The durable software result: digital quotient versus analog fiber

Pass 46 replaced four isolated digital points with four continuous legal fibers

```text
F_q = { c_q + a v : a in R }.
```

`q` is digital rail identity; `a` is an analog payload.

At L1024:

```text
learned soft:
    digital mean accuracy ~0.9624
    analog RMSE          ~0.00487

nearest-fiber reification:
    digital exact 10/10
    analog RMSE ~0.00474

exact point+tangent compile:
    digital exact 10/10
    analog RMSE ~3.86e-8
```

The paired-history audit is the clean statement:

```text
projection can put two histories on the same legal rail
while leaving a history-dependent displacement along that rail.
```

So

```text
projection onto the legal manifold
!=
correct dynamics along the legal manifold.
```

Geometrically:

```text
DIGITAL = quotient/equivalence directions
ANALOG  = tangent/fiber directions.
```

This is classical hybrid geometry used as a post-training compiler audit, not a new hybrid-systems theorem.

See `PASS46_RESULT_DIGITAL_QUOTIENT_ANALOG_FIBER.md`.

---

# 3. Exactness has a location

The branch now distinguishes four deployment contracts:

```text
state label
    explicit digital Q x R hybrid

body/operator
    exact finite-order relation, kernel or tangent action

port/interface
    calibrated/transported observable map

runtime relock
    periodic return to a legal attractor/basin.
```

They are not interchangeable.

Pure interface distortion can be repaired by recalibrating the port.

A body with the wrong recurrent phase relation can fit a finite calibration window and still drift later.

Exact nominal algebra does not itself correct physical noise.

At `sigma=.03` continuous state noise, relocking every 16 steps kept the rail task digitally exact through L1024, whereas never relocking fell to about `.858` overall.

Dynamic zero-mean noise behaved roughly as a diffusion clock

```text
T_dynamic ~ 1/sigma^2,
```

while static phase/frequency bias produced coherent drift

```text
T_static ~ 1/eta.
```

So even "relock rate" is error-mechanism dependent.

---

# 4. Important metacircuit correction

The first metacircuit backend pass made a physically important mistake and the branch now records it explicitly.

For the **discrete central-difference recurrence**

```text
u[t+1] = (2 - dt^2 D^-1Y)u[t] - u[t-1],
```

an exact sampled character uses

```text
D^-1Y = 2(1-cos(theta))/dt^2.
```

But the physical analog resonator obeys a continuous law

```text
D u'' + Y u = 0,
```

so exact sampled phase requires

```text
D^-1Y = (theta/dt)^2.
```

Therefore the earlier finite-difference `lambda=4` stability cliff, ~13.8x component-headroom improvement, ~12x relation-defect separation and ~2.5x tolerance-horizon gain are **discrete numerical-backend results only**.

They must not be cited as analog-circuit physics.

See `METACIRCUIT_CONTINUOUS_TIME_CORRECTION.md`.

---

# 5. Corrected metacircuit result: no easy robustness win

Under the correct continuous resonator law, backend-aware representation choice still changes sampled interface conditioning, but the dramatic robustness story disappears.

A physically constrained exact C101 bank

```text
[16,18,19,20,25,28,30,31]
margin 5.712879
max sampled lag->phase cond(T) 1.8406
```

was compared with a strong exact digital-only bank

```text
[3,5,12,21,22,23,37,50]
margin 5.888437
max cond(T) 64.2934.
```

The physical bank gives up only ~2.98% symbolic margin and has a vastly better sampled displacement/lag coordinate transform.

But robust state tracking does **not** clearly improve:

```text
bounded eta=1e-5 certificate:
990 cycles versus 990 cycles

L1024 after per-device readout calibration:
sigma       physical      digital-only
1e-5        .97705         .97477
2e-5        .69088         .73256
5e-5        .30353         .42783
```

So KYY has **not** earned the claim that a simple physical-conditioning objective beats a strongly optimized exact digital representation.

---

# 6. The metacircuit's demonstrated port is also the wrong direct port for C_n phase state

The fabricated metacircuit classifier measures selected oscillator voltages, rectifies them and integrates output energy over time.

For an exact cyclic state represented by a phase-shifted periodic trajectory

```text
h_q(t) = h_0(t+q),
```

any complete-period quadratic energy

```text
E(q) = sum_t |W h_q(t)|^2
```

is independent of `q`: changing `q` only permutes one full period.

The branch test gives

```text
instantaneous phase prototype accuracy: 1.000
full 101-step energy spread:             ~1.99e-12
16-step truncated energy spread:          ~100.82.
```

So the metacircuit body remains relevant as neighboring resonant hardware, but its demonstrated energy-classifier port is not a natural terminal C101 phase-state port.

It is demoted from "direct KYY backend" to **neighboring hardware architecture**.

See `METACIRCUIT_ENERGY_READOUT_BOUNDARY.md`.

---

# 7. Better semantic match: multi-phase SHIL / Potts oscillator hardware

Current multi-phase ring-oscillator Potts machines use subharmonic injection locking so one continuous oscillator phase has several stable discrete phase states, with phase-sensitive physical readout.

That primitive is prior art.

But it matches KYY's digital/analog seam directly:

```text
continuous physical variable = phase phi

digital state = which locking basin contains phi.
```

Small analog phase errors relax back toward the digital basin without a software nearest-state projection.

This makes it a much better test backend for KYY quotient semantics than the metacircuit energy classifier.

---

# 8. Pass-44 partial merge becomes a literal attractor-landscape compilation

Desired C4 merge:

```text
0 -> 0
1 -> 0
2 -> 2
3 -> 2.
```

Compile it physically by switching temporarily from four phase wells to two shifted phase wells, allowing relaxation, then restoring four-well locking.

The locally obvious midpoint choice

```text
alpha = pi/4
```

puts the coarse attractors exactly on the restored C4 separatrices.

It performs the local pair collapse but gives the next stage zero re-entry margin and about 50% final correctness under arbitrarily small symmetry breaking/noise.

Balancing source capture against fine-state re-entry gives

```text
alpha* = pi/8 = 22.5 degrees
worst composition margin = pi/8.
```

In the reduced phase SDE, `alpha=pi/8` stayed 100% correct through the largest tested phase diffusion `D=.02` over 16000 trajectories per point, while `pi/4` stayed near 50%.

The lesson is not that SHIL can merge phases; staged phase locking is prior art.

The useful compiler lesson is:

> **a physical stage can satisfy its local quotient and still hand the next stage a geometrically invalid state. Compile the composition, not each stage in isolation.**

See `SHIL_C4_QUOTIENT_BACKEND_RESULT.md`.

---

# 9. One-stage cyclic quotient compile/reject law

For a fine `C_n` phase code and a temporary uniform `m`-well SHIL landscape, one physical stage has `m` equal contiguous basins.

Therefore one-stage quotient kernel classes must be

```text
- contiguous in cyclic order;
- equal size r=n/m;
- one cyclic run per output class.
```

Interleaved or unequal kernel classes reject.

For equal contiguous blocks, with fine spacing

```text
Delta = 2*pi/n,
```

the optimum cross-stage margin is

```text
r odd:
    choose the middle fine state as representative
    attractor = that state
    margin = Delta/2

r even:
    choose either central fine state
    place attractor Delta/4 toward block center
    margin = Delta/4.
```

Examples:

```text
C4  -> C2,  r=2:  margin pi/8
C12 -> C4,  r=3:  margin pi/12
C12 -> C3,  r=4:  margin pi/24
C100-> C10, r=10: margin pi/200.
```

The scaling cost is therefore mainly **fine phase resolution n**, not the number of states merged by itself.

See `SHIL_CYCLIC_QUOTIENT_COMPILER_RESULT.md`.

---

# 10. A real instruction-set lower bound appears

Consider the alternating C4 quotient

```text
{0,2} -> A
{1,3} -> B.
```

It is a perfectly valid abstract digital equivalence relation, but its classes are interleaved on the physical phase circle.

Allow any number of the current one-circle primitives:

```text
- cyclic rotations/reflections;
- uniform equal-contiguous SHIL quotient stages.
```

Every such primitive is cyclic-monotone: it preserves/reverses cyclic order and may collapse only contiguous arcs.

Composition preserves contiguous fibers.

Therefore **no sequence of these one-circle instructions can realize the alternating kernel**.

The exhaustive C4 semigroup check agrees:

```text
adjacent [0,0,1,1]: reachable
alternating [0,1,0,1]: unreachable.
```

This gives the compiler a genuine resource answer rather than a failed parameter search.

See `SHIL_INSTRUCTION_SET_BOUNDARY.md`.

---

# 11. Harmonic phase is a different quotient instruction

A coherent `h`-th harmonic carries phase

```text
phi -> h phi mod 2*pi.
```

Its kernel is not a contiguous-arc kernel. It identifies congruence/subgroup classes.

For C4 with the second harmonic:

```text
q=0 -> 0
q=1 -> pi
q=2 -> 0
q=3 -> pi,
```

so it realizes exactly

```text
{0,2}/{1,3}.
```

Harmonic/frequency multiplication is established physical technology; KYY does not claim the primitive.

The interesting compiler distinction is that two equally small digital quotients require physically different instructions because their kernel geometries differ:

```text
contiguous kernel
    -> basin-collapse / SHIL instruction

congruence/interleaved kernel
    -> harmonic phase instruction candidate.
```

And the Pass-44 body/port warning still applies: merely *measuring* a harmonic quotient does not erase the original fine state. Body-level forgetting requires transferring/locking state into the coarse carrier and discarding or decoupling the old fine carrier.

---

# 12. Current compiler vocabulary

The physical backend is beginning to look like an instruction-selection problem:

```text
INPUT:
    exact behavioral transition/kernel

AUDIT:
    which distinctions survive?
    which must collapse?
    are analog tangent variables preserved?

MATCH KERNEL SHAPE:
    contiguous cyclic classes
        -> uniform SHIL quotient

    congruence/subgroup classes
        -> harmonic phase map/carrier candidate

    neither
        -> richer nonuniform forcing,
           auxiliary state dimension,
           another embedding,
           or reject backend

COMPOSE:
    place intermediate attractors away from next-stage separatrices

PRICE:
    phase resolution
    locking time/noise margin
    harmonic conversion cost
    extra carrier/state cost
    readout cost
    relock bandwidth.
```

This is the clearest KYY-shaped residual so far.

It is still surrounded by prior art in phase logic, oscillator computing, nonlinear frequency conversion and compiler theory; novelty is not established.

---

# 13. Current stopping pin

The next meaningful work is no longer another generic state-tracking benchmark.

It is to enlarge the **physical instruction set** and ask whether arbitrary small transition monoids can be lowered with explicit resource costs and impossibility certificates.

The first hard questions are:

- which kernels are single-stage basin quotients;
- which are harmonic/congruence quotients;
- which need extra state dimension or nonuniform forcing;
- when a port-only quotient is insufficient because the old carrier retains forbidden memory;
- how locking/noise and cross-stage basin margins compose.

Only after this compiler layer is coherent is transistor-level oscillator simulation worth the effort.

Until then:

> **keep this work on `agent/geometric-wave-state-v01`; do not promote to `main`.**
