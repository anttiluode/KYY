# KYY current frontier

Updated: 2026-08-11

This is the short map of the research branch after the phase-backend instruction-set closure, body/port carrier audit, program-level resource planner, and audible SHIL demo.

## One-sentence status

KYY has **not** discovered new automata theory, hybrid systems, oscillator computing, SHIL/Potts hardware, harmonic generation, EDA technology mapping, or generic analog optimization.

The surviving compiler question is narrower and now finite enough to terminate:

> **Given an exact behavioral transition, what distinctions must its physical body preserve or destroy, which declared substrate instruction has the same kernel geometry, and what resource is missing when no instruction matches?**

`main` remains untouched.

---

# 1. Durable software result: digital quotient versus analog fiber

Pass 46 replaced isolated digital points with continuous legal fibers

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

The paired-history audit gives the clean statement:

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

This is classical hybrid geometry used as a compiler audit, not a new hybrid-systems theorem.

See `PASS46_RESULT_DIGITAL_QUOTIENT_ANALOG_FIBER.md`.

---

# 2. Exactness has a location

The branch distinguishes four deployment contracts:

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

A body with the wrong recurrent relation can fit a finite calibration window and still drift later.

Exact nominal algebra does not itself correct physical noise.

Dynamic zero-mean noise produced a diffusion-style free-run clock

```text
T_dynamic ~ 1/sigma^2,
```

while static phase/frequency bias produced coherent drift

```text
T_static ~ 1/eta.
```

---

# 3. Important metacircuit correction and demotion

The first metacircuit backend pass conflated the paper's **finite-difference RNN recurrence** with the free-running law of its **continuous analog resonator**.

Discrete recurrence:

```text
D^-1Y = 2(1-cos(theta))/dt^2.
```

Continuous physical resonator:

```text
D u'' + Y u = 0
D^-1Y = (theta/dt)^2.
```

Therefore the earlier finite-difference `lambda=4` stability cliff, ~13.8x component-headroom improvement, ~12x relation-defect separation and ~2.5x tolerance-horizon gain are **discrete numerical-backend results only**.

Under the corrected continuous law, a physically conditioned exact character bank did not beat a strong exact digital baseline on robust state tracking.

The fabricated metacircuit's demonstrated port also rectifies/integrates oscillator energy. A complete-period quadratic energy readout erases a pure cyclic phase shift; the branch test gives full-period phase-state energy spread only ~`1.99e-12`.

So the metacircuit is retained as neighboring resonant hardware, not the direct KYY phase-state backend.

See:

- `METACIRCUIT_CONTINUOUS_TIME_CORRECTION.md`
- `METACIRCUIT_ENERGY_READOUT_BOUNDARY.md`

---

# 4. Better semantic match: multi-phase SHIL / Potts oscillator hardware

Multi-phase injection-locked oscillator hardware already uses continuous oscillator phase as a multivalued discrete state.

That primitive is prior art.

It matches the KYY seam naturally:

```text
continuous physical variable = phase phi

digital state = which locking basin contains phi.
```

Small phase errors can relax back toward the digital basin without software nearest-state projection.

---

# 5. Pass-44 merge becomes a physical composition-margin problem

Desired C4 merge:

```text
0 -> 0
1 -> 0
2 -> 2
3 -> 2.
```

Compile by switching temporarily from four phase wells to two shifted wells, relaxing, then restoring four-well locking.

The locally symmetric midpoint

```text
alpha = pi/4
```

performs the pair collapse but puts the coarse attractors exactly on the restored C4 separatrices.

Result under noise: about 50% final correctness.

Balancing source capture and fine-state re-entry gives

```text
alpha* = pi/8
worst composition margin = pi/8.
```

In the reduced phase SDE, `pi/8` stayed 100% correct through the largest tested diffusion `D=.02` over 16000 trajectories per source state, while `pi/4` stayed near 50%.

Compiler rule:

> **Compile the composition, not merely each physical stage.**

A stage can satisfy its local transition and still hand the next stage a geometrically invalid state.

See `SHIL_C4_QUOTIENT_BACKEND_RESULT.md`.

---

# 6. Uniform SHIL quotient compile/reject law

For fine `C_n` phase states and one uniform `m`-well temporary landscape, exact quotient classes must be:

```text
- contiguous in cyclic order;
- equal size r=n/m;
- one cyclic run per output class.
```

For fine spacing

```text
Delta = 2*pi/n,
```

the optimum cross-stage margin is

```text
r odd:
    representative = middle fine state
    attractor = representative
    margin = Delta/2

r even:
    representative = either central fine state
    attractor = representative + Delta/4 toward block center
    margin = Delta/4.
```

Examples:

```text
C4  -> C2,  r=2:  pi/8
C12 -> C4,  r=3:  pi/12
C12 -> C3,  r=4:  pi/24
C100-> C10, r=10: pi/200.
```

The cost scales mainly with fine phase resolution `n`, not directly with how many states are merged.

See `SHIL_CYCLIC_QUOTIENT_COMPILER_RESULT.md`.

---

# 7. One-circle instruction-set lower bound

With only

```text
cyclic rotations/reflections
+
uniform equal-contiguous SHIL quotient stages,
```

every primitive is cyclic-monotone and every fiber remains a contiguous arc under composition.

Therefore the alternating C4 quotient

```text
{0,2}/{1,3}
```

is unreachable by **any number** of those stages on one phase circle.

The exhaustive C4 semigroup check agrees.

A coherent second harmonic

```text
phi -> 2phi mod 2*pi
```

has exactly that alternating kernel.

So two equally small abstract quotients can require physically different instruction families:

```text
contiguous kernel
    -> basin-collapse / SHIL

cyclic congruence kernel
    -> quotient character / harmonic carrier.
```

Harmonic conversion is prior art; the compiler classification is the point.

See `SHIL_INSTRUCTION_SET_BOUNDARY.md`.

---

# 8. Current phase library is now closed as a finite checker

The declared library is:

```text
faithful cyclic phase re-encoding
uniform equal-basin SHIL collapse
pre-carried quotient-aligned character
runtime harmonic/carrier conversion for cyclic congruence kernels.
```

For every explicit partition the classifier now terminates with:

```text
identity / no-op
universal collapse
quotient-aligned character
uniform SHIL after faithful embedding
unsupported: unequal class sizes
unsupported: equal size but wrong topology.
```

Exhaustive small-n coverage:

```text
n    nontrivial kernels    supported by current exact library
3            3                 0
4           13                 3   = 23.08%
5           50                 0
6          201                 7   = 3.48%
7          875                 0
8         4138                14   = 0.338%.
```

For prime `n`, the current exact one-circle library has no nontrivial quotient: there is no proper cyclic subgroup character kernel and no nontrivial equal-block `m|n` SHIL quotient.

That is a lower bound for **this declared instruction library**, not for physics in general.

See `PHASE_KERNEL_LIBRARY_CLOSURE.md`.

---

# 9. Correction to "prefer re-encoding over harmonic"

A faithful character is a `C_n` automorphism.

A nontrivial proper character quotient is a coset partition of the unique subgroup of that order.

Every faithful automorphism maps that subgroup to itself.

Therefore a nontrivial congruence kernel does **not** become a contiguous SHIL kernel merely by choosing another faithful character.

The exhaustive audit through `n<=16` finds zero nontrivial character/SHIL collisions under faithful one-circle re-encoding.

For C4:

```text
adjacent [0,0,1,1]
    SHIL yes
    character no

alternating [0,1,0,1]
    character f=2 yes
    SHIL under faithful f no.
```

The useful design-time alternative is instead:

```text
carry a faithful full-state character
+
carry the non-faithful quotient character from the start.
```

Then the quotient can retire the modes that distinguish within a class rather than synthesize the harmonic at runtime.

So the real trade is

```text
standing redundant carrier/state
versus
runtime nonlinear conversion.
```

---

# 10. Body versus port: pre-carrying the quotient still does not forget

For C4 carry

```text
z1 = exp(i phi)      faithful fundamental
z2 = exp(i 2phi)     alternating quotient character.
```

The quotient port can read only `z2`, making `q` and `q+2` identical now.

But if `z1` remains future-observable, a later weak path

```text
y = z2 + epsilon*g*z1
```

recovers the forbidden distinction.

If the old carrier is damped as

```text
z1(t)=exp(-gamma t)z1(0),
```

the exact pairwise future gap is

```text
Delta(t)=2|epsilon|exp(-gamma t).
```

For `epsilon=.1, gamma=1`:

```text
t=0   .200000
t=1   .073576
t=2   .027067
t=4   .003663
t=8   .0000671.
```

To make the gap `<=delta`:

```text
t >= log(2|epsilon|/delta)/gamma.
```

But if every future path from `z1` is hard-gated to zero, behavioral forgetting is exact immediately even while `z1` remains physically different.

So the compiler has at least two retirement contracts:

```text
DAMP / ERASE
    spend settling time / dissipation

ISOLATE / DISCONNECT
    spend switching / isolation resource.
```

This is the hardware translation of the earlier observability correction: hidden-state equality is sufficient, future-unobservability is enough.

See `HARMONIC_BODY_PORT_FORGETTING_RESULT.md`.

---

# 11. Program-level resource planner

The transition classifier now lifts to a workload.

Within the restricted exact direct-character strategy:

```text
standing carrier lower bound
=
1 faithful full-state character
+
1 character for every distinct nontrivial cyclic congruence kernel
that must be directly available.
```

One exact character coordinate cannot serve two different kernels because its kernel is fixed by `gcd(n,f)`.

C12 demo:

```text
C2 congruence -> f=6
C3 congruence -> f=4
C4 contiguous quotient -> one SHIL stage
equal-size wrong topology -> reject
unequal kernel -> reject.
```

Pre-carried strategy:

```text
character bank [1,4,6]
standing carriers = 3.
```

Minimal-standing alternative:

```text
keep f=1
synthesize/transfer f=4 and f=6 at runtime when needed.
```

No winner is declared until hardware supplies area/power/latency/leakage costs.

See `PHASE_BACKEND_PROGRAM_PLANNER_RESULT.md`.

---

# 12. The SHIL composition result now has an audible artifact

Absolute steady-state phase is not directly audible, so the demo uses same-frequency stereo I/Q references.

The oscillator state tone stays at `220 Hz`; phase changes only its interference with the references.

Demo structure:

```text
110 Hz marker
    -> alpha=pi/4 midpoint
    -> 12 repeated noisy q=1 merge trials

660 Hz marker
    -> alpha=pi/8 compiled
    -> 12 repeated trials.
```

Fixed-seed generated result:

```text
pi/4: decoded [0,0,1,0,1,0,1,0,0,1,1,1]
      6/12 correct

pi/8: decoded [0,0,0,0,0,0,0,0,0,0,0,0]
      12/12 correct.
```

This is a sonification of the reduced phase model, not a hardware recording.

See `SHIL_MERGE_AUDIO_DEMO.md`.

---

# 13. Prior-art and novelty boundary

Do not claim as new:

- phase-coded oscillator logic;
- subharmonic injection locking;
- multi-phase Ising/Potts machines;
- harmonic/frequency multiplication;
- automata/state equivalence;
- technology mapping;
- hardware-aware optimization;
- observability/future-equivalence theory.

Targeted searches still have not found a direct established flow whose cell signature is an exact **dynamical kernel/attractor geometry** and which lowers arbitrary declared finite-state transitions into oscillator primitives with compile/reject and body/port retirement certificates.

That absence-of-find is **not** a novelty proof.

The closest conceptual analogy is technology mapping, except the backend cell signature here is not merely a Boolean truth table. It includes:

```text
kernel/equivalence relation
attractor geometry
body versus port semantics
cross-stage basin margin
retirement/isolation requirement
physical resource vector.
```

---

# 14. Current stopping pin

Do **not** enlarge the physical instruction set merely to make more abstract kernels compile.

The checker now rejects most kernels, and that is useful.

Next work should be forced by one of two things:

```text
A. a real target workload whose required transition is rejected
   -> then identify the cheapest missing physical primitive;

or

B. a real/audio-rate oscillator experiment measuring one existing resource:
   locking margin, carrier retirement time, isolation leakage, harmonic transfer cost, or phase readout.
```

The most buildable hardware experiment is now precisely specified:

```text
fundamental fine-state carrier
+
quotient-aligned harmonic carrier
+
retire fundamental by damping or isolation
+
measure whether forbidden phase memory leaks back.
```

Until real measurements exist:

> **keep the compiler/checker on `agent/geometric-wave-state-v01`; do not promote to `main`.**
