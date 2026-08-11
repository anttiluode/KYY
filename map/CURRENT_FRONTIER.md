# KYY current frontier

Updated: 2026-08-11

This is the short map of the research branch after the continuous-time metacircuit correction.

## One-sentence status

KYY has **not** discovered a new theory of automata, forgetting, Koopman representations, hybrid systems, resonant RNNs, analog optimization, or neuromorphic compilation.

The surviving question is narrower:

> Can a compiler carry exact **behavioral algebra** through a learned continuous machine and use it to decide what must be exact in the body, what can be calibrated at the port, what requires runtime relocking, and which behaviorally equivalent representation is cheapest for a specific physical substrate?

That remains a research hypothesis, not a novelty claim.

---

# 1. What has been subtracted

The branch has explicitly collided with mature work in:

- finite automata, Myhill-Nerode/state equivalence and synchronizing resets;
- RNN-to-automaton extraction, state quantization/reification and repair;
- reversible-RNN and machine-unlearning internal-state audits;
- group representations, equivariant ports and finite-state Koopman models;
- linear realization/state assignment;
- hybrid automata, quotient/gluing resets, tangent/nullspace geometry and switched systems;
- learned operator projection onto algebraic constraints;
- physical wave/oscillator recurrent neural networks;
- neuromorphic IR and hardware-aware analog optimization.

The rule remains:

> **Do not rename prior art and call it ours.**

`main` remains untouched.

---

# 2. Pass 44: useful failure witness, not new forgetting theory

The learned full-rank partial merge hid a digital distinction at the current output and a common future later exposed it.

That mechanism is classical observability/automata territory, and nearest-state reification repairs the finite four-point toy too.

At L1024:

```text
learned soft           ~0.734
merge-only reify       ~0.99984
every-step reify        1.000
exact operator compile  1.000
```

So exact singular surgery is not uniquely required on the finite-point task.

See:

- `PASS44_PARTIAL_MERGE_KERNEL_COMPILER.md`
- `CLOSURE_PARTIAL_MERGE_REIFICATION_BASELINE.md`

---

# 3. Pass 46: the useful digital / analog border

Replace four legal points by four continuous fibers

```text
F_q = { c_q + a v : a in R }.
```

`q` is digital rail identity. `a` is a continuous payload along the rail.

The learned coordinates are entangled.

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

The paired-history audit is the cleanest statement:

```text
nearest-fiber projection:
    digital mismatch = 0
    but analog history gap survives ~5.27e-4 mean

exact point+tangent operator:
    digital mismatch = 0
    analog gap -> numerical floor
```

Therefore

```text
projection onto the legal manifold
!=
correct dynamics along the legal manifold.
```

Geometrically:

```text
DIGITAL = quotient / equivalence directions
ANALOG  = tangent / fiber directions
```

This geometry is classical hybrid-systems territory; the KYY result is the controlled post-training audit/legalization experiment.

See `PASS46_RESULT_DIGITAL_QUOTIENT_ANALOG_FIBER.md`.

---

# 4. Exact nominal algebra is not physical error correction

Add continuous state noise after every exact update.

At L1024 and `sigma=.03`:

```text
bare exact continuous geometry: ~0.851 digital overall
reify every step:                1.000
explicit Q x R hybrid:           1.000
```

The analog error is similar across implementations.

So:

```text
exact operator = correct nominal law
reification    = runtime transverse error correction
explicit QxR   = digital error isolation
```

These are different deployment contracts.

---

# 5. Relocking turns the seam into an engineering dial

With the same noisy analog trace at `sigma=.03`:

```text
relock interval    digital overall
1                     1.0000
4                     1.0000
16                    1.0000
64                    0.9895
256                   0.8745
never                 0.8576
```

Relocking does not materially reduce analog tangent RMSE; it protects the digital quotient.

Across noise levels, the safe free-run interval behaves roughly like a diffusion clock:

```text
T_dynamic ~ 1 / sigma^2.
```

A static frequency/component bias instead creates coherent phase drift:

```text
T_static ~ 1 / eta.
```

So a physical compiler should distinguish error mechanisms rather than emit one generic robustness number.

---

# 6. Body versus port is now experimentally separated

A static invertible sensor/basis distortion is a port problem.

In the resonator deployment probe the old readout can fail almost completely under an unknown coordinate change, while a freshly calibrated linear port restores exact long-horizon decoding if the recurrent body itself is still exact.

By contrast, a persistent phase/frequency error changes the recurrent relation. A port can fit a finite calibration horizon, but it cannot make the body satisfy `C^n=I` forever.

The rule is therefore:

```text
interface distortion -> calibrate/transport the port
relation error        -> body tuning, different representation or relock
```

The 2025 analog HORN hardware result is an external real-hardware example of the first case: analog dynamics remained useful while the original digital readout did not transfer cleanly and had to be retrained.

See `PHYSICAL_OSCILLATOR_BODY_PORT_BOUNDARY.md`.

---

# 7. IMPORTANT metacircuit correction

The first KYY metacircuit passes conflated the paper's **finite-difference RNN recurrence** with the exact free-running law of its **continuous analog resonator**.

For the discrete central-difference recurrence:

```text
D^-1Y = 2(1-cos(theta))/dt^2
```

is the exact character lowering.

For the physical continuous resonator

```text
D u'' + Y u = 0,
```

exact sampled phase `theta` requires instead

```text
D^-1Y = (theta/dt)^2.
```

Therefore the earlier `lambda=4` stability edge, tiny near-Nyquist component headroom, ~13.8x headroom improvement, ~12x relation-defect improvement and ~2.5x tolerance-horizon improvement are **discrete numerical-surrogate results only**.

They must not be described as analog-circuit physics.

See `METACIRCUIT_CONTINUOUS_TIME_CORRECTION.md`.

---

# 8. What survives under the corrected continuous resonator law

Two earlier exact C101 banks were reevaluated under the continuous physical law with a fresh readout calibrated for every perturbed device.

At `sigma=1e-5` relative resonator-ratio spread, L1024:

```text
                         mean acc    worst      mean relation defect
old unconstrained        0.96545     0.76660       0.002510
old conditioned          0.99008     0.94531       0.002083
```

The conditioned bank still helps, but modestly.

Its bounded `eta=1e-5` static-tolerance certificate is

```text
709 cycles -> 900 cycles
```

about `1.27x`, not the earlier surrogate `2.5x`.

The continuous physical parameter relation is now

```text
lambda = D^-1Y = theta^2 / dt^2,
```

and relative component-to-phase sensitivity is simply

```text
lambda dtheta/dlambda = theta/2.
```

No finite-difference stability cliff remains.

See:

- `metacircuit_continuous_backend.py`
- `results/metacircuit_continuous_backend_summary.json`

---

# 9. Strong exact digital baseline removes the easy robustness claim

A corrected representation search compared two exact C101 8-character banks.

### physically constrained exhaustive optimum inside the declared candidate set

Constraints:

```text
cond(T) <= 2
f <= 31
```

Result:

```text
frequencies              [16,18,19,20,25,28,30,31]
symbolic margin           5.712879
max cond(T)               1.84063
max port-transform norm   1.48120
max phase sensitivity     0.96425
```

### strong digital-only exact heuristic

```text
frequencies              [3,5,12,21,22,23,37,50]
symbolic margin           5.888437
max cond(T)              64.2934
max port-transform norm  45.4678
max phase sensitivity     1.55524
```

The physically constrained bank loses only `2.98%` symbolic margin and is enormously easier to express through the sampled displacement/lag port.

But robustness does **not** clearly improve:

```text
bounded eta=1e-5 certified cycles:
physical constrained = 990
digital heuristic    = 990

L1024 after per-device calibration:
sigma       physical      digital heuristic
1e-5        0.97705           0.97477
2e-5        0.69088           0.73256
5e-5        0.30353           0.42783
```

So:

> **KYY has not shown that simple backend-conditioning constraints beat a strongly optimized exact digital representation on robust state tracking.**

What is real is the huge difference in port conditioning and physical parameter range among behaviorally exact representations.

Whether that matters enough to justify compiler-driven representation choice now depends on an actual hardware cost model.

See:

- `metacircuit_continuous_design.py`
- `results/metacircuit_continuous_design_summary.json`

---

# 10. Prior-art boundary for the hardware line

Do not claim novelty for:

- analog oscillator/resonator neural computation;
- training digital models and mapping them into physical dynamics;
- hardware-aware optimization under mismatch/noise;
- per-device readout retraining/calibration;
- robust physical-neural-network training;
- neuromorphic intermediate representations.

Shem in particular already performs generic hardware-aware optimization of analog systems with nonlinear dynamics, mismatch and oscillator examples.

Therefore KYY should **not** try to be a new generic analog optimizer.

The possible residual is as a constraint/certificate layer:

```text
behavioral algebra
    -> exact representation family
    -> target-substrate costs
    -> port/body/relock placement
    -> generic hardware optimizer works inside that contract
```

No direct owner for that exact contract has been established, but neither has KYY shown a decisive advantage yet.

---

# 11. Current stopping pin

The next useful physical experiment needs a cost the current toy phase model cannot manufacture by hand.

Good candidates are:

- actual circuit/component range;
- amplifier/ADC gain required by the physical port;
- saturation/dynamic range;
- parasitic cross-mode coupling;
- power/area;
- measured component tolerance and thermal drift;
- cost/frequency of injection locking or digital relock.

The 2026 metacircuit paper explicitly notes finite op-amp bandwidth, clipping, parasitic capacitance/resistance, component tolerances and thermal drift as practical issues, but the detailed circuit/hardware specifications needed for a faithful reproduction are not yet enough here to justify inventing a SPICE model.

Until a faithful backend or real measurement exists:

> **keep digging on the research branch; do not promote the metacircuit line to `main`.**

The broader KYY thread that still survives is simple:

> **exactness has a location, and behaviorally equivalent continuous realizations can have different physical interface costs.**

The first half is well demonstrated in the branch. The second half is now demonstrated for sampled port conditioning, but not yet as a decisive end-to-end hardware advantage.
