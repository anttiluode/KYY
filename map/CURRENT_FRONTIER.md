# KYY current frontier

Updated: 2026-08-11

This file is the short map of the research branch.

It exists because the branch now contains many useful negative and boundary results that should not require reconstructing forty separate pass notes.

## One-sentence status

KYY has **not** discovered a new theory of automata, forgetting, Koopman representations, hybrid systems, resonant RNNs, or neuromorphic compilation.

The surviving research question is now much narrower:

> Can a compiler take an already learned continuous dynamical machine, identify the exact behavioral algebra it is supposed to realize, and choose/legalize a physically cheap exact representation for a specific analog or mixed-signal substrate while deciding which errors belong in the body, the port, or a periodic digital correction?

That is still a hypothesis, not a novelty claim.

---

# 1. What was subtracted

The branch has explicitly collided with and subtracted mature work in:

- finite automata, state equivalence and distinguishing suffixes;
- transformation semigroups and synchronizing/reset automata;
- RNN -> automaton extraction and state quantization;
- state reification and post-hoc RNN repair;
- reversible-RNN limits on forgetting;
- representation-level machine-unlearning audits;
- differentiable/soft DFA learning;
- group representations and equivariant ports;
- finite-state Koopman representations;
- linear realization/state assignment;
- hybrid automata and mixed logical dynamical systems;
- quotient/gluing geometry for hybrid resets;
- rank-deficient/submersive resets and tangent/nullspace geometry;
- switched linear realization;
- learned operator projection onto exact constraints;
- neuromorphic / hardware-aware compilation;
- physical wave and oscillator recurrent neural networks.

The project rule remains:

> Do not rename prior art and call it ours.

See:

- `CLOSURE_PRIOR_ART_AUDIT_2026-08-11.md`
- `PASS45_KOOPMAN_STATE_CODE_BOUNDARY.md`

---

# 2. Pass 44 survived only as a clean audit example

Pass 44 showed a learned full-rank partial merge that looked forgotten at the current output but retained a hidden distinction which a common future exposed.

That is a useful two-dimensional demonstration, but the general phenomenon is classical observability/automata territory.

Worse for novelty, nearest-state reification repairs the toy too:

```text
L1024
learned soft          ~0.734
merge-only reify      ~0.99984
every-step reify       1.000
exact operator compile 1.000
```

So exact singular surgery is not uniquely needed on a finite four-point state set.

See:

- `PASS44_PARTIAL_MERGE_KERNEL_COMPILER.md`
- `CLOSURE_PARTIAL_MERGE_REIFICATION_BASELINE.md`

---

# 3. The useful digital / analog border: four rails

Pass 46 replaces four legal points by four legal continuous fibers

```text
F_q = { c_q + a v : a in R }.
```

`q` is digital state. `a` is a continuous payload.

The learned coordinates are randomly entangled so there is no explicit DIGITAL axis and ANALOG axis.

Tokens can:

- rotate the digital C4 base while preserving `a`;
- partially merge digital histories while preserving `a`;
- continuously scale `a` while preserving the digital state.

The correct comparison is now an explicit hybrid `Q x R` machine, not a DFA.

### learned machine

Perfect at the training horizon, but at L1024:

```text
mean digital accuracy: 0.96237
minimum seed:          0.85110
analog RMSE:           0.00487
```

All learned merge maps remain full rank.

### nearest-fiber reification every step

```text
digital: 10/10 exact through L1024
analog RMSE at L1024: 0.00474
```

It repairs the transverse/digital error but essentially not the along-fiber semantic error.

### exact point+tangent compiler

```text
digital: 10/10 exact through L1024
analog RMSE at L1024: 3.86e-8
```

The exact compiler specifies both where legal state centers go and how legal tangent directions must transform.

See:

- `PASS46_RESULT_DIGITAL_QUOTIENT_ANALOG_FIBER.md`
- `mixed_fiber_compiler_probe.py`

---

# 4. Same rail does not mean same analog state

The paired-history Pass-46 audit starts with two histories having the same analog payload but different digital state, then applies a required digital merge and an identical future.

Nearest-fiber reification gives

```text
rail/digital mismatch: 0
```

for the whole future.

Yet branch identity survives as a legal along-fiber displacement:

```text
mean analog gap after merge: ~5.27e-4
largest seed:                ~1.66e-3
```

The exact operator compiler reduces the same paired analog gap to the inherited float32/float64 numerical floor (~1e-7 or below).

So the clean geometric statement is:

```text
projection onto the legal manifold
!=
correct dynamics along the legal manifold.
```

See:

- `mixed_fiber_pair_audit.py`
- `results/mixed_fiber_pair_audit_summary.json`

---

# 5. Point + tangent lowering gives a compiler boundary

For digital centers `c_q`, tangent bases `V_q`, token transition `tau`, and required tangent maps `L_q`, one global linear token operator must satisfy

```text
A c_q = c_tau(q)
A V_q = V_tau(q) L_q.
```

Concatenate these source generators into `X` and required images into `Y`.

Then

```text
A X = Y
```

is solvable iff

```text
ker(X) subset ker(Y).
```

One useful resource consequence:

If several modes literally share the same physical tangent vector but a token requires different mode-dependent actions on that tangent, one global linear operator cannot do it.

The compiler must choose among:

```text
switch/condition the operator
use nonlinearity
or
spend extra dimension to separate tangent copies.
```

This is classical realization linear algebra turned into an explicit backend audit, not a new theorem.

See:

- `hybrid_fiber_lowering_audit.py`

---

# 6. Exact algebra does not provide physical error correction

The exact rail compiler was then subjected to additive continuous-state noise after every update.

At L1024:

```text
sigma=.003
bare continuous q: 1.000
reified q:        1.000
explicit QxR q:  1.000

sigma=.010
bare continuous q: 0.9986 overall / 0.9883 final
reified q:          1.000
explicit QxR q:     1.000

sigma=.030
bare continuous q: 0.8511 overall / 0.7461 final
reified q:          1.000
explicit QxR q:     1.000
```

The analog error is similar across all three at the same noise level.

Interpretation:

```text
exact operator = correct nominal law
reification    = runtime transverse error correction
explicit QxR   = digital error isolation
```

These are different deployment contracts.

See:

- `mixed_fiber_noise_boundary.py`
- `results/mixed_fiber_noise_boundary_summary.json`

---

# 7. The digital/analog border becomes a correction-bandwidth dial

With `sigma=.03`, use the exact continuous operators but relock/project to the legal fiber only periodically.

On an identical paired noise trace:

```text
relock interval    digital overall    digital final
1                     1.0000            1.0000
4                     1.0000            1.0000
16                    1.0000            1.0000
64                    0.9895            0.9727
256                   0.8745            0.7441
1024                  0.8576            0.7637
never                 0.8576            0.7637
```

Waiting until the end is too late: once diffusion crosses a digital basin boundary, nearest-fiber relocking can faithfully lock the wrong rail.

The analog RMSE stays essentially unchanged (~0.23556), because the relock preserves the tangent coordinate.

A noise sweep gives the empirical safe frontier (`>=.999` overall and `>=.99` final):

```text
sigma   largest tested safe interval
.010       256  (censored by grid)
.015       256  (censored by grid)
.020       128
.030        32
.040        16
.050        16
```

The boundary roughly follows a diffusion-like `sigma^2 * T` scale.

This turns "digital correction" into a physical resource:

```text
analog free-running time
        vs
digital correction bandwidth.
```

See:

- `mixed_fiber_relock_boundary.py`
- `mixed_fiber_relock_frontier.py`
- `results/mixed_fiber_relock_frontier_summary.json`

---

# 8. Real oscillator hardware makes the body/port split concrete

The 2025 analog HORN implementation is an external example where a digital oscillator network transfers to analog hardware but the original digital readout no longer matches well; retraining the readout recovers performance.

That independently demonstrates:

```text
body transfer != port transfer.
```

KYY then tests the stricter case where the physical body violates a recurrence relation the task requires.

For a C4 oscillator with only `delta=.001 rad` phase error per nominal increment, a static port calibrated on 16 winding cycles is initially perfect, but:

```text
cycles    calibrated accuracy   min margin
16            1.0000             +0.4837
64            1.0000             +0.3765
256           0.7979             -0.1446
1024          0.1995             -1.0003
```

By 1024 cycles, physical states belonging to different symbolic classes come within ~`2.04e-4` on the circle.

The port has no stable class geometry left to separate.

The legalized exact quarter-turn body remains exact with margin ~1.

So:

```text
port calibration can repair interface mismatch;
it cannot make the recurrent body satisfy a relation the body violates.
```

See:

- `PHYSICAL_OSCILLATOR_BODY_PORT_BOUNDARY.md`
- `physical_cycle_port_boundary.py`

---

# 9. A current 2026 metacircuit gives KYY a concrete analog backend

Zhou et al. (2026) demonstrate a fully analog resonant recurrent neural network implemented as coupled electrical local resonators.

Its second-order recurrence contains the block

```text
u[t+1] = (2I - dt^2 D^-1 Y)u[t] - u[t-1] + ...
```

For a KYY exact cyclic character

```text
theta = 2*pi*f/n,
```

the exact one-mode lowering is

```text
D^-1 Y = 2(1-cos(theta))/dt^2.
```

The physical lag-coordinate state `[u_t,u_{t-1}]` is analytically similar to the ordinary phase/quadrature rotation coordinates, giving exact port transport rather than mandatory retraining.

See:

- `METACIRCUIT_CYCLIC_BACKEND_BOUNDARY.md`
- `metacircuit_cyclic_backend.py`

---

# 10. This backend breaks a digital equivalence

For one faithful C101 character, all coprime frequencies have the same exact one-mode symbolic margin.

But the resonator backend sees them very differently:

```text
f      D^-1Y       cond(phase map)
1      .00387          32.14
4      .06160           8.00
25    1.96890           1.016
49    3.99130          21.42
50    3.99903          64.29
```

Near the upper stability edge, the positive relative component-error headroom also becomes tiny:

```text
f=25: ~103%
f=49: ~0.218%
f=50: ~0.024%
```

The already-learned exact C101 banks have worst-mode phase-map conditioning ranging from `2.44` to `32.14` across seeds.

So:

> exact software equivalence does not imply physical implementation equivalence.

That is the first concrete backend cost attached to KYY's earlier character geometry.

---

# 11. Backend-aware representation choice now has a natural objective

A simple greedy eight-character C101 design was run with and without a resonator conditioning cap.

```text
                       symbolic     max        minimum +component
                       margin       cond(T)    stability headroom
unconstrained          5.38965       7.10          1.99%
require cond(T)<=2     5.21618       1.99         27.39%
```

The physically constrained bank loses only ~3.2% of symbolic margin while gaining:

```text
3.57x better worst coordinate conditioning
3.22x lower worst port-transform norm
3.71x lower worst relative phase sensitivity
13.8x more positive stability headroom.
```

Both are algebraically faithful exact C101 codes.

The composite C100 test also produces a faithful backend-constrained bank.

See:

- `METACIRCUIT_FREQUENCY_DESIGN_RESULT.md`
- `metacircuit_frequency_design.py`

---

# 12. The current geometric interpretation

The useful phrase is not simply

> digital versus analog.

It is

> **where does exactness live?**

Exactness can live in different places:

```text
state label      -> explicit digital Q x R hybrid
body/operator    -> exact group/semigroup relation or kernel
port/interface   -> calibrated observable map
runtime relock   -> periodic return to a legal manifold/basin
```

The experiments show those locations are not interchangeable.

A second useful phrase is:

> **digital equivalence classes can contain physically inequivalent embeddings.**

The finite behavior may leave representation choices free that an analog substrate prices very differently.

This is the point where the original Geometric Neuron instinct has become a measurable compiler question rather than an analogy.

---

# 13. What is still unearned

KYY has **not** shown that its compiler beats an explicit hybrid implementation in software.

KYY has **not** demonstrated any of these ideas on a real resonator/metacircuit/oscillator circuit.

KYY has **not** established novelty over every hardware-aware compiler or physical-neural-network co-design method.

The next meaningful win would require a real or faithful physical backend model where carrying the behavioral algebra explicitly lets the compiler do something measurably better than:

- retrain/calibrate the readout;
- ordinary hardware-aware training;
- generic parameter projection;
- explicit digital mode storage;
- runtime state reification;
- standard injection/phase locking.

Until that exists, keep this work on the research branch.

## Current residual

The narrow residual worth testing is:

> A backend compiler whose IR carries not just neural parameters, but **behavioral algebra**: finite-order relations, irreversible kernel partitions, analog tangent semantics, port symmetry, physical tolerance and relock requirements; the compiler then chooses among exact representations based on a target substrate.

No direct owner for that exact contract has been found in the current search.

That sentence is a research target, not a novelty declaration.
