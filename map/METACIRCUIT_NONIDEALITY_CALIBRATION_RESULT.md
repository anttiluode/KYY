# Result — calibration boundary for a nonideal resonator backend

Date: 2026-08-11

This is a bounded backend experiment, not a claim that KYY has discovered hardware-aware analog optimization.

## Question

The previous metacircuit pass found many exactly correct cyclic character banks whose physical resonator conditioning differs sharply.

The harder question is:

> Does that backend-aware representation choice still matter after giving ordinary analog deployment its strongest cheap repair — a freshly calibrated linear readout on each physical device?

And, equally importantly:

> Does the same representation magically resist unrelated hardware errors too?

The answer in this idealized resonator model is selective:

- static interface/basis distortion is completely repaired by port calibration;
- static `D^-1Y` ratio mismatch changes the recurrent relation and survives finite-horizon port calibration;
- the backend-conditioned character bank reduces that relation defect by about 12.25x under paired component-error draws and extrapolates better;
- a separate pole-radius/Q mismatch negative control shows only a small advantage, so the condition cap is not a generic robustness trick;
- exact body trim is an upper bound and returns both banks to exact long-horizon behavior.

---

## 1. Two exactly legal digital representations

Use the two C101, eight-character banks from the previous backend-aware search.

### unconstrained symbolic-margin bank

```text
frequencies = [35,18,25,46,22,16,45,24]
exact symbolic margin = 5.38965
max cond(T) = 7.0976
minimum positive stability headroom = 1.985%
gcd = 1
```

### backend-conditioned bank (`cond(T) <= 2`)

```text
frequencies = [35,18,25,19,15,27,30,29]
exact symbolic margin = 5.21618
max cond(T) = 1.9855
minimum positive stability headroom = 27.386%
gcd = 1
```

Both are exact faithful C101 state codes. The conditioned bank gives up about 3.22% symbolic margin in exchange for a much friendlier resonator realization.

---

## 2. Fair deployment protocol

For every simulated physical device:

1. draw the same per-mode component-error vector for both banks;
2. apply the same unknown orthogonal sensor-coordinate transform;
3. expose only 16 winding cycles for calibration;
4. fit a fresh affine linear readout for that physical device;
5. evaluate the frozen calibrated port at 16, 64, 256 and 1024 winding cycles.

Eight paired devices are used at each mismatch level.

This matters because the comparison does **not** deny the standard analog-hardware repair of recalibrating/retraining the readout.

The toy backend separates three types of nonideality.

### A. sensor-only distortion

The recurrent body remains exact. Only the measured companion-state basis is changed by a fixed invertible orthogonal transform.

This belongs to the port.

### B. resonator-ratio mismatch

Each ideal scalar

```text
lambda = D^-1 Y = 2(1-cos(theta))
```

is perturbed multiplicatively.

This changes the physical phase increment and therefore the finite-order recurrence relation itself.

### C. pole-radius/Q mismatch

A separate per-mode pole-radius perturbation is used as a generic loss/gain negative control.

This is deliberately a toy nonideality, not a faithful circuit-level model of every loss mechanism in the 2026 metacircuit.

---

## 3. Port-only error is completely calibratable

Before recalibration, the unknown sensor basis destroys the original ideal port:

```text
unconstrained bank accuracy: 0.0000
conditioned bank accuracy:   0.0198
```

After fitting a new linear port at 16 cycles:

```text
L1024 calibrated accuracy:
unconstrained = 1.000
conditioned   = 1.000
```

This is an important negative result for unnecessary compiler surgery.

If the body still realizes the correct recurrence and only the interface coordinates moved, then the proper repair is simply:

```text
transport / recalibrate the port.
```

There is no reason to legalize or retune the recurrent body.

---

## 4. Relation-changing component error survives port calibration

Now perturb each resonator's `D^-1Y` ratio, still giving every physical device its own freshly calibrated 16-cycle linear readout.

### relative ratio sigma = 1e-5

```text
                         L1024 mean acc   worst device   mean max ||A^101-I||_F
unconstrained                 0.75765        0.45313          0.0412911
conditioned                   0.90518        0.72572          0.00337066
```

The conditioned bank has about **12.25x lower mean worst relation defect** and gains **14.75 percentage points** of mean L1024 accuracy.

### sigma = 2e-5

```text
unconstrained L1024 mean = 0.42458
conditioned   L1024 mean = 0.53532

relation defect:
0.0826130 vs 0.00674131
```

Again about 12.25x separation in the relation defect.

### sigma = 5e-5

```text
unconstrained L1024 mean = 0.18543
conditioned   L1024 mean = 0.25752

relation defect:
0.206720 vs 0.0168530
```

### sigma = 1e-4

```text
unconstrained L1024 mean = 0.10087
conditioned   L1024 mean = 0.13699

relation defect:
0.413777 vs 0.0337049
```

The classification advantage naturally shrinks once both physical machines are badly corrupted, but the relation-defect ratio remains near 12.3x across the sweep.

The geometric interpretation is the same as the earlier one-mode port boundary:

```text
static port calibration can rotate / rescale how a state is read;
it cannot make a body with the wrong phase increment satisfy C^101 = I.
```

The calibrated port can fit a finite winding window, but the wrong physical relation keeps accumulating outside that window.

---

## 5. Negative control: unrelated pole-radius/Q error

The backend-conditioned representation should not be advertised as generically more robust to every physical nonideality.

That prediction survives the stronger pole-radius sweep.

At L1024:

```text
radius sigma      unconstrained      conditioned
5e-5                 1.00000           1.00000
1e-4                 0.93792           0.94565
2e-4                 0.67990           0.70418
```

There is a small conditioned advantage, but nothing resembling the roughly 12x relation-defect separation seen for `D^-1Y` mismatch.

So the result is **specific**:

> a physical cost derived from phase/companion geometry helps mainly against the physical error channel that cost describes.

This is preferable to a vague claim that one character bank is simply "more robust."

---

## 6. Body trim upper bound

If the physical resonator ratios are restored to their exact algebraic values and only the sensor coordinate transform remains, then the freshly calibrated port is exact through L1024 for both banks:

```text
unconstrained = 1.000
conditioned   = 1.000
```

This is only an upper bound.

It does **not** claim that a real analog circuit can be trimmed perfectly.

It establishes the conceptual division:

```text
port error     -> port calibration
relation error -> body tuning / representation choice / locking
other body error -> requires its own physical objective or correction mechanism
```

---

## 7. Prior-art collision

Do not claim novelty for:

- analog hardware calibration or hardware-aware retraining;
- modeling fabrication mismatch and noise;
- differentiable optimization of nonlinear analog systems;
- choosing analog parameters to improve robustness or resource cost;
- oscillator-based physical neural networks.

In particular, **Shem: A Hardware-Aware Optimization Framework for Analog Computing Systems** already performs automated time-domain optimization of analog systems with nonlinear dynamics, noise, fabrication mismatch and discrete behavior, including an oscillator-based case study.

Recent physical-neural-network work likewise trains or retrains systems around hardware nonidealities.

Therefore the broad statement

```text
"optimize an analog neural system for hardware mismatch"
```

is occupied.

---

## 8. What survives after this subtraction

The narrower KYY question is now:

> Does explicitly carrying the task's exact behavioral algebra provide a useful optimization *constraint and equivalence class* that a generic hardware-aware optimizer does not otherwise receive?

The present result supports only the first half of that idea.

For C101, the finite behavior admits many exact character banks. The compiler can move within that exact class before deployment and choose a representation whose physical resonator parameters are less phase-sensitive. That choice remains useful even after per-device port calibration.

But this has **not** yet beaten a strong generic hardware-aware optimizer allowed to retune the physical body itself.

That is the next fair baseline.

The direct comparison should be:

```text
A. fixed exact representation + generic finite-horizon hardware optimization
B. algebra-aware exact representation selection
C. algebra-aware representation selection + body optimization constrained to the exact relation manifold
```

and evaluate all three on unseen component mismatch, long winding horizons, hardware cost and relation defect.

If A performs just as well without explicit algebra, KYY's compiler vocabulary is unnecessary for this backend.

If A fits the calibration horizon by moving off the finite-order relation and then drifts, while B/C retain comparable short-horizon performance with better certified long-horizon behavior, then exact behavioral algebra is doing a distinct job.

---

## Files

- `map/metacircuit_nonideality_calibration_boundary.py`
- `tests/test_metacircuit_nonideality_calibration_boundary.py`
- `.github/workflows/metacircuit-nonideality-calibration.yml`
- `results/metacircuit_nonideality_calibration_summary.json`

Workflow evidence: Actions run `31457741777`, focused tests green.
