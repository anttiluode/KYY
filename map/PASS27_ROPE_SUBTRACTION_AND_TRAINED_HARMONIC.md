# Pass 27 — subtract RoPE, correct the locality budget, and finally train the oscillator bank

Date: 2026-08-10

This pass addresses three criticisms of Passes 23–26:

1. the harmonic construction had not been explicitly subtracted against RoPE;
2. the mixed-radix locality comparison priced packing but not wiring carefully enough;
3. almost every harmonic result so far was an exact oracle rather than a trained recurrent experiment.

All three matter. One also changes the story.

---

## 1. RoPE owns the oscillator-bank mechanism

RoFormer (Su et al., 2021) explicitly partitions an even-dimensional representation into 2D planes and rotates each plane at a preset frequency. Standard RoPE uses a geometric schedule of the form

```text
theta_i = base^(-2 i / d)
```

(up to indexing convention).

Therefore KYY cannot claim novelty for:

```text
one scalar sequence coordinate
    -> several differently tuned 2D phase modes.
```

The collision is stronger in 2026.

Mamba-3 uses complex-valued recurrent dynamics for state tracking and derives an implementation through data-dependent rotary embeddings. A February 2026 RoPE analysis describes RoPE as phase modulation over a **bank of complex oscillators** and explicitly studies aliasing, phase drift, and finite-precision limits. A July 2026 paper studies how training selectively uses RoPE frequency bands and connects frequency scale to length generalization.

Subtract these ideas:

```text
oscillator bank
geometrically spaced rotary frequencies
phase aliasing / drift
frequency choice affects long-range behavior
training can prefer some frequency bands
complex rotations improve state tracking
```

---

## 2. What remains different in the KYY cyclic construction

Pass 23 restricts frequencies to characters of the finite cyclic group `C_n`:

```text
theta_i = 2*pi*f_i/n
f_i in Z_n.
```

That gives the exact algebraic relation

```text
A^n = I.
```

States separated by `n` increments are therefore the same recurrent state in exact arithmetic.

Standard RoPE does not impose that relation for a chosen modulus. It represents ordinary position/relative position, not an exact finite quotient `Z -> Z/nZ`.

The remaining question is narrower:

```text
RoPE:
    which oscillator spectrum gives useful positional geometry?

KYY finite-group compiler:
    among exact representations satisfying A^n = I,
    which character set gives the best
    state-separation / precision / training trade?
```

The character construction itself is known harmonic/group-frame mathematics. KYY does not claim it.

---

## 3. RoPE-shaped exact characters are a useful control

This pass adds

```text
geometric_characters
```

which geometrically spaces **integer character frequencies** and then uses exact cyclic angles `2*pi*f/n`.

This is not standard RoPE. It separates two design criteria:

```text
geometric frequency spacing
```

versus

```text
low cyclic coherence / large orbit margin.
```

Representative oracle radii from the current probes:

```text
C_31, k=8
    low-coherence search       ~0.65
    geometric characters       ~0.56

C_101, k=16
    low-coherence search       ~0.63
    geometric characters       ~0.48

C_1009, k=32
    low-coherence search       ~0.61
    geometric characters       ~0.38
```

These are search examples, not optimal coding records.

---

## 4. Correction to the locality story

The suggestion after Pass 26 was to rerun the mixed-radix comparison on wiring cost.

That is the right axis, but the specific `C_899` harmonic monolith exposes a stronger negative result.

The monolithic harmonic recurrence is already

```text
16 independent 2D rotations.
```

Its recurrent transition is block diagonal. It does **not** require every coordinate to communicate with every other coordinate.

For the pure counter, the monolithic recurrent body can be laid out as independent local rotators receiving a shared external increment/reset control.

The factorized mixed-radix realization instead has

```text
C_31 bank -> carry -> C_29 bank.
```

Even if the boundary locally decodes carry to one bit, that is an additional state-dependent communication edge the monolithic cyclic realization does not need.

So:

> **The mixed-radix pure counter is not a benchmark on which Sigma factorization should be expected to win either packing or recurrent wiring.**

This does not kill locality. It kills this benchmark as evidence for locality.

Keep it as a negative control: it is a behavior whose global algebra already has an unusually local representation.

---

## 5. First trained harmonic-state probe

New code:

- `map/harmonic_training_probe.py`
- `tests/test_harmonic_training_probe.py`

The model is deliberately tiny.

Input tokens are integer increments and the target is the running sum modulo `n`.

For fixed schedules:

```text
recurrent state: fixed 2D rotations only
learned:         linear readout only
```

For `learned`:

```text
learned: mode angles + linear readout
```

There is no drive, contraction, hidden MLP, attention, or projection back onto a symbolic orbit.

Schedules:

```text
low_coherence
random_characters
prime_characters
geometric_characters
rope
learned
```

The first four satisfy `A^n=I` exactly.

`rope` uses the standard geometric RoPE angular schedule and intentionally does not satisfy the modulo relation.

`learned` starts from unconstrained random angles.

---

## 6. Exploratory one-seed result

Configuration:

```text
n = 31
modes = 8
train length = 16
steps = 1000
input increments in {0,1,2,3,4}
test lengths = 16,64,256,1024
systematic evaluation angle error = 1e-3 rad / unit increment
seed = 0
```

This is one exploratory seed, not a benchmark result.

Clean all-token accuracy:

```text
                         L16     L64     L256    L1024
low_coherence           1.000   1.000   1.000   1.000
random_characters       1.000   1.000   1.000   1.000
prime_characters        1.000   1.000   1.000   1.000
geometric_characters    1.000   1.000   1.000   1.000
rope                    0.854   0.279   0.110   0.057
learned                 1.000   0.982   0.264   0.068
```

Do not interpret this as “harmonic frames beat RoPE.” RoPE is not designed to be a modulo-31 recurrent state machine. It is a structural control showing what happens when the oscillator bank does not obey the required group relation.

The unconstrained learned bank is more interesting: it fits the training horizon perfectly but does not discover the exact cyclic quotient strongly enough to extrapolate.

Define

```text
character defect = mean_i distance(n*theta_i/(2*pi), nearest integer).
```

For this run:

```text
before training  ~0.209
after training   ~0.045
```

Optimization moves the angles toward exact cyclic characters without reaching them. The remaining phase error compounds with horizon.

---

## 7. Controlled drift result

With the same trained readouts, add

```text
eta = 1e-3 rad per unit increment
```

to every mode during evaluation.

At length 1024, exploratory all-token accuracies were approximately:

```text
low_coherence           0.524
prime_characters        0.508
random_characters       0.493
geometric_characters    0.436
rope                    0.054
learned                 0.135
```

The exact-character ordering roughly follows the orbit-margin calculation in this seed.

This turns the Pass-24 oracle statement into a trained-readout falsifiable prediction:

> at matched exact cyclic algebra and mode count, larger orbit separation should buy a longer implementation-error runway.

---

## 8. Prime frequencies: useful heuristic, not magic

For `C_31, k=8`, prime characters give radius around `0.60`: better than the crude geometric-character schedule (~`0.56`), worse than the searched low-coherence set (~`0.64–0.65`).

So the old prime-frequency intuition lands in a known mathematical neighborhood:

```text
choose characters whose cyclic correlations are small.
```

Prime choices can be decent heuristics for some moduli. They are not a theorem and are not uniformly optimal.

---

## 9. New pin

Stop expanding exact oracles for a moment. The next empirical sweep is already specified by the new script:

```text
n in {31, 101, ...}
multiple seeds
train length = 16
eta in {0, 1e-4, 3e-4, 1e-3, 3e-3}
test lengths in {16, 64, 256, 1024}
```

Report together:

```text
accuracy
orbit noise radius
character relation defect
phase-error runway
mode count
```

Sharp hypotheses:

1. **Exact relation matters.** Unconstrained learned angles can fit short training while failing long extrapolation in proportion to relation defect.
2. **Margin matters under implementation error.** At matched exact algebra and mode count, low-coherence character sets should tolerate more phase error than poor-margin sets.
3. **RoPE-shaped spacing is a different objective.** Geometric exact-character spacing can have substantially worse cyclic state margin than a low-coherence schedule.
4. **The pure counter is a locality negative control.** The monolithic harmonic update is already block-local.

If any fail under the multi-seed sweep, keep the failure.

---

## Literature pin

Primary sources inspected in this pass:

- Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding*, arXiv:2104.09864.
- Peng et al. (2023), *YaRN: Efficient Context Window Extension of Large Language Models*, arXiv:2309.00071.
- Lahoti et al. (2026), *Mamba-3: Improved Sequence Modeling using State Space Principles*, arXiv:2603.15569 / ICLR 2026.
- Liu (2026), *Rotary Positional Embeddings as Phase Modulation: Theoretical Bounds on the RoPE Base for Long-Context Transformers*, arXiv:2602.10959.
- Wu, Liu & Jadbabaie (2026), *How Data Shapes RoPE Frequency Usage: From Positional Scale Matching to Length Generalization*, arXiv:2607.07678.

Status after Pass 27:

**THE OSCILLATOR-BANK IDEA IS OCCUPIED. THE EXACT-GROUP / MARGIN / TRAINED-RELATION QUESTION SURVIVES AS A NARROW EMPIRICAL COMPILER QUESTION.**
