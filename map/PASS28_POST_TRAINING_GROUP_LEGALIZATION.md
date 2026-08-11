# Pass 28 — post-training group legalization repairs the learned cyclic tracker

Date: 2026-08-10

This pass is the first KYY result in a while that changed the direction rather than merely narrowing it.

The experiment asks a compiler-style question:

> If short-horizon training learns an *approximately* correct recurrent group action, can we project the **operator parameters themselves** onto the exact task algebra after training, without retraining the readout, and thereby recover long-horizon behavior?

For the deliberately tiny `C_31` oscillator-bank probe, the answer is **yes in all three tested seeds**.

This is a narrow result. It is not yet a general state-tracking method.

Code:

- `map/harmonic_training_probe.py`
- `map/cyclic_relation_defect.py`
- `tests/test_harmonic_training_probe.py`
- `tests/test_cyclic_relation_defect.py`

Archived summary:

- `results/harmonic_n31_projection_summary.csv`

---

## 1. Setup

Task:

```text
running sum modulo 31
```

Input token at each step is an increment in

```text
{0,1,2,3,4}.
```

Recurrent state is an 8-mode complex/rotary bank, i.e. 16 real state coordinates.

For the unconstrained learned model,

```text
theta_1,...,theta_8
```

are ordinary trainable real angles. A token carrying increment `a` rotates every mode by

```text
a * theta_i.
```

There is no drive, contraction, MLP, hidden projection, attention, or state reset.

A linear 31-class readout sees the concatenated rotary state.

Training:

```text
sequence length = 16
steps           = 1000
seeds           = 0,1,2
```

Evaluation:

```text
lengths = 16,64,256,1024.
```

---

## 2. Before legalization: short fit, long drift

All three unconstrained learned oscillator banks fit the training horizon perfectly:

```text
seed   L16     L64     L256    L1024
0      1.000   0.982   0.264   0.068
1      1.000   0.628   0.229   0.077
2      1.000   0.768   0.246   0.083
```

Mean clean accuracy:

```text
L16      1.000
L64      0.793
L256     0.246
L1024    0.076
```

So the learned recurrent body has found a solution that is locally excellent but does not close exactly around the symbolic cycle.

---

## 3. The algebraic audit

For a block-diagonal rotary bank

```text
A = diag(R(theta_1),...,R(theta_k)),
```

an exact representation of `C_n` must satisfy

```text
A^n = I.
```

Per mode,

```text
||R(theta_i)^n - I||_2
    = 2 |sin(n theta_i / 2)|.
```

Therefore the full operator relation defect is

```text
D_op = ||A^n-I||_2
     = max_i 2 |sin(n theta_i/2)|.
```

For the equal-amplitude normalized phase state used here, two histories whose accumulated counts differ by exactly one symbolic cycle `n` have state distance

```text
D_state
  = 2 sqrt(mean_i sin^2(n theta_i/2)).
```

This quantity is independent of the starting count.

The learned models ended with one-cycle state relation defects:

```text
seed 0   0.3273
seed 1   0.6361
seed 2   0.4683
```

Their corresponding clean L64 accuracies were

```text
0.982, 0.628, 0.768.
```

With only three samples this is not a correlation study, but the ordering is exactly what the relation-defect picture predicts.

---

## 4. Legalization

For this diagonal cyclic representation, the exact legal operators are trivial to enumerate.

Each mode must be an `n`-th root of unity:

```text
theta_i = 2*pi*f_i/n
f_i in Z_n.
```

After training, independently replace every learned angle by the nearest legal character:

```text
f_i = round(n*theta_i/(2*pi))
theta_i <- 2*pi*f_i/n.
```

Crucial control:

> **Do not retrain the readout.**

No gradient step is taken after projection.

The trained linear classifier remains exactly as it was.

The projected frequency sets were:

```text
seed 0: [ 2,  3, 20, 27, 12, 22,  3,  4]
seed 1: [ 6, 17, 13, 22, 30,  6, 20,  1]
seed 2: [12, 20,  6, 17, 11, 14, 30, 11]
```

The resulting orbit noise radii are respectively approximately

```text
0.596, 0.560, 0.471.
```

---

## 5. Result: the unchanged readout becomes exact to length 1024

After operator legalization:

```text
seed   L16     L64     L256    L1024
0      1.000   1.000   1.000   1.000
1      1.000   1.000   1.000   1.000
2      1.000   1.000   1.000   1.000
```

So the same readout that previously gave only

```text
6.8%, 7.7%, 8.3%
```

at length 1024 gives

```text
100%, 100%, 100%
```

after changing **only the recurrent angles**.

The training-horizon result remains 100% in every seed.

This is the important part: the compiler projection did not trade away the short solution in order to impose the algebra. It preserved it while repairing the rollout in this experiment.

---

## 6. What the result means mechanistically

The linear decoder had already learned a decision geometry compatible with the nearby exact group orbit.

The recurrent update was the weak link.

During short training, a small eigenphase error is cheap: the approximate and exact trajectories remain near enough that the same classifier can label them correctly.

Over many wraps, the phase defect accumulates and histories representing the same symbolic modulo state no longer land at the same recurrent point.

Legalization collapses those drifting copies back onto one exact finite orbit:

```text
state s
state s+31
state s+62
...
```

become the same recurrent state again, up to floating-point roundoff.

The readout did not need to be repaired because it had already learned the local orbit labels.

That interpretation is strongly supported by the zero-shot nature of the projection, but it remains an inference from this controlled model rather than a general theorem about trained RNNs.

---

## 7. The RoPE projection control fails in the useful way

Standard RoPE angles were included only as a structural control; RoPE is not designed to implement modulo-31 state tracking.

Its unprojected short accuracy was about 0.85--0.89 at length 16 and collapsed with length.

Snapping those frequencies to the nearest `C_31` characters yields

```text
[5,2,0,0,0,0,0,0]
```

with a poor cyclic orbit radius around

```text
0.186.
```

The relation becomes exact, but the unchanged readout's L16 accuracy falls to about 0.26--0.29 and L1024 sits around 0.23--0.26.

So:

> **Exact algebra alone is not sufficient. The learned behavior must already lie near a useful exact representation.**

That negative control is important. Otherwise one could falsely conclude that snapping any oscillator bank onto roots of unity magically solves modular tracking.

---

## 8. The earlier margin result survives

Across the 12 already-exact character-bank runs (`low_coherence`, `random_characters`, `prime_characters`, `geometric_characters`) all clean runs stayed at 100% through L1024.

Under the same systematic phase perturbation

```text
eta = 1e-3 rad per unit increment,
```

the exact-code orbit radius was positively associated with L1024 accuracy across those 12 runs:

```text
Pearson r  ~= 0.76
Spearman rho ~= 0.77.
```

This is exploratory evidence from one modulus and one error model, not a resource law yet.

After legalization, the three learned codes also remain much better under this perturbation than before legalization through L256; at L1024 their perturbed accuracies are approximately

```text
0.495, 0.432, 0.428.
```

The clean algebra is exact; implementation error still consumes the orbit margin, exactly as Error Control Dynamics would lead us to expect.

---

## 9. Prior-art boundary

Several neighboring ideas are already occupied and must remain subtracted.

### Roots-of-unity recurrent tracking

AUSSM already uses unit-modulus complex recurrence and roots of unity for modulo tracking. KYY does not claim that construction.

### Complex diagonal group tracking

ICLR 2026 work already characterizes finite Abelian group tracking by complex diagonal SSMs. KYY does not claim that expressivity class.

### Separation versus accumulated drift

Error Control Dynamics already frames long-horizon state-tracking failure through within-state drift relative to between-state separation. KYY does not claim that general principle.

### Approximate representations

`||A^n-I||` is a special case of approximate-representation / Ulam-stability mathematics. KYY does not claim relation defect as new mathematics.

### Hard finite-group state projection

Lee (2026), *A Held-Out Transition-Pair Falsifier for Long-Horizon Non-Abelian State Tracking*, uses hard projection of a **continuous hidden state/readout** onto target finite-group representatives and obtains million-token non-Abelian tracking under its controlled protocol.

That is a close conceptual neighbor, but it projects a different object.

This KYY test performs a one-time **post-training projection of recurrent operator parameters** onto the exact algebra and then uses the original ordinary linear readout with no further optimization.

### Generic unitary projection / post-training quantization

Projected unitary training and post-training quantization are mature techniques. The operation here should therefore not be sold merely as "projection after training."

A targeted search in this pass did not locate a paper whose stated procedure is exactly:

```text
train a recurrent finite-group operator unconstrained
measure task-relation defect
snap its eigenphases to the nearest exact group characters after training
leave the trained decoder unchanged
recover long-horizon symbolic tracking.
```

That is a **search miss, not a novelty proof**.

---

## 10. This changes the KYY compiler picture

Before this experiment the compiler idea was mostly analytical:

```text
behavior -> algebra -> representation -> physical realization.
```

Now there is a concrete reverse direction from a trained approximate model:

```text
SHORT-HORIZON TRAINING
        |
        v
learn approximate operator
        |
        v
RELATION AUDIT
measure A^n - I
        |
        v
ALGEBRAIC LEGALIZATION
project operator onto exact C_n representation
        |
        v
RESOURCE SELECTION
among legal representations, price
margin / precision / wiring / state norm
        |
        v
DEPLOY EXACT RECURRENCE
```

The tiny `C_31` result shows that the legalization arrow can be behavior-preserving and can repair extrapolation.

That is the first part of this compiler picture that has now earned itself experimentally rather than by analogy.

---

## 11. What to do next

Do **not** immediately jump to a giant architecture.

The next falsifiers are small:

1. Repeat across several moduli with a training generator that exposes all symbolic states fairly.
2. Vary mode count and training horizon.
3. Measure how far a learned operator may be from the nearest exact representation before zero-shot legalization stops preserving the readout.
4. Compare zero-shot legalization with an explicit relation penalty during training.
5. Test a finite Abelian product with more than one relation.
6. Only then try non-Abelian/local-generator legalization, where the projection is no longer independent per 2D mode.

Important experimental caveat for the cross-modulus sweep: the current length-16 increment generator can visit every class for `C_31` but not for large `n` such as `C_101`. Before changing modulus, add a random starting offset or equivalent mechanism so every symbolic class is represented during short training. Otherwise the decoder comparison is confounded.

---

# Current pin

The strongest KYY statement supported **right now** is not a new architecture theorem.

It is this experimental observation:

> **On a tiny learned cyclic recurrent tracker, short-horizon optimization finds operators close enough to an exact finite-group representation that a zero-shot post-training projection of the recurrent eigenphases onto the task's exact characters preserves training behavior and restores perfect clean length extrapolation from 16 to 1024 tokens in all three tested seeds.**

Everything in that sentence is deliberately scoped.

Now try to break it.
