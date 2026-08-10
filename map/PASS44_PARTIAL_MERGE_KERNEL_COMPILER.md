# Pass 44 — partial-merge kernel compiler: forgetting only what the task says

Date: 2026-08-10

Pass 42 compiled the easiest irreversible primitive: a total reset.

A total reset has one universal kernel block, so the compiler can simply overwrite the entire old state.

Pass 44 tests the first genuinely different case:

> some histories must become exactly identical, while other histories must remain distinct.

This is a **partial merge**.

The experiment is intentionally tiny enough that every object can be inspected directly.

---

## 1. Prior art boundary

Nothing about singular transformations, non-uniform kernels, permutation groups combined with singular maps, or linear realization of finite automata is new.

Finite transformation-semigroup and synchronization literature studies groups together with noninvertible maps, including non-uniform kernel partitions.  Linear realization/state-assignment literature is decades old.

The KYY question remains post-training and compiler-specific:

> can a short-trained, full-rank continuous operator that only approximately implements a partial merge be replaced after training by the exact singular lowering implied by the behavioral transition and state geometry, while preserving the learned task interface and making the required forgetting literally future-proof?

---

## 2. Symbolic machine

There are four symbolic states arranged as a C4 cycle.

Increment tokens are rotations:

```text
q -> q+a mod 4,  a in {0,1,2,3}.
```

One special token `M` has transition

```text
0 -> 0
1 -> 0
2 -> 2
3 -> 2.
```

Its behavioral image is

```text
{0,2}
```

and its kernel partition is

```text
{0,1}
{2,3}.
```

So its finite-state behavioral rank is two.

This token must forget the difference `0 versus 1` and the difference `2 versus 3`, but it must preserve the distinction between the two blocks.

---

## 3. Geometric code

Use one planar C4 harmonic mode:

```text
z0 = ( 1, 0)
z1 = ( 0, 1)
z2 = (-1, 0)
z3 = ( 0,-1).
```

Pass 43's linear-lowering audit synthesizes the exact partial merge directly from the symbolic transition:

```text
M = [[1,1],
     [0,0]].
```

Check:

```text
M z0 = z0
M z1 = z0
M z2 = z2
M z3 = z2.
```

The continuous matrix has rank one even though its action on the legal four-point code has behavioral image size two.

That distinction matters:

```text
continuous operator rank != automaton transition rank.
```

The compiler contract is about the induced action on legal states.

---

## 4. The learned machine is deliberately allowed to cheat softly

The learned recurrent model has:

```text
one learned planar rotation angle
one learned arbitrary 2x2 merge matrix B
one learned 4-way linear readout.
```

The merge matrix is initialized full rank and unconstrained.

Nothing forces it to become singular.

Training length is only 16, with merge probability 0.15.

The compiler later ignores its exact learned coordinates and uses the task/code lowering from Pass 43.

---

## 5. Ten-seed training result

All ten learned models fit the short task perfectly:

```text
L16: 10 / 10 exact.
```

Short extrapolation also looks excellent:

```text
L64 exact seeds:   8 / 10
mean L64 accuracy: 0.999695

L256 exact seeds:  6 / 10
mean L256 accuracy: 0.996602
```

But long rollout breaks sharply:

```text
L1024 exact seeds: 0 / 10
mean L1024 accuracy: 0.733412
range roughly: 0.699 .. 0.765.
```

This is much stronger failure than the soft total-reset experiment.

---

## 6. What training actually learned

Every learned merge matrix remains full rank.

Across the ten seeds:

```text
learned det(B): -0.033 .. +0.860
full-rank learned merge seeds: 10 / 10
```

The smaller singular value ranges from roughly `0.009` to `0.229`, never zero.

So the machine often learns an *almost* singular direction, but it never actually creates the required kernel.

A simple residual along the exact kernel direction `(1,-1)` remains:

```text
0.087 .. 0.559.
```

The learned cycle relation is also approximate rather than exact:

```text
C4 relation defect: 0.0139 .. 0.0427.
```

Training has therefore found a soft continuous approximation, not the declared finite machine.

---

## 7. The leakage experiment isolates the failure

Construct two histories whose symbolic states immediately before `M` are:

```text
history A -> state 0
history B -> state 1.
```

The task says both must become state zero after `M`.

Then give both machines **exactly the same future**, containing only cyclic increments.

No further merge is allowed to clean up the residual.

### hidden state after the learned merge

The paired hidden vectors are still separated by:

```text
0.115 .. 0.772.
```

That distinction is real and persistent.

### but the port almost cannot see it immediately

At the merge event itself, probability TV distance is only:

```text
1.70e-4 .. 6.92e-4.
```

So by looking only at the current prediction one could easily conclude that reset/merge worked.

It did not.

### shared rotations reveal the hidden distinction

The future is made only of planar rotations.

Those rotations preserve hidden Euclidean distance.

Therefore the leftover distinction is not attenuated; it merely changes orientation relative to the learned decoder.

Later in the same shared future, maximum TV reaches:

```text
0.0020 .. 0.4065.
```

Four of ten seeds eventually produce **different predicted classes** for histories that the symbolic machine says were already merged.

Selected examples:

### seed 0

```text
hidden distance immediately after merge     0.577
TV immediately after merge                  0.000288
max later TV                                0.296
max future prediction mismatch rate         23.8%
learned L1024 accuracy                      69.9%
```

### seed 4

```text
hidden distance immediately after merge     0.772
TV immediately after merge                  0.000648
max later TV                                0.407
max future prediction mismatch rate         36.3%
learned L1024 accuracy                      71.1%
```

### seed 6

```text
hidden distance immediately after merge     0.115
TV immediately after merge                  0.000176
max later TV                                0.0020
future prediction mismatch                  none in this probe
learned L1024 accuracy                      76.0%
```

The exact-kernel residual strongly tracks later observable leakage in this ten-seed probe:

```text
Pearson(kernel residual, max future TV)   ~0.952, p ~2.2e-5
Spearman                                 ~0.988, p ~9.3e-8
```

This is a tiny controlled experiment, so those statistics are descriptive rather than a broad generalization.

The structural cause is already visible without them.

---

## 8. This is Pass 16 reduced to two dimensions

Pass 16 said:

> the orthogonal wave body learned to hide old history from the current port, not destroy it.

Pass 44 shows the same failure with almost nothing left to blame:

```text
soft full-rank merge leaves Δh != 0
        ↓
current decoder happens to suppress that direction
        ↓
TV nearly zero immediately
        ↓
subsequent lossless rotation preserves Δh
        ↓
orientation changes
        ↓
old history reappears at the port.
```

So the important distinction is not

```text
looks reset now / does not look reset now.
```

It is

```text
required kernel equality exists internally / does not exist internally.
```

That is the semigroup/compiler version of future-proof forgetting.

---

## 9. Compilation

The post-training compiler performs three surgeries.

### cycle

Snap the learned angle to the exact frequency-1 C4 character.

All ten seeds choose frequency `1`.

### partial merge

Replace the learned full-rank `B` with the exact Pass-43 lowering

```text
M = [[1,1],
     [0,0]].
```

This creates the required hidden kernel exactly.

The generator certificate uses only the square basis:

```text
M z0 = z0
M z1 = z0.
```

Because

```text
z2 = -z0
z3 = -z1,
```

linearity forces the other two required transitions.

Thus all four legal merge transitions are certified from two basis constraints rather than an arbitrary-word rollout.

### port

Audit:

```text
A. inherited learned readout
B. exact C4-equivariant readout projection
C. positive-kernel readout.
```

---

## 10. Compiled result

All three compiled ports are exact in all ten seeds at every tested length:

```text
                 L16   L64   L256   L1024
inherited        10/10 10/10 10/10 10/10
equivariant      10/10 10/10 10/10 10/10
positive kernel  10/10 10/10 10/10 10/10
```

Unlike Pass 42, the inherited decoder happens to survive the operator surgery in all ten runs.

The symmetry/canonical ports still provide a smaller and cleaner deployment contract.

Orbit margins:

```text
equivariant:     2.760 .. 3.198
positive kernel: 3.048 .. 3.385
```

All positive ports pass the algebraic C4 character certificate.

All exact generator certificates pass.

---

## 11. Compiled leakage is literally zero

Repeat the paired-history probe after compilation.

At the exact partial merge:

```text
hidden difference = 0.
```

Under the entire shared future:

```text
max hidden difference       = 0
max probability TV          = 0
prediction mismatch rate    = 0.
```

This is stronger than “good extrapolation.”

It follows from the actual kernel collision:

```text
M z0 = M z1.
```

Once the histories occupy the same hidden vector, deterministic future operators cannot separate them again.

That is exactly the declared automaton semantics.

---

## 12. Why the lowering audit matters

The exact pinch was not chosen because it looks geometric.

Pass 43 derives it from:

```text
state code Z
+ symbolic target columns Z_M
+ exact linear-realization condition.
```

So the emerging compiler front-end is now:

```text
symbolic transition
       ↓
behavioral image + kernel partition
       ↓
state geometry Z
       ↓
linear/affine realizability audit
       ↓
minimum exact lowering
       ↓
operator legalization / synthesis
       ↓
port legalization
       ↓
relation + kernel + port certificate.
```

That is a much more concrete object than “waves might implement finite-state computation.”

---

## 13. Resource interpretation

There are now at least three different notions of rank/collapse in play.

### behavioral rank

How many legal symbolic states remain in the image of a transition.

For this merge:

```text
behavioral rank = 2.
```

### continuous operator rank

Rank of the real-valued lowering matrix acting in the ambient state space.

For the exact pinch:

```text
continuous rank = 1.
```

### learned numerical near-rank

The trained matrix has one small singular value but remains full rank.

That is enough to imitate the merge at short horizons, but not enough to guarantee the correct kernel.

This separation is likely important for a physical backend:

> a tiny nonzero singular channel can be behaviorally invisible now and still be a future leakage channel.

So singular-value cost should be tied to the **required symbolic kernel**, not treated only as generic regularization.

---

## 14. Prior-art subtraction again

The mathematical ingredients are classical:

- finite transformation semigroups;
- kernel partitions and ranks;
- permutation groups plus singular transformations;
- linear realization / state assignment of finite automata;
- pseudoinverse synthesis of consistent linear maps;
- group-equivariant output parameter sharing.

No novelty claim is made for any of those.

The empirical/compiler result earned here is:

> Ten short-trained full-rank continuous machines perfectly fit a four-state partial-merge task at training length while retaining hidden distinctions the task requires them to destroy.  The residual can be nearly invisible at the merge port and later reappear under shared norm-preserving dynamics.  Post-training replacement by the exact singular lowering derived from the symbolic kernel and state code eliminates the hidden distinction exactly and yields 100% behavior through length 1024 for all ten seeds, with an algebraically certifiable port.

---

## 15. New stopping pin

The next question should **not** be another hand-picked singular map.

Pass 43 gives us the general lowering test.

The next useful step is to apply it automatically across a nontrivial finite task/transition set:

1. choose or learn a compact state code `Z`;
2. for every token, compute `(image, kernel partition)`;
3. test exact linear lowering;
4. if rejected, test affine lowering;
5. record required ambient rank / singular directions;
6. synthesize all legal operators;
7. verify their relations and port behavior.

Then state-code search itself becomes a compiler problem:

> find the smallest / most robust geometry in which the entire task alphabet has cheap legal lowerings.

That is where the original geometric instinct may finally become a measurable optimization objective rather than an analogy.

Files:

- `map/partial_merge_compiler_probe.py`
- `tests/test_partial_merge_compiler_probe.py`
- `.github/workflows/partial-merge-compiler.yml`
- `results/partial_merge_c4_compiler_summary.json`
