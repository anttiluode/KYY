# Pass 42 — trained permutation-reset compiler: make forgetting literal

Date: 2026-08-10

Pass 41 closed the easiest group-only certification loop.  This pass returns to the failure that mattered in Pass 16:

> a conservative recurrent body can make two histories look equal at the current port without making them equal internally, so the distinction can leak back under a shared future continuation.

The new question is deliberately smaller than a general semigroup compiler:

> Can a trained machine that only *approximately* implements one cyclic generator and one reset be compiled after training into an exact mixed reversible/irreversible machine, while retaining exact observable behavior?

For the controlled C101 experiment below, yes.

---

## 1. Prior art: permutation-reset automata are old

There is no novelty claim for the target automaton class.

Permutation-reset automata were studied explicitly at least as early as 1976, including series decompositions of their characteristic semigroups.  Modern synchronizing-automata work routinely treats automata through their transition/transformation monoids and calls a monoid synchronizing when it contains a constant map.

Likewise, rank, image, kernel/fiber partitions, Green relations, and singular transformations are standard semigroup/transformation-monoid language.

The KYY object under test is the **post-training compilation step**:

```text
soft learned recurrent machine
        ↓
identify nearby declared behavioral algebra
        ↓
legalize reversible and irreversible generators differently
        ↓
canonicalize / certify the observable interface
        ↓
run the exact compiled machine
```

---

## 2. The trained reset is deliberately not a reset

The learned model has eight complex/planar harmonic modes.

Increment tokens rotate each mode by a learned angle.

The reset token uses a residual affine blend

```text
h' = G h + (I-G) r
```

where

```text
G = diag(g_1,...,g_m),  0 < g_i < 1
```

and `r` is a learned reset target.

The gates are sigmoids, so at finite learned parameters every `g_i` is strictly positive.

Therefore, on hidden-state differences,

```text
Δh' = G Δh.
```

The learned reset can contract old distinctions, but it cannot make two distinct hidden states exactly equal in one reset event as long as `G` is full rank.

This is intentional.  Training is given an easy soft mechanism and the compiler is asked to perform the category change afterward.

The reset starts from `g_i = 0.5`, not from an almost-singular initialization.

---

## 3. Task

The exact symbolic state is `C101` with one reset token.

Ordinary tokens are small cyclic increments.

```text
increment a:  q -> q+a mod 101
reset R:      q -> 0 for every q
```

Training sequences have length 16.

The first token can be a random C101 increment so that short training sees the full symbolic orbit; later increments are local in `0..4`.

Reset probability is `0.12`.

Evaluation lengths are:

```text
16, 64, 256, 1024
```

---

## 4. Compiler

After training, no gradient step is used for compilation.

### reversible generator

Each learned harmonic angle is snapped to the nearest exact C101 character:

```text
θ_i -> 2π f_i / 101.
```

This makes the cycle relation exact:

```text
A^101 = I.
```

### irreversible generator

The learned soft reset is discarded and replaced by the literal overwrite

```text
Z(h) = h0.
```

So its action on hidden differences is exactly

```text
Δh -> 0.
```

This is the thing the Pass-16 orthogonal body could not do.

### port

The same three choices are audited:

```text
A. inherited learned 101-way decoder
B. exact C101-equivariant decoder projection
C. positive correlation-kernel decoder
```

The last two are the Pass-39/40 compiler interfaces.

The positive port has only

```text
8 nonnegative modal weights + 1 shared bias = 9 free scalars.
```

---

## 5. Ten-seed trained result

Configuration:

```text
C101
8 planar modes
10 seeds
2500 training steps
train length 16
batch 128
reset probability 0.12
```

### learned machine

Every seed is exact on the training-length sample:

```text
L16:  10 / 10 exact
```

But sampled exactness disappears as rollout grows:

```text
L64:   8 / 10 exact
L256:  0 / 10 exact
L1024: 0 / 10 exact
```

Mean sampled accuracy is still very high:

```text
L16      1.000000
L64      0.999939
L256     0.999298
L1024    0.999224
```

So this is not a case where compilation rescues a uselessly trained network.

The learned machines already behave extremely well.

They are simply not exact machines.

### their internals are not close to the declared algebra in a strict sense

Across the ten seeds:

```text
mean learned reset gate              ~0.21449
reset-gate range                     0.20890 .. 0.22546
||learned reset target - h0||        4.05498 .. 4.41948
C101 operator relation defect        0.54469 .. 1.59105
```

So the learned reset still preserves about one fifth of a hidden difference per reset in its linear part, and its preferred reset target is nowhere near the compiler's canonical harmonic seed.

The learned cyclic operator also violates `A^101=I` substantially.

Yet the task behavior is nearly perfect.

This is exactly the situation in which a post-training compiler is meaningful.

---

## 6. After operator surgery

Character snapping reduces the measured cyclic operator relation defect to numerical roundoff:

```text
max over ten seeds < 4e-14.
```

The exact overwrite makes the reset relations true by construction:

```text
Z² = Z
Z erases every incoming state
any common future begins from exactly the same hidden state after Z.
```

### inherited learned port

The large inherited decoder is not perfectly compatible with the surgery:

```text
9 / 10 seeds are sampled-exact at every tested length.
```

Seed 3 exposes the familiar failure:

```text
                 L16        L64        L256       L1024
inherited     0.994629    0.998535    0.999542    0.999710
```

The internal machine is now legal, but the old port is not quite the right interface to it.

### 17-parameter exact equivariant port

Projecting the decoder into exact C101 symmetry gives:

```text
10 / 10 exact at L16
10 / 10 exact at L64
10 / 10 exact at L256
10 / 10 exact at L1024
```

### 9-parameter positive-kernel port

The smaller self-certifying port gives exactly the same sampled result:

```text
10 / 10 exact at all four lengths.
```

All ten positive ports retain all eight active modes, all have character gcd 1, and all pass the Pass-40 algebraic port certificate.

Their legal-orbit minimum margins are:

```text
3.862 .. 5.250
```

The smallest learned positive modal coefficient across all ten runs is still about `1.96`, so this is not a numerical accident caused by clipping almost-zero negative modes to zero.

Summary artifact:

- `results/cyclic_reset_c101_compiler_summary.json`

Code:

- `map/cyclic_reset_compiler_probe.py`
- `tests/test_cyclic_reset_compiler_probe.py`
- `.github/workflows/cyclic-reset-compiler.yml`

---

## 7. The tiny exact normal form

Once compiled, this particular transition monoid has an elementary normal form.

Let `c` be the unit cycle and `R` the constant reset to state zero.

Every input word is behaviorally one of only two shapes:

```text
c^k       no reset has occurred; incoming state is translated by k
R c^k     a reset occurred; incoming state is gone and output is constant k
```

Only the suffix after the **last** reset matters in the second case.

If the harmonic C101 representation is faithful, there are therefore

```text
101 distinct cyclic permutations
101 distinct constant maps
--------------------------------
202 transformations total.
```

This is classical permutation-reset structure, not a new theorem.

For KYY it gives a compact certificate:

1. exact character relations for the group part;
2. faithful character orbit (`gcd=1` here);
3. reset is literally compiled as a constant overwrite;
4. port separates the legal orbit.

No enumeration of arbitrary words is needed.

Executable normal-form check:

- `map/cyclic_reset_monoid_certificate.py`
- `tests/test_cyclic_reset_monoid_certificate.py`

---

## 8. The deeper connection to Pass 16: kernel, not appearance

The reset-leakage experiment gave two histories with different prefixes, one common reset, and one shared continuation.

The orthogonal scatterer produced almost identical output immediately after reset, but the old distinction reappeared later.

The right exact finite-state object is the **kernel partition** of a transition.

For a task token `x`, define

```text
p ~_x q    iff    δ(p,x) = δ(q,x).
```

That says which behavioral distinctions the token genuinely destroys.

For a permutation:

```text
kernel blocks = all singletons
rank          = n
```

Nothing is forgotten.

For a total reset:

```text
kernel block = all n states together
rank         = 1
```

Everything about the prior state is forgotten.

And this forgetting is automatically future-proof:

```text
δ(p,x) = δ(q,x)
        =>
δ(p,xw) = δ(q,xw)
```

for every deterministic future word `w`.

Pass 16's orthogonal model never created that hidden-state collision.  It made the two states **port-indistinguishable at one time**, which is a weaker statement.

That is why the distinction could return.

---

## 9. Compiler front-end: inspect the task transition before choosing a primitive

KYY now contains a tiny audit that computes, for every exact task token:

```text
image
rank
kernel/fiber partition
idempotence
number of state distinctions merged
```

and emits a primitive hint:

```text
full rank + discrete kernel  -> permutation / reversible primitive
rank 1                      -> constant reset / overwrite
intermediate rank           -> partial merge / pinch primitive
```

Existing KYY toy tasks expose the split clearly:

### parity / mod3 / perm3

All token transitions are permutations.

```text
no irreversible merges
```

### permreset3

```text
identity    rank 3, singleton kernel blocks
cycle       rank 3, singleton kernel blocks
reset       rank 1, one kernel block {0,1,2}
```

### flipflop

```text
no-op       permutation
set         constant write to 1
reset       constant write to 0
toggle      permutation
```

Code:

- `map/task_transition_kernel_audit.py`
- `tests/test_task_transition_kernel_audit.py`

This is established transformation-semigroup information being used as a compiler front-end, not a new algebraic classification.

---

## 10. Rank is useful; kernel is sharper

Rank tells us **how many** distinctions survive a transition.

The kernel partition tells us **which** distinctions were destroyed.

That matters for a compiler.

Two different rank-two transitions can merge entirely different pairs of behavioral states and therefore require different singular operators or different state embeddings.

So the more useful target descriptor is not merely

```text
reversible / irreversible
```

but

```text
(image, kernel partition, induced action on image).
```

In the full transformation monoid this is closely related to the classical Green-relation coordinates: rank, image, and fiber partition.  KYY should use that language rather than inventing replacements.

---

## 11. What is actually new in this pass, if anything?

Not permutation-reset automata.

Not constant maps.

Not transformation monoids.

Not kernels, ranks, Green relations, or synchronizing automata.

Not harmonic characters.

The empirical/compiler statement earned here is narrower:

> A short-trained C101 harmonic machine with substantially illegal learned cyclic dynamics and only soft residual forgetting can be post-processed, without retraining, into an exact mixed reversible/constant-reset transition system; symmetry projection of the learned output port repairs the observable incompatibility introduced by that surgery, and a smaller positive orbit-kernel port admits an algebraic correctness certificate.

That is a post-training compiler experiment assembled from classical pieces.

It is still only one controlled family.

---

## 12. New stopping pin: partial merges

A total reset has the universal kernel, so the compiler can implement it by throwing away the entire old state.

That is the easiest irreversible map.

The next real boundary is an intermediate-rank transformation:

```text
some histories must become identical
others must remain distinct.
```

For example, on four states a token could require

```text
{0,1} -> one state
{2,3} -> another state
```

with rank two.

Now the compiler cannot simply overwrite the whole harmonic state.

It has to realize the **correct kernel partition** while retaining the quotient information that survives.

That is the first experiment where the new kernel language earns its keep.

So the next pass should not be a larger reset automaton and not another group.

It should be:

> train a soft/full-rank approximation of a partial merge, then ask whether post-training surgery can create the exact required kernel partition, preserve the surviving quotient action, canonicalize the port, and certify future-proof forgetting without enumerating arbitrary histories.

That is the next genuinely different compiler problem.
