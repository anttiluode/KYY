# Pass 8 — Coxeter/Dynkin IR: exact full-S5 locality without a learned architecture

Date: 2026-08-10

Pass 7 added an exact `S3` local oracle and changed the question from "can a large Lie algebra be generated?" to "what is the cheapest exact synthesis of the behaviorally required token operators?"

Pass 8 applies that discipline to the **full S5 permutation-composition benchmark used in modern LRNN state-tracking work**.

The result is not a new architecture. It is a compact exact resource floor built from classical Coxeter/representation theory and permutation routing.

---

## 1. Why full S5 is the right next audit

Grazzi et al. (ICLR 2025) use the word problem for `S5` as a state-tracking benchmark. In the standard/full condition, every input token is sampled uniformly from all 120 elements of `S5`, and the target is the running product.

They also study easier presentations:

- identity + swaps only;
- permutations moving at most three elements;
- four tokens per transition.

Their theory relates a permutation moving `k+1` elements to a product of `k` generalized Householder-style transition factors; for full `S5`, the relevant worst-case count is four factors.

Primary source:

- Grazzi et al., *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues* (ICLR 2025), https://arxiv.org/abs/2411.12537

This is useful because it gives KYY a field-standard problem where **the token operator can be globally simple but locally expensive**.

---

# 2. Natural 5D realization: perfect physical locality

The obvious faithful representation is the natural permutation action on

```text
R^5.
```

Each adjacent transposition

```text
s_i = (i,i+1)
```

is literally a nearest-neighbour 2-port swap/reflection.

So on a physical path

```text
0 -- 1 -- 2 -- 3 -- 4
```

the simple Coxeter generators have perfect locality.

But this 5D representation contains a behaviorally useless one-dimensional invariant direction:

```text
span{(1,1,1,1,1)}.
```

Permutations never change it.

That is exactly the kind of invisible/redundant direction Pass 6 says a compiler should be allowed to remove.

---

# 3. Quotient to 4D without losing graph locality

The standard irreducible/faithful representation lives on the zero-sum hyperplane

```text
V = {x in R^5 : sum_i x_i = 0},
```

which has dimension 4.

Choose the simple-root basis of type `A4`:

```text
alpha_1 = e_1 - e_2
alpha_2 = e_2 - e_3
alpha_3 = e_3 - e_4
alpha_4 = e_4 - e_5.
```

The basis graph is the Dynkin path

```text
alpha_1 -- alpha_2 -- alpha_3 -- alpha_4.
```

Now apply the adjacent transposition `s_i`.

On roots,

```text
s_i(alpha_i)     = -alpha_i
s_i(alpha_(i-1)) = alpha_(i-1) + alpha_i
s_i(alpha_(i+1)) = alpha_i + alpha_(i+1)
```

with all other roots unchanged.

Therefore, if the recurrent state stores simple-root coefficients `c`, one token primitive changes only one coordinate:

```text
c'_i = -c_i + c_(i-1) + c_(i+1)
```

with missing boundary terms omitted. Every other `c_j` is unchanged.

So the 4D behaviorally reduced representation still has **support radius 1 on the A4 path**.

There is a subtlety: this basis is not orthonormal. The generators preserve the Cartan/Gram metric

```text
[ 2 -1  0  0 ]
[-1  2 -1  0 ]
[ 0 -1  2 -1 ]
[ 0  0 -1  2 ]
```

rather than the ordinary coordinate Euclidean metric.

This gives an exact miniature of a KYY compiler trade-off:

```text
5D natural realization:
    + literal 2-port swaps
    - one behaviorally redundant channel

4D simple-root realization:
    + redundant mode removed
    + still radius-1 local
    - primitive becomes a three-coordinate stencil
    - natural preserved metric is non-Euclidean in these coordinates
```

Same group behavior. Different realization cost.

---

# 4. Compile all 120 tokens exactly onto the path

Any permutation is a word in the adjacent transpositions `s_i`.

For `S_n`, the minimum **sequential** number of adjacent transpositions is the inversion number. For `S5`:

```text
mean sequential depth = 5
worst sequential depth = 10.
```

But disjoint adjacent swaps can execute simultaneously. One layer is therefore a matching of the path edges.

`map/s5_coxeter_oracle.py` performs an exact BFS over the 120 permutations using every valid path matching as one primitive layer. For full `S5` it gives the exact minimum parallel-depth distribution:

| minimum local matching depth | number of S5 elements |
|---:|---:|
| 0 | 1 |
| 1 | 7 |
| 2 | 16 |
| 3 | 35 |
| 4 | 46 |
| 5 | 15 |

Thus:

```text
mean exact local parallel depth = 403/120 ~= 3.3583
worst exact local parallel depth = 5.
```

This agrees with the old routing-via-matchings viewpoint: Alon, Chung & Graham model a routing step as simultaneous swaps on a matching and note that an `n`-vertex path can route any permutation in `n` steps by odd-even sorting.

Primary anchor:

- Alon, Chung & Graham, *Routing Permutations on Graphs via Matchings* (STOC 1993 / SIAM J. Discrete Math. 1994), https://doi.org/10.1137/S0895480192236628

For `S5`, the BFS gives the exact per-element optimum rather than only the worst-case upper bound.

---

# 5. Exact recurrent state/readout oracle

The compiler also constructs a 4-channel recurrent state.

Take a distinct zero-sum vector, for example

```text
v0 = (-2,-1,0,1,2)
```

and express it in the simple-root basis.

Its orbit under `S5` contains all 120 permutations, so every group state is distinct.

Because all orbit vectors have equal Euclidean norm in the original 5D space, a fixed prototype linear readout recovers the state exactly. In simple-root coordinates the readout uses the Cartan Gram matrix so the logits are the same original-space inner products.

The resulting resource floor is:

```text
target:                     full S5 running product
behavioral state channels:  4
number of target states:    120
trainable recurrence params:0
primitive support radius:   1
sequential local depth:     mean 5, max 10
parallel local depth:       mean 403/120, max 5
operator relation error:    exact integer arithmetic
length drift:               zero in exact arithmetic
```

The test suite requires exact tracking over a 4096-token random full-S5 sequence and verifies every one of the 120 compiled token operators.

Again, the representation theory is classical. The contribution of the oracle is to make the **resource floor explicit inside the same benchmark harness** that we were about to use for learned models.

---

# 6. Comparison with the global Householder view

Grazzi et al. show why products of generalized Householders are powerful for permutation state tracking: global permutation operators admit short products of reflection-like factors.

For full `S5`, their theory motivates up to four global generalized-Householder factors for one transition.

KYY's exact path compiler instead has:

```text
up to 5 parallel nearest-neighbour layers
```

in a 4D behaviorally reduced representation.

These counts are **not the same cost model**:

- Householder factor: global dot/reduction + broadcast over the hidden state;
- Coxeter path layer: only radius-1 communication, with disjoint sites updated in parallel.

So the first honest hardware question becomes measurable:

> Is one extra layer of strictly local communication cheaper than several global reduction/broadcast operations on the target substrate?

On a GPU, maybe not. On a nearest-neighbour/physical wave fabric, maybe. That is a backend question, not an abstract FLOP claim.

---

# 7. A new realization/gauge connection — with old prior art

The 5D versus 4D forms make a broader point:

> equivalent realizations of the same behavior can have materially different implementation structure.

This is not new. Classical state-space engineering has optimized equivalent realizations for decades:

- balanced/modal/canonical forms;
- finite-word-length sensitivity under similarity transforms;
- sparse realizations;
- numerical conditioning.

A particularly direct 2026 boundary is:

- Du & Li, *Sparse State-Space Realizations of Linear Controllers* (2026), https://arxiv.org/abs/2603.28754

They ask whether a transfer function can be given a state-space realization with a desired sparsity pattern, reduce the problem to a similarity transform from a modal realization, and solve the resulting polynomial constraints using algebraic geometry.

Older finite-word-length work also optimizes similarity transformations while explicitly considering sparse realizations.

Therefore **"choose a gauge/basis that makes implementation sparse" is occupied control-engineering territory.**

The KYY residual must involve the *family* of input-switched transitions / behavioral automaton and a substrate-specific resource model, not merely one LTI matrix.

---

# 8. Search result: Coxeter IR in sequence models

Targeted searches on 2026-08-10 for combinations of

```text
Coxeter / Dynkin
recurrent neural network
state tracking
permutation composition
simple reflections
sequence model
```

did not locate a modern LRNN/state-tracking paper whose central construction is explicitly:

> minimize the behavioral permutation representation, express it in a Coxeter simple-root basis, and use the Dynkin graph as a hardware-local recurrent IR.

That is **not evidence of novelty**. Coxeter/root-system theory is ancient, permutation networks and routing are mature, and state-space realization/sparsification are mature. The exact conjunction may simply use different language.

Status: **BRIDGE / UNMAPPED.**

---

# 9. The more interesting general compiler object

The S5 example suggests a compiler pipeline:

```text
task transition monoid/group
          |
          v
behaviorally sufficient representation
          |
          +---- choose realization / basis ----+
          |                                     |
          v                                     v
small state dimension                    cheap local primitives
          \                                     /
           \                                   /
            +------ hardware-aware cost -------+
                           |
                           v
             exact / relation-aware token words
```

The optimization is multi-objective:

```text
state dimension
primitive support / wire span
parallel word depth
gate arithmetic
metric/conditioning
relation defect
long-horizon behavioral drift.
```

This is much closer to a **realization compiler** than to a new recurrent neural layer.

---

# Current pin after Pass 8

A candidate KYY/TWC-shaped question is now:

> **Among behaviorally equivalent recurrent realizations of a transition monoid/group, which realization minimizes execution cost on a declared local substrate, and can the compiler certify exact relations / long-horizon behavior rather than merely matrix approximation error?**

Every ingredient has deep prior art. The exact input-switched + behavioral + hardware-local joint optimization remains a search target, not a novelty claim.
