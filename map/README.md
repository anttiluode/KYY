# KYY / map

This folder is the **research cartography layer** for KYY.

It exists because the project repeatedly derived something exciting, built it, and only afterwards discovered the nearest established name. The rule is:

> **Search first. Locate the nearest known object. Compute the residual. Build only the residual.**

`UNMAPPED != NEW`.

Snapshot: **2026-08-10**.

## Where the map has moved

KYY started with:

> Can useful recurrent computation arise from **local geometry + propagation** rather than unrestricted global state mixing?

After repeated prior-art subtraction, the question is no longer one architecture.

It is a compiler/resource question across several layers:

```text
BEHAVIOR
 finite automaton / transition monoid / associative memory
        |
        v
ALGEBRAIC DECOMPOSITION
 group/permutation factors     reset/aperiodic factors
        \                       /
         \                     /
          v                   v
STATE REPRESENTATION
 one-hot / MinMax / harmonic orbit / other continuous code
                |
                v
PHYSICAL PRIMITIVE TYPE
 conservative transport  contraction  singular write/reset
                |
                v
LOCAL ORGANIZATION
 Sigma-chain predecessor links / graph routing / write-site placement
                |
                v
COST
 dimension + margin + precision + communication + reset + drift + garbage
```

The Geometric Neuron intuition survives only if geometry earns something **after all of these costs are visible**.

---

# Start here — current frontier

1. **[PASS23_HARMONIC_COUNTER.md](PASS23_HARMONIC_COUNTER.md)** — current mathematical seam: exact cyclic state tracking by a harmonic/group frame with `O(log n)` real dimensions and constant geometric state margin. Known frame mathematics; recurrent resource interpretation not located yet.
2. **[cyclic_harmonic_state_oracle.py](cyclic_harmonic_state_oracle.py)** — exact block-rotation recurrent construction, margin/coherence diagnostics, concentration bound, random frequency search.
3. **[PASS22_STATE_PACKING_FLOOR.md](PASS22_STATE_PACKING_FLOOR.md)** / **[state_packing_floor.py](state_packing_floor.py)** — bounded state dimension, dynamic range, and precision cannot all be free.
4. **[PASS20_SIGMA_AFFINE_WITNESS.md](PASS20_SIGMA_AFFINE_WITNESS.md)** / **[sigma_affine_compiler.py](sigma_affine_compiler.py)** — exact switched-affine lowering of the July-2026 Sigma-chain; includes the literal neighbour-propagating irreversible front witness.
5. **[PASS19_SIGMA_LOCAL_RECURRENT_COROLLARY.md](PASS19_SIGMA_LOCAL_RECURRENT_COROLLARY.md)** — immediate-predecessor-only permutation/reset recurrent realization for all regular languages, as a corollary/translation of the Sigma-chain theorem.
6. **[PASS18_ORBIT_PRECISION.md](PASS18_ORBIT_PRECISION.md)** — continuous group-orbit realization trades recurrent coordinate count against state separation/precision.
7. **[PASS16_RESET_LEAKAGE_RESULT.md](PASS16_RESET_LEAKAGE_RESULT.md)** / **[PASS15_OBSERVABLE_RESET_LEMMA.md](PASS15_OBSERVABLE_RESET_LEMMA.md)** — empirical and algebraic reset result: the lossless local scatterer learned to hide reset history, not forget it.
8. **[PASS17_WRITE_SITE_PARETO.md](PASS17_WRITE_SITE_PARETO.md)** — exact small-`n` trade-off between number/location of irreversible write sites and reversible routing depth.
9. **[PASS21_FSM_STATE_ASSIGNMENT_BOUNDARY.md](PASS21_FSM_STATE_ASSIGNMENT_BOUNDARY.md)** — subtracts forty years of hardware state-assignment / graph-embedding prior art.
10. **[PASS12_LAKE_FAST_WEIGHT_BOUNDARY.md](PASS12_LAKE_FAST_WEIGHT_BOUNDARY.md)** — read-before-write lake picture lands directly on fast-weight / DeltaNet and photorefractive delta-rule prior art.
11. **[PASS11_WAVE_RESET_SIGMA_CHAIN.md](PASS11_WAVE_RESET_SIGMA_CHAIN.md)** — Krohn-Rhodes permutation/reset split and the new Sigma-chain locality theorem.
12. **[PASS13_TRANSPORT_AND_PINCH.md](PASS13_TRANSPORT_AND_PINCH.md)** / **[PASS14_THERMODYNAMIC_BOUNDARY.md](PASS14_THERMODYNAMIC_BOUNDARY.md)** — reversible routing + singular merge compiler floor; forgetting/energy theory is already established.
13. **[PASS8_COXETER_IR.md](PASS8_COXETER_IR.md)** / **[s5_coxeter_oracle.py](s5_coxeter_oracle.py)** — exact full-`S5` state-tracking compiler in a 4D local Coxeter representation.
14. **[PASS7_DISCRETE_SYNTHESIS.md](PASS7_DISCRETE_SYNTHESIS.md)** / **[exact_perm3_local_oracle.py](exact_perm3_local_oracle.py)** — burns `perm3` as evidence by giving it a trivial exact local oracle.
15. **[PASS5_CONTROL_ALGEBRA.md](PASS5_CONTROL_ALGEBRA.md)** / **[RING_ALGEBRA_LEMMA.md](RING_ALGEBRA_LEMMA.md)** — graph/control placement determines reachable Lie algebra, but large Lie dimension is not itself the task.
16. **[OCCUPANCY_MATRIX.md](OCCUPANCY_MATRIX.md)** / **[LANDSCAPE.md](LANDSCAPE.md)** / **[SEARCH_LOG.md](SEARCH_LOG.md)** / **[SOURCES.md](SOURCES.md)** — wider map and search history.

---

# The strongest empirical result so far

The mixed benchmark `permreset3` has tokens:

```text
I: identity
C: 0 -> 1 -> 2 -> 0
R: every prior state -> 0
```

Three-seed, 1000-step result, state dimension 8:

| model | L=16 | L=64 | L=256 |
|---|---:|---:|---:|
| complex diagonal | 1.0000 | 1.0000 | 1.0000 |
| Householder-2 | 0.9204 | 0.6766 | 0.4714 |
| local orthogonal scatter | **0.9946** | 0.6096 | 0.4213 |
| GRU | 1.0000 | 1.0000 | 1.0000 |

For the local orthogonal scatterer, two different histories become almost indistinguishable **at the reset token**:

```text
lag 0 mean probability TV ~= 0.000169
argmax mismatch = 0
```

but the same continuation rotates the preserved hidden difference back into the readout:

```text
lag 4   mismatch ~= 2.5%
lag 8   mismatch ~= 11.9%
lag 16  mismatch ~= 15.3%
lag 64  mismatch ~= 16.3%
```

So the clean sentence is:

> **The lossless local KYY body learned to hide reset history, not to forget it.**

This is a result about this model/task pair, not a universal claim about waves.

`complex_diag` is technically invertible at each finite step but contractive; it achieved sampled 100% through L=256 with zero post-reset prediction mismatches. Therefore the useful physical split is not simply invertible/noninvertible:

```text
CONSERVATIVE
    distinctions persist

CONTRACTIVE
    distinctions fade

SINGULAR RESET
    distinctions are explicitly removed in the effective realization
```

---

# The newest exact scaling result: harmonic cyclic state

The simplest phase counter stores `n` cyclic states around one circle:

```text
v_j = (cos(2*pi*j/n), sin(2*pi*j/n)).
```

It uses only two real coordinates, but nearest-state noise radius shrinks as

```text
sin(pi/n) ~ pi/n.
```

Pass 23 instead chooses `k` Fourier characters/frequencies and stores

```text
v_j = 1/sqrt(k) [
 cos(2*pi*f_1*j/n), sin(...),
 ...,
 cos(2*pi*f_k*j/n), sin(...)
].
```

The increment token remains a **fixed block-diagonal bank of 2D rotations**.

If frequencies are sampled uniformly, character orthogonality + Hoeffding + a union bound gives

```text
P[max nontrivial state inner product >= alpha]
    <= (n-1) exp(-k alpha^2 / 2).
```

Thus some frequency multiset exists with constant state separation whenever

```text
k > 2 log(n-1) / alpha^2.
```

So, in ideal arithmetic:

> **`C_n` has an exact norm-preserving recurrent realization in `O(log n)` real dimensions with constant Euclidean symbolic-state margin.**

This is harmonic-frame / random-character mathematics, not claimed as a new theorem.

What has not yet been located is this exact **recurrent state-resource interpretation** in the modern 2026 state-tracking literature.

Important caveat:

```text
constant margin != error correction.
```

Affine exact group motion does not create an attracting symbolic state. It only gives accumulated implementation error a larger geometric runway.

---

# Why the harmonic result matters to the current literature

Two extremely recent papers bracket it.

## Complex State Propagator — August 4, 2026

Uses complex phase rotations for deterministic state tracking and explicitly represents cyclic/parity behavior in phase.

Occupied claim:

```text
complex rotations can implement modular state transitions.
```

Current gap:

```text
how many phase modes are required at fixed state-separation margin as modulus grows?
```

## Error Control Dynamics — May 2026

Shows that affine state-tracking failures are governed by accumulated within-state error relative to minimum between-state separation, and that exact affine return dynamics cannot selectively contract the state-separating error directions.

So Pass 23's harmonic frame should be interpreted as an **error-budget code**, not a restoring memory.

---

# The Sigma-local bridge

The July-2026 Sigma-chain theorem says every regular language can be represented by a chain of permutation-reset automata in which each component depends only on:

```text
external token
    +
immediate predecessor state.
```

KYY now has an exact recurrent lowering of those components:

```text
permutation -> orthogonal update
reset       -> singular affine overwrite.
```

This suggests a compiler rather than one monolithic recurrent layer:

```text
DFA / regular behavior
        |
        v
Sigma-chain decomposition
        |
        +--> group/permutation factor
        |       -> choose a robust compact group-frame state code
        |       -> conservative phase/rotation update
        |
        +--> reset/aperiodic factor
                -> contractive/singular update only where required
        |
        v
neighbour-only recurrent chain
```

The Sigma-chain paper is only weeks old and targeted searches have not located a neural/physical lowering or a combination with the May-2026 MinMax recurrent-cascade work.

That is a chronology-based reason to keep digging, **not evidence of novelty**.

---

# Hard prior-art walls now on the map

```text
local 2-port recurrent mesh             -> EUNN / Givens / optical meshes
Householder recurrence                  -> oRNN / DeltaProduct
oscillator / graph-wave RNN             -> coRNN / GraphCON / wave-RNN work
input-dependent recurrent operator      -> ISAN / Mamba / dynamic filters
read-before-write delta memory          -> fast-weight programmers / DeltaNet
sparse delta memory                     -> Sparse Delta Memory 2026
physical delta learning                 -> photorefractive optical NNs 1989+
Lie algebra explains state order        -> 2026 sequence Lie-algebra theory
few local controls generate global flow -> classical network/quantum control
permutation + reset -> arbitrary FSA    -> Krohn-Rhodes
neighbour permutation-reset chain       -> Sigma-chain July 2026
S_n + one singular map -> T_n           -> transformation-semigroup theory
graph-local reset word length           -> digraph transformation semigroups
state encoding for physical cost        -> FSM state assignment / VLSI graph embedding
harmonic group orbits                   -> harmonic/group frame theory
state separation as robustness budget   -> coding/frame theory + 2026 error-control work
```

The map is crowded. Good.

---

# Current KYY hypothesis — version 2026-08-10 afternoon

The strongest surviving research direction is not:

> geometry replaces attention.

It is:

> **Compile behavior into algebraically typed local recurrent factors, and choose a geometric state code for each factor that optimizes the trade among dimension, symbolic-state margin, communication, irreversible reset cost, and long-horizon error control.**

The most concrete possible next step is the **abelian group-factor case**:

```text
finite abelian factor G
    -> choose O(log |G|) characters
    -> harmonic-frame state code
    -> diagonal phase token update
    -> constant geometric state margin
```

then insert those factors into an exact Sigma-local permutation/reset chain.

Before implementing a learned version, close the prior-art search on:

```text
harmonic/group frames as recurrent finite-group state codes
robust logarithmic modular counting state dimension
character-code recurrent automata
Sigma-chain + neural recurrent cascades
```

---

# Mandatory build gate

Before another learned architecture is added to `kyy/models.py`, add this to `SEARCH_LOG.md`:

```text
DATE:
PROPOSED OBJECT:
NEAREST KNOWN OBJECTS:
EXACT OVERLAP:
RESIDUAL DIFFERENCE:
WHY THE RESIDUAL MATTERS:
PRIMARY-SOURCE SEARCH QUERIES:
PAPERS INSPECTED:
WHAT WAS NOT LOCATED:
CHEAPEST TEST WITHOUT A NEW ARCHITECTURE:
CHEAPEST FALSIFIER IF CODE IS REQUIRED:
BASELINES THAT CAN KILL IT:
RESULT THAT STOPS THE BRANCH:
```

If `RESIDUAL DIFFERENCE` cannot be written precisely, **keep mapping**.
