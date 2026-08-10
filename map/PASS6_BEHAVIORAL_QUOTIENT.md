# Pass 6 — hidden operator algebra versus the behaviorally visible quotient

Date: 2026-08-10

Pass 5 asks what operator algebra a small set of local controls can generate. Pass 6 adds an equally important warning:

> **The full hidden algebra is not automatically the computation.**

A recurrent model is judged through its outputs. Hidden directions may carry nuisance rotations, clocks, redundant coordinates, or other dynamics that never change the task answer. Therefore a model can solve a finite-state task without its full hidden matrices forming an exact representation of the target finite group.

This is not new mathematics. It is the recurrent/automata version of the same realization lesson that TWC hit in physical inverse problems: **ports identify an input/output equivalence class, not arbitrary internal coordinates.**

---

## 1. Why the obvious group-relation test is too strong

KYY's `perm3` task uses three symbols:

```text
e = identity
s = transposition
r = 3-cycle
```

The abstract target group obeys

```text
s^2 = e
r^3 = e
s r s = r^-1.
```

If a hidden linear model were literally a faithful matrix representation `rho` of `S3`, its token transitions would satisfy those equations everywhere in hidden space.

But supervised state tracking only requires

```text
readout(hidden_after_word) = correct finite state.
```

It does **not** require every unused hidden direction to obey the `S3` relations.

A model can instead realize a larger extension:

```text
hidden dynamics H
      |
      v
behavioral quotient
      |
      v
     S3 state
```

Two hidden states are behaviorally equivalent when no relevant future input can make their outputs differ. Exact hidden-matrix equality is therefore stronger than task equivalence.

This is why [operator_algebra_audit.py](operator_algebra_audit.py) reports global relation defects but labels them **diagnostics, not correctness conditions**.

---

## 2. This territory is heavily mapped

### Automata extraction from RNNs

- Weiss, Goldberg & Yahav, *Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples* (2017), https://arxiv.org/abs/1711.09576
- Okudono et al., *Weighted Automata Extraction from Recurrent Neural Networks via Regression on State Spaces* (2019), https://arxiv.org/abs/1904.02931

These works explicitly recover finite/weighted automata that summarize trained recurrent behavior.

### Weighted automata and bilinear/second-order RNNs

- Rabusseau, Li & Precup, *Connecting Weighted Automata and Recurrent Neural Networks through Spectral Learning* (2018), https://arxiv.org/abs/1807.01406

For discrete sequences, weighted finite automata and linear second-order RNNs have an exact expressive connection, with Hankel/spectral methods supplying minimal linear representations.

### Realization/minimality

- Defourneau & Petreczky, *Realization theory of recurrent neural networks and rational systems* (2019), https://arxiv.org/abs/1903.05609

Classical realization theory already separates reachable/observable behavior from redundant hidden coordinates.

### Modern state tracking

The current LRNN literature studies group/automaton tracking directly, including permutation composition, code execution, and partially observable probabilistic automata.

- Grazzi et al., negative-eigenvalue state tracking (ICLR 2025)
- Shakerinava et al., diagonal-SSM group limits (ICLR 2026)
- Siems et al., *Learning State-Tracking from Code: REPL Traces and Probabilistic Automata* (2026)

So **"a recurrent net has an automaton hidden inside it" is not a KYY opening.**

---

## 3. Why this still changes the control-algebra question

Pass 5 could be misread as:

> maximize `dim Lie{G,B1,...,Br}` with as few controls as possible.

That is not the actual objective.

A full `so(N)` hidden algebra may be impressive and completely wasteful if the task needs only a six-state quotient.

The more relevant object is something like:

```text
local control system
      |
      v
hidden reachable dynamics
      |
      v
observable / predictive quotient
      |
      v
target transition monoid/group
```

The architecture should be priced by the smallest local control system whose **behavioral quotient** contains the target task dynamics.

This makes `control-algebra efficiency` stricter:

```text
not:  Lie dimension / number of ports

but:  task-relevant quotient capability
      --------------------------------
      ports × local depth × communication cost
```

The numerator needs a careful definition; raw hidden dimension is not enough.

---

## 4. A finite-horizon diagnostic we can actually run

For an affine recurrent family

```text
h' = A_x h + b_x
y  = C h + d
```

augment the hidden state with a constant `1` so every token becomes a linear matrix `A~_x`.

For a future word `w`, the future output depends on

```text
C~ A~_w h~.
```

Stack these maps over all words up to horizon `H`:

```text
O_H = [
    C~
    C~ A~_x
    C~ A~_x A~_y
    ...
].
```

Directions in `ker(O_H)` are invisible to all tested futures up to that horizon.

This gives a practical, finite approximation to a behavioral quotient:

```text
hidden state / future-output-indistinguishable directions.
```

This is ordinary observability logic applied to an input-switched family, not a new theorem.

A better KYY audit can therefore report both:

```text
full hidden noncommutativity
behaviorally visible noncommutativity
```

rather than confusing the two.

---

## 5. An even cheaper task-specific check

For `perm3`, compare pairs of words that are equal in `S3`:

```text
ss       ~ empty
rrr      ~ empty
srs      ~ r^-1
```

Instead of requiring the hidden vectors or matrices to be equal, ask whether the **future behavior distributions / logits / classifications remain equivalent under continuations**.

This is closer to a Myhill-Nerode style test:

```text
u ~ v
iff
for every tested continuation z,
output(uz) == output(vz).
```

The continuation test can expose whether a model really learned a stable state quotient or merely memorized the training horizon.

That may be especially useful for the current observation that `householder2` extrapolates much more reliably than some local-scatter seeds despite both fitting the training length.

---

## 6. New design target after Pass 5 + Pass 6

The map now points to a three-layer optimization problem:

```text
GEOMETRY / LOCAL CONTROLS
        |
        v
GENERATED HIDDEN ALGEBRA
        |
        v
BEHAVIORAL QUOTIENT
        |
        v
TARGET AUTOMATON / STATE-TRACKING TASK
```

The interesting architecture/compiler question is not whether the hidden algebra is huge.

It is:

> **Can a very small, locally addressable physical/control algebra realize the required behavioral quotient with short control words and stable length extrapolation?**

This is still a bridge among known areas: graph controllability, bilinear RNNs, automata/minimal realization, and modern state-tracking benchmarks.

Status: **BRIDGE / UNMAPPED. NOT A NOVELTY CLAIM.**

---

## 7. Relation to TWC

This is the first point where the analogy is almost literal rather than poetic.

TWC learned the hard way that different internal reciprocal networks can be indistinguishable at the measured ports. The honest object is therefore an identifiable response/realization class.

KYY has the same structural warning:

```text
hidden operator != behaviorally identifiable operator.
```

That does **not** make TWC machinery directly applicable to AI. It says the same mathematical discipline should be used:

> optimize and interpret the part of the operator that survives the declared ports/readout, and report the invisible freedom separately.

---

# Current combined pin

Pass 5:

> Can a small local control interface generate the right noncommutative algebra cheaply?

Pass 6:

> Which part of that generated algebra is actually required and visible at the task readout?

Together:

> **Find the smallest local control geometry whose behaviorally visible quotient implements the required state-transition algebra.**

That is now the sharpest KYY formulation on the map.
