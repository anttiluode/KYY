# Pass 27C — subtract AUSSM, Error Control Dynamics, and approximate-representation stability

Date: 2026-08-10

Pass 27/27A still left three neighboring literatures insufficiently explicit. They materially narrow the harmonic-state story again.

---

## 1. AUSSM already owns unit-modulus adaptive modulo tracking

Karuvally et al., *Bridging Expressivity and Scalability with Adaptive Unitary SSMs* (NeurIPS 2025 / arXiv:2507.05238), introduce AUSSM: an input-dependent unitary/complex recurrent SSM.

For modulo `k`, their construction explicitly chooses a `k`-th root of unity as the recurrent update. Their finite-precision discussion also treats numerical error in repeated unitary rotations.

Therefore KYY must subtract:

```text
unit-modulus complex recurrence for modulo counting
roots of unity as exact cyclic state transitions
adaptive unitary SSMs for Abelian/cyclic automata
finite-precision concerns for repeated root-of-unity updates
Krohn-Rhodes combinations of cyclic and reset-like factors in SSMs
```

This is a much closer neighbor than RoPE for the finite-state use case.

The distinction left by Pass 23 is not "use roots of unity." AUSSM already does that.

The narrower representation question is whether **multiple exact characters used redundantly as one symbolic state code** can buy a better state-separation / implementation-error trade at a specified mode budget than the one-phase construction.

---

## 2. Error Control Dynamics already owns the separation-vs-drift framing

Chung, Choi & Kim, *Rethinking State Tracking in Recurrent Models Through Error Control Dynamics* (arXiv:2605.07755, May 2026), argue that state-tracking failure is governed by accumulated within-state error relative to between-state separation. They show that affine recurrent trackers cannot self-correct state-separating drift once the symbolic representation is preserved, and empirically connect a distinguishability ratio to failure horizon.

Therefore KYY must also subtract the broad claim:

```text
state-tracking robustness depends on
between-state separation versus accumulated drift.
```

That is already a current state-tracking result.

What KYY can test more narrowly is a **representation-design intervention**:

```text
hold the exact symbolic algebra fixed
hold mode count / norm fixed
change only the character set
measure whether the resulting orbit margin changes the drift runway.
```

The first `C_31`, 8-mode, three-seed sweep gives exploratory evidence for this narrow intervention: under the same systematic phase-error model, larger exact-code orbit radius is positively associated with length-1024 accuracy. This needs replication across moduli, mode budgets and noise models.

---

## 3. Relation defect is approximate-representation / Ulam-stability territory

The quantity

```text
||A^n - I||
```

is not a new mathematical concept. It is a special-case relation defect for an approximate representation of the cyclic group.

The general question "if group multiplication/relations are only approximately satisfied, is the map close to a genuine representation?" belongs to the theory of approximate representations and Ulam stability. Relevant examples include Burger, Ozawa & Thom (2010) and Gowers & Hatami (2015/2016), with much older antecedents.

For KYY's diagonal `C_n` bank the projection is elementary: each learned angle can be rounded to its nearest exact character,

```text
f_i = round(n*theta_i/(2*pi))
theta_i <- 2*pi*f_i/n.
```

So KYY must not claim novelty for:

```text
measuring approximate satisfaction of group relations
seeking a nearby exact unitary representation
projecting a near cyclic rotation onto an n-th root of unity
```

---

## 4. The compiler question becomes operational

These collisions suggest a much more concrete KYY operation:

```text
FREE TRAINING
    learn approximate recurrent dynamics
        |
        v
ALGEBRA AUDIT
    measure task relations, e.g. ||A^n-I||
        |
        v
LEGALIZATION / PROJECTION
    map the learned operator onto a nearby exact representation
        |
        v
RESOURCE SELECTION
    among exact legal representations,
    choose one with favorable margin / precision / wiring cost
        |
        v
ROLLOUT TEST
```

None of the individual arrows is novel.

The useful empirical question is whether **this compiler sequence repairs or predicts trained state tracking in practice**, and whether selecting among equivalent exact representations by geometric margin improves implementation robustness.

---

## 5. Projection experiment now running in KYY

`map/harmonic_training_probe.py` now includes a zero-shot legalization test for the unconstrained learned oscillator bank and standard RoPE control:

1. train exactly as before;
2. leave the learned linear readout untouched;
3. snap each recurrent angle to its nearest exact `C_n` character;
4. evaluate again from length 16 through 1024;
5. report projected orbit radius and relation defects.

No readout fine-tuning is allowed in this first projection test.

This makes the result sharp:

- if short-horizon accuracy collapses, nearest-representation legalization is not behavior-preserving enough to be useful in this setting;
- if short accuracy survives but long accuracy remains poor, exact algebra alone does not repair the learned decoder/orbit geometry;
- if long accuracy returns, KYY has a concrete **train approximately -> compile exactly** operation worth testing much harder.

---

## 6. Current novelty posture

Occupied:

```text
roots-of-unity modulo recurrence
adaptive unitary SSMs
Abelian group tracking by diagonal complex recurrence
Krohn-Rhodes cyclic/reset SSM decompositions
separation-vs-drift state-tracking theory
approximate unitary representations / Ulam stability
projection of a near cyclic phase onto an exact root of unity
```

Residual worth testing:

```text
compiler-level legalization of a trained recurrent operator
plus resource-aware choice among behaviorally equivalent exact representations
using orbit margin / precision / wiring / horizon as explicit costs.
```

Status:

**THE THEORY INGREDIENTS ARE OCCUPIED. THE POSSIBLE CONTRIBUTION IS NOW A COMPILER PROCEDURE + EMPIRICAL RESOURCE LAW.**
