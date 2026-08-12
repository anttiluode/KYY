# Physics does not supply semantics

Date: 2026-08-12

This is a standing guardrail for the Geometric Neuron -> WidePresent -> KYY line.

> **A physical substrate can supply trajectories, delays, bases, timescales, schedules,
> conservation laws and locality constraints. It does not automatically supply the
> task semantics of those states.**

That distinction has now been rediscovered by several independent failures in this
project. Writing it down should prevent the next version of the same mistake.

---

## 1. GeometricNeuron_V20: geometry did not automatically supply directed meaning

The V20 recomposition already preserves several relevant kills.

- The dendrite-as-delay-line picture survived as a useful physical abstraction.
- The stronger claim that branch length itself gives a clean phase-delay wall did not.
- A passive geometric medium did not create one-way flow: the V13 reciprocity test was
  exactly symmetric.
- Direction only appeared once active/history-dependent dynamics were introduced.

The lesson was already present there:

```text
physical geometry
    !=
directed computational semantics
```

A delay, angle or silhouette constrains what trajectories are possible. It does not by
itself say what a trajectory *means* to a receiver.

---

## 2. WidePresent: local field geometry did not automatically supply temporal meaning

The `LAKE_VS_LEDGER` attack found that a useful slow diffusive field diagonalized into
independent exponential modes.

In the linear regime the graph supplied:

```text
an eigenbasis
+
a spectrum of decay rates
```

but a linear readout had access to the same information in modal coordinates. The
geometry did not create a special temporal semantics; it selected a basis for a
familiar fading-memory computation.

Again:

```text
physical/local representation
    !=
automatically privileged meaning
```

---

## 3. KYY deadline gate: asynchrony did not automatically supply usable partial answers

The strongest recent failure is even cleaner.

`geom_scatter` is a product of local two-port operations. Removing global checkerboard
barriers makes partially completed work physically available earlier while preserving
the exact completed computation.

The naive prediction was:

> if partial work is available, the existing decoder should degrade gracefully as a
> deadline is swept.

It was false.

An ordinary final-only model could solve `perm3` perfectly after the full transition
while its own final readout mapped intermediate states catastrophically below chance.
Separate linear probes showed that the answer was already strongly decodable in those
states.

So the failure was not primarily lack of information. It was **semantic/frame
misalignment**:

```text
information present
    !=
information legible to this receiver
```

Shared phase supervision repaired legibility, but that repair is a learned alignment
constraint and has direct neighbors in deep-supervision / early-exit work. It is not a
free gift from the delay mesh.

---

## 4. The general decomposition

For future claims, separate at least four questions.

### Availability

Is useful information physically present somewhere in the current state?

### Identifiability

Could an appropriate observer recover it under the actual noise and state budget?

### Legibility

Can the *existing downstream receiver* interpret it without a phase/depth-specific
coordinate transform?

### Causal usefulness

Does making it available/legible improve behavior, latency, robustness, energy or some
other measured objective?

A substrate can win one and lose the others.

---

## 5. Biological caution

Do **not** claim that brains literally deep-supervise checkerboard phases.

The KYY training trick is an engineering intervention:

```text
same linear readout
+
loss after several homogeneous computation phases
```

Biological brains do not expose an obvious analogue of that exact objective.

The neuroscience-side hypothesis should therefore be phrased only at the functional
level:

> **If downstream circuits can act on upstream populations before all local processing
> has settled, some mechanism must keep receiver-relevant variables sufficiently
> interpretable across the encountered maturation states.**

Possible mechanisms could involve learned communication subspaces, recurrent feedback,
neuromodulation, attractor structure, population geometry, synchronization, or simply
receiver-specific adaptation. Which mechanism biology uses is an empirical question.

Output-null / output-potent language is useful only relative to a specified receiver.
A direction can be null for one target and potent for another.

---

## 6. A useful design rule for KYY

Before attributing a capability to local geometry, ask:

```text
Did the substrate create information?
Did it merely schedule when information became available?
Did the readout/training create the semantics?
Would a non-geometric sparse dataflow graph get the same benefit?
```

The current deadline work has evidence for the second item and strong evidence that the
third matters. The fourth is the next gate.

---

## 7. What the original accident still contributes

This rule does not erase the Geometric Neuron line.

The original accident kept attention on things mainstream vector abstractions often
idealize away:

```text
finite propagation
local interactions
history-dependent physical state
interference / phase
receiver geometry
energy and irreversibility
```

Those constraints can still be useful inductive biases or implementation advantages.
But the project should never again infer:

```text
physical structure exists
therefore
its computational interpretation comes for free
```

The honest remaining question is narrower and stronger:

> **When semantics are trained fairly for every competitor, does the physical/local
> constraint buy a measurable systems advantage?**

That is the gate KYY should now finish.