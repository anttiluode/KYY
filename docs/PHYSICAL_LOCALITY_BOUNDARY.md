# Physical locality boundary

Date: 2026-08-12

This note records where the KYY locality question currently stops, and what would be
required to reopen it honestly.

It is deliberately **not** another architecture proposal.

---

## 1. What the software deadline work already killed

The mixed-maturity / deadline experiments found that:

```text
sparse dependency structure + asynchronous scheduling
    -> can expose useful partial work before a global barrier
```

but arbitrary non-geometric random matchings can obtain the same qualitative effect and
in scratch runs could match or exceed the fixed ring.

Therefore KYY's fixed local ring has **not** earned a software anytime advantage.

The existing control explicitly does not charge:

```text
wire length
placement / routing
energy
physical propagation distance
analog locality
noise
```

That omission is now the entire remaining locality question.

---

## 2. What PerceptionLab added

The public `PerceptionLab` repo now contains a simple finite-propagation calibration:

```text
nodes/wavefieldnode.py
graph workflows/wave_field_probe.json
nodes/temporalmultiscopenode.py
```

A second-order field sends one pulse past spatially separated probes. The field predicts
adjacent-probe lag from distance and propagation speed. With the checked settings the
adjacent prediction was approximately `13.2` frames.

The Temporal Multi Scope averages best lag across all probe pairs. For four equally
spaced probes, the mean pair separation is `10/6` gaps, giving an expected mean absolute
lag of:

```text
13.2 * 10/6 = 22.0 frames
```

Manual runs reported approximately `21.3-22.2` frames.

This is not evidence for KYY. It is a **known-answer calibration** showing that the
instrumentation can recover a distance-derived propagation time when such a time really
exists in the model.

That removes one excuse from any future KYY physical-delay test: lag measurement itself
can be checked first on a system with known ground truth.

---

## 3. The only KYY-specific locality gate left

If this question is reopened, compare the fixed local KYY graph against an equally sparse
nonlocal graph **while charging physical distance**.

A minimal local-only experiment can be entirely synthetic and run without APIs, remote
models or paid services.

Give every state channel a physical coordinate `p_i`.

For an operation connecting `i,j`, define for example:

```text
wire_length_ij = distance(p_i, p_j)
prop_delay_ij  = base_gate_delay + wire_length_ij / propagation_speed
wire_cost_ij   = wire_length_ij
```

Then compare at matched:

```text
state dimension
number of two-port operations
number of trainable parameters
training objective
receiver/readout help
```

at least:

```text
A. KYY ring / local-neighbour scatter
B. arbitrary sparse random matchings
C. synchronized version of the same computation
```

Score:

```text
final task accuracy
deadline accuracy / AUC
critical-path latency
total messages / operations
total wire length
energy proxy if one is explicitly defined
```

Do **not** let the random graph teleport for free.

---

## 4. A stronger version: propagation state, not scalar edge cost

The cheap gate above assigns a scalar delay to each edge.

A more physical version would let an edge contain state while a signal is travelling,
rather than merely scheduling its arrival later.

That would distinguish:

```text
message scheduled for future delivery
```

from

```text
signal physically present along the connection now
```

The PerceptionLab Wave Field is the simplest demonstration of the second case: one current
spatial state contains the same event at different propagation stages.

KYY does not need this stronger version unless the cheap distance-cost gate first shows
that locality buys something.

---

## 5. Relation to the standing rule

The result does not alter `PHYSICS_DOES_NOT_SUPPLY_SEMANTICS.md`.

A physical propagation substrate can supply:

```text
delay
causal order
locality
wire cost
signals in flight
heterogeneous maturity
```

It still does not supply the task meaning of those states.

Receiver/training alignment remains a separate problem.

---

## 6. Stop rule

This is the boundary, not a mandate.

The KYY locality seam should remain closed unless somebody wants to test a real
physical-cost hypothesis.

A future result is interesting only if the local graph wins after nonlocal connections
are charged for distance / propagation, or if it demonstrates a hardware-relevant
robustness or energy advantage under a stated model.

If random sparse graphs remain as good after those costs are included, close the locality
claim.

Because the repositories are public, this gate is intentionally written so that anyone
can implement and rerun it locally without access to an API model.
