# Readout fibers — a cleaner language for unfinished computation

Date: 2026-08-12

This note is not a novelty claim. It is a compact mathematical language for a pattern
that has now appeared repeatedly from GeometricNeuron_V20 through KYY.

The recurring mistake has been to conflate:

```text
what information the physical state contains
```

with

```text
what a particular downstream receiver can read from that state
```

Those are different maps.

---

## 1. Separate dynamics from semantics

Let the internal physical/computational state be `h` and let its ongoing dynamics be

```text
hdot = F(h, u)
```

A receiver does not usually see or interpret all of `h`. It applies some map

```text
y = R(h)
```

where `y` is the receiver-relevant quantity: a class logit, motor command, message to
another area, threshold variable, etc.

The substrate `F` determines which trajectories are possible.
The receiver `R` determines which distinctions in those trajectories matter to that
receiver.

This is the simplest mathematical statement of the standing rule:

> **physics supplies state trajectories; semantics is receiver-relative.**

---

## 2. A meaning is a fiber / level set of the readout

For one receiver and one public value `y`, define

```text
M_y = { h : R(h) = y }
```

Many very different hidden states can therefore mean the same thing to the receiver.

If the machine keeps computing while its public meaning remains stable, its trajectory
can move substantially inside (or near) the same fiber.

For a differentiable readout, an infinitesimal hidden-state motion `dh` is locally
invisible to the receiver when

```text
J_R(h) dh ~= 0
```

where `J_R` is the readout Jacobian.

So the local private/receiver-null directions are

```text
N_R(h) = ker J_R(h)
```

For KYY's linear readout

```text
R(h) = W h + b
```

this reduces exactly to

```text
N_R = null(W)
```

which is the output-null diagnostic already measured in
`experiments/output_null_motion.py`.

---

## 3. Why this helps with mixed maturity

A barrier-free or asynchronous machine does not need every hidden coordinate to have
the same computational age.

But if an existing receiver must remain useful while local work continues, two kinds of
motion matter:

```text
potent motion
    J_R(h) dh != 0
    updates what the receiver sees

null/private motion
    J_R(h) dh ~= 0
    changes hidden computation without strongly changing the receiver's current value
```

A useful asynchronous computation can therefore look like:

```text
first:
    enough potent motion to establish/update the public variable

then:
    continued large private motion mostly tangent to the receiver's current fiber
```

This is only one possible solution. Synchronization, receiver adaptation, recurrent
feedback or other mechanisms could solve the same functional problem differently.

---

## 4. Receiver relative means receiver relative

There is no globally output-null direction.

With two receivers

```text
y_A = R_A(h)
y_B = R_B(h)
```

a motion can satisfy

```text
J_A dh ~= 0
```

while simultaneously

```text
J_B dh != 0
```

So the same ongoing hidden computation can be:

```text
private to receiver A
public to receiver B
```

This is the right way to use the neuroscience analogy. Output-null/output-potent is
always defined relative to a specified downstream target.

The random-receiver control in `experiments/output_null_transfer_control.py` exists
precisely to enforce this guardrail.

---

## 5. The V13 directionality failure was already a readout problem

GeometricNeuron_V20's V13 active-relay test attempted to create directionality with an
asymmetric read/write relay.

The relay read only scalar amplitude at one point.
A wave arriving from either direction can produce the same local amplitude.
So the receiver had discarded the variable required to distinguish direction before
its asymmetric write rule could use it.

In this language:

```text
physical wave state may distinguish direction
             |
             v
R_amplitude(h) collapses those states together
             |
             v
relay cannot act on direction it cannot observe
```

That is structurally the same lesson as the KYY phase-specific probe result:

```text
information can exist in h
while the chosen R makes it unusable
```

---

## 6. The original PerceptionLab accident looks different through this lens

The seed loop was roughly:

```text
homeostatic coupler
    -> checkerboard geometry
    -> image-to-vector (256 values)
    -> splitter exposing only 4 values
    -> feedback to coupler
```

The important operation is not merely that a geometric pattern existed.
A rich state was passed through a **restricted observation map** and then the observed
coordinates were closed back into the dynamics.

So the accidental spike-like behavior can be framed as a closed loop

```text
hdot = F(h, R(h))
```

where the choice of `R` is dynamically consequential.

This is a safer and more general interpretation than saying that geometry itself came
with a semantic code.

---

## 7. Readout becomes part of physics when the loop closes

In a feed-forward classifier, `R` can be treated as an observer attached after the
computation.

In a recurrent organism or control loop, receiver outputs alter future state:

```text
hdot = F(h, y, u)
y    = R(h)
```

Now the distinction between dynamics and readout remains conceptually useful, but the
readout is also part of the closed-loop dynamics.

This is exactly where the Geometric Neuron / PresentMoment lines can meet without
claiming a new neural architecture:

```text
upstream state evolves
    -> receiver sees only a projection
    -> receiver acts / feeds back
    -> that changes the future state and which signals are in flight
```

Different body and brain loops can have different receivers, delays and gains.
There need not be one global semantic vector or one global computational maturity.

---

## 8. A possible metric

For a trajectory `h(t)` and receiver `R`, measure receiver-relative semantic velocity:

```text
v_public(t) = || J_R(h(t)) hdot(t) ||
v_hidden(t) = || hdot(t) ||
```

and a local privacy ratio

```text
privacy_R(t) = 1 - ||Proj_row(J_R) hdot||^2 / ||hdot||^2
```

For linear KYY this is exactly the potent/null motion decomposition.
For nonlinear receivers it generalizes using the Jacobian.

The useful question is not whether hidden state is moving.
It is:

> **How much continuing motion changes what each receiver currently sees?**

---

## 9. Guardrail

Do not turn this vocabulary into a claim that brains explicitly optimize fibers,
Jacobians or null spaces.

These are analysis objects.

The empirical question is whether real interacting populations exhibit the functional
property:

> substantial ongoing internal change can coexist with relatively stable/useful
> receiver-specific variables before every contributing process has globally settled.

KYY supplies a toy system in which that property can be measured and attacked. It does
not establish the biological mechanism.