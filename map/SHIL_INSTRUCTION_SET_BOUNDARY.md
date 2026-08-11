# Boundary — one phase circle cannot realize every digital quotient

Date: 2026-08-11

This is a substrate/instruction-set result, not a novelty claim for monotone circle maps, harmonic generation, frequency multiplication, SHIL, or oscillator logic.

## Question

The equal-block SHIL compiler rejects the C4 alternating quotient

```text
{0,2} -> A
{1,3} -> B
```

because each target class is interleaved around the physical phase circle.

Could a longer sequence of otherwise legal one-circle operations realize it anyway?

For the current instruction set, no.

---

## 1. One-circle instruction set

Allow any sequence of:

### reversible phase permutations

```text
q -> +/- q + a  (mod k)
```

These are the dihedral rotations/reflections of the current cyclic phase code.

### uniform SHIL quotients

For `m | k`, collapse equal contiguous blocks of `C_k` into `m` phase basins.

These are the quotient stages compiled in `SHIL_CYCLIC_QUOTIENT_COMPILER_RESULT.md`.

Both primitive types are cyclic-monotone:

- reversible operations preserve cyclic order up to reversal;
- irreversible operations collapse contiguous arcs but do not interleave them.

---

## 2. Closure gives an impossibility result

A composition of cyclic-monotone maps remains cyclic-monotone.

Equivalently, the preimage/fiber of every final state remains a contiguous cyclic interval.

Reason:

1. each primitive has connected/contiguous point fibers;
2. the preimage of a contiguous interval under a cyclic-monotone map is contiguous;
3. therefore for `g o f`,

```text
(g o f)^-1(y) = f^-1(g^-1(y))
```

is contiguous whenever `g^-1(y)` is contiguous.

Thus an interleaved final kernel cannot be produced by any length sequence in this instruction set.

This is standard monotone-map geometry expressed as a backend lowering rule, not a new topological theorem.

---

## 3. Exhaustive C4 check

`map/shil_instruction_set_boundary.py` enumerates the finite instruction semigroup for C4.

Starting from four distinct phase states, only three canonical partitions are reachable under arbitrary dihedral permutations and uniform equal-block quotients.

The relevant controls are:

```text
adjacent pair quotient
[0,0,1,1]
reachable: YES
witness:   Q4->2

alternating quotient
[0,1,0,1]
reachable: NO
```

So the alternating case is not merely absent from a short search.

It is outside the one-circle instruction set.

---

## 4. A harmonic operation changes the topology

Let the physical state be a phase

```text
phi_q = 2*pi*q/n.
```

A coherent `h`-th harmonic carries phase

```text
h phi_q  (mod 2*pi).
```

Two fine states become identical in that harmonic exactly when

```text
h(q1-q2) = 0 mod n.
```

So the harmonic map has a subgroup/congruence kernel rather than a contiguous-arc kernel.

For C4 with `h=2`:

```text
q=0 -> phase 0
q=1 -> phase pi
q=2 -> phase 0
q=3 -> phase pi
```

therefore

```text
{0,2} -> 0
{1,3} -> pi.
```

The exact rejected alternating quotient appears immediately.

The exhaustive artifact reports

```text
second harmonic partition = [0,1,0,1]
output phase count = 2
matches target = true.
```

Frequency/harmonic multiplication is established physical technology. KYY does not claim this primitive as new.

---

## 5. Port versus body still matters

Merely **measuring** the second harmonic is only a new observable/port.

The original fundamental phase can still retain the full C4 state.

To implement the irreversible symbolic transition in the body, a physical backend must do something stronger, for example:

```text
fundamental C4 carrier
    -> coherent second-harmonic / doubled-phase carrier
    -> transfer or lock state into the two-phase carrier
    -> discard / decouple the original fine carrier.
```

Only then have `0` and `2` actually become the same physical future state rather than merely producing the same measurement.

That is the same distinction Pass 44 exposed in software:

```text
quotient observed at the port
!=
quotient enforced in the body.
```

---

## 6. Two different physical quotient geometries

We now have two qualitatively different backend instructions.

### Basin quotient

Uniform SHIL landscape:

```text
merges contiguous arcs of phase
```

Example:

```text
{0,1}/{2,3}.
```

This is geometric adjacency on the circle.

### Harmonic quotient

Phase multiplication:

```text
phi -> h phi
```

merges congruence/subgroup classes.

Example:

```text
{0,2}/{1,3}.
```

These classes are interleaved in the original circle.

So the abstract algebra can ask for the same cardinality reduction while the physical backend needs completely different instructions depending on the **shape of the kernel**.

That is a concrete compiler distinction at the digital/analog border.

---

## 7. Resource diagnostic

For a requested transition kernel, the compiler can now answer:

```text
1. contiguous equal cyclic fibers?
      -> one uniform SHIL quotient may lower directly

2. subgroup/congruence fibers compatible with phase multiplication?
      -> harmonic carrier is a natural candidate

3. neither?
      -> require richer nonuniform forcing,
         auxiliary physical state,
         a different embedding,
         or reject this backend.
```

That begins to look like actual instruction selection rather than saying "use oscillators."

The hardware cost is also meaningful: harmonic extraction, extra carrier/state, locking circuitry, and destruction of the old carrier are not free.

---

## 8. Prior-art boundary

Do not claim as new:

- monotone/cyclic-order map closure;
- harmonic or frequency multiplication;
- coherent second-harmonic phase;
- phase-encoded oscillator logic;
- multi-phase oscillator memories;
- SHIL/Potts oscillator hardware.

Current oscillator/Potts work already establishes multi-phase physical states and phase-sensitive readout, while nonlinear electronics/optics routinely generate harmonic carriers.

The KYY-shaped residual is the compilation problem:

> **classify the declared behavioral kernel by the geometry of its fibers, choose a physical quotient instruction whose kernel matches, and distinguish a mere observable quotient from actual body-level forgetting.**

This is still a research direction, not an established novelty claim.

## Files

- `map/shil_instruction_set_boundary.py`
- `tests/test_shil_instruction_set_boundary.py`
- `.github/workflows/shil-instruction-set-boundary.yml`
- `results/shil_instruction_set_boundary_summary.json`

Workflow evidence: Actions run `31459247444`, focused tests green.
