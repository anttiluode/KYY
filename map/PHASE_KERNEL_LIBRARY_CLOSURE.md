# Phase-kernel library closure: decide first, optimize second

Date: 2026-08-11

This note closes the current one-circle phase backend **as a finite decision procedure**.

It does **not** claim a complete physical instruction set for arbitrary finite-state quotients.

The point is the opposite: the current library is deliberately small enough that the compiler can terminate with an exact `supported / unsupported` answer and name the missing resource.

## 1. Current declared library

The backend currently knows four ideas:

```text
faithful cyclic phase re-encoding
uniform equal-basin SHIL collapse
pre-carried quotient-aligned character
runtime harmonic/carrier conversion for congruence kernels
```

The last two share the same abstract cyclic-character kernel but have different resource costs.

## 2. Correction to "prefer re-encoding over harmonic"

A faithful character `f` with `gcd(f,n)=1` is an automorphism of `C_n`.

A nontrivial proper cyclic character quotient is the coset partition of the unique subgroup of `C_n` of the corresponding order.

Every automorphism maps that subgroup to itself.

Therefore a nontrivial subgroup/coset kernel that is interleaved around the phase circle does **not** become a contiguous-arc kernel under any faithful re-encoding.

The exhaustive audit through `n<=16` finds zero collisions between:

```text
nontrivial proper character/congruence kernels
and
uniform contiguous-SHIL kernels under faithful one-circle re-encoding.
```

This is the finite check of a general cyclic-group fact, not a new group-theory theorem.

For `C4`:

```text
adjacent quotient     [0,0,1,1]
    character kernel: no
    uniform SHIL: yes

alternating quotient  [0,1,0,1]
    character f=2: yes
    uniform SHIL after any faithful f: no
```

So the cheap design-time alternative to runtime harmonic conversion is **not** "pick another faithful phase order."

It is:

```text
carry a faithful state character
+
carry a non-faithful quotient-aligned character from the start

then, when the quotient is required,
retire/decouple the modes that still distinguish inside the quotient class.
```

That creates a real resource trade:

```text
standing redundant state / extra carrier now
versus
runtime nonlinear harmonic/carrier conversion later.
```

Merely reading the quotient character is still a port operation. Body-level forgetting requires the distinguishing carrier/modes to be physically retired, damped, disconnected or otherwise made future-unobservable.

## 3. Total classifier for the current library

For any explicit finite partition of `C_n`, `phase_kernel_lowering_classifier.py` returns one of:

```text
identity / no-op
universal collapse
quotient-aligned cyclic character
uniform SHIL after a faithful phase embedding
unsupported: unequal class sizes
unsupported: equal size but wrong kernel topology
```

The decision terminates. There is no optimizer fallback hidden behind `unsupported`.

That is the desired behavior for a compiler checker.

## 4. Exhaustive small-n coverage

Every set partition was enumerated once using restricted-growth strings.

```text
n    Bell partitions   nontrivial   character   SHIL   unsupported   supported fraction
3          5               3            0         0         3              0
4         15              13            1         2        10           23.08%
5         52              50            0         0        50              0
6        203             201            2         5       194            3.48%
7        877             875            0         0       875              0
8       4140            4138            2        12      4124            0.338%
```

For `C6`, 18 of the unsupported kernels even have equal class sizes: equal size is necessary for the current uniform primitives, but very far from sufficient.

For `C8`, 126 unsupported kernels have equal class sizes.

So this library rapidly covers a vanishingly small fraction of abstract finite quotients.

That is not a defect in the checker. It is the physical restriction the checker is supposed to expose.

## 5. Prime-state impossibility for this library

For prime `n`:

- `C_n` has no nontrivial proper subgroup, so there is no nontrivial character/harmonic quotient;
- a uniform `m`-well quotient with equal blocks requires `m|n`, giving only `m=1` or `m=n`.

Therefore this declared library has **no nontrivial exact quotient instruction for prime-state `C_n`**.

`C3`, `C5`, and `C7` exhaustively confirm the statement.

A nontrivial quotient of a prime-state phase machine therefore immediately requires a richer physical resource, such as:

```text
nonuniform forcing
auxiliary state / another carrier
non-circle embedding
state-dependent switching
another nonlinear map
```

This is an instruction-set lower bound, not a claim that physics cannot realize the quotient.

## 6. Compiler order after this closure

A sensible lowering order is now:

```text
INPUT exact behavioral kernel

1. Is it already a coordinate/kernel carried by the chosen representation?
      yes -> retire distinguishing modes when the body must forget

2. Can a faithful physical embedding make it a legal uniform SHIL basin quotient?
      yes -> compile basin landscape and composition margin

3. Is it a cyclic congruence kernel but the quotient coordinate was not pre-carried?
      yes -> runtime harmonic/carrier transfer candidate

4. Otherwise:
      reject current library
      and say whether the first obstruction is
          unequal class size
          or equal-size wrong topology.
```

Only after this structural decision should a hardware optimizer tune locking strength, phase offsets, transistor parameters, noise margins or energy.

## 7. What survives from Claude's useful suggestion

The broad rule survives, but in a corrected form:

> **Prefer representation resources already paid for over inserting a runtime nonlinear instruction.**

For cyclic congruence quotients this means carrying a quotient-aligned non-faithful character beside a faithful state code if future transitions justify the standing cost.

This is analogous to compiler precomputation / redundant representation selection, not new mathematics.

The interesting physical cost question becomes:

```text
extra standing oscillator/carrier/state dimension
versus
runtime harmonic conversion latency / power / leakage / relock cost.
```

That cost has not yet been measured on hardware.

## 8. Files

- `map/phase_kernel_lowering_classifier.py`
- `tests/test_phase_kernel_lowering_classifier.py`
- `.github/workflows/phase-kernel-lowering-classifier.yml`
- `results/phase_kernel_lowering_classifier_summary.json` (Actions artifact)

Focused CI is green.
