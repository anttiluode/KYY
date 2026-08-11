# Harmonic carrier body/port forgetting audit

Date: 2026-08-11

This is the reduced modal version of the hardware experiment suggested by the Pass-44/body-port discussion.

It is **not** a transistor-level oscillator simulation.

The purpose is to make the physical question exact before building hardware:

> If a quotient-aligned harmonic is present, what must happen to the old faithful carrier for the body—not merely the port—to forget the forbidden distinction?

## 1. Two-character C4 state code

Represent the four fine states with

```text
z1 = exp(i phi)      faithful fundamental character
z2 = exp(i 2phi)     quotient-aligned second character
```

for

```text
phi = q*pi/2.
```

Then the desired alternating quotient

```text
{0,2} / {1,3}
```

is already carried by `z2`:

```text
q=0,2 -> z2=+1
q=1,3 -> z2=-1.
```

But `z1` still distinguishes the histories:

```text
q=0 vs 2 -> z1=+1 vs -1
q=1 vs 3 -> z1=+i vs -i.
```

Therefore choosing a quotient-aligned character at design time avoids runtime phase multiplication, but does **not** by itself implement body-level forgetting.

## 2. Port-only quotient

Define the current coarse output to read only

```text
y_now = z2.
```

The merged pairs have zero output gap to numerical precision.

Now allow a later physical/readout path with a small coupling from the old faithful carrier:

```text
y_future = z2 + epsilon * g * z1.
```

`g=1` means the old carrier remains future-observable.

For either merged pair `q` and `q+2`, `z1` differs by sign, so the future gap is

```text
Delta_future = 2 |epsilon|.
```

With `epsilon=.1`, the apparently perfect quotient exposes a future gap of `.2` immediately.

This is the physical-mode version of the Pass-44 point:

```text
current port equality
!=
behavioral forgetting.
```

## 3. Damping the old carrier gives a settling-time certificate

Model retirement of the old fundamental carrier by

```text
z1(t) = exp(-gamma t) z1(0).
```

Then

```text
Delta_future(t) = 2 |epsilon| exp(-gamma t).
```

The focused test matches this law to floating-point precision.

For `epsilon=.1`, `gamma=1`:

```text
t       future gap
0       0.200000
1       0.073576
2       0.027067
4       0.003663
8       0.0000671
```

To guarantee a residual future gap no larger than `delta`, the required damping time is

```text
t >= (1/gamma) log(2|epsilon|/delta).
```

Examples:

```text
target gap    minimum t  (epsilon=.1, gamma=1)
1e-1          0.6931
1e-2          2.9957
1e-3          5.2983
1e-6         12.2061
```

Thus body erasure has an explicit latency/settling cost.

## 4. But physical erasure is not necessary

Set the future coupling gate to

```text
g=0.
```

Then the future gap is exactly zero even while the two hidden `z1` states remain maximally different.

This is the important correction already learned earlier in the software work:

> hidden-state equality is sufficient for forgetting, but future observational equivalence is enough.

In hardware language, the old carrier may retain energy and phase memory while becoming behaviorally irrelevant if every future path from it is truly isolated.

So the compiler has at least two distinct body-level retirement instructions:

```text
DAMP / ERASE
    spend settling time / dissipation
    residual distinguishability decays exponentially

ISOLATE / DISCONNECT
    spend switching / routing / isolation resource
    future observability can become zero immediately
```

They are not the same physical contract.

## 5. This sharpens "prefer re-encoding"

For a congruence quotient, a pre-carried non-faithful character can save the runtime cost of generating a harmonic carrier.

But that optimization creates a standing resource and a retirement obligation:

```text
before transition:
    faithful carrier + quotient carrier

after transition:
    quotient carrier survives
    faithful carrier must be erased or made future-unobservable.
```

So the actual compiler trade is

```text
extra standing carrier/state dimension
+
retirement cost

versus

runtime harmonic conversion
+
carrier-transfer cost.
```

No superiority claim is made yet because actual oscillator isolation, damping, harmonic-conversion and leakage costs are hardware-specific.

## 6. What a physical experiment should measure

An audio/RF oscillator experiment can now have a precise target rather than merely showing phase locking:

1. encode fine state in a fundamental phase carrier;
2. establish a quotient-aligned second-harmonic/coarse carrier;
3. stop driving or otherwise retire the fundamental;
4. measure residual fundamental amplitude/phase versus time;
5. intentionally open a known weak coupling from fundamental back to coarse output;
6. measure whether the forbidden pairwise distinction follows the predicted retirement curve;
7. compare damping retirement with hard electrical isolation.

If residual fundamental information leaks back, that is a physical Pass-44 witness.

If hard isolation makes it permanently future-unobservable, that is equally informative: the substrate supports behavioral forgetting without literal hidden-state equality.

## 7. Files

- `map/harmonic_body_port_forgetting_audit.py`
- `tests/test_harmonic_body_port_forgetting_audit.py`
- `.github/workflows/harmonic-body-port-forgetting.yml`
- `results/harmonic_body_port_forgetting_summary.json` (Actions artifact)

Focused CI is green.
