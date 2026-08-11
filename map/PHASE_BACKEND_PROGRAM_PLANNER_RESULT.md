# Phase backend program planner: standing state versus runtime instructions

Date: 2026-08-11

The single-transition classifier is useful only if it can be lifted to a workload.

This pass asks:

> If a machine may need several exact quotient kernels, which physical resources should be carried permanently and which should be synthesized only when a transition occurs?

No hardware cost is guessed. The planner emits a **resource vector**.

## 1. Restricted exact strategy

For cyclic congruence kernels, a direct-coordinate strategy carries characters

```text
z_f(q) = exp(2 pi i f q / n).
```

A faithful character (`gcd(n,f)=1`) is needed to retain all `C_n` distinctions before any irreversible quotient.

A non-faithful character has one exact congruence kernel determined by `gcd(n,f)`.

Distinct congruence kernels therefore require distinct character gcds.

Within this restricted strategy, the exact standing lower bound is

```text
1 faithful character carrier
+
1 quotient-aligned character carrier
for every distinct nontrivial congruence kernel
that must be directly available.
```

The bound is intentionally scoped. More general nonlinear/state-space encodings can change it.

## 2. Why one quotient character cannot serve two exact kernels

For `C_n`, the kernel of character `f` is

```text
q ~ q' iff f(q-q') = 0 mod n.
```

Its class count is

```text
n / gcd(n,f).
```

Changing output labels does not change that equivalence relation.

So a single exact character coordinate has one fixed kernel.

If two desired quotient transitions have different exact cyclic congruence kernels, a direct-coordinate representation needs two corresponding coordinates or a runtime transformation between carriers.

This is elementary cyclic-group structure, not a new theorem.

## 3. Runtime alternative

Instead of carrying every quotient character permanently, keep only a faithful carrier and synthesize the needed coarse carrier when required:

```text
pre-carried strategy:
    standing carrier cost now
    + retirement/isolation cost at quotient

runtime strategy:
    no standing quotient carrier
    + harmonic/nonlinear conversion cost when transition occurs
    + transfer/locking/retirement cost.
```

The planner does not decide which is cheaper until a target backend supplies those costs.

## 4. C12 demo workload

The CI demo asks for five transitions.

### C2 congruence

```text
[0,1,0,1,0,1,0,1,0,1,0,1]
```

Exact character:

```text
f=6.
```

### C3 congruence

```text
[0,1,2,0,1,2,0,1,2,0,1,2]
```

Exact character:

```text
f=4.
```

### contiguous C4 block quotient

```text
[0,0,0,1,1,1,2,2,2,3,3,3]
```

This is not a character kernel.

It maps to one uniform SHIL basin-collapse stage under the faithful `f=1` embedding.

### equal-size wrong-topology control

```text
[0,0,1,0,1,1,2,2,3,2,3,3]
```

All classes have size three, but the kernel is neither cyclic-congruence nor equal contiguous arcs under any faithful embedding in the current library.

Result:

```text
REJECT.
```

### unequal control

The last control has unequal class sizes.

Result:

```text
REJECT.
```

## 5. Planner output

For the pre-carried direct-coordinate strategy:

```text
faithful/full-state carrier: f=1
C3 quotient carrier:         f=4
C2 quotient carrier:         f=6

standing bank = [1,4,6]
standing carrier count = 3
exact restricted lower bound = 3.
```

The same workload, if quotient characters are *not* pre-carried, has runtime harmonic/carrier obligations

```text
f=4
f=6.
```

It also has

```text
1 SHIL stage type
2 transitions rejected by the current library.
```

## 6. The compiler rule after this pass

Claude's useful heuristic can now be stated without calling re-encoding free:

> **Prefer an already-paid representation resource over inserting a runtime physical instruction, but account for the standing carrier and its eventual retirement.**

The mapper should therefore return alternatives rather than a premature winner:

```text
PLAN A
    more standing carriers
    fewer runtime conversions

PLAN B
    minimal standing state
    more runtime nonlinear conversion

BOTH
    same exact behavioral kernel contract
```

A real backend can later price:

- oscillator/carrier area and power;
- idle holding power;
- harmonic conversion energy/latency;
- locking time;
- isolation/damping cost;
- leakage;
- readout complexity.

## 7. Files

- `map/phase_backend_program_planner.py`
- `tests/test_phase_backend_program_planner.py`
- `.github/workflows/phase-backend-program-planner.yml`
- `results/phase_backend_program_planner_summary.json` (Actions artifact)

Focused CI is green.
