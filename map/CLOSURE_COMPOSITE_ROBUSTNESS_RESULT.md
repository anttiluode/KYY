# Closure result — composite moduli and positive-port robustness

Date: 2026-08-11

This is closure work for Pass 40, not a new numbered architecture/pass.

The goals were to test the two holes left after the C101 positive-kernel result:

1. `gcd(n, active frequencies)=1` had only been exercised on prime `n=101`, where failure is unusually hard;
2. algebraic correctness had not been cleanly separated from cumulative phase-error robustness for the 9-parameter positive port.

## Setup

Train the existing learned harmonic tracker at length 16, then compile exactly as in the recent cyclic passes:

```text
learned angles
    -> nearest exact C_n characters
    -> midpoint/commutant recentering
    -> positive correlation-kernel output port
```

Test:

```text
n = 100 and 105
modes = 8
seeds = 0..4
random-start training
lengths = 16, 64, 256, 1024
```

The positive port has `m+1 = 9` free scalars; the general exact-equivariant port has `2m+1 = 17`.

## Composite gcd controls

Explicit non-faithful/faithful controls now exercise both outcomes of the certificate:

```text
C100  [2,4,6,8]    gcd = 2   -> REJECT
C100  [2,4,6,7]    gcd = 1   -> CERTIFY
C105  [3,6,9,12]   gcd = 3   -> REJECT
C105  [3,6,10,14]  gcd = 1   -> CERTIFY
```

Thus the certificate is not merely being tested in the prime-modulus easy case.

## Trained composite-modulus result

All ten trained/compiled models happened to land on faithful active character sets:

```text
10 / 10 character gcd = 1
10 / 10 algebraically certified
10 / 10 active modes = 8
```

Clean compiled runtime:

```text
C100: 5/5 exact at L16, L64, L256, L1024
C105: 5/5 exact at L16, L64, L256, L1024
```

So the bounded C101 result survives the first composite-modulus test.

## Margin column

The positive-kernel port again has a larger complete-orbit minimum margin than the 17-parameter equivariant port in **every** run.

```text
C100 positive margin: min 4.043, mean 4.434
C105 positive margin: min 4.269, mean 5.107
```

This is consistent with the already-recorded C101 ten-seed Pass-40 table and corrects any older summary implying the 9-parameter port's margin was not measured.

## Cumulative systematic phase error

This audit uses a runtime error added to every mode's per-unit rotation increment and therefore lets phase error accumulate along the actual token path.

### error = 1e-4 radians / unit increment

All ten remain exact at every tested horizon through L1024:

```text
C100: 5/5 exact through L1024
C105: 5/5 exact through L1024
```

### error = 1e-3 radians / unit increment

All ten remain exact through L256, but none remains exact at L1024.

Mean L1024 accuracy:

```text
C100: 0.3830   range 0.3276 .. 0.4547
C105: 0.4160   range 0.4002 .. 0.4316
```

This is a useful hard boundary.

The gcd/stabilizer certificate proves correctness of the **exact legalized machine**. It does not certify arbitrary implementation error. Deployment robustness remains a separate geometric/margin/error-budget question.

Do not conflate:

```text
exact symbolic correctness
with
finite-precision / systematic-drift robustness.
```

## Interpretation

The closure table is now:

```text
                           exact?      compact?      robust to 1e-4/L1024?   robust to 1e-3/L1024?
positive C100/C105 port      yes          yes                 yes                      no
```

So the algebraic port result survives composite moduli, but it is not magically immune to accumulating phase error.

That distinction is exactly what a physical/compiler story would eventually need to price.

## Scope

This does not add a novelty claim.

The cyclic harmonic representation, character faithfulness/gcd condition, group-equivariant ports and matched/correlation kernels are classical. This closure only establishes that the current KYY implementation behaves as its stated certificate predicts outside the prime-101 case and exposes a concrete robustness limit.

## Files

- `map/closure_composite_robustness_audit.py`
- `tests/test_closure_composite_robustness_audit.py`
- `.github/workflows/closure-composite-robustness.yml`
- workflow artifact: `closure_composite_robustness.json`

## Current decision

Do not promote this to `main` as new theory.

Together with `CLOSURE_PRIOR_ART_AUDIT_2026-08-11.md`, this is evidence supporting a narrower research residual around post-training exact operator/kernel legalization + port transport/canonicalization + behavioral certification. That residual still needs comparison against the nearest automata extraction/discretization/repair baselines before any novelty-facing claim is earned.
