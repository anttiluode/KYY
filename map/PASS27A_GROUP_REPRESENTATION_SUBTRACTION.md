# Pass 27A — subtract diagonal-SSM group expressivity and learned group representations

Date: 2026-08-10

Pass 27 removed any claim that KYY invented a multi-frequency rotary/oscillator bank. A second subtraction is necessary.

## 1. Complex diagonal recurrence as an Abelian group tracker is already occupied

Shakerinava, Khavari, Ravanbakhsh & Chandar, *The Expressive Limits of Diagonal SSMs for State-Tracking* (ICLR 2026 / arXiv:2603.01959), study input-dependent complex-valued diagonal SSMs directly as group state trackers.

Their expressivity result is much stronger than the narrow observation needed for KYY's cyclic counter:

- a single complex diagonal layer can track finite Abelian group behavior at finite precision;
- no single such layer can track a non-Abelian group at finite precision;
- multi-layer expressivity is characterized through a subnormal series with Abelian factors;
- the paper also reports a gap between expressivity and what optimization actually learns.

Therefore KYY must also subtract:

```text
complex diagonal SSMs can implement Abelian group state tracking
cyclic counting is a special Abelian case
layered diagonal recurrence connects to group-factor structure
expressivity does not imply learnability
```

This is very close to the algebraic half of Passes 23–25.

## 2. Learned recurrent group representations are older still

Gao et al., *On Path Integration of Grid Cells: Group Representation and Isotropic Scaling* (NeurIPS 2021 / arXiv:2006.10259), formulate recurrent path integration with a group-representation condition, derive a rotational matrix-Lie-group structure for the linear case, and learn representations by optimization.

Zhao et al., *A minimalistic representation model for head direction system* (NeurReps 2024), explicitly use a learned high-dimensional representation of the rotation group `U(1)` in a recurrent/path-integration setting.

So KYY must not present

```text
"train a recurrent representation until its transformations satisfy a group law"
```

as a new learning principle either.

## 3. What is left after both subtractions

The surviving KYY question is no longer whether rotary diagonal recurrence can represent a cyclic/Abelian state machine. It can, and that is known.

The sharper compiler question is:

> **Given a required finite-group behavior that already lies inside the expressivity class, which exact representation should a compiler choose when the implementation has finite precision, drift/noise, fixed state norm, a mode budget, and a required horizon?**

For `C_n`, Pass 23 gives many legal exact representations:

```text
frequency set F subset/multiset of Z_n
A_F^n = I.
```

All can implement the same symbolic counter. They can nevertheless have very different geometric separation:

```text
minimum orbit distance / nearest-prototype radius.
```

Pass 27's trained probe turns that representation choice into an empirical question by comparing legal exact character schedules under controlled phase error.

That is a **resource selection problem inside a known expressivity class**, not a new expressivity theorem.

## 4. The learned-angle result must also be phrased conservatively

The exploratory learned oscillator bank moved its angles closer to exact `C_n` characters while fitting short-horizon data, but did not reach an exact quotient and lost long-horizon accuracy.

That is consistent with the known expressivity/learnability gap. It is not evidence that KYY discovered group-law learning.

What may still be useful is the diagnostic itself:

```text
character relation defect
    = distance of learned angles from an exact finite-group representation
```

reported jointly with

```text
train accuracy
length extrapolation
orbit margin
controlled implementation error.
```

The falsifiable question is whether relation defect and margin explain extrapolation better than training loss alone.

## 5. Current novelty posture

Occupied:

```text
rotary oscillator banks
RoPE frequency geometry
complex recurrent rotations for state tracking
Abelian group tracking by complex diagonal SSMs
learned recurrent group representations
harmonic/group frames
```

Still worth testing:

```text
compiler-level selection among equivalent exact finite-group representations
using explicit margin / precision / drift / wiring costs,
and diagnosing trained recurrence by algebraic relation defect.
```

Status:

**KNOWN REPRESENTATIONS + A NARROW RESOURCE/COMPILER EXPERIMENT.**

That is a smaller claim than Pass 23 initially suggested, and a better one.
