# Result — compile/reject law for cyclic quotients on a uniform SHIL phase backend

Date: 2026-08-11

This is a substrate-specific lowering law inside a known multi-phase oscillator/SHIL geometry. It is **not** a novelty claim for Potts machines, phase discretization, phase-shifted SHIL, oscillator memory, or staged phase computation.

## Question

For a fine digital phase code `C_n`, which irreversible quotients can be implemented by one temporary uniform `m`-well phase-locking landscape and then returned to the original `n`-well code?

The answer is simple enough to be useful as a compiler audit.

---

## 1. One uniform SHIL landscape imposes a geometric kernel

A uniform `m`-well locking potential partitions the phase circle into `m` equal contiguous attraction basins.

Therefore, sampled on `n` equally spaced fine phase states, one deterministic quotient stage can only have kernel classes that are

```text
- contiguous in cyclic order;
- all the same size r=n/m;
- one contiguous run per quotient class,
```

up to cyclic rotation and relabeling of the outputs.

This gives a direct compile/reject test.

### legal

```text
C4: [0,0,1,1]
```

preimages are the two contiguous pairs

```text
{0,1}, {2,3}.
```

### rejected

```text
C4: [0,1,0,1]
```

preimages are

```text
{0,2}, {1,3},
```

which are interleaved around the physical circle.

The abstract digital quotient is perfectly legal, but one uniform two-basin phase landscape cannot realize that geometry.

Similarly, unequal run sizes such as

```text
[0,0,0,1,1,2]
```

are rejected by one uniform SHIL stage.

So:

> **abstract equivalence does not imply substrate-realizable equivalence.**

---

## 2. Equal consecutive blocks

Let

```text
Delta = 2*pi/n
r = n/m.
```

Consider one canonical quotient block

```text
0,1,...,r-1.
```

Its temporary `m`-well basin has half-width

```text
r Delta / 2.
```

Let the temporary attractor be at phase `a` and let `k Delta` be the fine-state representative to which the system should return under `n`-well locking.

The source-block capture margin is

```text
Delta/2 - |a - block_center|,
```

while fine-state re-entry margin is

```text
Delta/2 - |a - k Delta|.
```

The compiler chooses the representative nearest the block center and puts the temporary attractor midway between that representative and the block center.

That yields a parity law.

---

## 3. Odd block size

If `r` is odd, the block center is itself a fine C_n state.

Choose it as the representative and place the temporary coarse attractor exactly there.

Then

```text
certified composition margin = Delta/2.
```

Example:

```text
C12 -> C4
r=3
Delta=pi/6
representative of {0,1,2} = state 1
attractor = pi/6
margin = pi/12.
```

The same construction repeats by coarse-well spacing for every block.

---

## 4. Even block size

If `r` is even, the block center lies halfway between the two central fine states.

Putting the coarse attractor at the exact block center would reproduce the C4 midpoint failure: the coarse state would lie on a fine-state separatrix when `n`-well locking is restored.

Choose either central fine state as representative. Using the lower one by convention, place the temporary attractor

```text
Delta/4
```

toward the block center.

Then capture margin and fine re-entry margin are equal:

```text
certified composition margin = Delta/4.
```

Example:

```text
C12 -> C3
r=4
Delta=pi/6
representative of {0,1,2,3} = state 1
coarse attractor = pi/6 + pi/24
margin = pi/24.
```

The Pass-44 C4 pair merge is the special case

```text
n=4, r=2, Delta=pi/2
coarse offset = Delta/4 = pi/8
margin = pi/8.
```

---

## 5. Block size versus phase resolution

A useful consequence is that the ideal composition margin depends on the **fine phase spacing** and block parity, not directly on how many states are merged:

```text
odd r  -> Delta/2
even r -> Delta/4.
```

For example:

```text
C12 -> C4, r=3:  margin = pi/12
C12 -> C3, r=4:  margin = pi/24
C16 -> C4, r=4:  margin = pi/32
C100 -> C10,r=10: margin = pi/200
```

So on this ideal backend the expensive scaling variable is primarily

```text
fine phase resolution n,
```

not the cardinality of a quotient block by itself.

That is a concrete physical cost law for the state encoding.

---

## 6. What this means for the digital / analog border

The digital specification supplies only a kernel partition:

```text
which fine states must become behaviorally identical?
```

The physical oscillator supplies extra structure:

```text
cyclic order
basin contiguity
uniform basin width
phase resolution
separatrices
locking order.
```

Therefore the compiler has three possible outcomes:

```text
COMPILE
    kernel classes fit one physical basin partition
    -> solve attractor phases and margins

MULTI-STAGE
    quotient is legal but cannot fit one uniform landscape
    -> factor transition through several physical stages

REJECT / RICHER PRIMITIVE
    backend cannot realize the kernel without nonuniform forcing,
    extra state, switching, or another substrate primitive.
```

That is a more operational version of

> digital equivalence classes can be physically inequivalent embeddings.

---

## 7. Prior-art boundary

The 2025 multi-stage CMOS ring-oscillator Potts-machine work already shows that:

- phase-shifted SHIL creates different discrete phase sets;
- locked oscillator phases can act as memory;
- alternating SHILs create multi-stage computation;
- multiple phases represent multivalued Potts spins;
- phase is read physically with reference signals/DFFs.

So KYY does not own the physical primitives or the staged-phase idea.

The bounded residual is the **compiler direction**:

```text
arbitrary declared transition kernel
    -> test against backend basin geometry
    -> compile or reject
    -> choose physical representative/attractor phases
    -> certify composition margin.
```

This experiment only establishes that workflow for equal-block cyclic quotients in the ideal uniform phase-potential abstraction.

---

## 8. Next boundary

The next hard case is not another equal-block example.

It is a symbolic quotient that the one-stage audit rejects, then ask whether the compiler can automatically factor it into a minimum-cost sequence of legal physical stages.

Candidate:

```text
C4 alternating quotient
{0,2} -> A
{1,3} -> B.
```

A single uniform two-well potential rejects it because the classes are interleaved.

The interesting question is whether a short sequence of allowed physical operations—cyclic rotation, reflection, phase locking with different well counts, or an auxiliary phase degree of freedom—can realize it, and what extra hardware/state is required.

That begins to look like an actual **instruction selection / lowering** problem rather than another oscillator demonstration.

## Files

- `map/shil_cyclic_quotient_compiler.py`
- `tests/test_shil_cyclic_quotient_compiler.py`
- `.github/workflows/shil-cyclic-quotient-compiler.yml`
- `results/shil_cyclic_quotient_compiler_summary.json`

Workflow evidence: Actions run `31459023543`, focused tests green.
