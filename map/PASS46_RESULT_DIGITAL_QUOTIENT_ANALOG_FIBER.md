# Pass 46 result — exact digital quotient with a surviving analog fiber

Date: 2026-08-11

Pass 44 used four isolated legal points. That made an exact DFA replacement and nearest-state reification devastatingly strong baselines.

Pass 46 asks a different question:

> what changes when the legal state itself contains a continuum that must survive a digital merge?

The answer is not a new theory of hybrid systems. The digital/continuous seam, reset maps, quotient/gluing constructions, rank-deficient resets, tangent compatibility, switched linear realizations, and hybrid-model repair all have substantial prior art.

The useful result is a controlled compiler boundary inside that known geometry.

---

## 1. Four rails instead of four points

The legal state is a union of four affine fibers

```text
F_q = { c_q + a v : a in R }.
```

`q` is a C4-valued digital state. `a` is a real analog payload. `v` is the common analog tangent.

Before a fixed random orthogonal mixing of coordinates, the rails are

```text
q=0: ( 1, 0, a)
q=1: ( 0, 1, a)
q=2: (-1, 0, a)
q=3: ( 0,-1, a).
```

The random mixing is important only to prevent the implementation from receiving an axis labeled DIGITAL and another labeled ANALOG.

The token alphabet is

```text
0..3 : C4 increments; change q, preserve a
M    : {0,1}->0 and {2,3}->2; preserve a
S    : q fixed; a -> 0.9 a.
```

The model must output both the finite class `q` and the continuous value `a`.

A finite DFA can reproduce the `q` component but cannot reproduce arbitrary real `a`. The natural software baseline is therefore an explicit hybrid state `Q x R`, not a DFA.

---

## 2. Learned approximate machine

The learned latent has dimension three and is deliberately allowed small structural errors:

- approximate C4 angle;
- full-rank approximate merge;
- small coupling of digital coordinates into the analog coordinate;
- approximate analog gain at merge;
- approximate `a -> 0.9a` scale.

Training is only at length 16.

Ten seeds all fit the training-length digital task perfectly.

```text
L16 q accuracy:  10/10 exact
L64 q accuracy:  10/10 exact
L256 q exact:     7/10
L1024 q exact:    5/10
```

At L1024:

```text
mean q accuracy: 0.962374
minimum:         0.851097
mean analog RMSE: 0.004874
```

Every learned merge remains full rank.

Its residual digital kernel error is

```text
0.0599 .. 0.0903
```

and its analog-tangent preservation defect is

```text
2.42e-4 .. 3.38e-3.
```

The task can therefore look solved at the training horizon while both the digital quotient relation and the analog tangent semantics remain approximate.

---

## 3. Nearest-fiber reification

The Pass-44 nearest-state baseline is generalized to the legal union of rails.

After every learned update:

1. infer the analog coordinate along `v`;
2. select the nearest legal rail;
3. project transversely onto that rail;
4. keep the inferred coordinate along `v`.

This completely repairs the digital classification drift:

```text
q exact seeds
L16:    10/10
L64:    10/10
L256:   10/10
L1024:  10/10
```

But the analog error remains:

```text
L1024 analog RMSE: 0.004743
```

The learned machine itself was at `0.004874`.

So manifold reification repairs transverse state error but does not repair incorrect dynamics *along* a legal manifold.

This is the first KYY toy in which the strong Pass-44 reification baseline and exact operator legalization clearly separate.

---

## 4. Exact point+tangent compilation

Recover the three latent generators

```text
c0, c1, v
```

inside the entangled learned coordinates.

Synthesize each token operator by specifying its action on that basis.

Cycle:

```text
C c0 =  c1
C c1 = -c0
C v  =  v
```

Merge:

```text
M c0 = c0
M c1 = c0
M v  = v
```

Scale:

```text
S c0 = c0
S c1 = c1
S v  = 0.9 v.
```

This is not a new linear-algebra theorem. It is Pass 43's exact lowering contract extended from state points to state points plus continuous tangent generators.

The compiled defects are at numerical precision:

```text
max C^4 relation defect:       1.04e-15
max merge-kernel defect:       3.15e-16
max merge fiber defect:        2.71e-16
```

Runtime result:

```text
q exact: 10/10 through L1024
L1024 analog RMSE: 3.86e-8
```

The remaining analog error is the numerical floor inherited from float32 legal initial coordinates before the float64 compiled runtime.

---

## 5. Stronger paired-history audit

Pass 44 asked whether a forbidden digital distinction could survive a merge and reappear later.

On fibers there is a sharper possibility:

> the digital distinction can disappear while information about it survives as a perfectly legal displacement along the analog fiber.

Construct two histories with

```text
same analog a
q=0 versus q=1
```

then give both

```text
M + exactly the same future.
```

The learned merge contains tiny source-dependent digital-to-analog couplings. Across ten trained seeds the resulting paired analog-history gap is

```text
mean: 5.27e-4
range: 9.36e-6 .. 1.66e-3.
```

### after nearest-fiber reification

Digital identity is completely merged:

```text
q mismatch at merge:           0
max q mismatch in common future: 0
```

But the analog gap survives because both points are legal positions on the same rail:

```text
mean analog gap at merge:       5.27e-4
mean max future analog gap:     5.29e-4
largest seed max future gap:    1.66e-3.
```

So the projection has correctly answered

```text
which rail?
```

while failing to answer

```text
where on that rail should this history be?
```

### after exact compilation

The exact merge kills the forbidden digital-history direction while preserving the *shared* analog payload:

```text
q mismatch:                     0
mean analog gap at merge:       2.18e-8
mean max future analog gap:     8.16e-8
largest max future gap:         1.06e-7.
```

Again this is the inherited numerical floor, not a semantic memory channel.

---

## 6. The geometric border, stated carefully

The clean picture is now:

```text
DIGITAL
= equivalence / quotient directions
= distinctions a token is allowed or required to destroy

ANALOG
= tangent / fiber directions
= distinctions that must remain continuously resolvable
```

A correct irreversible reset must align its nullspace with the required digital-collapse directions while keeping the required analog tangents out of that nullspace.

This geometry is classical hybrid-systems territory. Rank-deficient resets and tangent/nullspace relationships were studied long before KYY.

What Pass 46 contributes as an experiment is the post-training failure mode and repair inside a learned entangled latent coordinate system:

```text
learned approximate operator
    -> digital and tangent semantic defects

nearest-manifold projection
    -> digital/transverse repair
    -> tangent semantic defect remains

exact point+tangent lowering
    -> digital quotient exact
    -> analog tangent action exact
    -> no runtime projection.
```

---

## 7. A general lowering audit for fibers

For digital centers `c_q`, tangent bases `V_q`, digital transition `tau(q)`, and required tangent-coordinate maps `L_q`, a single global linear token map must satisfy

```text
A c_q = c_tau(q)
A V_q = V_tau(q) L_q.
```

Concatenate all source generators into `X` and all required images into `Y`.

Then the exact linear-lowering question is again

```text
A X = Y
```

and therefore is solvable iff

```text
ker(X) subset ker(Y).
```

`map/hybrid_fiber_lowering_audit.py` implements this as an automatic compiler check.

One useful resource consequence follows immediately.

If several digital modes literally share the same physical tangent vector, but one token requires different source-mode-dependent actions on that tangent, one global linear operator cannot implement the specification: the same source vector cannot have two different images.

The compiler must then choose among

```text
switch / condition the operator
use a nonlinear operator
or
separate the tangent copies by spending extra state dimension.
```

The unit tests exercise both the rejected shared-tangent case and the higher-dimensional separated-tangent realization.

This is not claimed as new realization theory. It is a concrete geometry/cost diagnostic for the KYY compiler.

---

## 8. Prior-art subtraction

Do not claim novelty for any of the following:

- hybrid systems as discrete + continuous state;
- mixed logical dynamical systems;
- switched linear state-space realization;
- reset maps and quotient/gluing geometry;
- continuous latent embeddings of hybrid dynamics;
- tangent/velocity compatibility across resets;
- rank-deficient or dimension-dropping resets;
- nullspace/tangent alignment;
- synthesizing reset controllers;
- repairing hybrid models to satisfy specifications;
- projecting learned operators onto exact algebraic constraint sets.

Recent nearby examples include CHyLL and *Embedding Hybrid Systems into Continuous Latent Vector Fields* for quotient/continuous latent embeddings, while older geometric hybrid-systems work already studies reset-induced gluing and rank-deficiency geometry.

The broad pattern

```text
learn approximate operator -> project to constrained operator -> certify
```

is also independently occupied in current operator-learning literature.

---

## 9. What survives

The residual is narrower than before:

> Given an already-trained, entangled continuous latent machine whose deployment geometry is worth retaining, can we infer its digital quotient directions and analog tangent semantics, legalize a mixed alphabet of reversible and irreversible token operators directly in those learned coordinates, transport its observable interface, and produce a lower-cost or more robust implementation than explicit hybrid switching or runtime manifold projection?

Pass 46 answers only the first controlled part:

```text
point + tangent semantics can be audited and legalized;
nearest-manifold reification is not enough to correct along-manifold dynamics;
the compiled toy preserves continuous payload while enforcing exact digital collapse.
```

It does **not** yet show that this is cheaper or better than an explicit `Q x R` hybrid machine.

That comparison is now the right boundary rather than a DFA baseline.

## Files

- `map/mixed_fiber_compiler_probe.py`
- `map/mixed_fiber_pair_audit.py`
- `map/hybrid_fiber_lowering_audit.py`
- `tests/test_mixed_fiber_compiler_probe.py`
- `tests/test_mixed_fiber_pair_audit.py`
- `tests/test_hybrid_fiber_lowering_audit.py`
- `results/mixed_fiber_compiler_summary.json`
- `results/mixed_fiber_pair_audit_summary.json`
- `.github/workflows/mixed-fiber-compiler.yml`
- `.github/workflows/mixed-fiber-pair-audit.yml`
