# Pass 43 — geometric transition lowering: ask what this code can realize

Date: 2026-08-10

Pass 42 classified symbolic task transitions by their image and kernel partition before choosing an operator primitive.

That immediately raises a second compiler question:

> Given the geometric state code we have already chosen, can this symbolic transition actually be realized by one linear or affine operator on that code?

This question is old finite-automata realization / state-assignment territory.  The point here is to put the exact test in KYY's compiler pipeline.

---

## Prior art boundary

There is explicit literature titled *Linear realization of finite automata* going back at least to Reusch (1977), and much older/later work on state assignment and linear/affine realization of finite-state machines.

So neither linear realization nor the solvability criterion below is a novelty claim.

KYY uses it as a **lowering audit** between symbolic behavior and a geometric/operator backend.

---

## 1. Code matrix

Let the declared legal symbolic states be represented by vectors

```text
z_1, ..., z_n in R^d
```

and collect them as columns

```text
Z = [z_1 ... z_n] in R^{d x n}.
```

For a task token `x`, construct the target code matrix

```text
Z_x = [z_{delta(1,x)} ... z_{delta(n,x)}].
```

We want to know whether one linear operator can realize this entire symbolic transition on the legal code:

```text
A_x Z = Z_x.
```

---

## 2. Exact linear solvability

The matrix equation has a solution iff every linear dependency among legal state vectors remains a dependency after the symbolic transition:

```text
ker(Z) subset ker(Z_x).
```

Why this must hold is immediate:

```text
Z c = 0
=>
A_x Z c = 0
=>
Z_x c = 0.
```

It is also sufficient for the matrix equation.

When consistent, a convenient minimum-Frobenius solution is

```text
A_x = Z_x Z^+
```

with `Z^+` the Moore–Penrose pseudoinverse.

The repo audit measures both the dependency violation and the direct reconstruction residual.

---

## 3. Affine fallback

If a transition is not linear on the chosen centered code, try

```text
A_x z + b_x.
```

Homogeneous augmentation gives

```text
Z_bar = [ Z
          1 ... 1 ].
```

The same condition becomes

```text
ker(Z_bar) subset ker(Z_x).
```

and the compiler solves

```text
[A_x  b_x] Z_bar = Z_x.
```

This creates an exact hierarchy:

```text
linear lowering exists
    else
exact affine lowering exists
    else
this code/operator class cannot realize the token exactly.
```

The third case is useful.  It says to change the code, increase dimension, or admit a nonlinear/gated primitive rather than adding training knobs to an impossible lowering.

---

## 4. The C4 square makes the distinction visible

Use the one-frequency C4 code

```text
z0 = ( 1, 0)
z1 = ( 0, 1)
z2 = (-1, 0)
z3 = ( 0,-1).
```

### cycle

```text
[0,1,2,3] -> [1,2,3,0]
```

is exactly linear:

```text
A = 90-degree rotation
continuous rank = 2.
```

### partial merge

```text
0 -> 0
1 -> 0
2 -> 2
3 -> 2
```

is also exactly linear, but singular:

```text
M = [[1,1],
     [0,0]]
```

with continuous rank one.

On the **four legal code points**, however, its behavioral image has two states:

```text
{0,2}
```

and behavioral rank two.

This is an important warning:

> continuous matrix rank and finite-automaton transition rank are different resources.

The compiler cares about the induced map on the legal state manifold, not matrix rank by itself.

### total reset

```text
0,1,2,3 -> 0
```

cannot be linear on this centered square.

The reason is exactly the dependency criterion.  The code has relations such as

```text
z0 + z2 = 0,
```

but after reset the same coefficient combination becomes

```text
z0 + z0 != 0.
```

So a linear map is impossible.

Affine lowering succeeds immediately:

```text
A = 0
b = z0.
```

That is the exact constant overwrite used in Pass 42.

---

## 5. What this adds to the compiler

The front end can now separate three questions that had been mixed together:

### symbolic irreversibility

From the automaton transition:

```text
image / behavioral rank / kernel partition.
```

This says **what distinctions must survive or disappear**.

### geometric realizability

From `(Z, Z_x)`:

```text
linear?
affine?
neither?
```

This says **what operator class can implement the required transition on this code**.

### hardware/backend cost

Only after those two audits should a physical compiler ask:

```text
what primitive implements that operator cheaply?
```

That is a much cleaner division than calling everything a recurrent weight matrix.

---

## 6. Connection back to Geometric Neuron / TWC

The old Geometric Neuron instinct was that the geometry of the state/readout changes what dynamics are available.

Here that idea becomes exact:

```text
same symbolic transition
+ different state code Z
=> different linear/affine realizability and different operator cost.
```

So state-code geometry is not decoration.

It is part of the instruction set.

TWC contributed the complementary lesson:

> internal coordinates should be chosen/compiled according to the declared observable behavior rather than treated as sacred learned parameters.

The combination now looks like:

```text
behavioral transition
       ↓
image + kernel partition
       ↓
chosen state geometry Z
       ↓
linear / affine realizability audit
       ↓
exact synthesized operator
       ↓
port canonicalization / certification
       ↓
backend realization and error budget.
```

That is much closer to an actual compiler pipeline than the original “geometric RNN” framing.

---

## 7. Executable audit

Files:

- `map/transition_lowering_audit.py`
- `tests/test_transition_lowering_audit.py`

The tests require:

```text
C4 cycle          -> exact linear
C4 partial merge  -> exact linear singular pinch
C4 total reset    -> linear rejection, exact affine overwrite
```

---

## 8. Next falsifier

The criterion itself is elementary and old.

The empirical question is whether it is useful **after learning**.

That is why the next experiment trains the C4 partial-merge machine with a full-rank unconstrained learned 2x2 merge operator, then ignores its exact coordinates and compiles the task transition through the legal square code.

The sharp test is not just classification accuracy.

For the required kernel block `{0,1}`, after the compiled merge we require

```text
h(0 after M) == h(1 after M)
```

exactly, followed by zero hidden and port divergence under every identical future cycle.

If only the outputs match momentarily while the hidden vectors remain distinct, then Pass 16 has merely returned in a smaller costume.
