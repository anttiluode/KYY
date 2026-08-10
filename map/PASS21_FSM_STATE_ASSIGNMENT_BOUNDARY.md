# Pass 21 — FSM state-assignment wall: representation-aware hardware mapping is old

Date: 2026-08-10

Recent KYY passes repeatedly arrived at the idea:

> choose a behaviorally equivalent state representation so that common transitions are physically cheap, local, low-switching, and easy to implement.

That is not a new compiler principle.

It is very close to the classical **finite-state-machine state assignment / state encoding** problem in digital logic synthesis.

---

# 1. Classical state assignment

A symbolic FSM has abstract states

```text
q_0, q_1, ..., q_(N-1).
```

A digital implementation must assign each symbolic state a binary code.

The encoding changes the cost of the next-state logic and state-register transitions even though the external finite-state behavior is unchanged.

Thus the hardware designer already solves a problem of the form

```text
behavioral machine
    -> choose equivalent internal coordinates
    -> minimize implementation cost.
```

De Micheli, Brayton & Sangiovanni-Vincentelli, *Optimal State Assignment for Finite State Machines* (IEEE TCAD, 1985), explicitly optimize the encoding of internal FSM states for area-effective VLSI implementation.

This is a direct prior-art wall for any KYY claim based only on "state representation is a compiler variable."

---

# 2. Transition geometry and power are also old

Later low-power state-assignment work explicitly makes states connected by likely transitions close in code space.

Hong, Park, Hwang & Kyung, *State assignment in finite state machines for minimal switching power consumption* (Electronics Letters 1994), minimize switching activity by assigning **smaller Hamming distance** to state pairs with higher transition probability.

Bacchetta et al. and many later methods likewise optimize state encodings against transition activity/power.

Therefore the statement

```text
"the geometry of state codes should reflect the transition graph"
```

is established digital-design practice.

---

# 3. There is even literal graph-embedding state assignment

A 1997 state-assignment method, *State assignment based on two-dimensional placement and hypercube mapping*, treats the main state-assignment step as a graph-embedding problem:

1. place the state-transition graph in a 2D array while minimizing total edge length;
2. map the placement into a hypercube with bounded dilation.

This is uncomfortably close to a broad reading of KYY's phrase

> "put automaton states into a geometry so transitions become local."

That phrase is therefore occupied.

---

# 4. Modern state assignment is multi-objective

The literature also jointly optimizes combinations such as:

```text
area
power / switching activity
critical-path delay
logic complexity
testability.
```

So simply adding a Pareto cost vector does not create a new problem either.

---

# 5. What KYY still adds to the question

The classical digital problem usually starts from a fixed hardware state model:

```text
N symbolic states
    -> binary register codewords
    -> combinational next-state logic.
```

KYY is currently varying more structural layers at once:

```text
behavioral automaton
        |
        +--> decompose into local Sigma-chain factors
        |
        +--> choose representation of each factor
        |       binary / one-hot / MinMax / continuous orbit / other
        |
        +--> type transitions
        |       conservative / contractive / singular reset
        |
        +--> place scarce irreversible sites
        |
        +--> compile to a declared local propagation substrate
```

Even that larger joint problem may have ancestors in FSM decomposition, logic synthesis, distributed automata, and physical design. It remains a search target rather than a novelty claim.

---

# 6. The important difference introduced by continuous geometric state

Binary state assignment largely treats the separation of codewords as a digital/noise-margin issue handled by the logic technology.

A compact continuous realization makes **state separation itself an explicit resource**.

Examples already in KYY:

```text
C_n group state:
    2D regular n-gon orbit
    exact arithmetic state dimension = 2
    nearest-state separation -> 0 as n grows

length-h threshold:
    h robust binary latch cells
    versus one normalized scalar with h+1 levels
    scalar level spacing -> 0 as h grows.
```

So the KYY representation question is not merely

```text
how many bits / coordinates?
```

but

```text
how many coordinates
x what dynamic range
x what precision/noise margin
x what transition complexity
x what communication depth?
```

That is where the next pass should focus.

---

# 7. Prior-art references to keep on the map

- De Micheli, Brayton & Sangiovanni-Vincentelli, *Optimal State Assignment for Finite State Machines* (IEEE TCAD, 1985).
- Hong, Park, Hwang & Kyung, *State assignment in finite state machines for minimal switching power consumption* (Electronics Letters, 1994), DOI 10.1049/el:19940422.
- *State assignment based on two-dimensional placement and hypercube mapping* (Integration, 1997), DOI 10.1016/S0167-9260(97)00027-8.
- Tsui, Pedram & Despain, *Low-power state assignment targeting two- and multilevel logic implementations* (IEEE TCAD, 1998), DOI 10.1109/43.736568.

The field is much larger than these examples.

---

# Current pin after Pass 21

Do not claim:

> choose state encodings according to transition geometry.

Old.

The sharper KYY question is:

> **What state representation minimizes physical recurrent cost when representation dimension, analog precision, transition algebra, irreversible-site placement, and neighbour communication are all allowed to trade against one another?**

The next thing to establish is a representation-independent lower bound showing that `low dimension` and `many robust states` cannot both be free.