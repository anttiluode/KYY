# Pass 14 — thermodynamic boundary: forgetting already has an energy theory

Date: 2026-08-10

Pass 13 separated a finite deterministic transition into two idealized resource types:

```text
reversible transport
    +
irreversible rank-lowering merge / pinch.
```

It is tempting to jump directly from the second line to an energy-efficiency claim.

Do not.

There is already a substantial thermodynamics-of-computation literature, including work specifically on finite-state automata and regular languages.

---

# 1. Logical many-to-one maps and Landauer cost are old territory

Landauer's principle links logical information erasure to a minimum thermodynamic cost under specified physical/statistical assumptions.

Useful modern references:

- Sagawa & Ueda, *Minimal Energy Cost for Thermodynamic Information Processing: Measurement and Information Erasure* (PRL 2009), https://doi.org/10.1103/PhysRevLett.102.250602
- Giorgini et al., *Thermodynamic cost of erasing information in finite time* (Physical Review Research 2023), https://doi.org/10.1103/PhysRevResearch.5.023084

The important caution is that a logical reset by itself does not determine one universal finite-time energy number. Cost depends on the physical realization, state distribution, protocol, speed, error tolerance, and thermodynamic setting.

So KYY's transformation rank is **not an energy meter**.

---

# 2. Finite-state automata already have a dissipation theory

DeDeo, *Irreversibility and dissipation in finite-state automata* (Physics Letters A 2014), develops a physical-information-theoretic lower bound on average energy dissipated per transition for deterministic FSAs driven by a random input source.

The computational irreversibility is tied to information about past inputs that is lost from the automaton state.

This directly occupies the broad statement:

```text
"measure an automaton's energy cost from how much history its transitions forget."
```

That is not a KYY opening.

---

# 3. Regular languages already have an energy-complexity classification

Kutrib, Malcher & Wendlandt, *Energy complexity of regular languages* (Theoretical Computer Science 2024), study the energy expenditure associated with real-time deterministic and quantum finite automata.

They show that regular languages can be classified by intrinsic energy/forgetting requirements and give upper/lower bounds.

This occupies another tempting claim:

```text
"classify sequence tasks by how much irreversible finite-state work they inherently require."
```

Again: already a research subject.

---

# 4. What Pass 13's rank defect does and does not say

Let a deterministic map `f` act on an `n`-state set and have

```text
rank(f) = |image(f)| = r.
```

If the only singular primitive allowed is a defect-1 merge, then at least

```text
n - r
```

such merge events are needed: one merge can lower the number of distinct images by at most one.

With free permutations and arbitrary defect-1 merges, `n-r` merge events are also sufficient: collapse every kernel class onto a chosen representative, then permute the surviving representatives to their desired images.

This gives an exact **combinatorial irreversible-event floor** under that primitive library.

But it does not specify thermodynamic energy.

Two maps with the same rank can discard very different amounts of Shannon information under a nonuniform input distribution. Likewise, a physical implementation can have very different finite-time work/error costs.

KYY should therefore keep separate columns:

```text
transformation rank / kernel structure
number of singular primitive events
input/state probability distribution
information lost per transition
physical reset protocol
energy / dissipation estimate
```

---

# 5. The residual remains spatial/resource-aware

The literature above largely asks:

```text
how irreversible is this logical computation?
```

KYY's surviving hardware question is different:

```text
WHERE are irreversible operations physically available?
HOW FAR must distinctions be reversibly routed to reach them?
CAN a behaviorally equivalent state realization move the pinch sites closer?
HOW MUCH hidden garbage is retained if reset is emulated reversibly?
```

This suggests a compiler objective with at least two independent costs:

```text
logical / thermodynamic irreversibility
        +
spatial reversible transport.
```

Even this conjunction is not claimed novel. Reversible-circuit synthesis already prices ancillas/garbage and local gate costs, while automata thermodynamics prices information loss. The exact joint finite-state recurrent/local-geometry problem still needs deeper search.

---

# 6. Current discipline

Do not say:

> `n-rank(f)` joules, bits, or Landauer units.

Say:

> `n-rank(f)` is the minimum count of defect-1 merge primitives in the idealized permutation+merge compiler.

Only after specifying a state distribution and a physical reset model may we translate logical irreversibility into an information/thermodynamic cost.

---

# Current pin after Pass 14

The physical question is now cleanly split:

> **What forgetting is behaviorally unavoidable, and what extra communication is imposed by the geometry on where that forgetting can physically occur?**

The first half already has automata/thermodynamic theory.

The second half is where KYY should keep digging.