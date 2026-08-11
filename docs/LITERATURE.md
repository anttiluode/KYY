# Literature checkpoint

A minimal list of papers that KYY must compare against before making novelty claims.

1. **Mamba-3: Improved Sequence Modeling using State Space Principles** — Lahoti et al. (2026), arXiv:2603.15569. More expressive discretization, complex state update, MIMO recurrence.
2. **Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues** — Grazzi et al. (2025), arXiv:2411.12537. Negative/complex/non-triangular transition structure and state tracking; products of reflection-like transformations.
3. **DeltaProduct: Increasing the Expressivity of DeltaNet Through Products of Householders** — Siems et al. (2025), arXiv:2502.10297. Directly relevant Householder-product baseline.
4. **Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models** — Terzić et al. (2025), arXiv:2509.22284. PD-SSM; sparse transitions with strong finite-state expressivity.
5. **Flash PD-SSM: Memory-Optimized Structured Sparse State-Space Models** — Terzić et al. (2026), arXiv:2605.19150. Hardware/throughput-oriented follow-up.
6. **Improved state mixing in higher-order and block diagonal linear recurrent networks** — Dubinin, Orvieto, Effenberger (2025/26), OpenReview. Richer local/block state mixing.
7. **Orthogonal Recurrent Neural Networks with Scaled Cayley Transform** — Helfrich, Willmott, Ye (2018). Orthogonal recurrent dynamics with negative eigenvalues.
8. **Unitary Evolution Recurrent Neural Networks** — Arjovsky, Shah, Bengio (2016), arXiv:1511.06464. Structured products for unitary recurrence.

There are also multiple Graph-SSM / message-passing SSM lines. Therefore "graph + SSM" is not a novelty claim.

## KYY's narrower question

The candidate transition is constrained by an explicit *physical-style local geometry*: state coordinates only interact across declared edges, and global transformations arise by composition/propagation. The comparison must determine whether that locality earns anything against generic sparse and Householder transition families.
