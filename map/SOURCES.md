# Primary-source bibliography

This is a working bibliography for `map/`. It favors papers / proceedings / publisher pages over secondary summaries.

The annotation after each source says **why it constrains KYY**.

Snapshot: 2026-08-10.

---

## Unitary / orthogonal recurrent networks

1. **Arjovsky, Shah, Bengio — Unitary Evolution Recurrent Neural Networks** (2015/2016)  
   https://arxiv.org/abs/1511.06464  
   Early efficient unitary recurrent transition; complex hidden state and norm-preserving long-memory motivation.

2. **Jing et al. — Tunable Efficient Unitary Neural Networks (EUNN) and their application to RNNs** (ICML 2017)  
   https://proceedings.mlr.press/v70/jing17a.html  
   Alternating local 2x2 unitary/Givens-style transformations; tunable mesh depth/capacity. Direct prior art for `geom_scatter` as a matrix parameterization.

3. **Mhammedi et al. — Efficient Orthogonal Parametrisation of Recurrent Neural Networks Using Householder Reflections** (ICML 2017)  
   https://proceedings.mlr.press/v70/mhammedi17a.html  
   Householder-product orthogonal RNN. Direct ancestor of the `householder2` control.

4. **Vorontsov et al. — On orthogonality and learning recurrent networks with long term dependencies** (ICML 2017)  
   https://proceedings.mlr.press/v70/vorontsov17a.html  
   Important negative context: hard orthogonality can hurt convergence/performance.

5. **Jing et al. — Gated Orthogonal Recurrent Units: On Learning to Forget** (2017)  
   https://arxiv.org/abs/1706.02761  
   Adds forgetting/gating to unitary recurrence; warning that perfect norm preservation is not universally desirable.

6. **Dorobantu et al. — DizzyRNN: Reparameterizing Recurrent Neural Networks for Norm-Preserving Backpropagation** (2016)  
   https://arxiv.org/abs/1612.04035  
   Givens rotations as norm-preserving recurrent parameterization; another reason not to claim local rotation products as new.

---

## Optical / physical realization of unitary transforms

7. **Clements et al. — Optimal design for universal multiport interferometers** (Optica 2016)  
   https://opg.optica.org/optica/article.cfm?uri=optica-3-12-1460  
   Rectangular local 2-port optical mesh for arbitrary unitary transforms; physical analogue of a Givens network.

8. **Miller et al. / photonic circuit simulation paper — Highly parallel simulation and optimization of photonic circuits...** (Scientific Reports 2019)  
   https://www.nature.com/articles/s41598-019-42408-2  
   Explicitly discusses implementing unitary matrices with cascaded 2x2 mixing units and looping the unitary transform into a recurrent neural network.

9. **Pai et al. — Matrix optimization on universal unitary photonic devices** (2018)  
   https://arxiv.org/abs/1808.00458  
   Shows optimization behavior and biases arising from locally interacting interferometer mesh components.

10. **Radford et al. — Inverse Design of Unitary Transmission Matrices in Silicon Photonic Coupled Waveguide Arrays** (2024)  
    https://arxiv.org/abs/2409.18284  
    Geometry/pattern inverse design of physical coupled-waveguide arrays toward target unitary transmission matrices.

11. **RecLight — A Recurrent Neural Network Accelerator with Integrated Silicon Photonics** (2022)  
    https://arxiv.org/abs/2209.00084  
    Hardware reminder: photonic recurrent acceleration is an established engineering line.

12. **OREO — An optoacoustic field-programmable perceptron for recurrent neural networks** (Nature Communications 2024)  
    https://www.nature.com/articles/s41467-024-47053-6  
    Traveling acoustic wave supplies recurrent latency/memory in an optical NN.

---

## Input-dependent recurrent operators / modern SSMs

13. **Foerster et al. — Input Switched Affine Networks: An RNN Architecture Designed for Interpretability** (2016/2017)  
    https://arxiv.org/abs/1611.09434  
    Explicit input-switched recurrent matrices. Strong prior art against claiming token-dependent `Q_t` itself.

14. **Ha, Dai, Le — HyperNetworks** (2016)  
    https://arxiv.org/abs/1609.09106  
    A compact network generates weights of another network; includes recurrent applications. Constrains the "small controller generates large operator" idea.

15. **Dao & Gu — Transformers are SSMs / Structured State Space Duality (Mamba-2)** (2024)  
    https://arxiv.org/abs/2405.21060  
    Modern bridge between selective SSMs and attention; establishes current efficient recurrent context.

16. **Lahoti et al. — Mamba-3: Improved Sequence Modeling using State Space Principles** (2026)  
    https://arxiv.org/abs/2603.15569  
    More expressive recurrence, complex-valued state, and MIMO aimed directly at state tracking and inference efficiency.

---

## State tracking / transition algebra

17. **Grazzi et al. — Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues** (2024/2025)  
    https://arxiv.org/abs/2411.12537  
    Positive-eigenvalue restrictions block parity; complex eigenvalues enter modular counting; transition algebra matters directly.

18. **Siems et al. — DeltaProduct: Increasing the Expressivity of DeltaNet Through Products of Householders** (2025)  
    https://arxiv.org/abs/2502.10297  
    Token-conditioned generalized Householder products for modern state tracking / length extrapolation.

19. **Terzić et al. — Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models (PD-SSM)** (2025)  
    https://arxiv.org/abs/2509.22284  
    Sparse permutation-diagonal transition family with finite-state expressivity guarantees and linear recurrence cost.

20. **Terzić et al. — Flash PD-SSM** (2026)  
    https://arxiv.org/abs/2605.19150  
    Efficient modern implementation and benchmark reference for expressive structured sparse transitions.

21. **Siems et al. — Learning State-Tracking from Code Using Linear RNNs** (2026)  
    https://arxiv.org/abs/2602.14814  
    Shows state-tracking questions moving into code-like next-token settings rather than remaining only synthetic sequence-to-sequence toys.

---

## Oscillator recurrent computation

22. **Rusch & Mishra — Coupled Oscillatory Recurrent Neural Network (coRNN)** (2020)  
    https://arxiv.org/abs/2010.00951  
    Second-order controlled oscillator RNN. Very close ancestor of KYY's naive wave recurrence.

23. **Rusch et al. — Graph-Coupled Oscillator Networks (GraphCON)** (2022)  
    https://arxiv.org/abs/2202.02296  
    Second-order damped/controlled oscillators coupled by graph adjacency. Direct prior art for "geometry defines oscillator coupling."

24. **Lanthaler, Rusch, Mishra — Neural Oscillators are Universal** (2023)  
    https://arxiv.org/abs/2305.08753  
    Broad universality result encompassing oscillator architectures in sequence modeling, graphs, and physical networks.

25. **Pasqualetti & Guo — Attention by Synchronization in Coupled Oscillator Networks** (2026)  
    https://arxiv.org/abs/2606.12059  
    Very current physical-substrate attempt to replace expensive attention operations with oscillator synchronization.

26. **Guo & Pasqualetti — Learnable Sequential Memory in Coupled Oscillator Networks** (2026)  
    https://arxiv.org/abs/2607.18439  
    Current multi-timescale oscillator sequential-memory architecture. Important live neighbour, not historical trivia.

---

## Graph / topology structured recurrent computation

27. **Li et al. — State Space Models on Temporal Graphs: A First-Principles Study (GraphSSM)** (2024)  
    https://arxiv.org/abs/2406.00943  
    Brings graph structural information/Laplacian regularization into SSM formulation for temporal graphs.

28. **Gallicchio & Micheli — Ring Reservoir Neural Networks for Graphs** (2020)  
    https://arxiv.org/abs/2005.05294  
    Explicit ring topology as reservoir design.

29. **d'Andrea et al. — Complex topological features of reservoirs shape learning performances...** (2022)  
    https://arxiv.org/abs/2211.00161  
    Studies how reservoir topology/connectome structure affects computation.

30. **Mastrogiuseppe & Ostojic — Linking connectivity, dynamics and computations in low-rank recurrent neural networks** (2017/2018)  
    https://arxiv.org/abs/1711.09672  
    Low-dimensional recurrent connectivity structure produces low-dimensional dynamics; major neighbour to any "small operator basis generates computation" claim.

---

## Geometry-generated connectivity / indirect encoding

31. **Stanley, D'Ambrosio, Gauci — A hypercube-based encoding for evolving large-scale neural networks (HyperNEAT)** (2009)  
    PubMed: https://pubmed.ncbi.nlm.nih.gov/19199382/  
    Connectivity patterns generated as functions of geometry; can scale regular connection patterns to large substrates.

32. **Risi & Stanley — An Enhanced Hypercube-Based Encoding for Evolving the Placement, Density, and Connectivity of Neurons** (Artificial Life 2012)  
    https://direct.mit.edu/artl/article/18/4/331/2720/An-Enhanced-Hypercube-Based-Encoding-for-Evolving  
    Makes the geometry-to-connectivity mechanism explicit: endpoint coordinates are fed to a CPPN that outputs connection weights.

**Why these matter:** A1 cannot claim "geometry generates a large network from a compact rule" as new. Its possible residual is dynamic/token-conditioned **state-transition** generation under modern sequence benchmarks and strict control dimension.

---

## Delay / reservoir computing

33. **Appeltant et al. — Information processing using a single dynamical node as complex system** (Nature Communications 2011)  
    https://www.nature.com/articles/ncomms1476  
    Single nonlinear node + delay loop creates virtual high-dimensional reservoir; core prior art for delay-as-computation.

34. **Duan et al. — Embedding Theory of Reservoir Computing and Reducing Reservoir Network Using Time Delays** (2023)  
    https://arxiv.org/abs/2303.09042  
    Explicit relation between delay embeddings and reservoir dimension/reduction.

35. **Maksymov — Physical Reservoir Computing Enabled by Solitary Waves...** (2024)  
    https://arxiv.org/abs/2402.03319  
    Physical wave dynamics as reservoir computation.

---

## Linear wave scattering as nonlinear computation

36. **Wanjura & Marquardt — Fully nonlinear neuromorphic computing with linear wave scattering** (Nature Physics 2024)  
    https://www.nature.com/articles/s41567-024-02534-9  
    Crucial neighbour: input is encoded in physical parameters that alter the scattering matrix; other physical parameters are trainable. Linear waves therefore produce nonlinear input-output computation.

37. **Hammami et al. — Expressivity of Programmable-Metasurface-Based Physical Neural Networks** (2026)  
    https://arxiv.org/abs/2603.13602  
    Structural input encoding, mutual coupling, and depth in a physics-consistent rich-scattering network.

38. **Valantinas & Vettenburg — Scaling Up Wave Calculations with a Scattering Network** (2024)  
    https://spj.science.org/doi/10.34133/icomputing.0098  
    Maps multiple-scattering physics onto deterministic neural-network computation; relevant compiler/representation neighbour.

---

## Realization, symmetry, identifiability

39. **Defourneau & Petreczky — Realization theory of recurrent neural networks and rational systems** (2019)  
    https://arxiv.org/abs/1903.05609  
    RNN realization/minimality theory already exists.

40. **Vlačić & Bölcskei — Affine symmetries and neural network identifiability** (2020)  
    https://arxiv.org/abs/2006.11727  
    Formal neural-network identifiability under activation symmetries.

41. **Lengyel et al. — GENNI: Visualising the Geometry of Equivalences for Neural Network Identifiability** (2020)  
    https://arxiv.org/abs/2011.07407  
    Explicitly identifies and visualizes functionally equivalent parameter subspaces.

42. **Hashimoto, Hirono, Sannai — Unification of Symmetries Inside Neural Networks** (2024)  
    https://arxiv.org/abs/2402.02362  
    Gauge-symmetry framing across feedforward networks, neural ODEs, and transformers.

43. **Lim et al. — The Empirical Impact of Neural Parameter Symmetries, or Lack Thereof** (2024)  
    https://arxiv.org/abs/2405.20231  
    Parameter symmetries and symmetry-reduced architectures; relevant to any claim that "rotation orbit" is uniquely ours.

44. **Zhao, Walters, Yu — Symmetry in Neural Network Parameter Spaces** (2025/2026 survey)  
    https://arxiv.org/abs/2506.13018  
    Current survey of the parameter-symmetry field.

45. **Zhao et al. — Finding Symmetry in Neural Network Parameter Spaces** (2025/2026)  
    https://openreview.net/forum?id=0XhWusHpLq  
    Automated symmetry discovery across architectures. Strong constraint on the "audit-topology for AI" idea.

46. **Gerasimov et al. — Unstable Features, Reproducible Subspaces** (2026)  
    https://arxiv.org/abs/2606.12138  
    SAE features vary individually across seeds while lower-rank subspaces can be reproducible; basis ambiguity is empirically current.

---

## Controllability / observability / reduction inside modern ML

47. **Hamdan et al. — Sparse Mamba: Introducing Controllability, Observability, And Stability To Structural State Space Models** (2024)  
    https://arxiv.org/abs/2409.00563

48. **Moon — From Black-Box to White-Box: Control-Theoretic Neural Network Interpretability** (2025)  
    https://arxiv.org/abs/2511.12852

49. **Ezoe & Sato — Balanced truncation for S4/DSS** (2024)  
    https://arxiv.org/abs/2402.15993

50. **Schwerdtner et al. — Hankel singular value regularization for SSM compression** (2025)  
    https://arxiv.org/abs/2510.22951

---

# Reading priority for KYY

If there is time to read only ten things before the next architecture experiment:

1. EUNN
2. oRNN
3. coRNN
4. GraphCON
5. Input Switched Affine Networks
6. negative-eigenvalue state tracking
7. DeltaProduct
8. PD-SSM
9. Wanjura & Marquardt linear-wave scattering
10. HyperNEAT / geometry-generated connectivity

Those ten define most of the walls around the current KYY idea.