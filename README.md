# Neural Quantum State (RBM) for the 1D Transverse-Field Ising Model

## Project Overview

This project implements a **Restricted Boltzmann Machine (RBM)** as a neural quantum state (NQS) to approximate the ground state of the 1D transverse-field Ising model (TFIM). The implementation connects quantum many-body physics to statistical mechanics and machine learning, providing concrete code for an exam project on:

- **Path integrals and variational quantum mechanics**
- **Molecular/stochastic dynamics on parameter manifolds**
- **Statistical mechanics of RBMs and neural network phase transitions**

### Key References

- **Carleo, G.** (2017). *Neural-Network Quantum States* lecture notes ([csrc.ac.cn](https://www.csrc.ac.cn/upload/file/20170703/1499072201537152.pdf))
- **Carleo & Troyer** (2017). Science 355, 602–606 ([phys.org](https://phys.org/news/2017-02-artificial-neural-network-simulate-quantum.html))
- **Decelle et al.** (2021). *Restricted Boltzmann machine: Recent advances and mean-field theory*. Chinese Physics B ([cpb.iphy.ac.cn](https://cpb.iphy.ac.cn/EN/10.1088/1674-1056/abd160))

---

## Physics Background

### 1D Transverse-Field Ising Model (TFIM)

The Hamiltonian is:
$$H = -J \sum_i \sigma_i^z \sigma_{i+1}^z - g \sum_i \sigma_i^x$$

- $J$: Ising coupling strength
- $g$: Transverse field strength
- Periodic boundary conditions: $\sigma_{L+1} = \sigma_1$

The phase diagram features:
- **Ordered phase** ($g < J$): Ferromagnetic correlations dominate
- **Disordered phase** ($g > J$): Transverse field disorder dominates
- **Critical point** ($g = J$): Second-order quantum phase transition

### RBM Neural Quantum State Ansatz

The wave function is represented as:
$$\Psi_{\text{RBM}}(\sigma) \propto \exp\left(\sum_i a_i \sigma_i\right) \prod_{j=1}^M 2\cosh\left(b_j + \sum_i W_{ij}\sigma_i\right)$$

**Parameters:**
- $a_i \in \mathbb{R}$: visible biases (one per spin)
- $b_j \in \mathbb{R}$: hidden biases (one per hidden unit)
- $W_{ij} \in \mathbb{R}$: weights (visible-to-hidden connections)

**Advantages of RBM ansatz:**
1. Exact normalization via hidden unit marginalisation
2. Efficient sampling via Metropolis algorithm
3. Differentiable → amenable to variational optimization
4. Statistical mechanics interpretation: acts like a spin-glass energy landscape

---

## Implementation Structure

### Core Modules

#### `tfim_exact.py` – Exact Diagonalization Benchmark
- Class `TFIMExact`: builds full Hilbert space Hamiltonian via Kronecker products
- Methods:
  - `groundstate_energy()`: Dense eigendecomposition (for $L \leq 12$)
  - `benchmark()`: Compute exact E₀ for multiple L and g values
- **Usage:** provides reference energies for comparison

#### `rbm_nqs.py` – RBM Implementation
- Class `RBMNQS`: wave function representation
- Key methods:
  - `log_psi(sigma, theta)`: $\log|\Psi_{\text{RBM}}(\sigma)|$
  - `psi_squared_ratio()`: efficient probability ratio for Metropolis
  - `compute_gradients()`: natural gradient terms $\partial \log|\Psi|/\partial\theta$
  - `update_parameters()`: SGD parameter updates
- **Features:** numerical stability via cosh terms, stochastic gradient descent

#### `vmc.py` – Variational Monte Carlo Solver
- Class `VMCSolver`: combines RBM and TFIM Hamiltonian
- Key methods:
  - `local_energy(sigma)`: $E_{\text{loc}}(\sigma) = \langle\Psi|H|\sigma\rangle / \langle\Psi|\sigma\rangle$
  - `metropolis_step()`: Metropolis update on $|\Psi|^2$
  - `sample_configs()`: generate MC samples
  - `optimize_step()`: natural gradient descent on energy functional
  - `train()`: full optimization loop
- **Algorithm:** 
  1. Sample configurations from $|\Psi(\theta)|^2$ via Metropolis
  2. Compute local energies for each sample
  3. Compute energy gradients using natural gradient rule
  4. Update parameters to minimize $\langle E \rangle(\theta)$

#### `utils.py` – Utilities
- `generate_all_configs()`: enumerate all $2^L$ spin configurations
- `order_parameter_weights()`: compute RBM weight statistics
- `save_json()`, `load_json()`: file I/O
- `print_header()`: formatted output

---

## Running the Experiment

### Installation

```bash
cd /path/to/tfim
pip install -r requirements.txt
```

### Quick Start

```bash
python main.py
```

This runs:
1. **Exact benchmark:** compute exact ground-state energies for $L \in \{8, 10, 12\}$ and $g \in \{0.5, 1.0, 1.5, 2.0\}$
2. **VMC with RBM:** train RBMs with various hidden unit ratios $M/L$
3. **Analysis:** tabulate energy errors and order parameters
4. **Output:** save results to `results/` directory

### Configuration

Edit `src/config.py` to control:
- System sizes: `SYSTEM_SIZES`
- Transverse fields: `TRANSVERSE_FIELDS`
- RBM architecture: `HIDDEN_UNIT_RATIOS`
- Training parameters: `N_EPOCHS`, `N_SAMPLES`, `LEARNING_RATE`

---

## Expected Results and Interpretation

### Energy Error vs. Hidden Units

For each $(L, g)$ pair, the energy error $\Delta E = E_{\text{RBM}} - E_0^{\text{exact}}$ should:

1. **Decrease monotonically** with more hidden units (improved expressivity)
2. **Plateau** at a minimum error (limited by RBM expressivity vs. TFIM complexity)
3. **Show phase-dependent trends:**
   - In **ordered phase** ($g < 1$): fewer hidden units needed (simpler state)
   - In **disordered phase** ($g > 1$): more hidden units needed (entanglement stronger)

### Order Parameters (RBM Weight Statistics)

- **Mean weight magnitude:** $\langle |W_{ij}| \rangle$
  - Relates to "interaction strength" in spin-glass picture
  - Increases with hidden units → more complex interactions captured
  
- **Sparsity:** fraction of $|W_{ij}| < \epsilon$
  - Low sparsity ($g < 1$): many active hidden units form "ferromagnetic" features
  - High sparsity ($g > 1$): sparse localized features capture disorder

- **Correlations between hidden activations:**
  - Measures if hidden units cooperatively encode system properties
  - Relates to "compositional" phases in statistical mechanics literature

### Connection to Exam Topics

#### 1. Path Integrals & Variational Manifolds

The VMC energy functional:
$$E(\theta) = \frac{\langle \Psi(\theta) | H | \Psi(\theta) \rangle}{\langle \Psi(\theta) | \Psi(\theta) \rangle}$$

is a **projection operator** onto the variational manifold spanned by $\{\Psi(\theta)\}$. This connects to:

- **Imaginary-time evolution:** $e^{-\tau H} \to \text{ground state}$ as $\tau \to \infty$ can be viewed as **stochastic descent** on $E(\theta)$
- **Euclidean path integral:** $Z = \int D[\phi] e^{-S_E[\phi]}$ with auxiliary fields → RBM Trace formula
- **Quantum ergodicity breaking:** Phase transitions manifest as *barriers* in the $\theta$-landscape that VMC must traverse

#### 2. Molecular/Stochastic Dynamics

Parameter updates via natural gradient:
$$\dot{\theta} = -\eta \, \mathcal{F}^{-1} \nabla_\theta E(\theta)$$

- $\mathcal{F}$: Fisher information matrix (metric on NQS manifold)
- Interprets as **Langevin dynamics** on rugged potential $E(\theta)$
  - **Discrete time:** SGD step is a stochastic discretization
  - **Friction:** learning rate $\eta$ acts like inverse viscosity
  - **Noise:** MC sampling introduces stochastic fluctuations ~$O(1/\sqrt{N_{\text{samples}}})$

This is analogous to **molecular dynamics** in classical statistical mechanics:
- Particles (parameters) explore configuration space
- Phase transitions appear as *slow modes* in the dynamics
- Dynamics can get "trapped" in local minima (glassy behavior)

#### 3. Statistical Mechanics of RBMs

The RBM Hamiltonian (in statistical physics notation):
$$H_{\text{RBM}}(\sigma, h) = -\sum_i a_i \sigma_i - \sum_j b_j h_j - \sum_{ij} W_{ij} \sigma_i h_j$$

can be studied via **mean-field theory** and **replica methods**:

- **Mean-field approximation:** $\langle \sigma_i h_j \rangle \approx \langle \sigma_i \rangle \langle h_j \rangle$
  - Predicts phase diagram: **paramagnetic**, **mixed spin-glass**, **ferrimagnetic** phases
  - Entropy and energy curves exhibit typical S-shaped transitions

- **Replica symmetry breaking (RSB):**
  - For large $M/L$ (overparameterized), RBM exhibits **spin-glass** behavior
  - Multiple local minima in energy landscape → difficult optimization
  - Observed as plateaus in $E(\theta)$ vs. training step

- **Compositional phase:** 
  - Hidden units can organize into **feature groups** (e.g., domain-wall detectors)
  - Differs from amorphous spin-glass: *structured* disorder
  - Signaled by low but non-zero sparsity in weight matrix

**Your numerical study should reveal:**
1. Energy error scaling with $M$ varies smoothly across phase transition
2. Order parameter discontinuities near $g \approx 1$ (critical point)
3. Optimization difficulty increases in disordered phase (dynamics slow down)

---

## File Structure

```
tfim/
├── README.md                 (this file)
├── main.py                   (experiment driver)
├── requirements.txt          (Python dependencies)
├── src/
│   ├── __init__.py
│   ├── config.py            (hyperparameters)
│   ├── tfim_exact.py        (exact diagonalization)
│   ├── rbm_nqs.py           (RBM ansatz)
│   ├── vmc.py               (VMC optimizer)
│   └── utils.py             (utilities)
├── notebooks/
│   └── analysis.ipynb       (plotting and analysis)
└── results/
    ├── exact_benchmark.json
    ├── vmc_results.json
    └── summary.json
```

---

## Further Improvements & Extensions

1. **Diagnostic Tools:**
   - Plot energy convergence curves
   - Visualize hidden unit activation patterns
   - Compute entanglement entropy estimates

2. **Advanced Algorithms:**
   - Natural gradient (Fisher metric) instead of vanilla SGD
   - Adam optimizer with weight decay
   - Annealed learning rates

3. **Theoretical Analysis:**
   - Mean-field theory predictions vs. numerics
   - Replica analysis of overcomplete RBMs
   - Phase diagram mapping

4. **Application Scope:**
   - 2D lattices (Ising or Heisenberg models)
   - Fermionic systems with sign structure
   - Time evolution with real-time NQS dynamics

---

## References & Reading List

### Core Papers
1. Carleo & Troyer (2017) – Original RBM-NQS paper
2. Carleo (2017) – Lecture notes with full derivations
3. Decelle et al. (2021) – Statistical mechanics of RBMs

### Key Textbooks & Reviews
- Landau & Lifshitz, *Statistical Physics* – foundational phase transitions
- Mehta et al., *A high-bias, low-variance introduction to Machine Learning for physicists* – NQS review
- Recent NQS reviews (e.g., [arxiv:2402.11014](https://arxiv.org/abs/2402.11014))

### TFIM Resources
- TeNPy documentation: [tenpy.readthedocs.io](https://tenpy.readthedocs.io/en/v0.7.1/intro/examples/tfi_exact.html)
- Sachdev–Ye–Kitaev and transverse-field effects

---

## Author Notes

This code is designed to be:
- **Self-contained:** no external NQS libraries required
- **Pedagogical:** readable and well-commented
- **Exam-ready:** explicit connections to three exam topics

The implementation prioritizes clarity over peak performance; for production use, consider:
- Vectorized NumPy operations
- JAX for autodiff
- GPU acceleration (if available)

---

**Last Updated:** June 2026  
**Status:** Ready for exam submission
