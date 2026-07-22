# Physics Background & Derivations

## Table of Contents

1. [1D Transverse-Field Ising Model](#1d-transverse-field-ising-model-tfim)
2. [Restricted Boltzmann Machines](#restricted-boltzmann-machines-rbm)
3. [Variational Monte Carlo](#variational-monte-carlo-vmc)
4. [Connection to Path Integrals](#connection-to-path-integrals)
5. [Stochastic Dynamics & Optimization](#stochastic-dynamics--optimization)
6. [Statistical Mechanics of RBMs](#statistical-mechanics-of-rbms)

---

## 1D Transverse-Field Ising Model (TFIM)

### Hamiltonian

$$H = -J \sum_{i=1}^L \sigma_i^z \sigma_{i+1}^z - g \sum_{i=1}^L \sigma_i^x$$

with periodic boundary conditions ($\sigma_{L+1} = \sigma_1$).

**Parameters:**

- $L$: number of spins (lattice size)
- $J > 0$: Ising coupling (ferromagnetic)
- $g > 0$: transverse field strength
- $\sigma_i^x, \sigma_i^z$: Pauli matrices on site $i$

### Phase Diagram

The TFIM exhibits a **quantum phase transition** at $g_c = J$:

| Region           | $g < J$                           | $g = J$    | $g > J$                        |
| ---------------- | --------------------------------- | ---------- | ------------------------------ |
| **Phase**        | Ordered                           | Critical   | Disordered                     |
| **Order**        | $\langle \sigma_z \rangle \neq 0$ | Power-law  | $\langle \sigma_z \rangle = 0$ |
| **Correlations** | Ferromagnetic                     | Long-range | Exponential decay              |
| **Entanglement** | Area law                          | Enhanced   | Area law                       |

### Ground State Properties

For large $L$, exact results (Ising + XY duality):

- **Ordered phase ($g < J$):**
  - GS energy per spin: $e_0 \approx -J(1 + \text{const} \cdot e^{-\pi(1-g/J)})$
  - Ferromagnetic order: $m = \sqrt{1 - (g/J)^2}$ for $g/J < 1$
- **Disordered phase ($g > J$):**
  - GS energy per spin: $e_0 \approx -g + O(J/g)$
  - Spins predominantly in $x$-eigenstates; $z$-correlations exponential

### Why TFIM for NQS?

1. **Canonical benchmark:** well-understood exact solution
2. **Phase transition:** tests RBM's ability to handle two regimes
3. **Small system sizes:** exact diagonalization is fast ($L \lesssim 15$)
4. **Clear physics:** order parameter and correlation functions interpretable

---

## Restricted Boltzmann Machines (RBM)

### Definition

An RBM is a bipartite graphical model with:

- **Visible units** $\sigma = (\sigma_1, \ldots, \sigma_L) \in \{-1, +1\}^L$ — spin configuration
- **Hidden units** $h = (h_1, \ldots, h_M) \in \{-1, +1\}^M$ — latent variables

### Energy Function (Standard Form)

$$E(\sigma, h) = -\sum_i a_i \sigma_i - \sum_j b_j h_j - \sum_{ij} W_{ij} \sigma_i h_j$$

### Associated Probability Distribution

$$P(\sigma, h) = \frac{\exp(-E(\sigma,h))}{Z}$$

where $Z = \sum_{\sigma,h} \exp(-E(\sigma,h))$ is the partition function.

### Marginal Over Hidden Units

Integrating out hidden units (using $h_j \in \{-1,+1\}$):

$$P(\sigma) \propto \exp\left(\sum_i a_i \sigma_i\right) \prod_{j} 2\cosh\left(b_j + \sum_i W_{ij} \sigma_i\right)$$

This is **exactly** the NQS ansatz used in the project!

**Key insight:** Because hidden units are independent given $\sigma$, we can compute the marginal analytically.

### Quantum Wave Function Interpretation

Identifying:
$$\Psi_{\text{RBM}}(\sigma) \propto \exp\left(\sum_i a_i \sigma_i\right) \prod_{j} 2\cosh\left(b_j + \sum_i W_{ij}\sigma_i\right)$$

The amplitude $|\Psi(\sigma)|^2$ is identical to $P(\sigma)$ for an RBM—this is the Carleo–Troyer ansatz.

**Why this works:**

1. **Expressivity:** RBM can represent broad class of states (if $M$ is large)
2. **Efficient sampling:** can sample $\sigma \sim |\Psi(\sigma)|^2$ in polynomial time via Metropolis
3. **Differentiability:** amplitude is smooth in parameters $\theta = (a, b, W)$
4. **Interpretation:** relates to statistical physics — RBM is a spin-glass model

### Numerical Stability of $\log|\Psi|$

In code, we compute $\log|\Psi|$ rather than $|\Psi|$ directly to avoid overflow. The natural form $\sum_j \log(2\cosh(x_j))$ overflows for $|x_j| \gtrsim 710$. The numerically stable equivalent is:

$$\log(2\cosh(x)) = \text{logaddexp}(x, -x) = \log(e^x + e^{-x})$$

which NumPy evaluates without overflow for all finite $x$. This is implemented in `RBMNQS.log_psi()` in `src/rbm_nqs.py` and in the `log_psi` function in the notebook.

---

## Variational Monte Carlo (VMC)

### Variational Principle

For any ansatz $\Psi(\theta)$, the ground state energy satisfies:

$$E_0 \leq \frac{\langle \Psi(\theta) | H | \Psi(\theta) \rangle}{\langle \Psi(\theta) | \Psi(\theta) \rangle} \equiv E(\theta)$$

So finding optimal $\theta^*$ that minimizes $E(\theta)$ is an upper bound on the true ground state.

### Local Energy

For a fixed configuration $\sigma$, the **local energy** is:

$$E_{\text{loc}}(\sigma) = \frac{\langle \sigma | H | \Psi(\theta) \rangle}{\langle \sigma | \Psi(\theta) \rangle}$$

**Key property:** For any state $|\Psi\rangle$,
$$\langle E \rangle = \sum_\sigma |\Psi(\sigma)|^2 \, E_{\text{loc}}(\sigma)$$

So the energy expectation can be computed as a **Monte Carlo average** over configurations sampled from $|\Psi|^2$.

### Computation in Code

For TFIM, $E_{\text{loc}}(\sigma)$ is computed by summing contributions:

$$E_{\text{loc}}(\sigma) = \underbrace{\sum_i (-J\sigma_i\sigma_{i+1})}_{\text{ZZ terms}} + \underbrace{\sum_i \frac{\langle \sigma | (-g\sigma^x_i) | \Psi \rangle}{\langle \sigma | \Psi \rangle}}_{\text{X-flip terms}}$$

The X-flip term involves evaluating $\Psi$ on configurations where spin $i$ is flipped:
$$\text{X-flip contribution} = -g \sum_i \frac{\Psi(\sigma^i_{\text{flip}})}{\Psi(\sigma)}$$

where $\sigma^i_{\text{flip}}$ has spin $i$ flipped. Note this is the **amplitude ratio** (not the probability ratio $|\Psi|^2/|\Psi|^2$) — this is the correct off-diagonal matrix element.

### Energy Gradient and Log-Derivative Operators

Define the **log-derivative operators** $O_k(\sigma) = \partial_k \log|\Psi(\sigma)|$:

$$O_{a_i}(\sigma) = \sigma_i$$
$$O_{b_j}(\sigma) = \tanh\!\left(b_j + \sum_i W_{ij} \sigma_i\right)$$
$$O_{W_{ij}}(\sigma) = \sigma_i \cdot \tanh\!\left(b_j + \sum_i W_{ij} \sigma_i\right)$$

The **variational energy gradient** (plain SGD direction) is then:
$$F_k = \langle O_k \, E_{\text{loc}} \rangle - \langle O_k \rangle \langle E_{\text{loc}} \rangle$$

The **Stochastic Reconfiguration** update solves $S \delta\theta = F$ where:
$$S_{kl} = \langle O_k O_l \rangle - \langle O_k \rangle \langle O_l \rangle$$
This is fully implemented in `VMCSolver._compute_sr_update()`.

For RBMs, the Fisher metric $\mathcal{F}$ has a known structure that relates to shot-noise in MC sampling.

### Monte Carlo Sampling

**Metropolis algorithm** to sample $\sigma \sim |\Psi(\theta)|^2$:

1. Start from random $\sigma_0$
2. Propose: flip random spin $i$ to get $\sigma' = (\sigma_1, \ldots, -\sigma_i, \ldots, \sigma_L)$
3. Acceptance probability: $\alpha = \min(1, |\Psi(\sigma')|^2 / |\Psi(\sigma)|^2)$
4. Accept with probability $\alpha$, else keep $\sigma$
5. Repeat

Since $|\Psi|^2 \propto e^{2\log|\Psi|}$, this is efficient due to **single-spin flips** breaking the RBM factorization minimally.

---

## Connection to Path Integrals

### Imaginary-Time Evolution & Ground State Projection

The **imaginary-time evolution operator** is:
$$e^{-\tau H} = \sum_n e^{-\tau E_n} |\psi_n\rangle\langle\psi_n|$$

As $\tau \to \infty$, this projects onto the ground state: $e^{-\tau H} |\psi\rangle \to |\psi_0\rangle \langle \psi_0 | \psi \rangle$.

### Euclidean Path Integral Form

The ground state amplitude can be written as a path integral:
$$\langle \psi_f | e^{-\beta H} | \psi_i \rangle = \int_{\text{const BC}} D[\psi(t)] \, e^{-\int_0^\beta dt \, H_t[\psi]}$$

where the Euclidean action is:
$$S_E = \int_0^\beta dt \left[ \frac{1}{2}\langle \psi | \dot{\psi} \rangle + \langle \psi | H | \psi \rangle \right]$$

### Variational Manifold as Path Integrand

By minimizing $E(\theta)$ over the RBM manifold:
$$\min_\theta E(\theta) = \min_\theta \frac{\langle \Psi(\theta) | H | \Psi(\theta) \rangle}{\langle \Psi(\theta) | \Psi(\theta) \rangle}$$

we are **effectively summing** $S_E$ over configurations constrained to lie on the NQS manifold.

This is a **projection** of the full path integral onto a low-dimensional subspace — hence "variational."

### Quantum-Classical Mapping

The TFIM in $d=1$ dimension maps to a classical Ising model in $d=2$ (space × imaginary-time):
$$Z = e^{\beta E_0} \times \text{const} + \text{subspace to higher energies}$$

The RBM ansatz can be viewed as an **effective classical model** on this imaginary-time lattice that approximately reproduces the quantum ground state.

---

## Stochastic Dynamics & Optimization

### Natural Gradient Descent on the NQS Manifold

The **natural gradient** on the parameter manifold is:
$$\dot{\theta} = -\eta \, S^{-1}(\theta) \, F(\theta)$$

where:

- $S_{kl} = \text{Cov}_{\sigma \sim |\Psi|^2}(O_k, O_l)$ is the **Fisher information matrix** (also called the quantum geometric tensor for real wave functions)
- $O_k(\sigma) = \partial_k \log|\Psi(\sigma)|$ are the **log-derivative operators**
- $F_k = \text{Cov}_{\sigma}(O_k, E_{\text{loc}}) = \langle O_k E_{\text{loc}} \rangle - \langle O_k \rangle \langle E_{\text{loc}} \rangle$ is the plain energy gradient

**Interpretation:**

- $F$ points uphill in energy in flat parameter space
- $S^{-1}$ re-weights directions by the curvature of the probability manifold
- Result: steepest descent _in KL-divergence_, not in Euclidean parameter space

### Stochastic Reconfiguration (SR) — Implemented in Code

The **Stochastic Reconfiguration** algorithm (Sorella 1998, Carleo & Troyer 2017) is the practical implementation of the above natural gradient. It is implemented in `VMCSolver._compute_sr_update()` and `train_vmc()` (notebook):

1. **Collect log-derivatives** for all $N$ MC samples into matrix $O \in \mathbb{R}^{N \times n_\theta}$
2. **Centre:** $\delta O_i = O_i - \bar{O}$, $\delta E_i = E_i - \bar{E}$
3. **Build Fisher matrix:** $S = \frac{1}{N} \delta O^\top \delta O + \epsilon I$
4. **Build gradient:** $F = \frac{1}{N} \delta O^\top \delta E$
5. **Solve:** $S\, \delta\theta = F$ (small linear system of size $n_\theta \times n_\theta$)
6. **Update:** $\theta \leftarrow \theta - \eta\, \delta\theta$

The diagonal regularisation $\epsilon I$ (controlled by `SR_REG`) is added to prevent singularity when the sample size is small relative to $n_\theta = L + M + LM$.

**Comparison with plain SGD:** SGD only uses $F$ (step 4), ignoring the metric $S$. SR converges significantly faster near the critical point where the Fisher matrix is highly anisotropic.

### Stochastic Discretization & Molecular Dynamics Analogy

In either optimizer, finite MC samples introduce noise:
$$\theta_{t+1} = \theta_t - \eta \, \hat{g}_t$$

where $\hat{g}_t$ is estimated from $N_{\text{samples}}$ configurations. In the **continuous limit**:

This is equivalent to **stochastic differential equation** (in continuous limit):
$$d\theta = -\eta \nabla_\theta E(\theta) \, dt + \sqrt{2\eta D(\theta)} \, d W_t$$

where:

- **Drift term:** $-\eta \nabla_\theta E$ = energy gradient
- **Diffusion term:** $D(\theta) \sim 1/N_{\text{samples}}$ depends on MC noise
- **$dW_t$:** standard Wiener process (random noise)

This is a **Langevin-type dynamics** in parameter space.

### Analogy to Molecular Dynamics

In classical MD, particles follow:
$$m \ddot{x}_i = -\nabla_i V(x) - \gamma \dot{x}_i + F_{\text{noise}}$$

where:

- **$-\nabla V$:** potential gradient (the "force")
- **$\gamma \dot{x}$:** friction/damping
- **$F_{\text{noise}}$:** thermal noise

Similarly, in parameter-space dynamics:

- **$\nabla_\theta E$:** energy gradient (like force on landscape)
- **$\eta$:** inverse "viscosity" (learning rate)
- **MC noise:** thermal fluctuations from sampling

### Phase Transitions & Slow Dynamics

When the system approaches a **phase transition**, the energy landscape becomes increasingly **rugged** with multiple local minima. This manifests as:

1. **Slow convergence:** gradient descent gets trapped in local minima
2. **Long correlation times:** MC sampling becomes inefficient (critical slowing down)
3. **Diverging viscosity:** effective friction increases — dynamics "slow down"

These are hallmarks of **glassy dynamics** and appear in our VMC study.

The **integrated autocorrelation time** $\tau_{\text{int}}$ of the Metropolis chain is measured in `VMCSolver.measure_autocorrelation()` and the notebook cell _Autocorrelation Time and Critical Slowing Down_. A peak in $\tau_{\text{int}}$ near $g = J$ directly signals the phase transition.

---

## Statistical Mechanics of RBMs

### RBM as a Spin-Glass Model

Recall the RBM energy:
$$E_{\text{RBM}}(\sigma, h) = -\sum_i a_i \sigma_i - \sum_j b_j h_j - \sum_{ij} W_{ij} \sigma_i h_j$$

**Physical interpretation:**

- **First term:** external field on spins
- **Second term:** hidden-spin "internal degrees of freedom"
- **Third term:** quenched disorder (in the spirit of spin-glasses)

Via **spin-glass analogy:**

- Weights $W_{ij}$ act like **quenched random couplings**
- Hidden units act like **auxiliary spins in replica trick**

### Mean-Field Theory & Phase Transitions

For large $M$ (number of hidden units), mean-field approximation gives:

$$F_{\text{MF}}(\sigma) = -\sum_i a_i \sigma_i + M \log 2 - \langle \text{log-partition of hidden units} \rangle_{\text{MF}}$$

**Predictions:**

1. For small $W$ (weak coupling): paramagnetic phase
2. For intermediate $W$: ferrimagnetic phase (mixed order)
3. For large $W$: spin-glass phase (glassy disorder)

These correspond to:

- **Small hidden units:** RBM acts like effective single-spin model
- **Large hidden units:** RBM exhibits compositional phases

### Replica Symmetry Breaking (RSB)

In the limit $M/L \to \infty$ (overparameterized RBM), replica analysis predicts:
$$S_{\text{config}} = \frac{1}{M} \log \text{(number of distinguished minima)}$$

**Implication:** exponentially many local minima in parameter space → **hard optimization**.

Observed in our study as:

- Plateau in energy vs. training step
- Non-monotonic convergence rates
- Dependence on initialization

### Entropy & Information Theory

The **effective entropy** trapped in an RBM configuration is:
$$S_{\text{eff}} = \frac{1}{T} \left( \langle E \rangle - F_{\text{MF}} \right)$$

For large $M$:

- $S_{\text{eff}}$ can become **negative** (entropy crisis)
- Related to **compositional breakdown** — hidden units no longer independently encode features

**Connection to NQS task:**

- Low $S_{\text{eff}}$: hidden units are "busy" — each encodes crucial information
- High $S_{\text{eff}}$: redundant hidden units — easy optimization

### Compositional Phase

Unlike "glassy" phases (disordered), RBMs can exhibit a **compositional** phase where:

1. Hidden units self-organize into **feature groups**
   - Group 1: detects domain walls
   - Group 2: detects spin correlations
   - Group 3: detects boundary effects
   - etc.

2. **Sparsity structure:** many $(i,j)$ pairs have $|W_{ij}| \ll 1$

3. **Entropy remains finite:** but _concentrated_ in a subset of hidden units

**Relevance to TFIM:**

- Ordered phase ($g < 1$): RBM learns ferromagnetic structure (few feature groups)
- Critical region ($g \approx 1$): feature proliferation (many groups needed)
- Disordered phase ($g > 1$): disorder-locked features (few but complex groups)

---

## References for Derivations

1. **Carleo, G.** (2017) – Lecture notes: full VMC and natural gradient derivations
2. **Carleo & Troyer** (2017) – Section II: RBM ansatz and local energy formula
3. **Mehta et al.** (2019) – Comprehensive review of machine learning for physics
4. **Decelle et al.** (2021) – Statistical mechanics of RBMs: replica and mean-field analysis
5. **Lange et al.** (2024) – Modern NQS review: path-integral and stochastic-dynamics perspectives

---

**Last Updated:** June 2026
