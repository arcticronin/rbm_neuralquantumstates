# Neural Quantum States & the Quantum Fisher Matrix

## RBM Approximation of the 1D Transverse-Field Ising Model

> **Reference paper:** Valenti, Mansfield, Koch-Janusz (2019) — _Scalable Neural-Network Quantum States and Correlations in the Transverse-Field Ising Model_
>
> **Notebooks:** `analysis.ipynb` (exam project) · `paper_reproduction.ipynb` (paper reproduction)

---

## Slide 1 — The Curse of Dimensionality

**The problem:** simulating $L$ spin-$\tfrac{1}{2}$ particles requires a Hilbert space of dimension $2^L$.

| $L$ | $2^L$     | Memory (complex128) |
| --- | --------- | ------------------- |
| 10  | 1 024     | 8 KB                |
| 20  | 1 048 576 | 8 MB                |
| 30  | ~10⁹      | 8 GB                |
| 50  | ~10¹⁵     | **impossible**      |

### Why classical methods fail

| Method                         | Strength                | Failure mode                                   |
| ------------------------------ | ----------------------- | ---------------------------------------------- |
| **Exact Diagonalization**      | Exact, systematic       | Memory $\sim 2^L$ — limited to $L \lesssim 30$ |
| **Tensor Networks (MPS/DMRG)** | Efficient for gapped 1D | Fail at volume-law entanglement, 2D            |
| **Quantum Monte Carlo**        | Polynomial cost         | Sign problem in frustrated/fermionic systems   |

### The NQS Solution

Replace the $2^L$-vector with a neural network:
$$\Psi_\theta(\sigma) = F(\sigma;\,\theta)$$

- $\sigma \in \{-1,+1\}^L$: spin configuration
- $\theta$: learnable parameters (polynomial count)
- Train by minimising $E(\theta) = \langle\Psi_\theta|H|\Psi_\theta\rangle / \langle\Psi_\theta|\Psi_\theta\rangle$

**Intuition:** NQS is a JPEG compressor for quantum states — instead of storing every pixel (every amplitude), it learns the underlying pattern.

---

## Slide 2 — The RBM Architecture

### Structure

An RBM is a **bipartite generative network** with two layers:

- **Visible layer** — $L$ spins $\sigma = (\sigma_1,\ldots,\sigma_L) \in \{-1,+1\}^L$: the physical degrees of freedom
- **Hidden layer** — $M$ auxiliary spins $h = (h_1,\ldots,h_M) \in \{-1,+1\}^M$: introduced to mediate long-range correlations

Hidden units are never observed — they are **traced out (marginalised) analytically**.

```
Visible:  σ₁  σ₂  σ₃  …  σ_L
           ╲╱  ╲╱  ╲╱
Hidden:    h₁  h₂  …  h_M
```

### Parameters

| Symbol       | Shape          | Meaning          |
| ------------ | -------------- | ---------------- |
| $\mathbf{a}$ | $(L,)$         | Visible biases   |
| $\mathbf{b}$ | $(M,)$         | Hidden biases    |
| $W$          | $(L \times M)$ | Coupling weights |

Total parameters: $L + M + LM$ — **polynomial** in system size.

### Wavefunction Formula (Exact Marginalisation)

Summing over all $2^M$ hidden configurations:
$$\boxed{\Psi_\theta(\sigma) = e^{\sum_i a_i \sigma_i} \prod_{j=1}^M 2\cosh\!\left(\chi_j(\sigma)\right)}$$

where $\chi_j(\sigma) = b_j + \sum_i W_{ij}\sigma_i$ is the hidden-unit field.

### Numerical Stability

Direct evaluation overflows for large $|\chi_j|$.  
Work in the **log-domain** using:
$$\log 2\cosh(x) = \operatorname{logaddexp}(x,\,-x) = |x| + \log(1 + e^{-2|x|})$$

**Intuition:** Hidden neurons are like unseen gravitational forces. By observing how the visible spins correlate, the hidden units are accounted for perfectly — without ever measuring them.

---

## Slide 3 — The Physical Model: 1D Transverse-Field Ising Model (TFIM)

### Hamiltonian

$$\boxed{H = -J \sum_{i=1}^{L} \sigma_i^z \sigma_{i+1}^z - h \sum_{i=1}^{L} \sigma_i^x}$$

with **periodic boundary conditions** ($\sigma_{L+1} \equiv \sigma_1$).

| Term                           | Operator     | Effect                             |
| ------------------------------ | ------------ | ---------------------------------- |
| $-J \sigma_i^z \sigma_{i+1}^z$ | Diagonal     | Favours ferromagnetic alignment    |
| $-h \sigma_i^x$                | Off-diagonal | Flips spins → quantum fluctuations |

### Quantum Phase Transition

$$\boxed{g_c = J \quad (g_c/J = 1)}$$
| Phase              | $g < J$                                   | $g = J$          | $g > J$                    |
| ------------------ | ----------------------------------------- | ---------------- | -------------------------- |
| Name               | Ferromagnet (FM)                          | Critical         | Paramagnet (PM)            |
| Ground state       | GHZ-like $\frac{1}{\sqrt{2}}(\vert\uparrow\cdots\uparrow\rangle + \vert\downarrow\cdots\downarrow\rangle)$ | Highly entangled | Product state $\vert+\cdots+\rangle$ |
| Correlation length | Finite                                    | $\xi \to \infty$ | Finite                     |

For finite $L$, the pseudo-critical point is **slightly shifted** below $g_c = 1$ due to finite-size effects.

### Exact Diagonalization (for $L \leq 14$)

Build $H$ as a $2^L \times 2^L$ matrix via Kronecker products:
$$H = -J \sum_i \sigma^z_i \otimes \sigma^z_{i+1} - h \sum_i \sigma^x_i$$

Exact ground state from `numpy.linalg.eigvalsh`.

**Numerical results (N=8, J=1):**
| $h$ | $E_0$ | $E_0/N$ |
|---|---|---|
| 0.0 | −8.0000 | −1.0000 |
| 1.0 | −10.2517 | −1.2815 |
| 2.0 | −17.0182 | −2.1273 |

---

## Slide 4 — Variational Monte Carlo (VMC)

### The Variational Principle

$$E(\theta) = \frac{\langle\Psi_\theta|H|\Psi_\theta\rangle}{\langle\Psi_\theta|\Psi_\theta\rangle} \geq E_0$$

The VMC estimator replaces exact summation with Monte Carlo:
$$\langle H\rangle_\theta = \sum_\sigma p_\theta(\sigma)\,E_\text{loc}(\sigma) \approx \frac{1}{N_s}\sum_{k=1}^{N_s} E_\text{loc}(\sigma^{(k)})$$

where samples $\sigma^{(k)} \sim p_\theta$.

### Born's Rule

The RBM defines a probability distribution:
$$p_\theta(\sigma) = \frac{|\Psi_\theta(\sigma)|^2}{\sum_{\sigma'} |\Psi_\theta(\sigma')|^2}$$

The normalisation sum over all $2^L$ states is avoided by sampling.

**Intuition:** To find the average height of mountains, don't measure every point — drop probes randomly with higher probability of landing on peaks (Born's rule).

---

## Slide 5 — Metropolis–Hastings Sampling

### Algorithm

Starting from $\sigma$, propose $\sigma'$ by flipping one randomly chosen spin.  
Accept with probability:
$$A(\sigma \to \sigma') = \min\!\left(1,\; \frac{|\Psi_\theta(\sigma')|^2}{|\Psi_\theta(\sigma)|^2}\right) = \min\!\Bigl(1,\; e^{2\,[\log|\Psi(\sigma')| - \log|\Psi(\sigma)|]}\Bigr)$$

The **normalisation constant cancels** — only amplitude ratios are needed.

### Acceptance Rate as Phase Diagnostic

| Phase                   | Acceptance Rate       | Reason                               |
| ----------------------- | --------------------- | ------------------------------------ |
| Ferromagnet ($h \ll 1$) | $\ll 1\%$             | All weight on a single configuration |
| Critical ($h = 1$)      | $\sim 25\text{–}35\%$ | Broadly distributed state            |
| Paramagnet ($h \gg 1$)  | $\sim 70\%$           | Near-product state, easy to move     |

**Critical slowing down:** As $h \to 1$, the correlation length $\xi \to \infty$ and local spin-flip moves cannot traverse system-wide correlations — the chain takes exponentially many steps to decorrelate. This is measured by the **autocorrelation time** $\tau_\text{int}$ (see supplementary section).

---

## Slide 6 — Local Energy

### Formula

$$E_\text{loc}(\sigma) = \frac{\langle\sigma|H|\Psi_\theta\rangle}{\langle\sigma|\Psi_\theta\rangle} = \sum_{\sigma'} H_{\sigma\sigma'} \frac{\Psi_\theta(\sigma')}{\Psi_\theta(\sigma)}$$

For the TFIM, matrix elements split into:

| Type             | $\sigma'$                          | $H_{\sigma\sigma'}$               |
| ---------------- | ---------------------------------- | --------------------------------- |
| **Diagonal**     | $\sigma' = \sigma$                 | $-J \sum_i \sigma_i \sigma_{i+1}$ |
| **Off-diagonal** | $\sigma'_i = -\sigma_i$, rest same | $-h$ for each spin $i$            |

In practice:
$$E_\text{loc}(\sigma) = -J \sum_i \sigma_i \sigma_{i+1} - h \sum_i \frac{\Psi_\theta(\sigma_{\text{flip},i})}{\Psi_\theta(\sigma)}$$

Each amplitude ratio $\Psi_\theta(\sigma')/\Psi_\theta(\sigma) = \exp\!\left[\log|\Psi(\sigma')| - \log|\Psi(\sigma)|\right]$ costs $\mathcal{O}(LM)$.

---

## Slide 7 — Stochastic Reconfiguration (SR): Motivation

### Problem with Plain Gradient Descent

Standard update:
$$\theta \leftarrow \theta - \eta \,\nabla_\theta \langle H \rangle$$

This **fails** because the parameter space of quantum probability distributions is **non-Euclidean** — straight-line gradient steps traverse curved geometry inefficiently. The optimiser oscillates in steep valleys rather than converging.

**Geometric interpretation:** Standard gradient descent navigates a globe by drawing straight lines on a flat map. The Quantum Fisher Matrix (QFM) is the globe itself — it encodes the true information-geometric curvature of the probability manifold.

### The Natural Gradient Solution

$$\boxed{\theta \leftarrow \theta - \eta\, (S + \epsilon I)^{-1} \nabla_\theta \langle H \rangle}$$

where $S$ is the **Quantum Fisher Matrix** and $\epsilon$ is a small diagonal regularisation. This is the **Stochastic Reconfiguration (SR)** method — the quantum analogue of Amari's natural gradient.

---

## Slide 8 — Log-Derivative Operators $O_\alpha$

### Definition

$$O_\alpha(\sigma) = \frac{\partial \log \Psi_\theta(\sigma)}{\partial \theta_\alpha}$$

These measure the **sensitivity of the log-amplitude** to each parameter.

### RBM Explicit Forms

With $\chi_j(\sigma) = b_j + \sum_i W_{ij}\sigma_i$:

$$\boxed{O_{a_i}(\sigma) = \sigma_i}$$

$$\boxed{O_{b_j}(\sigma) = \tanh\!\bigl(\chi_j(\sigma)\bigr)}$$

$$\boxed{O_{W_{ij}}(\sigma) = \sigma_i\,\tanh\!\bigl(\chi_j(\sigma)\bigr)}$$

These follow directly from differentiating $\log|\Psi_\theta(\sigma)|$ with respect to each parameter.

### Implementation

In code (`compute_gradients()`):

1. Compute $\chi_j(\sigma) = b_j + W^T \sigma$
2. Compute $\tanh(\chi)$
3. Return $\{O_a = \sigma,\; O_b = \tanh(\chi),\; O_W = \sigma \otimes \tanh(\chi)\}$

---

## Slide 9 — The Quantum Fisher Matrix (QFM)

### Definition

$$\boxed{S_{\alpha\beta} = \langle O_\alpha^\dagger O_\beta \rangle - \langle O_\alpha^\dagger \rangle \langle O_\beta \rangle}$$

This is the **covariance matrix of the log-derivative operators** over sampled configurations.

**Physical meaning:** $S$ encodes the curvature of the probability manifold $\{|\Psi_\theta|^2\}$ via the Fubini-Study metric. It is the information-geometric "ruler" of the parameter space.

### Monte Carlo Estimator

From $N_s$ samples $\{\sigma^{(k)}\}$:

$$S \approx \frac{1}{N_s} \overline{O}^\dagger \overline{O}, \qquad \overline{O}_{k,\alpha} = O_\alpha(\sigma^{(k)}) - \langle O_\alpha\rangle$$

Matrix shape: $(L + M + LM) \times (L + M + LM)$ — for N=8, α=3: $224 \times 224$.

### Energy Gradient (Force Vector)

$$F_\alpha = \frac{\partial \langle H \rangle}{\partial \theta_\alpha} = \langle O_\alpha^\dagger H\rangle - \langle O_\alpha^\dagger\rangle\langle H\rangle = \overline{\text{Cov}}(O_\alpha,\, E_\text{loc})$$

---

## Slide 10 — SR Update Rule & Physical Meaning

### The SR Update

$$\boxed{\theta \leftarrow \theta - \eta\, (S + \epsilon I)^{-1} F}$$

**In practice:**

1. Sample $N_s$ configurations via Metropolis
2. Compute $E_\text{loc}(\sigma^{(k)})$ for each
3. Build $S$ and $F$ from log-derivatives
4. Solve $(S + \epsilon I)\,\delta = F$ for $\delta$
5. Update $\theta \leftarrow \theta - \eta\,\delta$

Regularisation $\epsilon I$ prevents singularity when $S$ is rank-deficient (common in simple phases).

### Physical Meaning: Imaginary-Time Evolution

**Profound result:** the SR update is mathematically equivalent to applying the imaginary-time evolution operator $(1 - \epsilon H)$ to $|\Psi_\theta\rangle$ and **projecting back onto the RBM parameter manifold**:

$$|\Psi_\theta\rangle \xrightarrow{\text{imag. time}} (1 - \epsilon H)|\Psi_\theta\rangle \xrightarrow{\text{projection}} |\Psi_{\theta - \eta\delta}\rangle$$

**Physics interpretation:** In quantum mechanics, imaginary time $t \to -i\tau$ acts as extreme friction that damps all excited states, leaving only the ground state. SR translates this physical cooling into weight updates — the network literally cools into the ground state.

**Stochastic dynamics connection:** the SR update with MC noise reads
$$d\theta = -\eta\, S^{-1} \nabla_\theta E\, dt + \sqrt{\eta/N_s}\; dW_t$$
This is **Langevin dynamics with anisotropic diffusion tensor** $S^{-1}$.

---

## Slide 11 — Numerical Results I: SGD vs SR

### Setup

$L=8$, $g=1.0$ (critical point), $M=8$, 80 epochs, 300 samples.

### Results

| Optimizer                 | Final Energy | $\Delta E = E - E_0$ |
| ------------------------- | ------------ | -------------------- |
| **SGD** (plain gradient)  | −10.057      | +0.194               |
| **SR** (natural gradient) | −10.248      | +0.003               |

**60× improvement** from respecting the Fisher geometry.

### What the plots show

- **Left panel:** SGD oscillates and stagnates; SR descends smoothly to the exact energy
- **Right panel (log scale):** SR error decays monotonically; SGD error plateaus at ~0.2

### Why such a large difference at the critical point?

At $g=1.0$, the QFM has a **dense, broadly distributed spectrum** — the parameter space has many directions of similar importance. Plain SGD treats all directions equally; SR identifies the natural gradient direction in this curved space.

---

## Slide 12 — Numerical Results II: QFM Eigenspectrum (Paper Finding)

_Core result of Valenti et al. (2019): the QFM eigenspectrum is the primary physics diagnostic of the NQS._

### Key measurements (N=8, α=3=M/N)

| $h$ | Phase              | Non-zero eigenvalues | Interpretation                  |
| --- | ------------------ | -------------------- | ------------------------------- |
| 0.0 | FM (product state) | 0 / 224              | Extreme rank-deficiency (acc→0) |
| 0.6 | FM (ordered)       | 56 / 224             | **Rank-deficient**              |
| 1.0 | Critical           | 183 / 224            | **Dense, exponential decay**    |
| 1.4 | PM                 | 217 / 224            | Nearly full, kink visible       |
| 2.0 | PM                 | 224 / 224            | **Fully populated**             |

### Three signatures (all confirmed numerically):

**1. Rank deficiency in ordered phases** ($h < 1$)

- $S$ has only $O(N)$ significant eigenvalues
- The wavefunction lives in a low-dimensional parameter subspace
- Learning is straightforward

**2. Dense exponential decay at criticality** ($h \approx 1$)

- All 224 eigenvalues non-zero, smoothly decaying
- The optimizer must balance all parameter directions simultaneously
- Learning is slow and the natural gradient is ill-conditioned

**3. Symmetry kink at $N(N+1)/2 = 36$**

- In the converged critical/PM spectra, a change of slope appears at eigenvalue index 36
- This separates the **$\mathbb{Z}_2$-symmetric** subspace (indices 0–35) from the **antisymmetric** subspace (indices 36–79)
- Reflects the $\mathbb{Z}_2$ symmetry $\sigma_i \to -\sigma_i$ of the TFI Hamiltonian

---

## Slide 13 — Numerical Results III: Universal Initial Dynamics (Paper Finding)

### Claim (Valenti et al., 2019)

> _At random initialisation (before any SR step), the QFM spectrum is universal — identical for all values of $h$, regardless of the physical phase._

### Why universality holds

At initialisation, parameters are tiny ($\sim 0.01$), so:
$$\log|\Psi_\theta(\sigma)| \approx \text{const} \quad \forall\,\sigma$$

The log-derivative operators $O_\alpha(\sigma)$ are nearly identical for all $\sigma$, and the empirical covariance $S$ is close to a **rescaled identity matrix** — independent of $h$.

### Numerical confirmation (N=8, after 1 SR step)

| $h$ | Non-zero eigenvalues | Max eigenvalue |
| --- | -------------------- | -------------- |
| 0.5 | 6                    | 5.29 × 10⁰     |
| 1.0 | 6                    | 5.29 × 10⁰     |
| 2.0 | 6                    | 5.29 × 10⁰     |

**All three cases produce virtually identical spectra** — the universality is exact at this scale.

### Consequence

Physical information **only enters the QFM as training progresses**. The spectrum reorganises from a universal initial shape into a phase-specific converged shape. Watching this reorganisation is like watching the network "discover" the physics of the system.

---

## Slide 14 — Numerical Results IV: Eigenvector Entanglement (Paper Finding)

### Setup

For each converged QFM eigenvector $v_k$, extract the W-block (size $L \times M$), reshape to matrix $A_k$, compute SVD $A_k = U \Sigma V^T$, and compute the **von Neumann entropy**:

$$S_E(v_k) = -\sum_i \lambda_i \log \lambda_i, \qquad \lambda_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}$$

This measures entanglement between visible and hidden parameter directions in eigenvector $v_k$.

### Key finding (confirmed numerically)

**Large eigenvalues → near-zero entanglement:**

- The 5–10 largest eigenvalues correspond to eigenvectors with $S_E \approx 0$–0.5
- These directions adjust the **mean magnetization** (first moments) of spins — simple, unentangled operations

**Small eigenvalues → high entanglement:**

- The bulk of eigenvalues (small) have $S_E \approx 1.5$–2.0
- Physical correlations are encoded in these directions — wide, flat valleys of the parameter landscape

**Implication:** SR naturally decomposes learning into:

1. Fast, easy directions (large eigenvalues, low entanglement) — handle bias corrections
2. Slow, correlated directions (small eigenvalues, high entanglement) — encode physical correlations in stable, robust flat valleys

---

## Slide 15 — The h=0 Frozen Chain Problem

### What happens at $h=0$

The Hamiltonian $H = -J\sum_i \sigma_i^z\sigma_{i+1}^z$ is purely diagonal.  
Ground state: $|\uparrow\uparrow\cdots\uparrow\rangle$ (or $|\downarrow\cdots\downarrow\rangle$).

The RBM correctly learns this state ($\Delta E = 0$), but concentrates all probability on **one configuration**. Local spin-flip Metropolis then rejects every proposal:

$$A(\sigma \to \sigma') = \min\!\left(1, \frac{|\Psi(\sigma')|^2}{|\Psi(\sigma)|^2}\right) \approx \min(1, e^{-\Delta}) \approx 0 \quad (\Delta \gg 1)$$

**Acceptance rate drops to 0.000 at epoch ~150.** All QFM samples are identical → $S \approx 0$ → 0 non-zero eigenvalues.

### Fix: Parallel Tempering

Run $K$ chains at inverse temperatures $\beta_k = k/K$, $k=1,\ldots,K$:

| Chain               | $\beta_k$ | Distribution | Role     |
| ------------------- | --------- | ------------ | -------- | ----------------- | --------------------- |
| Chain 1 (hottest)   | $1/K$     | $\propto     | \Psi     | ^{2/K}$ ≈ uniform | Explores freely       |
| $\vdots$            | $\vdots$  | $\vdots$     | $\vdots$ |
| Chain $K$ (coldest) | $1$       | $\propto     | \Psi     | ^2$               | Physical distribution |

**Swap acceptance** between adjacent chains $c$ (hotter, $\beta_c$) and $c+1$ (colder, $\beta_{c+1}$):
$$A_\text{swap} = \min\!\left(1,\; \exp\!\Bigl[2(\beta_c - \beta_{c+1})\bigl(\log|\Psi(\sigma_{c+1})| - \log|\Psi(\sigma_c)|\bigr)\Bigr]\right)$$

Hot chains tunnel freely between ferromagnetic domains and pass diverse configurations to the cold chain via swaps.

**Paper recipe:** $K = 16$ chains, $\beta_k = k/16$ for $k = 1, \ldots, 16$.

### Result with PT (N=8, h=0)

| Sampler               | Final $\Delta E$ | Final QFM non-zero | Cold chain acc.    |
| --------------------- | ---------------- | ------------------ | ------------------ |
| Plain Metropolis      | 0.000            | 0/224              | **0.000** (frozen) |
| Parallel Tempering    | ~0.004           | 6/224              | ~0.001             |
| **Exact enumeration** | 0.000            | **rank-1**         | N/A                |

**Note:** For $N \leq 14$, exact QFM computation from full $2^N$-state enumeration is the most reliable approach for $h=0$.

---

## Slide 16 — The QFM Spectrum: Three Phases Side by Side

```
Eigenvalue
  │
  │ h=0.6 (FM, rank-deficient)
  │ ████████
  │         ───────── (near zero after ~50 eigenvalues)
  │
  │ h=1.0 (critical, exponential)
  │ ████████████████████████████████████████
  │ (smooth monotone decay across all 224 eigenvalues)
  │
  │ h=2.0 (PM, kink at N(N+1)/2=36)
  │ ███████████████████│████████████████████
  │ (change of slope at index 36, Z₂ symmetry subspace boundary)
  └──────────────────────────────────────────→  eigenvalue index
              36         224
```

### What the spectrum reveals

| Phase       | Spectrum shape        | Physical meaning                                                    |
| ----------- | --------------------- | ------------------------------------------------------------------- |
| Ferromagnet | **Rank-deficient**    | State is connected to a simple product state; few directions matter |
| Critical    | **Dense exponential** | Long-range entanglement requires all parameter directions           |
| Paramagnet  | **Full with kink**    | $\mathbb{Z}_2$ symmetry partitions the parameter space              |

### The irrelevance of raw weights

A key result of Valenti et al. (2019): the **bare weight values $W_{ij}$ reveal almost nothing** about the physical system. Extreme parameter redundancy means many different weight configurations represent the same ground state. **The QFM eigenspectrum is the true diagnostic.**

---

## Slide 17 — MCMC Autocorrelation Time (Supplementary)

> _Not a paper result — supplementary connection to Langevin dynamics._

### Setup

Measure integrated autocorrelation time $\tau_\text{int}$ of the **magnetisation** observable $|m(\sigma)| = |\sum_i \sigma_i|/L$ along a Metropolis chain on the trained RBM.

$$\tau_\text{int} = \frac{1}{2} + \sum_{t=1}^\infty \frac{C(t)}{C(0)}, \qquad C(t) = \langle m(t_0)\,m(t_0+t)\rangle - \langle m\rangle^2$$

**Why magnetisation, not $E_\text{loc}$?** A converged RBM has near-zero variance in $E_\text{loc}$ — the energy autocorrelation appears short even when the chain is frozen. The magnetisation is sensitive to domain-wall crossings, which is exactly what slows down near criticality.

### Numerical results (N=8, α=1, seed=7)

| $h$      | $\tau$ (MC steps) |
| -------- | ----------------- |
| 0.50     | 9.16              |
| **0.75** | **9.57** (peak)   |
| 1.00     | 8.99              |
| 1.25     | 7.55              |
| 1.50     | 6.10              |
| 2.00     | 3.46              |

Peak at $h \approx 0.75$ (slightly below $g_c = 1$) — consistent with the finite-size pseudo-critical shift.

### Connection to Langevin dynamics

Near the phase transition, the Metropolis chain satisfies a **stochastic dynamics equation** analogous to overdamped Langevin dynamics. The diverging autocorrelation time mirrors critical slowing down in classical MD simulations.

---

## Slide 18 — Wavefunction Fidelity (Supplementary)

> _Not a paper result — supplementary test of ansatz expressivity._

### Definition

$$\mathcal{F} = |\langle \Psi_\text{RBM} | \psi_0\rangle|^2 \in [0,1]$$

Computed by full enumeration over all $2^L$ states (feasible for $L \leq 14$).

**Why fidelity $>$ energy?** Energy is a single scalar; a structurally wrong wavefunction can still have low energy. Fidelity measures exact structural overlap.

### Numerical results (N=8)

| $h$      | $M=2$ (sparse, M/L=0.25) | $M=8$ (unit-density) |
| -------- | ------------------------ | -------------------- |
| 0.50     | 0.50                     | 1.00                 |
| 0.75     | 0.51                     | 1.00                 |
| **1.00** | **0.95**                 | **1.00**             |
| 1.25     | 0.94                     | 1.00                 |
| 1.50     | 0.94                     | 1.00                 |
| 2.00     | 0.96                     | 1.00                 |

**Observation:** The sparse ($M=2$) RBM struggles most in the ferromagnetic phase ($h < 1$), not at the critical point. This is because the ferromagnetic ground state is a GHZ-like superposition — hard to encode with only 2 hidden units. Unit-density RBM achieves $\mathcal{F} = 1$ everywhere at $N=8$.

---

## Slide 19 — Energy Error Heatmap (Supplementary)

> _Not a paper result — supplementary visualisation of how ansatz size and phase jointly determine accuracy._

### Setup

Sweep over $g \in \{0.5, 0.75, 1.0, 1.25, 1.5, 2.0\}$ and $M/L \in \{0.5, 1.0, 1.5, 2.0\}$.

**Color (log scale):** $|\Delta E| = |E_\text{VMC} - E_0|$ — brighter = worse approximation.

### What to look for

- **Row (M/L):** more hidden units = more expressive ansatz = smaller error
- **Column (g):** near-critical region shows elevated error for small M/L
- **Connection to QFM:** at $g \approx 1$, the Fisher spectrum is broadly distributed → optimizer needs more parameter directions → more hidden units required

**Ferromagnetic region ($g < 1$) also shows elevated error** — caused by the frozen Metropolis chain (acceptance ~3–5%), not by intrinsic complexity.

---

## Slide 20 — Experiment Setup: Paper Reproduction

### System parameters (paper values)

| Parameter                    | Value                                   |
| ---------------------------- | --------------------------------------- |
| Visible units $N$            | 8 (small) → 28 (paper)                  |
| Hidden ratio $\alpha = M/N$  | 3                                       |
| Hidden units $M = \alpha N$  | 24 (N=8) → 84 (N=28)                    |
| Total parameters             | 224 (N=8) → 2464 (N=28)                 |
| Learning rate $\eta$         | 0.01                                    |
| SR regularisation $\epsilon$ | 0.001                                   |
| Epochs                       | 300 (N=8) → 2000+ (paper)               |
| Sampler                      | Metropolis → Parallel Tempering (paper) |

### Transverse fields swept

$$h \in \{0.0,\; 0.6,\; 1.0,\; 1.4,\; 2.0\}$$

Spanning ferromagnet → critical → paramagnet.

### QFM snapshot epochs

QFM spectrum recorded at epochs $\{0, 10, 25, 50, 100, 200\}$ to watch geometry evolve.

---

## Slide 21 — Energy Convergence Results

### Summary table (N=8)

| $h$ | Phase    | $E_0$ (exact) | $E_\text{VMC}$ | $\Delta E$ |
| --- | -------- | ------------- | -------------- | ---------- |
| 0.0 | FM       | −8.0000       | −8.0000        | **0.0000** |
| 0.6 | FM       | −8.7408       | −8.7392        | +0.0016    |
| 1.0 | Critical | −10.2517      | −10.2504       | +0.0013    |
| 1.4 | PM       | −12.6962      | −12.6969       | −0.0007    |
| 2.0 | PM       | −17.0182      | −17.0174       | +0.0007    |

All $|\Delta E| < 0.002$ — excellent agreement with exact diagonalization.

### Convergence speed by phase

- **PM ($h=2.0$):** fastest — simple product-like state
- **Critical ($h=1.0$):** slowest non-trivial — dense QFM makes SR ill-conditioned
- **FM ($h=0.0$):** exact at epoch ~150, then chain freezes

---

## Slide 22 — Connection to Exam Topics

### Topic 1: Path Integrals and Variational Projection

$$E(\theta) = \frac{\langle\Psi_\theta|H|\Psi_\theta\rangle}{\langle\Psi_\theta|\Psi_\theta\rangle} \geq E_0$$

- The Euclidean path integral $\int \mathcal{D}[\psi]\,e^{-S_E[\psi]}$ projects onto the ground state
- The RBM restricts this to a variational subspace
- Minimising $E(\theta)$ = variational projection onto the NQS manifold
- SR update = imaginary-time evolution $(1 - \epsilon H)|\Psi\rangle$ projected back onto the manifold

### Topic 2: Stochastic Dynamics and Molecular Dynamics

The SR update with MC noise:
$$d\theta = -\eta\, S^{-1} \nabla_\theta E\, dt + \sqrt{\eta/N_s}\; dW_t$$

This is **Langevin dynamics** with anisotropic diffusion tensor $S^{-1}$. Near phase transitions, the QFM becomes broadly distributed (dense spectrum, large condition number) → learning dynamics slow down in parameter space, mirroring critical slowing down in physical systems.

### Topic 3: Statistical Mechanics of RBMs

The RBM energy functional:
$$\mathcal{E}(\sigma, h) = -\sum_i a_i \sigma_i - \sum_j b_j h_j - \sum_{ij} W_{ij}\sigma_i h_j$$

maps to a **bipartite spin-glass** model. The QFM eigenspectrum is the physical diagnostic:

- Ordered phase → rank-deficient $S$ (simple ground state)
- Critical point → dense, exponential $S$ (maximal complexity)
- Disordered phase → $S$ with $\mathbb{Z}_2$ symmetry kink

---

## Slide 23 — Summary of Key Results

### Paper findings reproduced

| Claim                                | Numerical evidence                                      |
| ------------------------------------ | ------------------------------------------------------- |
| Universal initial QFM spectrum       | All 3 phases give 6/224 non-zero, max=5.29 after 1 step |
| Rank deficiency in ordered phase     | $h=0.6$: 56/224 non-zero (25% populated)                |
| Dense exponential at criticality     | $h=1.0$: 183/224 populated, smooth decay                |
| Symmetry kink at $N(N+1)/2 = 36$     | Visible kink in PM and critical spectra                 |
| Large eigenvalues = low entanglement | Confirmed in all 5 phases                               |
| Raw weights don't encode physics     | Weight statistics show ambiguous trends                 |

### Supplementary analyses (not in paper)

| Analysis                 | Finding                                                                |
| ------------------------ | ---------------------------------------------------------------------- |
| SGD vs SR at criticality | SR: $\Delta E = 0.003$, SGD: $\Delta E = 0.19$ — 60× improvement       |
| MCMC autocorrelation     | $\tau$ peaks at $h \approx 0.75$ (finite-size shift of critical point) |
| Wavefunction fidelity    | $M=2$ shows low fidelity in FM phase; $M=8$ perfect everywhere         |
| Energy error heatmap     | Elevated errors near $g \approx 1$ for small $M/L$                     |

---

## Appendix A — Formula Reference

### RBM Log-Amplitude

$$\log\Psi_\theta(\sigma) = \sum_i a_i \sigma_i + \sum_j \log 2\cosh\!\left(b_j + \sum_i W_{ij}\sigma_i\right)$$

### Log-Derivative Operators

$$O_{a_i} = \sigma_i \qquad O_{b_j} = \tanh(\chi_j) \qquad O_{W_{ij}} = \sigma_i \tanh(\chi_j)$$

### Quantum Fisher Matrix

$$S_{\alpha\beta} = \frac{1}{N_s}\sum_k \overline{O}_\alpha(\sigma^{(k)})^* \overline{O}_\beta(\sigma^{(k)}), \qquad \overline{O}_\alpha = O_\alpha - \langle O_\alpha\rangle$$

### SR Update

$$\delta = (S + \epsilon I)^{-1} F, \qquad F_\alpha = \text{Cov}(O_\alpha,\, E_\text{loc})$$

### Local Energy (TFIM)

$$E_\text{loc}(\sigma) = -J\sum_i \sigma_i\sigma_{i+1} - h\sum_i e^{\log|\Psi(\sigma^{(i)}_\text{flip})| - \log|\Psi(\sigma)|}$$

### Metropolis Acceptance

$$A(\sigma \to \sigma') = \min\!\left(1,\; e^{2[\log|\Psi(\sigma')| - \log|\Psi(\sigma)|]}\right)$$

### Parallel Tempering Swap Acceptance

$$A_\text{swap}(c, c{+}1) = \min\!\left(1,\; e^{2(\beta_c - \beta_{c+1})[\log|\Psi(\sigma_{c+1})| - \log|\Psi(\sigma_c)|]}\right)$$

### Eigenvector Entanglement Entropy

$$S_E(v_k) = -\sum_i \lambda_i \log \lambda_i, \quad \lambda_i = \frac{\sigma_i^2(W\text{-block of }v_k)}{\sum_j \sigma_j^2}$$

### Integrated Autocorrelation Time

$$\tau_\text{int} = \frac{1}{2} + \sum_{t=1}^\infty \frac{C(t)}{C(0)}, \qquad C(t) = \langle m(t_0)\,m(t_0+t)\rangle - \langle m\rangle^2$$

---

## Appendix B — Code Structure

```
rbm_neuralquantumstates/
├── src/
│   ├── complex_rbm.py      # RBM core: init, log_psi, gradients, QFM,
│   │                       #   SR, parallel_tempering_sampler
│   ├── tfim_exact.py       # Exact diagonalization, TFIMExact class
│   ├── rbm_nqs.py          # RBMNQS class (OOP interface)
│   ├── vmc.py              # VMCSolver class
│   └── utils.py            # generate_all_configs, load/save JSON
├── notebooks/
│   ├── analysis.ipynb      # Exam project notebook
│   └── paper_reproduction.ipynb  # Valenti et al. reproduction
└── results/
    ├── paper_energy_convergence.png
    ├── paper_qfm_evolution.png
    ├── paper_qfm_converged.png
    └── paper_entanglement.png
```

### Key functions in `src/complex_rbm.py`

| Function                            | Purpose                                 |
| ----------------------------------- | --------------------------------------- |
| `init_rbm(N, M, ...)`               | Initialise parameters (real or complex) |
| `log_psi(σ, θ)`                     | Stable log-amplitude                    |
| `compute_gradients(σ, θ)`           | Log-derivative operators $O_\alpha$     |
| `metropolis_sampler(...)`           | Standard Metropolis-Hastings            |
| `parallel_tempering_sampler(...)`   | PT with $K$ temperature replicas        |
| `local_energy_tfim(σ, θ, N, h)`     | TFIM local energy                       |
| `build_qfm(configs, θ)`             | Build QFM matrix $S$                    |
| `qfm_spectrum(S)`                   | Eigendecompose $S$ (descending)         |
| `eigenvector_entanglement(v, N, M)` | von Neumann entropy of W-block          |
| `sr_update(configs, E_locs, θ, ε)`  | SR natural gradient step                |

---

## Appendix C — Scaling to N=28 (Paper Regime)

To reproduce the full paper results, change `N=8` to `N=28` in the hyperparameters cell.

| Quantity                  | N=8              | N=28                |
| ------------------------- | ---------------- | ------------------- |
| Hilbert space $2^N$       | 256              | $2.7 \times 10^8$   |
| Hidden units $M = 3N$     | 24               | 84                  |
| Parameters                | 224              | 2464                |
| QFM matrix                | $224 \times 224$ | $2464 \times 2464$  |
| Epochs                    | 300              | 2000+               |
| Parallel tempering chains | 16               | 16                  |
| Runtime                   | ~5 min           | ~hours (single CPU) |

Additional requirement for N=28: **parallel tempering is mandatory** for the ferromagnetic phases. Exact enumeration for QFM is not feasible; only PT-collected samples can be used.
