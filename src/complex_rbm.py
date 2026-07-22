"""Complex-valued RBM neural quantum state — for reproducing Valenti et al. (2019).

Supports both real and complex parameters.  For the 1D TFIM, the ground state is
real, so real parameters suffice; complex parameters are needed for general systems.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------


def init_rbm(
    L: int,
    M: int,
    scale: float = 0.01,
    complex_params: bool = False,
    seed: int = None,
) -> Dict[str, np.ndarray]:
    """Initialise RBM parameters with small random values.

    Args:
        L: number of visible units
        M: number of hidden units
        scale: standard deviation of initialisation
        complex_params: if True, parameters are complex-valued
        seed: random seed

    Returns:
        dict with keys 'a' (L,), 'b' (M,), 'W' (L, M)
    """
    rng = np.random.RandomState(seed)
    if complex_params:
        dtype = complex
        a = (rng.randn(L) + 1j * rng.randn(L)) * scale
        b = (rng.randn(M) + 1j * rng.randn(M)) * scale
        W = (rng.randn(L, M) + 1j * rng.randn(L, M)) * scale
    else:
        dtype = float
        a = rng.randn(L) * scale
        b = rng.randn(M) * scale
        W = rng.randn(L, M) * scale

    return {"a": a.astype(dtype), "b": b.astype(dtype), "W": W.astype(dtype)}


def flatten_params(theta: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten parameter dict to 1-D array."""
    return np.concatenate([theta["a"], theta["b"], theta["W"].ravel()])


def unflatten_params(flat: np.ndarray, L: int, M: int) -> Dict[str, np.ndarray]:
    """Unflatten 1-D array back to parameter dict."""
    a = flat[:L]
    b = flat[L : L + M]
    W = flat[L + M :].reshape(L, M)
    return {"a": a, "b": b, "W": W}


# ---------------------------------------------------------------------------
# Log-amplitude (numerically stable for real and complex parameters)
# ---------------------------------------------------------------------------


def _log2cosh(z: np.ndarray) -> np.ndarray:
    """Numerically stable log(2 cosh(z)) for real or complex z."""
    abs_re = np.abs(np.real(z))
    # = abs_re + log( e^{z - abs_re} + e^{-z - abs_re} )
    return abs_re + np.log(np.exp(z - abs_re) + np.exp(-z - abs_re))


def log_psi(sigma: np.ndarray, theta: Dict[str, np.ndarray]):
    """Log amplitude log Ψ(σ) for real or complex RBM.

    Ψ(σ) = exp(Σ_i a_i σ_i) Π_j 2cosh(χ_j(σ))
    χ_j(σ) = b_j + Σ_i W_ij σ_i

    Args:
        sigma: spin configuration (L,) in {-1, +1}
        theta: RBM parameter dict

    Returns:
        log Ψ(σ)  — complex if params are complex, real otherwise
    """
    a, b, W = theta["a"], theta["b"], theta["W"]
    sigma_cast = sigma.astype(a.dtype)

    term1 = np.dot(a, sigma_cast)
    chi = b + W.T @ sigma_cast  # (M,)
    term2 = np.sum(_log2cosh(chi))
    return term1 + term2


def psi_ratio(sigma_old: np.ndarray, sigma_new: np.ndarray, theta: dict):
    """Ψ(σ_new) / Ψ(σ_old)."""
    return np.exp(log_psi(sigma_new, theta) - log_psi(sigma_old, theta))


def psi_squared_ratio(
    sigma_old: np.ndarray, sigma_new: np.ndarray, theta: dict
) -> float:
    """|Ψ(σ_new)|² / |Ψ(σ_old)|²  (always real, always non-negative)."""
    log_old = log_psi(sigma_old, theta)
    log_new = log_psi(sigma_new, theta)
    return float(np.exp(2.0 * np.real(log_new - log_old)))


# ---------------------------------------------------------------------------
# Log-derivative operators  O_α(σ) = ∂ log Ψ / ∂ θ_α
# ---------------------------------------------------------------------------


def compute_gradients(sigma: np.ndarray, theta: dict) -> Dict[str, np.ndarray]:
    """Log-derivative operators for all parameters.

    O_{a_i}(σ) = σ_i
    O_{b_j}(σ) = tanh(χ_j(σ))
    O_{W_ij}(σ) = σ_i tanh(χ_j(σ))

    Returns:
        dict with arrays matching shapes of a, b, W
    """
    a, b, W = theta["a"], theta["b"], theta["W"]
    sigma_cast = sigma.astype(a.dtype)

    chi = b + W.T @ sigma_cast  # (M,)
    tanh_chi = np.tanh(chi)  # (M,)

    return {
        "a": sigma_cast.copy(),
        "b": tanh_chi.copy(),
        "W": np.outer(sigma_cast, tanh_chi),  # (L, M)
    }


# ---------------------------------------------------------------------------
# Metropolis sampler
# ---------------------------------------------------------------------------


def metropolis_sampler(
    n_samples: int,
    n_steps: int,
    theta: dict,
    L: int,
    rng: np.random.RandomState = None,
    n_burnin: int = 50,
) -> Tuple[np.ndarray, float]:
    """Metropolis-Hastings sampling from |Ψ(θ)|².

    Args:
        n_samples: number of samples to collect
        n_steps:   Metropolis steps between samples
        theta:     RBM parameters
        L:         system size
        rng:       RandomState (default: fresh)
        n_burnin:  burn-in steps

    Returns:
        configs (n_samples, L)  int8
        acceptance rate
    """
    if rng is None:
        rng = np.random.RandomState()

    sigma = rng.choice([-1, 1], size=L)
    configs = np.zeros((n_samples, L), dtype=np.int8)
    n_accepted = 0

    # Burn-in
    for _ in range(n_burnin):
        sigma_new = sigma.copy()
        sigma_new[rng.randint(L)] *= -1
        if rng.rand() < min(1.0, psi_squared_ratio(sigma, sigma_new, theta)):
            sigma = sigma_new

    # Sampling
    for s in range(n_samples):
        for _ in range(n_steps):
            sigma_new = sigma.copy()
            sigma_new[rng.randint(L)] *= -1
            if rng.rand() < min(1.0, psi_squared_ratio(sigma, sigma_new, theta)):
                sigma = sigma_new
                n_accepted += 1
        configs[s] = sigma

    return configs, n_accepted / (n_samples * n_steps)


# ---------------------------------------------------------------------------
# Local energy for 1D TFIM
# H = -J Σ σ_i^z σ_{i+1}^z  - h Σ σ_i^x
# ---------------------------------------------------------------------------


def local_energy_tfim(
    sigma: np.ndarray,
    theta: dict,
    L: int,
    J: float = 1.0,
    h: float = 1.0,
):
    """Local energy E_loc(σ) = <σ|H|Ψ> / <σ|Ψ>.

    Args:
        sigma: configuration {-1,+1}^L
        theta: RBM parameters
        L:     system size
        J:     Ising coupling (default 1.0)
        h:     transverse field

    Returns:
        E_loc (complex if params complex, else real)
    """
    dtype = theta["a"].dtype
    E = dtype.type(0.0)

    # ZZ diagonal terms
    for i in range(L):
        E += dtype.type(-J * sigma[i] * sigma[(i + 1) % L])

    # X off-diagonal terms: flip spin i
    log_psi_sigma = log_psi(sigma, theta)
    for i in range(L):
        sigma_flip = sigma.copy()
        sigma_flip[i] *= -1
        ratio = np.exp(log_psi(sigma_flip, theta) - log_psi_sigma)
        E += dtype.type(-h) * ratio

    return E


# ---------------------------------------------------------------------------
# Quantum Fisher Matrix and spectrum
# ---------------------------------------------------------------------------


def build_qfm(
    configs: np.ndarray,
    theta: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build QFM from sampled configurations.

    S_αβ = <O†_α O_β> - <O†_α><O_β>

    Args:
        configs: (n_samples, L) configurations
        theta:   RBM parameters

    Returns:
        S: QFM matrix  (n_params × n_params), Hermitian
        O: log-derivative matrix (n_samples × n_params) — for reuse
    """
    n_samples = len(configs)
    O_rows = []
    for sigma in configs:
        g = compute_gradients(sigma, theta)
        O_rows.append(np.concatenate([g["a"], g["b"], g["W"].ravel()]))
    O = np.array(O_rows)  # (n_samples, n_params), possibly complex

    O_mean = O.mean(axis=0)
    dO = O - O_mean[None, :]
    S = (dO.conj().T @ dO) / n_samples  # Hermitian
    return S, O


def qfm_spectrum(
    S: np.ndarray,
    n_top: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Eigendecompose QFM (Hermitian), return descending eigenvalues + eigenvectors.

    Args:
        S:     QFM matrix (n_params × n_params)
        n_top: keep only this many largest eigenvalues (None = all)

    Returns:
        eigs  (n_params,)  real, descending
        evecs (n_params, n_params) columns = eigenvectors
    """
    eigs_raw, evecs_raw = np.linalg.eigh(S)
    idx = np.argsort(eigs_raw)[::-1]
    eigs = np.real(eigs_raw[idx])
    evecs = evecs_raw[:, idx]

    if n_top is not None:
        eigs = eigs[:n_top]
        evecs = evecs[:, :n_top]

    return eigs, evecs


# ---------------------------------------------------------------------------
# Eigenvector entanglement entropy
# ---------------------------------------------------------------------------


def eigenvector_entanglement(
    evec: np.ndarray,
    L: int,
    M: int,
) -> float:
    """Von Neumann entanglement entropy of a QFM eigenvector.

    Extracts the W-block (shape L×M) of *evec*, interprets it as a
    bipartite state across visible / hidden parameter directions,
    computes SVD singular values σ_k, and returns:
        S_E = -Σ_k λ_k log λ_k   where λ_k = σ_k² / Σ σ_k²

    This mirrors the paper's measure of how entangled each principal
    optimisation direction is between visible and hidden units.

    Args:
        evec: eigenvector of length L+M+L*M
        L:    visible units
        M:    hidden units

    Returns:
        von Neumann entropy (float, ≥ 0)
    """
    w_start = L + M
    w_block = evec[w_start : w_start + L * M].reshape(L, M)

    sv = np.linalg.svd(w_block, compute_uv=False)
    sv2 = np.real(sv * sv.conj())
    total = sv2.sum()
    if total < 1e-14:
        return 0.0

    lam = sv2 / total
    lam = lam[lam > 1e-14]
    return float(-np.sum(lam * np.log(lam)))


# ---------------------------------------------------------------------------
# SR update
# ---------------------------------------------------------------------------


def sr_update(
    configs: np.ndarray,
    local_energies: np.ndarray,
    theta: dict,
    sr_reg: float = 0.001,
) -> Dict[str, np.ndarray]:
    """Stochastic Reconfiguration natural gradient.

    Builds S (with regularisation S → S + ε I) and solves
    S δ = F  where F is the plain energy gradient.

    Args:
        configs:        (n_samples, L) configurations
        local_energies: (n_samples,) local energies
        theta:          RBM parameters
        sr_reg:         diagonal regularisation ε

    Returns:
        delta: dict matching theta shapes (same dtype)
    """
    n_samples = len(configs)
    L = len(theta["a"])
    M = len(theta["b"])
    E_mean = np.mean(local_energies)

    S, O = build_qfm(configs, theta)
    S += sr_reg * np.eye(S.shape[0], dtype=S.dtype)

    O_mean = O.mean(axis=0)
    dO = O - O_mean[None, :]
    dE = local_energies - E_mean

    # F_α = (1/N) Σ_i dO*_{αi} dE_i = Re( dO†.dE ) / N
    F = (dO.conj().T @ dE) / n_samples

    delta_flat = np.linalg.solve(S, F)

    idx = 0
    delta = {}
    delta["a"] = delta_flat[idx : idx + L]
    idx += L
    delta["b"] = delta_flat[idx : idx + M]
    idx += M
    delta["W"] = delta_flat[idx : idx + L * M].reshape(L, M)
    return delta


# ---------------------------------------------------------------------------
# Parallel Tempering sampler
# ---------------------------------------------------------------------------


def parallel_tempering_sampler(
    n_samples: int,
    n_steps: int,
    theta: dict,
    L: int,
    n_chains: int = 16,
    rng: np.random.RandomState = None,
    n_burnin: int = 200,
) -> Tuple[np.ndarray, float]:
    """Parallel tempering MCMC for |Ψ(θ)|².

    Runs n_chains parallel Markov chains at inverse temperatures β_k
    linearly spaced from 1/n_chains (hot) to 1 (physical, cold).
    Each chain k samples from |Ψ(σ)|^{2β_k}.  Adjacent chains periodically
    attempt configuration swaps, allowing the physical (cold) chain to escape
    regions with near-zero acceptance — the key failure mode when h≈0.

    Paper recipe: 16 chains, β ∈ {1/16, 2/16, …, 1}.

    Args:
        n_samples: configurations to collect from the cold (β=1) chain
        n_steps:   Metropolis steps per chain between swap attempts
        theta:     RBM parameters
        L:         system size
        n_chains:  number of temperature replicas (paper: 16)
        rng:       RandomState
        n_burnin:  burn-in sweeps

    Returns:
        configs:     (n_samples, L)  int8, from the β=1 chain
        accept_rate: local-move acceptance rate on the cold chain
    """
    if rng is None:
        rng = np.random.RandomState()

    # β_k linearly spaced from 1/n_chains (hottest) to 1 (physical/coldest)
    betas = np.linspace(1.0 / n_chains, 1.0, n_chains)

    # Initialise all replicas at random configurations
    sigmas = [rng.choice([-1, 1], size=L) for _ in range(n_chains)]
    log_psis = [float(np.real(log_psi(s, theta))) for s in sigmas]

    configs = np.zeros((n_samples, L), dtype=np.int8)
    n_accepted_cold = 0
    n_steps_cold = 0

    def _sweep(n_sw):
        """One sweep: n_sw local moves per chain + all adjacent swaps."""
        nonlocal n_accepted_cold, n_steps_cold
        for _ in range(n_sw):
            # ── Local moves ─────────────────────────────────────────────
            for c in range(n_chains):
                sigma_new = sigmas[c].copy()
                sigma_new[rng.randint(L)] *= -1
                lp_new = float(np.real(log_psi(sigma_new, theta)))
                # acceptance at temperature 1/β_c:
                # A = min(1, |Ψ_new|^{2β_c} / |Ψ_old|^{2β_c})
                log_a = 2.0 * betas[c] * (lp_new - log_psis[c])
                if log_a >= 0.0 or rng.rand() < np.exp(log_a):
                    sigmas[c] = sigma_new
                    log_psis[c] = lp_new
                    if c == n_chains - 1:
                        n_accepted_cold += 1
                if c == n_chains - 1:
                    n_steps_cold += 1

        # ── Swap attempts between adjacent replicas ──────────────────────
        for c in range(n_chains - 1):
            # detailed balance: swap (σ_c, σ_{c+1})
            # log-acceptance = 2(β_c - β_{c+1})(log|Ψ(σ_{c+1})| - log|Ψ(σ_c)|)
            log_a_swap = (
                2.0 * (betas[c] - betas[c + 1]) * (log_psis[c + 1] - log_psis[c])
            )
            if log_a_swap >= 0.0 or rng.rand() < np.exp(log_a_swap):
                sigmas[c], sigmas[c + 1] = sigmas[c + 1], sigmas[c]
                log_psis[c], log_psis[c + 1] = log_psis[c + 1], log_psis[c]

    # Burn-in
    _sweep(n_burnin)

    # Collect samples from the cold (β=1) chain
    for s_idx in range(n_samples):
        _sweep(n_steps)
        configs[s_idx] = sigmas[-1]  # cold chain is last (β=1)

    accept_rate = n_accepted_cold / max(1, n_steps_cold)
    return configs, accept_rate
