"""Variational Monte Carlo solver for NQS wave functions."""

import numpy as np
from typing import Tuple, Dict, List
from rbm_nqs import RBMNQS
from tfim_exact import TFIMExact


class VMCSolver:
    """Variational Monte Carlo with NQS ansatz."""

    def __init__(
        self, L: int, M: int, J: float = 1.0, g: float = 1.0, seed: int = None
    ):
        """Initialize VMC solver.

        Args:
            L: System size
            M: Number of hidden units
            J: Ising coupling
            g: Transverse field
            seed: Random seed
        """
        self.L = L
        self.M = M
        self.J = J
        self.g = g

        self.nqs = RBMNQS(L, M, seed=seed)
        self.rng = np.random.RandomState(seed)

        # For tracking
        self.energies = []
        self.variances = []

    def local_energy(self, sigma: np.ndarray) -> float:
        """Compute local energy E_loc(σ) for TFIM.

        E_loc(σ) = <ψ|H|σ> / <ψ|σ>

        Computed via sum over single-spin flips that contribute to H |σ>:
        - ZZ terms: -J σ^z_i σ^z_{i+1}
        - X terms: -g σ^x_i (flip i-th spin)

        Args:
            sigma: Configuration {-1, +1}^L

        Returns:
            Local energy
        """
        E_loc = 0.0
        sigma_copy = sigma.copy()

        # ZZ coupling: -J σ_i σ_{i+1}
        for i in range(self.L):
            i_next = (i + 1) % self.L
            E_loc += -self.J * sigma[i] * sigma[i_next]

        # Transverse field X term: -g * <σ| σ^x_i |ψ> / <σ|ψ>
        # On basis state |σ>, σ^x_i flips spin i, contributing the amplitude
        # ratio Ψ(σ_flip) / Ψ(σ) (NOT the squared probability ratio).
        for i in range(self.L):
            sigma_flip = sigma.copy()
            sigma_flip[i] *= -1

            # Amplitude ratio: <σ_flip|ψ> / <σ|ψ>
            psi_ratio = self.nqs.psi_ratio(sigma, sigma_flip)
            E_loc += -self.g * psi_ratio

        return E_loc

    def metropolis_step(self, sigma: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Single Metropolis step.

        Args:
            sigma: Current configuration

        Returns:
            New configuration and acceptance flag
        """
        sigma_new = sigma.copy()

        # Propose: flip random spin
        i = self.rng.randint(0, self.L)
        sigma_new[i] *= -1

        # Accept/reject based on |ψ|^2
        accept_prob = self.nqs.psi_squared_ratio(sigma, sigma_new)
        accept_prob = min(1.0, accept_prob)

        if self.rng.rand() < accept_prob:
            return sigma_new, True
        else:
            return sigma, False

    def sample_configs(
        self, n_samples: int, n_steps: int = 1
    ) -> Tuple[np.ndarray, float]:
        """Generate Metropolis samples from |ψ|^2.

        Args:
            n_samples: Number of samples to generate
            n_steps: Metropolis steps between samples

        Returns:
            Array of configurations (n_samples, L) and acceptance rate
        """
        configs = np.zeros((n_samples, self.L), dtype=np.int8)

        # Random initial configuration
        sigma = self.rng.choice([-1, 1], size=self.L)

        # Thermalization
        for _ in range(100):
            sigma, _ = self.metropolis_step(sigma)

        # Sampling
        n_accepted = 0
        config_idx = 0

        for sample_idx in range(n_samples):
            for step in range(n_steps):
                sigma, accepted = self.metropolis_step(sigma)
                if accepted:
                    n_accepted += 1

            configs[sample_idx] = sigma

        accept_rate = n_accepted / (n_samples * n_steps)

        return configs, accept_rate

    def estimate_energy(self, n_samples: int = 1000) -> Tuple[float, float]:
        """Estimate variational energy from MC samples.

        Args:
            n_samples: Number of samples

        Returns:
            Mean energy and standard error
        """
        configs, _ = self.sample_configs(n_samples, n_steps=self.L)

        local_energies = np.array([self.local_energy(sigma) for sigma in configs])

        mean_E = np.mean(local_energies)
        std_E = np.std(local_energies)
        err_E = std_E / np.sqrt(n_samples)

        return mean_E, err_E

    def _compute_sr_update(
        self, configs: np.ndarray, local_energies: np.ndarray, sr_reg: float
    ) -> Dict[str, np.ndarray]:
        """Stochastic Reconfiguration (natural gradient) update direction.

        Computes the natural gradient S^{-1} F where:
          S_{kl} = Cov_{sigma}(O_k, O_l)  -- Fisher information matrix
          F_k    = Cov_{sigma}(O_k, E_loc) -- variational energy gradient

        O_k(sigma) = d log|Psi| / d theta_k are the log-derivatives collected
        into a flat vector of length n_params = L + M + L*M.

        Regularisation: S -> S + sr_reg * I  (diagonal shift) prevents
        singularity and stabilises early training when S is near-rank-deficient.

        Args:
            configs: MC configurations of shape (n_samples, L)
            local_energies: Local energy for each sample, shape (n_samples,)
            sr_reg: Diagonal regularisation added to S

        Returns:
            Dict with parameter-shaped natural-gradient update directions
        """
        n_samples = len(configs)
        E_mean = np.mean(local_energies)
        L, M = self.nqs.L, self.nqs.M

        # --- collect log-derivative matrix O: shape (n_samples, n_params) ---
        O_rows = []
        for sigma in configs:
            grads = self.nqs.compute_gradients(sigma, self.nqs.theta)
            flat = np.concatenate([grads["a"], grads["b"], grads["W"].ravel()])
            O_rows.append(flat)
        O = np.array(O_rows)  # (n_samples, n_params)

        O_mean = O.mean(axis=0)  # (n_params,)
        dO = O - O_mean[None, :]  # centred log-derivatives
        dE = local_energies - E_mean  # centred energies

        # Fisher / SR matrix  S = (1/N) dO^T dO
        S = (dO.T @ dO) / n_samples
        S += sr_reg * np.eye(S.shape[0])

        # Energy gradient (force vector)  F = (1/N) dO^T dE
        F = (dO.T @ dE) / n_samples

        # Solve S * delta = F
        delta_flat = np.linalg.solve(S, F)

        # Unpack back into parameter-shaped dicts
        idx = 0
        grad_avg = {}
        grad_avg["a"] = delta_flat[idx : idx + L]
        idx += L
        grad_avg["b"] = delta_flat[idx : idx + M]
        idx += M
        grad_avg["W"] = delta_flat[idx : idx + L * M].reshape(L, M)

        return grad_avg

    def _compute_sgd_gradient(
        self, configs: np.ndarray, local_energies: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Plain variational energy gradient (vanilla SGD direction).

        grad_k = (1/N) sum_i (E_loc_i - <E>) * O_k(sigma_i)

        Args:
            configs: MC configurations of shape (n_samples, L)
            local_energies: Local energy for each sample, shape (n_samples,)

        Returns:
            Dict with parameter-shaped gradient update directions
        """
        n_samples = len(configs)
        E_mean = np.mean(local_energies)

        grad_avg = {key: np.zeros_like(self.nqs.theta[key]) for key in self.nqs.theta}

        for i, sigma in enumerate(configs):
            force = local_energies[i] - E_mean
            grad = self.nqs.compute_gradients(sigma, self.nqs.theta)
            for key in grad_avg:
                grad_avg[key] += force * grad[key]

        for key in grad_avg:
            grad_avg[key] /= n_samples

        return grad_avg

    def optimize_step(
        self,
        n_samples: int = 1000,
        learning_rate: float = 0.01,
        clip_norm: float = None,
        use_sr: bool = False,
        sr_reg: float = 1e-4,
    ) -> Dict:
        """Single optimization step.

        Supports two optimizers:
          - SGD (use_sr=False, default): plain variational energy gradient.
          - Stochastic Reconfiguration (use_sr=True): natural gradient that
            pre-conditions the update by the inverse Fisher information matrix,
            i.e. the geometry of the probability manifold |Psi|^2.

        Args:
            n_samples: MC samples for gradient estimation
            learning_rate: Step size / learning rate
            clip_norm: Clip gradient L2-norm to this value (None = no clipping)
            use_sr: Use Stochastic Reconfiguration (natural gradient) update
            sr_reg: Diagonal regularisation added to Fisher matrix (SR only)

        Returns:
            Dict with diagnostics
        """
        configs, accept_rate = self.sample_configs(n_samples, n_steps=self.L)

        # Compute local energies
        local_energies = np.array([self.local_energy(sigma) for sigma in configs])
        E_mean = np.mean(local_energies)

        # Choose optimizer
        if use_sr:
            grad_avg = self._compute_sr_update(configs, local_energies, sr_reg)
        else:
            grad_avg = self._compute_sgd_gradient(configs, local_energies)

        # Update parameters (clip_norm applied inside update_parameters)
        self.nqs.update_parameters(
            grad_avg, learning_rate=learning_rate, clip_norm=clip_norm
        )

        # Re-estimate energy after update
        E_new, err_E = self.estimate_energy(n_samples=500)

        self.energies.append(E_new)

        return {
            "energy": E_new,
            "energy_err": err_E,
            "accept_rate": accept_rate,
            "grad_norm_a": float(np.linalg.norm(grad_avg["a"])),
            "grad_norm_b": float(np.linalg.norm(grad_avg["b"])),
            "grad_norm_W": float(np.linalg.norm(grad_avg["W"])),
        }

    def measure_autocorrelation(self, n_steps: int = 2000) -> float:
        """Estimate Metropolis autocorrelation time for the current RBM.

        Runs a single long Metropolis chain and computes the integrated
        autocorrelation time tau of the local energy observable.  Near the
        TFIM critical point (g ~ J) tau diverges — critical slowing down.

        Args:
            n_steps: Length of Markov chain to record

        Returns:
            Integrated autocorrelation time tau (in units of MC steps)
        """
        sigma = self.rng.choice([-1, 1], size=self.L)
        # Burn-in
        for _ in range(200):
            sigma, _ = self.metropolis_step(sigma)

        # Collect E_loc time-series
        series = np.zeros(n_steps)
        for t in range(n_steps):
            sigma, _ = self.metropolis_step(sigma)
            series[t] = self.local_energy(sigma)

        # Compute normalised autocorrelation function C(t) up to n_steps/2
        series -= series.mean()
        var = np.var(series)
        if var < 1e-12:
            return 1.0

        max_lag = n_steps // 2
        c = np.correlate(series, series, mode="full")
        c = c[n_steps - 1 :]  # Keep lags 0, 1, 2, ...
        c = c[:max_lag] / (var * n_steps)

        # Integrated autocorrelation time: sum until c(t) < threshold
        tau = 0.5
        for t in range(1, max_lag):
            if c[t] < 0.05:
                break
            tau += c[t]

        return float(tau)

    def wavefunction_overlap(self) -> float:
        """Compute |<Psi_RBM | Psi_exact>|^2 for small systems (L <= 14).

        Enumerates all 2^L configurations, builds the exact ground-state
        vector via full diagonalisation, and computes the squared overlap.
        This is the most direct quality metric for the NQS ansatz.

        Returns:
            Squared overlap in [0, 1].  1.0 means perfect representation.
        """
        from utils import generate_all_configs

        n_configs = 2**self.L
        configs_all = generate_all_configs(self.L)

        # RBM amplitudes (unnormalised)
        log_psis = np.array([self.nqs.log_psi(s) for s in configs_all])
        log_psis -= log_psis.max()  # shift for numerical stability
        psi_rbm = np.exp(log_psis)
        psi_rbm /= np.linalg.norm(psi_rbm)

        # Exact ground state
        tfim = TFIMExact(self.L, J=self.J, g=self.g)
        psi_exact = tfim.groundstate_vector()
        # Enforce consistent sign convention (largest component positive)
        if psi_exact[np.argmax(np.abs(psi_exact))] < 0:
            psi_exact = -psi_exact
        psi_exact /= np.linalg.norm(psi_exact)

        overlap = float(np.dot(psi_rbm, psi_exact) ** 2)
        return overlap

    def train(
        self,
        n_steps: int = 100,
        n_samples: int = 1000,
        learning_rate: float = 0.01,
        verbose: bool = True,
        clip_norm: float = None,
        use_sr: bool = False,
        sr_reg: float = 1e-4,
    ) -> List[Dict]:
        """Full VMC training loop.

        Args:
            n_steps: Number of optimization steps
            n_samples: MC samples per step
            learning_rate: SGD / SR learning rate
            verbose: Print progress
            clip_norm: Gradient norm clip threshold (None = disabled)
            use_sr: Use Stochastic Reconfiguration (natural gradient)
            sr_reg: SR diagonal regularisation (only used when use_sr=True)

        Returns:
            List of diagnostics dicts
        """
        history = []

        for step in range(n_steps):
            diag = self.optimize_step(
                n_samples=n_samples,
                learning_rate=learning_rate,
                clip_norm=clip_norm,
                use_sr=use_sr,
                sr_reg=sr_reg,
            )
            history.append(diag)

            if verbose and (step % 10 == 0 or step == n_steps - 1):
                print(
                    f"Step {step:3d}: E = {diag['energy']:8.4f} ± {diag['energy_err']:.4f}, "
                    f"acc_rate = {diag['accept_rate']:.3f}"
                )

        return history
