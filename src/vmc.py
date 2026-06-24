"""Variational Monte Carlo solver for NQS wave functions."""

import numpy as np
from typing import Tuple, Dict, List
from rbm_nqs import RBMNQS
from tfim_exact import TFIMExact


class VMCSolver:
    """Variational Monte Carlo with NQS ansatz."""
    
    def __init__(self, L: int, M: int, J: float = 1.0, g: float = 1.0,
                 seed: int = None):
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
        # On basis state |σ>, σ^x_i flips spin i
        for i in range(self.L):
            sigma_flip = sigma.copy()
            sigma_flip[i] *= -1
            
            # Ratio: <σ_flip|ψ> / <σ|ψ>
            psi_ratio = self.nqs.psi_squared_ratio(sigma, sigma_flip)
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
    
    def sample_configs(self, n_samples: int, n_steps: int = 1) -> Tuple[np.ndarray, float]:
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
        configs, _ = self.sample_configs(n_samples)
        
        local_energies = np.array([self.local_energy(sigma) for sigma in configs])
        
        mean_E = np.mean(local_energies)
        std_E = np.std(local_energies)
        err_E = std_E / np.sqrt(n_samples)
        
        return mean_E, err_E
    
    def optimize_step(self, n_samples: int = 1000,
                     learning_rate: float = 0.01) -> Dict:
        """Single optimization step using natural gradient.
        
        Args:
            n_samples: MC samples for gradient estimation
            learning_rate: SGD learning rate
            
        Returns:
            Dict with diagnostics
        """
        configs, accept_rate = self.sample_configs(n_samples)
        
        # Compute local energies
        local_energies = np.array([self.local_energy(sigma) for sigma in configs])
        E_mean = np.mean(local_energies)
        
        # Compute gradients for all samples
        gradients = []
        for sigma in configs:
            grad = self.nqs.compute_gradients(sigma, self.nqs.theta)
            gradients.append(grad)
        
        # Average gradients, weighted by energy (force):
        # We update in direction that reduces energy
        grad_avg = {key: np.zeros_like(self.nqs.theta[key]) for key in self.nqs.theta}
        
        for i, sigma in enumerate(configs):
            force = local_energies[i] - E_mean
            grad = self.nqs.compute_gradients(sigma, self.nqs.theta)
            
            for key in grad_avg:
                grad_avg[key] += force * grad[key]
        
        for key in grad_avg:
            grad_avg[key] /= n_samples
        
        # Update parameters: move opposite to gradient (steepest descent on energy)
        self.nqs.update_parameters(grad_avg, learning_rate=learning_rate)
        
        # Re-estimate energy
        E_new, err_E = self.estimate_energy(n_samples=500)
        
        self.energies.append(E_new)
        
        return {
            'energy': E_new,
            'energy_err': err_E,
            'accept_rate': accept_rate,
            'grad_norm_a': np.linalg.norm(grad_avg['a']),
            'grad_norm_b': np.linalg.norm(grad_avg['b']),
            'grad_norm_W': np.linalg.norm(grad_avg['W']),
        }
    
    def train(self, n_steps: int = 100, n_samples: int = 1000,
             learning_rate: float = 0.01, verbose: bool = True) -> List[Dict]:
        """Full VMC training loop.
        
        Args:
            n_steps: Number of optimization steps
            n_samples: MC samples per step
            learning_rate: SGD learning rate
            verbose: Print progress
            
        Returns:
            List of diagnostics dicts
        """
        history = []
        
        for step in range(n_steps):
            diag = self.optimize_step(n_samples=n_samples,
                                     learning_rate=learning_rate)
            history.append(diag)
            
            if verbose and (step % 10 == 0 or step == n_steps - 1):
                print(f"Step {step:3d}: E = {diag['energy']:8.4f} ± {diag['energy_err']:.4f}, "
                      f"acc_rate = {diag['accept_rate']:.3f}")
        
        return history
