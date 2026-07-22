"""RBM-based Neural Quantum State for TFIM."""

import numpy as np
from typing import Dict, Tuple
import copy


class RBMNQS:
    """Restricted Boltzmann Machine as Neural Quantum State.

    Wave function ansatz:
    Ψ_RBM(σ) ∝ exp(Σ_i a_i σ_i) ∏_j 2*cosh(b_j + Σ_i W_{ij} σ_i)

    Parameters:
    - a: visible biases (L,)
    - b: hidden biases (M,)
    - W: weights (L, M)
    """

    def __init__(self, L: int, M: int, seed: int = None):
        """Initialize RBM with random parameters.

        Args:
            L: Number of visible (spin) units
            M: Number of hidden units
            seed: Random seed
        """
        self.L = L
        self.M = M

        if seed is not None:
            np.random.seed(seed)

        # Initialize parameters with small random values
        scale = 0.01
        self.theta = {
            "a": np.random.randn(L) * scale,
            "b": np.random.randn(M) * scale,
            "W": np.random.randn(L, M) * scale,
        }

    def log_psi(self, sigma: np.ndarray, theta: Dict = None) -> float:
        """Compute log|Ψ_RBM(σ)| for a spin configuration.

        Args:
            sigma: Spin configuration (L,) with values in {-1, +1}
            theta: Parameters dict. If None, use self.theta

        Returns:
            log|Ψ|
        """
        if theta is None:
            theta = self.theta

        a = theta["a"]
        b = theta["b"]
        W = theta["W"]

        # First term: Σ_i a_i σ_i
        term1 = np.dot(a, sigma)

        # Second term: Σ_j log(2*cosh(b_j + Σ_i W_{ij} σ_i))
        # Numerically stable form: log(2 cosh(x)) = logaddexp(x, -x)
        # avoids overflow for large |x| that plagues log(2*cosh(x))
        h_fields = b + np.dot(W.T, sigma)  # Shape (M,)
        term2 = np.sum(np.logaddexp(h_fields, -h_fields))

        return term1 + term2

    def psi_squared(self, sigma: np.ndarray, theta: Dict = None) -> float:
        """Compute |Ψ_RBM(σ)|^2 for a spin configuration.

        Args:
            sigma: Spin configuration
            theta: Parameters dict

        Returns:
            |Ψ|^2
        """
        return np.exp(2.0 * self.log_psi(sigma, theta))

    def psi_squared_ratio(
        self, sigma_old: np.ndarray, sigma_new: np.ndarray, theta: Dict = None
    ) -> float:
        """Compute |Ψ(σ_new)|^2 / |Ψ(σ_old)|^2 for efficient MC.

        Args:
            sigma_old: Previous configuration
            sigma_new: Proposed configuration
            theta: Parameters dict

        Returns:
            Ratio of probabilities
        """
        if theta is None:
            theta = self.theta

        log_psi_new = self.log_psi(sigma_new, theta)
        log_psi_old = self.log_psi(sigma_old, theta)

        return np.exp(2.0 * (log_psi_new - log_psi_old))

    def psi_ratio(
        self, sigma_old: np.ndarray, sigma_new: np.ndarray, theta: Dict = None
    ) -> float:
        """Compute the amplitude ratio Ψ(σ_new) / Ψ(σ_old).

        Needed for off-diagonal local-energy terms. The RBM ansatz used here
        is real and positive, so the ratio is exp(logΨ_new - logΨ_old).

        Args:
            sigma_old: Previous configuration
            sigma_new: Proposed configuration
            theta: Parameters dict

        Returns:
            Amplitude ratio
        """
        if theta is None:
            theta = self.theta

        log_psi_new = self.log_psi(sigma_new, theta)
        log_psi_old = self.log_psi(sigma_old, theta)

        return np.exp(log_psi_new - log_psi_old)

    def local_magnetization(self, sigma: np.ndarray) -> float:
        """Compute local magnetization for analysis.

        Returns:
            Mean spin: <σ_i>
        """
        return np.mean(sigma)

    @staticmethod
    def compute_gradients(sigma: np.ndarray, theta: Dict) -> Dict[str, np.ndarray]:
        """Compute ∂log|Ψ|/∂θ for parameter updates.

        These are the "natural gradient" terms for RBM:
        ∂log|Ψ|/∂a_i = σ_i
        ∂log|Ψ|/∂b_j = tanh(b_j + Σ_i W_{ij} σ_i)
        ∂log|Ψ|/∂W_{ij} = σ_i * tanh(b_j + Σ_i W_{ij} σ_i)

        Args:
            sigma: Spin configuration
            theta: Parameters

        Returns:
            Dict with gradients for each parameter
        """
        a = theta["a"]
        b = theta["b"]
        W = theta["W"]

        h_fields = b + np.dot(W.T, sigma)
        tanh_h = np.tanh(h_fields)

        grads = {
            "a": sigma.copy(),
            "b": tanh_h.copy(),
            "W": np.outer(sigma, tanh_h),  # Shape (L, M)
        }

        return grads

    def update_parameters(
        self,
        grad_avg: Dict[str, np.ndarray],
        learning_rate: float,
        clip_norm: float = None,
    ) -> None:
        """Update parameters via stochastic gradient descent.

        Args:
            grad_avg: Average gradients over MC samples
            learning_rate: SGD learning rate
            clip_norm: Gradient norm clip threshold
        """
        # Clip gradients if requested
        if clip_norm is not None:
            for key in grad_avg:
                norm = np.linalg.norm(grad_avg[key])
                if norm > clip_norm:
                    grad_avg[key] = grad_avg[key] * (clip_norm / norm)

        # Update: θ → θ - η * ∂E/∂θ  (gradient descent to minimize energy).
        # grad_avg holds the (variational) energy gradient, so we subtract it.
        for key in self.theta:
            self.theta[key] -= learning_rate * grad_avg[key]

    def get_parameters_copy(self) -> Dict:
        """Return a deep copy of current parameters."""
        return {k: v.copy() for k, v in self.theta.items()}

    def set_parameters(self, theta: Dict) -> None:
        """Set parameters from dict."""
        self.theta = {k: v.copy() for k, v in theta.items()}
