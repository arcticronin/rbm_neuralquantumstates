"""Utility functions for the NQS-RBM TFIM project."""

import numpy as np
from typing import Tuple, List
import json
from pathlib import Path


def generate_all_configs(L: int) -> np.ndarray:
    """Generate all 2^L spin configurations for size L.
    
    Args:
        L: Number of spins
        
    Returns:
        Array of shape (2^L, L) with configurations in {-1, +1}
    """
    n_configs = 2 ** L
    configs = np.zeros((n_configs, L), dtype=np.int8)
    
    for i in range(n_configs):
        # Binary to spin conversion: 0 -> +1, 1 -> -1
        binary = format(i, f'0{L}b')
        configs[i] = np.array([1 if b == '0' else -1 for b in binary])
    
    return configs


def log_psi_exact_to_vector(log_psi_func, configs: np.ndarray, theta: dict) -> np.ndarray:
    """Evaluate log psi on all configurations.
    
    Args:
        log_psi_func: Function that computes log|Psi| for single config
        configs: Array of shape (n_configs, L)
        theta: RBM parameters dict
        
    Returns:
        Log amplitudes for all configs
    """
    return np.array([log_psi_func(sigma, theta) for sigma in configs])


def periodic_boundary(i: int, L: int) -> int:
    """Apply periodic boundary conditions."""
    return i % L


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: dict, filepath: str) -> None:
    """Save dict to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_results(results: dict, name: str, basedir: str = "results") -> str:
    """Save results to timestamped file."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{basedir}/{name}_{timestamp}.json"
    save_json(results, filename)
    return filename


def order_parameter_weights(theta: dict) -> dict:
    """Compute order parameters related to RBM weights.
    
    Args:
        theta: Parameter dict with 'W' key
        
    Returns:
        Dict with statistics on weight matrix
    """
    W = theta.get('W', np.array([]))
    if W.size == 0:
        return {}
    
    return {
        'mean_abs_W': float(np.mean(np.abs(W))),
        'std_W': float(np.std(W)),
        'sparsity': float(np.sum(np.abs(W) < 0.01) / W.size),
        'max_W': float(np.max(np.abs(W))),
    }


def print_header(msg: str) -> None:
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {msg}")
    print("="*60)
