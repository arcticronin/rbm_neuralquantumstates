"""Main experiment script: NQS-RBM approximation of TFIM ground state."""

import numpy as np
import sys
import json
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tfim_exact import TFIMExact
from rbm_nqs import RBMNQS
from vmc import VMCSolver
from utils import save_json, print_header, order_parameter_weights
import config


def run_exact_benchmark(sizes: List[int], fields: List[float]) -> Dict:
    """Compute exact ground state energies for benchmark."""
    print_header("EXACT DIAGONALIZATION BENCHMARK")
    
    exact_results = {}
    
    for L in sizes:
        for g in fields:
            tfim = TFIMExact(L, J=config.J, g=g)
            E0 = tfim.groundstate_energy()
            e_per_spin = E0 / L
            
            exact_results[(L, g)] = {
                'L': L,
                'g': g,
                'E0': float(E0),
                'E0_per_spin': float(e_per_spin),
            }
            
            print(f"L={L:2d}, g={g:.2f}: E0 = {E0:8.4f}, e/L = {e_per_spin:8.4f}")
    
    return exact_results


def run_vmc_study(sizes: List[int], fields: List[float],
                 m_ratios: List[float]) -> Dict:
    """Run VMC with various RBM sizes and system parameters."""
    print_header("VARIATIONAL MONTE CARLO WITH RBM")
    
    vmc_results = {}
    
    for L in sizes:
        # Load exact benchmark for this size
        exact_tfim = TFIMExact(L, J=config.J, g=config.TRANSVERSE_FIELDS[0])
        
        for g in fields:
            # Re-compute exact for this g
            exact_tfim_g = TFIMExact(L, J=config.J, g=g)
            E0_exact = exact_tfim_g.groundstate_energy()
            
            for m_ratio in m_ratios:
                M = max(1, int(L * m_ratio))
                
                print(f"\nL={L}, g={g:.2f}, M={M} (ratio={m_ratio}):")
                
                # Initialize VMC solver
                vmc = VMCSolver(L, M, J=config.J, g=g, seed=42)
                
                # Initial energy estimate
                E_init, _ = vmc.estimate_energy(n_samples=500)
                print(f"  Initial energy: {E_init:.4f}")
                
                # Train
                history = vmc.train(
                    n_steps=config.N_SWEEPS,
                    n_samples=config.N_SAMPLES,
                    learning_rate=config.LEARNING_RATE,
                    verbose=False
                )
                
                # Final energy
                E_final, err_final = vmc.estimate_energy(n_samples=1000)
                E_error = E_final - E0_exact
                
                # Order parameters
                order_params = order_parameter_weights(vmc.nqs.theta)
                
                result = {
                    'L': L,
                    'g': float(g),
                    'M': M,
                    'm_ratio': m_ratio,
                    'E0_exact': float(E0_exact),
                    'E_nqs': float(E_final),
                    'E_error': float(E_error),
                    'E_init': float(E_init),
                    'error_err': float(err_final),
                    'history': [(h['energy'], h['accept_rate']) for h in history],
                    'order_params': order_params,
                }
                
                vmc_results[(L, g, M)] = result
                
                print(f"  Final energy:   {E_final:.4f} ± {err_final:.4f}")
                print(f"  Error vs exact: ΔE = {E_error:.4f}")
                print(f"  Order params: {order_params}")
    
    return vmc_results


def analyze_results(exact_results: Dict, vmc_results: Dict) -> Dict:
    """Analyze trends in results."""
    print_header("ANALYSIS AND INTERPRETATION")
    
    analysis = {}
    
    # Group by system size
    for L in config.SYSTEM_SIZES:
        L_data = {k: v for k, v in vmc_results.items() if k[0] == L}
        
        if L_data:
            print(f"\n=== System Size L={L} ===")
            
            # Analyze hidden unit scaling
            for g in config.TRANSVERSE_FIELDS:
                g_data = [(v['M'], v['E_error']) for k, v in L_data.items() if abs(k[1] - g) < 1e-6]
                g_data.sort()
                
                if g_data:
                    print(f"\nField g={g:.2f}:")
                    print("  M (hidden units) vs Energy Error:")
                    for M, E_err in g_data:
                        print(f"    M={M:2d}: ΔE = {E_err:8.5f}")
                    
                    # Trend across hidden-unit count
                    print(f"  Trend: {'improving' if g_data[-1][1] < g_data[0][1] else 'degrading'} "
                          f"with more hidden units")
    
    analysis['summary'] = "See console output for details"
    return analysis


def save_all_results(exact_results: Dict, vmc_results: Dict, analysis: Dict) -> None:
    """Save results to JSON files."""
    print_header("SAVING RESULTS")
    
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save exact benchmark
    exact_file = results_dir / "exact_benchmark.json"
    exact_dict = {str(k): v for k, v in exact_results.items()}
    save_json(exact_dict, str(exact_file))
    print(f"Saved exact results to {exact_file}")
    
    # Save VMC results
    vmc_file = results_dir / "vmc_results.json"
    vmc_dict = {str(k): v for k, v in vmc_results.items()}
    save_json(vmc_dict, str(vmc_file))
    print(f"Saved VMC results to {vmc_file}")
    
    # Summary statistics
    summary = {
        'n_exact_configs': len(exact_results),
        'n_vmc_configs': len(vmc_results),
        'best_error': min(v['E_error'] for v in vmc_results.values()),
        'worst_error': max(v['E_error'] for v in vmc_results.values()),
    }
    
    summary_file = results_dir / "summary.json"
    save_json(summary, str(summary_file))
    print(f"Saved summary to {summary_file}")
    
    print("\nAll results saved to results/ directory")


def main():
    """Main experiment driver."""
    print("\n" + "="*60)
    print("  RBM-NQS Approximation of TFIM Ground State")
    print("  Exam Project: Statistical Mechanics & Neural QS")
    print("="*60 + "\n")
    
    print("Configuration:")
    print(f"  System sizes: {config.SYSTEM_SIZES}")
    print(f"  Transverse fields: {config.TRANSVERSE_FIELDS}")
    print(f"  Hidden unit ratios: {config.HIDDEN_UNIT_RATIOS}")
    print(f"  VMC samples per step: {config.N_SAMPLES}")
    print(f"  Optimization steps: {config.N_SWEEPS}")
    
    # Run exact diagonalization
    exact_results = run_exact_benchmark(config.SYSTEM_SIZES, config.TRANSVERSE_FIELDS)
    
    # Run VMC with RBM
    vmc_results = run_vmc_study(config.SYSTEM_SIZES,
                                config.TRANSVERSE_FIELDS,
                                config.HIDDEN_UNIT_RATIOS)
    
    # Analyze
    analysis = analyze_results(exact_results, vmc_results)
    
    # Save
    save_all_results(exact_results, vmc_results, analysis)
    
    print_header("EXPERIMENT COMPLETE")


if __name__ == '__main__':
    main()
