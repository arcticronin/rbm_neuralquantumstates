# Quick Start Guide

## Installation

```bash
cd /path/to/tfim
pip install -r requirements.txt
```

## Running the Experiment

### Option 1: Full Experiment (All Configurations)

```bash
python main.py
```

This runs the complete pipeline:
1. Exact diagonalization for all $(L, g)$ pairs
2. VMC with RBM for varying hidden unit counts
3. Analysis and results saved to `results/`

**Estimated time:** 10-30 minutes (depending on system)

### Option 2: Interactive Jupyter Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

Run cells sequentially to:
- Visualize the TFIM phase diagram
- Train a single RBM NQS with plots
- Compute order parameters
- Analyze convergence behavior

**Best for:** understanding the physics and debugging

### Option 3: Custom Run

Create a Python script:

```python
import sys
sys.path.insert(0, 'src')

from tfim_exact import TFIMExact
from vmc import VMCSolver

# Exact benchmark
tfim = TFIMExact(L=8, J=1.0, g=1.0)
E0 = tfim.groundstate_energy()
print(f"Exact E0: {E0}")

# VMC optimization
vmc = VMCSolver(L=8, M=8, J=1.0, g=1.0)
history = vmc.train(n_steps=50, n_samples=500, learning_rate=0.1)
print(f"Final energy: {history[-1]['energy']}")
```

---

## Output Files

Results are saved to `results/`:

| File | Description |
|------|-------------|
| `exact_benchmark.json` | Exact ground state energies |
| `vmc_results.json` | VMC results for each (L, g, M) triplet |
| `summary.json` | Summary statistics |
| `exact_phase_diagram.png` | Phase diagram plot |
| `vmc_demo_convergence.png` | Training convergence curves |

---

## Key Metrics to Report

From `vmc_results.json`, each entry contains:

```json
{
  "L": 8,
  "g": 1.0,
  "M": 8,
  "E0_exact": -5.123456,
  "E_nqs": -5.087654,
  "E_error": 0.035802,
  "order_params": {
    "mean_abs_W": 0.156,
    "sparsity": 0.234,
    "max_W": 0.893
  }
}
```

**For your exam:**

1. **Energy error table:** vs. $M$ for each $g$
2. **Order parameters plot:** weight statistics vs. transverse field
3. **Convergence analysis:** training curves showing $E(\theta)$ vs. step
4. **Physical interpretation:** connect trends to RBM phases

---

## Customization

Edit `src/config.py` to modify:

```python
SYSTEM_SIZES = [8, 10, 12]          # Lattice sizes
TRANSVERSE_FIELDS = [0.5, 1.0, 1.5] # Magnetic field values
HIDDEN_UNIT_RATIOS = [0.5, 1.0, 2.0] # M/L ratios
N_SWEEPS = 100                       # Optimization steps
LEARNING_RATE = 0.1                  # SGD learning rate
N_SAMPLES = 500                      # MC samples per step
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Slow convergence** | Increase learning rate (0.05 → 0.2) or add more MC samples |
| **NaN in gradients** | Reduce learning rate or check for numerical overflow |
| **Memory issues** | Reduce `N_SAMPLES` or use smaller `SYSTEM_SIZES` |
| **Poor accuracy** | Increase `N_SWEEPS` or add more hidden units |

---

##References for Your Exam

**Core papers to cite:**

1. Carleo & Troyer (2017) – original NQS paper
2. Carleo (2017) – lecture notes with VMC derivations
3. Decelle et al. (2021) – RBM statistical mechanics
4. Chinese Physics B 30, 040202 (2021) – mean-field RBM theory

**Links in this project:**

- [PHYSICS.md](PHYSICS.md) – detailed derivations
- [README.md](README.md) – comprehensive overview
- Notebook: [analysis.ipynb](notebooks/analysis.ipynb) – interactive walkthrough

---

**Get started:**
```bash
python main.py          # Full experiment
# or
jupyter notebook notebooks/analysis.ipynb  # Interactive mode
```

Questions? See PHYSICS.md for theoretical background or README.md for architecture details.
