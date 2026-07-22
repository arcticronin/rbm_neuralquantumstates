# Plot Placement Guide for project.typst (N=12 Results)

## Summary of Updates

All numerical values in project.typst have been updated from N=8 to N=12 results. Below is a slide-by-slide guide for which plots to include.

---

## Main Presentation Slides

### **Slide 3 — Exact Diagonalization**

- **Plot:** `results/paper_energy_convergence.png` (right panel)
- **Purpose:** Show exact energies as reference lines
- **What to show:** The 5 horizontal dashed lines representing exact ground state energies for h=0.0, 0.6, 1.0, 1.4, 2.0
- **Updated table:** ✅ Now shows N=12 exact energies

### **Slide 11 — SGD vs SR**

- **Plot:** From `analysis.ipynb` results (if available)
- **Purpose:** Show energy convergence comparison between plain gradient descent and SR
- **Fallback:** If analysis.ipynb plot not available, describe results verbally (SGD fails at criticality, SR succeeds)
- **Table:** N=8 results retained (this is a specific experiment from analysis.ipynb)

### **Slide 12 — QFM Eigenspectrum**

- **Plot:** `results/paper_qfm_converged.png` ⭐ **CRITICAL PLOT**
- **Purpose:** Show the three phase signatures side-by-side
- **What to show:** Log-log plot of converged QFM eigenvalues for all 5 h values
- **Key features:**
  - h=0.6: 71/480 non-zero (rank-deficient)
  - h=1.0: 480/480 non-zero (dense exponential)
  - h=2.0: kink at index 78 (symmetry boundary)
- **Updated table:** ✅ Now shows N=12 QFM counts (71/480, 480/480, etc.)

### **Slide 13 — Universal Initial Dynamics**

- **Plot:** `results/paper_qfm_evolution.png` ⭐ **CRITICAL PLOT**
- **Purpose:** Show QFM spectrum evolution from universal initial shape to phase-specific final shape
- **What to show:** Multi-panel plot showing spectra at epochs {0, 10, 25, 50, 100, 200} for all 5 h values
- **Key observation:** All 5 phases start identical (universal), then diverge as training progresses
- **Updated text:** ✅ Now references N=12 (480 parameters)

### **Slide 14 — Eigenvector Entanglement**

- **Plot:** `results/paper_entanglement.png`
- **Purpose:** Show entanglement entropy $S_E$ vs eigenvalue index
- **What to show:** 5 curves (one per h value) showing large eigenvalues → low $S_E$, small eigenvalues → high $S_E$
- **Key message:** SR decomposes learning into fast (low-entanglement) and slow (high-entanglement) directions

### **Slide 15 — h=0 Frozen Chain Problem**

- **Plot:** `results/paper_pt_fix_h0.png` (if this exists as a separate plot)
- **Purpose:** Show PT successfully breaks the frozen chain
- **Alternative:** Use left panel of `results/paper_energy_convergence.png` highlighting h=0.0 curve
- **What to show:**
  - Energy convergence to exact
  - QFM spectrum showing 10/480 non-zero with PT (vs 0/480 without)
  - (Optional) Acceptance rate dropping to 0.000
- **Updated table:** ✅ Now shows N=12 PT results (10/480 non-zero)

### **Slide 16 — Three Phases Side by Side**

- **Plot:** `results/paper_qfm_converged.png` (same as Slide 12)
- **Purpose:** Reinforce the three signature patterns with visual ASCII diagram
- **Updated diagram:** ✅ Now shows kink at index 78 (was 36)

### **Slide 20 — Experiment Setup**

- **Plot:** None needed (hyperparameters table only)
- **Updated table:** ✅ Now shows N=12, M=36, 480 parameters, α=3, 1000 epochs

### **Slide 21 — Energy Convergence Results**

- **Plot:** `results/paper_energy_convergence.png` (both panels) ⭐ **CRITICAL PLOT**
- **Purpose:** Show VMC training converges to exact energies
- **Left panel:** Energy vs epoch for all 5 h values, with exact energies as dashed lines
- **Right panel:** |ΔE| vs epoch on log scale showing convergence quality
- **Updated table:** ✅ Now shows N=12 final energies (all |ΔE| < 0.0012)

### **Slide 23 — Summary of Key Results**

- **Plot:** Optional summary figure combining key panels
- **Updated tables:** ✅ Both tables updated with N=12 results
- **New observation added:** Stronger phase signatures with N=12 (rank-deficiency 14.8% vs 25%, criticality 100% vs 82%)

---

## Appendices

### **Appendix C — Scaling to N=28**

- **Plot:** None needed (comparison table only)
- **Updated table:** ✅ Now includes N=12 column between N=8 and N=28

---

## Key Numerical Changes Summary

| Quantity                | Old (N=8)     | New (N=12)     |
| ----------------------- | ------------- | -------------- |
| **System Parameters**   |
| Hidden units M (α=3)    | 24            | 36             |
| Total parameters        | 224           | 480            |
| Symmetry kink index     | 36            | 78             |
| **Exact Energies**      |
| E₀(h=0.0)               | −8.0000       | −12.0000       |
| E₀(h=0.6)               | −8.7408       | −13.1072       |
| E₀(h=1.0)               | −10.2517      | −15.3226       |
| E₀(h=1.4)               | −12.6962      | −19.0253       |
| E₀(h=2.0)               | −17.0182      | −25.5251       |
| **QFM Non-Zero Counts** |
| h=0.0 (Metropolis)      | 0/224         | 0/480          |
| h=0.0 (PT)              | 6/224         | 10/480         |
| h=0.6 (FM)              | 56/224 (25%)  | 71/480 (14.8%) |
| h=1.0 (critical)        | 183/224 (82%) | 480/480 (100%) |
| h=1.4 (PM)              | 217/224       | 480/480        |
| h=2.0 (PM)              | 224/224       | 480/480        |
| **Energy Errors (ΔE)**  |
| h=0.0                   | 0.0000        | 0.0000         |
| h=0.6                   | +0.0016       | +0.0011        |
| h=1.0                   | +0.0013       | +0.0010        |
| h=1.4                   | −0.0007       | −0.0003        |
| h=2.0                   | +0.0007       | +0.0002        |

---

## Plot Files Generated by paper_reproduction.ipynb

The notebook should have generated the following plots in `results/`:

1. ✅ `paper_energy_convergence.png` — Energy vs epoch (left) + |ΔE| vs epoch (right)
2. ✅ `paper_qfm_evolution.png` — QFM spectra evolution at 6 epochs × 5 phases
3. ✅ `paper_qfm_converged.png` — Final converged QFM spectra for all 5 phases
4. ✅ `paper_entanglement.png` — Eigenvector entanglement $S_E$ vs eigenvalue index
5. ❓ `paper_pt_fix_h0.png` — (Optional) Parallel tempering diagnostics for h=0

**Action needed:** Verify all 4-5 plots exist in `results/` folder. If any are missing, they can be regenerated by running the corresponding cells in paper_reproduction.ipynb.

---

## Recommended Slide Order for Plot Insertion

When adding figures to the typst file, insert in this order:

1. **Slide 3:** Add reference to energy convergence plot
2. **Slide 12:** ⭐ Add QFM converged spectrum plot (most important result)
3. **Slide 13:** ⭐ Add QFM evolution plot (shows learning dynamics)
4. **Slide 14:** Add entanglement plot
5. **Slide 15:** Add PT fix plot (if available) or reference energy plot
6. **Slide 21:** ⭐ Add energy convergence plot (both panels)

The three starred plots (Slides 12, 13, 21) are the **core visual evidence** for the paper reproduction.

---

## Notes on Scaling Behavior (N=8 → N=12)

**Improvements:**

- Energy accuracy improved: max |ΔE| from 0.0016 → 0.0011
- Critical phase now **fully populated** (100% vs 82%)
- Parallel tempering more effective: 10/480 vs 6/224 non-zero

**Expected behavior:**

- Ferromagnetic phase more rank-deficient: 25% → 14.8%
- Larger kink index: 36 → 78
- More training epochs needed: 300 → 1000 (for stability)

**Physical interpretation:**
Larger systems show **stronger signatures of phase structure** in the QFM spectrum, validating the paper's core thesis that QFM geometry encodes physics.
