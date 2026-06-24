"""Configuration and hyperparameters for the NQS-RBM TFIM project."""

# System parameters
SYSTEM_SIZES = [8, 10]               # Number of spins L
TRANSVERSE_FIELDS = [0.5, 1.0, 1.5]  # Transverse field g
J = 1.0                              # Ising coupling (set to 1 as reference)

# RBM architecture: number of hidden units M = round(m_ratio * L)
HIDDEN_UNIT_RATIOS = [1.0, 2.0]      # M/L ratios to test

# Variational Monte Carlo / optimization parameters
LEARNING_RATE = 0.05                 # SGD learning rate
N_SWEEPS = 150                       # Number of optimization steps
N_SAMPLES = 500                      # MC samples per optimization step
GRADIENT_CLIP = 1.0                  # Gradient norm clip threshold

# Output and logging
RESULTS_DIR = "results"
VERBOSE = True
