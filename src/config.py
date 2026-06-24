"""Configuration and hyperparameters for the NQS-RBM TFIM project."""

# System parameters
SYSTEM_SIZES = [8, 10, 12]          # Number of spins L
TRANSVERSE_FIELDS = [0.5, 1.0, 1.5]  # Transverse field g
J = 1.0                              # Ising coupling (set to 1 as reference)

# RBM hyperparameters
HIDDEN_UNITS_LIST = [4, 8, 16, 32]   # Number of hidden units M to vary
LEARNING_RATE = 0.1                  # SGD learning rate
N_EPOCHS = 500                       # Training epochs
BATCH_SIZE = 500                     # MC sample batch size

# Variational Monte Carlo parameters
N_SAMPLES = 1000                     # Number of MC samples for energy estimation
MC_SWEEPS = 100                      # Metropolis sweeps between samples
THERMALIZATION = 50                  # Thermalization sweeps at start

# Optimization parameters
OPTIMIZER = "sgd"                    # "sgd" or "adam"
GRADIENT_CLIP = 1.0                  # Gradient clipping threshold

# Physical parameters
THERMAL_FACTOR = 2.0                 # Factor for finite-T effects in analysis

# Output and logging
VERBOSE = True
SAVE_CHECKPOINTS = True
CHECKPOINT_FREQ = 50                 # Save every N epochs
