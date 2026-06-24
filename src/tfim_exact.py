"""Exact diagonalization for 1D transverse-field Ising model (TFIM)."""

import numpy as np
from typing import Tuple


class TFIMExact:
    r"""Exact diagonalization for 1D TFIM Hamiltonian.
    
    H = -J \sum_i \sigma^z_i \sigma^z_{i+1} - g \sum_i \sigma^x_i
    
    With periodic boundary conditions.
    """
    
    def __init__(self, L: int, J: float = 1.0, g: float = 1.0):
        """Initialize TFIM system.
        
        Args:
            L: Number of spins
            J: Ising coupling
            g: Transverse field strength
        """
        self.L = L
        self.J = J
        self.g = g
        self._build_hamiltonian()
    
    def _pauli_z(self) -> np.ndarray:
        """Pauli Z matrix."""
        return np.array([[1, 0], [0, -1]], dtype=np.float64)
    
    def _pauli_x(self) -> np.ndarray:
        """Pauli X matrix."""
        return np.array([[0, 1], [1, 0]], dtype=np.float64)
    
    def _pauli_i(self) -> np.ndarray:
        """Identity matrix."""
        return np.eye(2, dtype=np.float64)
    
    def _build_hamiltonian(self) -> None:
        """Build the full Hamiltonian matrix using Kronecker products."""
        # This can be slow for large L (hilbert space ~ 2^L)
        # For L > 15, consider using sparse matrix tricks or DMRG
        
        Ham = None
        
        # ZZ coupling terms: -J \sum_i \sigma^z_i \sigma^z_{i+1}
        for i in range(self.L):
            ii_next = (i + 1) % self.L
            
            # Build operator for Z_i Z_{i+1}
            ops = []
            for j in range(self.L):
                if j == i:
                    ops.append(self._pauli_z())
                elif j == ii_next:
                    ops.append(self._pauli_z())
                else:
                    ops.append(self._pauli_i())
            
            # Kronecker product
            term = ops[0]
            for op in ops[1:]:
                term = np.kron(term, op)
            
            if Ham is None:
                Ham = -self.J * term
            else:
                Ham = Ham - self.J * term
        
        # X field terms: -g \sum_i \sigma^x_i
        for i in range(self.L):
            ops = []
            for j in range(self.L):
                if j == i:
                    ops.append(self._pauli_x())
                else:
                    ops.append(self._pauli_i())
            
            term = ops[0]
            for op in ops[1:]:
                term = np.kron(term, op)
            
            Ham = Ham - self.g * term
        
        self.H = Ham
    
    def groundstate_energy(self) -> float:
        """Compute ground state energy via exact diagonalization.
        
        Returns:
            Ground state energy E_0
        """
        # Using eigsh for sparse matrices would be more efficient
        # For small L (up to ~15), dense diagonalization is okay
        eigenvalues = np.linalg.eigvalsh(self.H)
        return float(eigenvalues[0])
    
    def groundstate_vector(self) -> np.ndarray:
        """Compute ground state wavefunction.
        
        Returns:
            Ground state vector of length 2^L
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.H)
        return eigenvectors[:, 0]
    
    def energy_per_spin(self) -> float:
        """Ground state energy per spin."""
        return self.groundstate_energy() / self.L
    
    @staticmethod
    def benchmark(sizes: list, fields: list, J: float = 1.0) -> dict:
        """Benchmark exact diagonalization for various system sizes and fields.
        
        Args:
            sizes: List of system sizes L to test
            fields: List of transverse fields g to test
            J: Ising coupling
            
        Returns:
            Dictionary of results indexed by (L, g)
        """
        results = {}
        
        for L in sizes:
            for g in fields:
                tfim = TFIMExact(L, J=J, g=g)
                E0 = tfim.groundstate_energy()
                e_per_spin = tfim.energy_per_spin()
                
                results[(L, g)] = {
                    'L': L,
                    'g': g,
                    'J': J,
                    'E0': float(E0),
                    'energy_per_spin': float(e_per_spin),
                }
                
                print(f"L={L}, g={g:.2f}: E0 = {E0:.6f}, e/spin = {e_per_spin:.6f}")
        
        return results
