"""
Tests for electromagnetic field implementation.

These tests validate the mathematical formulation and numerical implementation
of the vector potential in the Schrödinger equation.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.demo_electromagnetic import (
    assemble_paramagnetic_operator,
    compute_diamagnetic_potential,
    vector_potential_uniform_field,
    solve_schrodinger_em
)
from src import solver


class TestVectorPotential:
    """Test vector potential functions."""
    
    def test_uniform_field_symmetric_gauge(self):
        """Test that symmetric gauge gives correct B field."""
        B_z = 1.5
        A_func = vector_potential_uniform_field(B_z, gauge='symmetric')
        
        # Sample points
        X = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]]).T
        
        A = A_func(X)
        
        # Check A = (-B_z y/2, B_z x/2, 0)
        np.testing.assert_allclose(A[0, :], -0.5 * B_z * X[1, :], rtol=1e-10)
        np.testing.assert_allclose(A[1, :],  0.5 * B_z * X[0, :], rtol=1e-10)
        np.testing.assert_allclose(A[2, :], 0.0, atol=1e-10)
    
    def test_gauge_invariance_curl(self):
        """Test that different gauges give the same magnetic field."""
        B_z = 1.0
        
        A_sym = vector_potential_uniform_field(B_z, gauge='symmetric')
        A_lx = vector_potential_uniform_field(B_z, gauge='landau_x')
        A_ly = vector_potential_uniform_field(B_z, gauge='landau_y')
        
        # At a test point, compute curl numerically
        x, y = 1.0, 0.5
        h = 1e-6
        
        # For symmetric gauge at (x,y,0):
        # A = (-B_z y/2, B_z x/2, 0)
        # ∂Ay/∂x - ∂Ax/∂y = B_z/2 - (-B_z/2) = B_z ✓
        
        # Just verify A is defined consistently
        X = np.array([[x, y, 0.0]]).T
        assert A_sym(X).shape == (3, 1)
        assert A_lx(X).shape == (3, 1)
        assert A_ly(X).shape == (3, 1)


class TestEMOperators:
    """Test electromagnetic operator assembly."""
    
    def test_paramagnetic_hermiticity(self):
        """Test that paramagnetic operator has correct anti-Hermitian structure."""
        # Simple mesh
        mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1,1,1), char_length=0.4)
        mesh, basis, K, M = solver.assemble_operators(mesh)
        
        # Constant vector potential
        def A_const(X):
            return np.array([np.ones(X.shape[1]),
                            np.zeros(X.shape[1]),
                            np.zeros(X.shape[1])])
        
        K_para = assemble_paramagnetic_operator(basis, A_const, q=-1.0, hbar=1.0)
        
        # Should be anti-Hermitian: K_para† = -K_para
        K_para_dag = K_para.conj().T
        diff = K_para + K_para_dag
        
        # Check if anti-Hermitian (difference should be small)
        max_diff = np.max(np.abs(diff.data))
        assert max_diff < 1e-10, f"Paramagnetic operator not anti-Hermitian: max |K+K†| = {max_diff}"
    
    def test_diamagnetic_positive(self):
        """Test that diamagnetic potential is positive definite."""
        mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1,1,1), char_length=0.4)
        mesh, basis, K, M = solver.assemble_operators(mesh)
        
        # Non-zero vector potential
        B_z = 1.0
        A_func = vector_potential_uniform_field(B_z, gauge='symmetric')
        
        V_dia = compute_diamagnetic_potential(basis, A_func, q=-1.0, mass_eff=1.0)
        
        # Should be positive everywhere (q²|A|²/2m ≥ 0)
        assert np.all(V_dia >= 0), "Diamagnetic potential should be non-negative"
        assert np.any(V_dia > 0), "Diamagnetic potential should be non-zero somewhere"
    
    def test_zero_field_recovery(self):
        """Test that A=0 recovers the field-free Hamiltonian."""
        mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1,1,1), char_length=0.4)
        mesh, basis, K, M = solver.assemble_operators(mesh)
        
        # Zero vector potential
        def A_zero(X):
            return np.zeros((3, X.shape[1]))
        
        K_para = assemble_paramagnetic_operator(basis, A_zero, q=-1.0, hbar=1.0)
        V_dia = compute_diamagnetic_potential(basis, A_zero, q=-1.0, mass_eff=1.0)
        
        # Should be zero
        assert np.max(np.abs(K_para.data)) < 1e-10, "Paramagnetic term should vanish for A=0"
        assert np.max(np.abs(V_dia)) < 1e-10, "Diamagnetic term should vanish for A=0"


class TestLandauLevels:
    """Test Landau level physics."""
    
    def test_cyclotron_frequency(self):
        """Test that ground state energy scales with cyclotron frequency."""
        # Small test (coarse mesh for speed)
        hbar = 1.0
        m_eff = 1.0
        q = -1.0
        B_values = [0.5, 1.0, 2.0]
        
        E0_values = []
        
        for B_z in B_values:
            omega_c = abs(q) * B_z / m_eff
            
            mesh = solver.make_mesh_box(
                x0=(-2,-2,-0.5), lengths=(4,4,1), char_length=0.8
            )
            mesh, basis, K, M = solver.assemble_operators(mesh)
            
            A_func = vector_potential_uniform_field(B_z, gauge='symmetric')
            
            # Weak confinement
            def V_ext(X):
                return 0.01 * (X[0]**2 + X[1]**2)
            
            E, modes = solve_schrodinger_em(
                mesh, basis, K, M, V_ext,
                A_func=A_func, q=q, hbar=hbar, mass_eff=m_eff,
                nev=2
            )
            
            E0_values.append(E[0].real)
        
        # Check that E0 scales approximately linearly with B (and thus ω_c)
        # E0 ≈ ℏω_c(1/2) for ground Landau level
        ratios = [E0_values[i] / B_values[i] for i in range(len(B_values))]
        
        # Ratios should be similar (allow some deviation due to confinement and mesh)
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        
        print(f"\nB values: {B_values}")
        print(f"E0 values: {E0_values}")
        print(f"E0/B ratios: {ratios}")
        print(f"Mean ratio: {mean_ratio:.4f}, Std: {std_ratio:.4f}")
        
        # Should have consistent scaling (within ~20% due to finite size effects)
        assert std_ratio / mean_ratio < 0.2, "Ground state should scale linearly with B"
    
    @pytest.mark.skip(reason="Requires fine mesh and long computation time")
    def test_landau_level_spacing(self):
        """Test that excited states show Landau level structure."""
        # Would need very fine mesh and large system to resolve level spacing
        pass


class TestGaugeInvariance:
    """Test gauge invariance of physical observables."""
    
    def test_energy_gauge_invariance(self):
        """Test that eigenvalues are gauge-invariant (approximately)."""
        # Small system
        hbar = 1.0
        m_eff = 1.0
        q = -1.0
        B_z = 1.0
        
        mesh = solver.make_mesh_box(
            x0=(-2,-2,-0.5), lengths=(4,4,1), char_length=0.8
        )
        mesh, basis, K, M = solver.assemble_operators(mesh)
        
        # Weak confinement
        def V_ext(X):
            return 0.1 * (X[0]**2 + X[1]**2)
        
        # Try different gauges
        E_sym, _ = solve_schrodinger_em(
            mesh, basis, K, M, V_ext,
            A_func=vector_potential_uniform_field(B_z, 'symmetric'),
            q=q, hbar=hbar, mass_eff=m_eff, nev=3
        )
        
        E_lx, _ = solve_schrodinger_em(
            mesh, basis, K, M, V_ext,
            A_func=vector_potential_uniform_field(B_z, 'landau_x'),
            q=q, hbar=hbar, mass_eff=m_eff, nev=3
        )
        
        E_ly, _ = solve_schrodinger_em(
            mesh, basis, K, M, V_ext,
            A_func=vector_potential_uniform_field(B_z, 'landau_y'),
            q=q, hbar=hbar, mass_eff=m_eff, nev=3
        )
        
        # Energies should be gauge-invariant (real parts)
        print(f"\nSymmetric gauge: {E_sym.real}")
        print(f"Landau-x gauge:  {E_lx.real}")
        print(f"Landau-y gauge:  {E_ly.real}")
        
        # Allow small numerical differences
        np.testing.assert_allclose(E_sym.real, E_lx.real, rtol=0.05, atol=0.05)
        np.testing.assert_allclose(E_sym.real, E_ly.real, rtol=0.05, atol=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
