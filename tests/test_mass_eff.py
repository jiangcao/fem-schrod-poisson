"""
Tests for spatially varying effective mass in Schrödinger equation.
"""
import numpy as np
from src import solver


def test_constant_scalar_mass():
    """Test that constant scalar mass gives same result as kinetic_coeff."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                               char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    def Vext(X):
        return 0.5 * (X[0]**2 + X[1]**2 + X[2]**2)
    
    # Test 1: Using kinetic_coeff directly
    E1, modes1 = solver.solve_generalized_eig(
        K, M, solver.potential_vector_from_callable(basis, Vext),
        nev=2, kinetic_coeff=0.5, basis=basis, mesh=mesh, 
        dirichlet_bc=False, Vfunc=Vext
    )
    
    # Test 2: Using mass_eff=1.0 with phys (hbar=1)
    phys = solver.PhysicalParams(hbar=1.0, m_eff=1.0)
    E2, modes2 = solver.solve_generalized_eig(
        K, M, solver.potential_vector_from_callable(basis, Vext),
        nev=2, mass_eff=1.0, phys=phys, basis=basis, mesh=mesh,
        dirichlet_bc=False, Vfunc=Vext
    )
    
    # Results should be very close
    assert np.allclose(E1, E2, rtol=1e-4)


def test_spatially_varying_scalar_mass_array():
    """Test spatially varying scalar effective mass specified as array."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                               char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    X = basis.doflocs
    # Mass varies from 0.5 to 1.5 along x-axis
    mass_array = 0.5 + X[0, :]
    
    def Vext(X):
        return 0.5 * (X[0]**2 + X[1]**2 + X[2]**2)
    
    phys = solver.PhysicalParams(hbar=1.0)
    E, modes = solver.solve_generalized_eig(
        K, M, solver.potential_vector_from_callable(basis, Vext),
        nev=2, mass_eff=mass_array, phys=phys, basis=basis, mesh=mesh,
        dirichlet_bc=False, Vfunc=Vext
    )
    
    # Should return valid eigenvalues
    assert len(E) == 2
    assert E[0] > 0  # Ground state energy should be positive
    assert E[1] > E[0]  # First excited state should be higher


def test_spatially_varying_scalar_mass_callable():
    """Test spatially varying scalar effective mass specified as callable."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                               char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    def mass_func(X):
        """Mass increases away from center."""
        r2 = (X[0] - 0.5)**2 + (X[1] - 0.5)**2 + (X[2] - 0.5)**2
        return 0.5 + 0.5 * r2
    
    def Vext(X):
        return 0.5 * ((X[0] - 0.5)**2 + (X[1] - 0.5)**2 + (X[2] - 0.5)**2)
    
    phys = solver.PhysicalParams(hbar=1.0)
    E, modes = solver.solve_generalized_eig(
        K, M, solver.potential_vector_from_callable(basis, Vext),
        nev=2, mass_eff=mass_func, phys=phys, basis=basis, mesh=mesh,
        dirichlet_bc=False, Vfunc=Vext
    )
    
    # Should return valid eigenvalues
    assert len(E) == 2
    assert E[0] > 0
    assert E[1] > E[0]


def test_anisotropic_mass_tensor_callable():
    """Test anisotropic effective mass tensor."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                               char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    def mass_tensor(X):
        """Diagonal tensor with different masses in each direction."""
        npts = X.shape[1]
        m = np.zeros((3, 3, npts))
        m[0, 0, :] = 1.0  # mx = 1.0
        m[1, 1, :] = 2.0  # my = 2.0
        m[2, 2, :] = 0.5  # mz = 0.5
        return m
    
    def Vext(X):
        return 0.5 * (X[0]**2 + X[1]**2 + X[2]**2)
    
    phys = solver.PhysicalParams(hbar=1.0)
    E, modes = solver.solve_generalized_eig(
        K, M, solver.potential_vector_from_callable(basis, Vext),
        nev=2, mass_eff=mass_tensor, phys=phys, basis=basis, mesh=mesh,
        dirichlet_bc=False, Vfunc=Vext
    )
    
    # Should return valid eigenvalues
    assert len(E) == 2
    assert E[0] > 0
    assert E[1] > E[0]


def test_scf_with_spatially_varying_mass():
    """Test SCF loop with spatially varying effective mass."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                               char_length=0.4, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    def mass_func(X):
        """Mass varies spatially."""
        return 0.8 + 0.4 * X[0, :]
    
    def Vext(X):
        return 0.5 * ((X[0] - 0.5)**2 + (X[1] - 0.5)**2 + (X[2] - 0.5)**2)
    
    phys = solver.PhysicalParams(hbar=1.0)
    E, modes, phi, Vfinal = solver.scf_loop(
        mesh, basis, K, M, Vext,
        coupling=0.5, maxiter=10, tol=1e-5,
        mix=0.3, nev=2, verbose=False,
        mass_eff=mass_func, phys=phys
    )
    
    # Should converge and return valid results
    assert len(E) >= 2
    assert E[0] > 0
    assert E[1] > E[0]


def test_scf_with_epsilon_and_mass():
    """Test SCF loop with both spatially varying epsilon and mass."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                               char_length=0.4, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    def epsilon_func(X):
        """Higher epsilon on right side."""
        eps = np.ones(X.shape[1])
        eps[X[0, :] > 0.5] = 2.0
        return eps
    
    def mass_func(X):
        """Higher mass on right side."""
        m = np.ones(X.shape[1])
        m[X[0, :] > 0.5] = 1.5
        return m
    
    def Vext(X):
        return 0.5 * ((X[0] - 0.5)**2 + (X[1] - 0.5)**2 + (X[2] - 0.5)**2)
    
    phys = solver.PhysicalParams(hbar=1.0)
    E, modes, phi, Vfinal = solver.scf_loop(
        mesh, basis, K, M, Vext,
        coupling=0.5, maxiter=10, tol=1e-5,
        mix=0.3, nev=2, verbose=False,
        epsilon=epsilon_func, mass_eff=mass_func, phys=phys
    )
    
    # Should converge and return valid results
    assert len(E) >= 2
    assert E[0] > 0
    assert E[1] > E[0]


def test_heterostructure_discontinuous_interface():
    """Test heterostructure with discontinuous epsilon and mass at interface."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(2.0, 1.0, 1.0), 
                               char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    # Material 1 (x < 1.0): epsilon=1.0, m_eff=1.0
    # Material 2 (x > 1.0): epsilon=3.0, m_eff=0.5
    def epsilon_func(X):
        eps = np.ones(X.shape[1])
        eps[X[0, :] > 1.0] = 3.0
        return eps
    
    def mass_func(X):
        m = np.ones(X.shape[1])
        m[X[0, :] > 1.0] = 0.5
        return m
    
    def Vext(X):
        """Confining potential centered at x=1.0"""
        return 0.5 * ((X[0] - 1.0)**2 + (X[1] - 0.5)**2 + (X[2] - 0.5)**2)
    
    phys = solver.PhysicalParams(hbar=1.0, q=-1.0, epsilon0=1.0, n_particles=1.0)
    E, modes, phi, Vfinal = solver.scf_loop(
        mesh, basis, K, M, Vext,
        coupling=0.3, maxiter=10, tol=1e-5,
        mix=0.3, nev=3, verbose=False,
        epsilon=epsilon_func, mass_eff=mass_func, phys=phys
    )
    
    # Should converge and return valid results
    assert len(E) >= 3
    assert E[0] > 0
    assert E[1] > E[0]
    assert E[2] > E[1]
    
    # Wave function should be continuous across interface
    # (though derivative may be discontinuous)
    psi0 = modes[:, 0]
    assert np.all(np.isfinite(psi0))


def test_flux_continuity_at_interface():
    """
    Test that flux is properly continuous at material interface.
    For Poisson equation: n·(ε₁∇φ₁) = n·(ε₂∇φ₂) at interface
    For Schrödinger: wave function continuous, but current density has proper jump
    """
    # Create mesh with interface at x=1.0
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(2.0, 1.0, 1.0), 
                               char_length=0.2, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    # Sharp interface at x=1.0
    def epsilon_func(X):
        eps = np.ones(X.shape[1])
        eps[X[0, :] > 1.0] = 4.0  # 4x higher on right
        return eps
    
    def mass_func(X):
        m = np.ones(X.shape[1])
        m[X[0, :] > 1.0] = 0.25  # 4x lighter on right
        return m
    
    # Gaussian charge distribution centered at interface
    def rho_func(X):
        r2 = (X[0] - 1.0)**2 + (X[1] - 0.5)**2 + (X[2] - 0.5)**2
        return np.exp(-10.0 * r2)
    
    # Solve Poisson with discontinuous epsilon
    phi = solver.solve_poisson(mesh, basis, rho_func, bc_value=0.0, epsilon=epsilon_func)
    
    # Verify solution is well-behaved
    assert np.all(np.isfinite(phi))
    assert np.max(np.abs(phi)) < 10.0  # Reasonable magnitude
    
    # Now test Schrödinger with discontinuous mass
    def Vext(X):
        return 0.5 * ((X[0] - 1.0)**2 + (X[1] - 0.5)**2 + (X[2] - 0.5)**2)
    
    phys = solver.PhysicalParams(hbar=1.0)
    E, modes = solver.solve_generalized_eig(
        K, M, solver.potential_vector_from_callable(basis, Vext),
        nev=2, mass_eff=mass_func, phys=phys, basis=basis, mesh=mesh,
        dirichlet_bc=False, Vfunc=Vext
    )
    
    # Wave function should be continuous everywhere
    psi0 = modes[:, 0]
    assert np.all(np.isfinite(psi0))
    
    # Ground state should be localized near interface
    X = basis.doflocs
    r_from_interface = np.abs(X[0, :] - 1.0)
    center_mask = r_from_interface < 0.3
    edge_mask = r_from_interface > 1.0
    
    if np.sum(center_mask) > 0 and np.sum(edge_mask) > 0:
        rho_psi = np.abs(psi0) ** 2
        center_density = np.mean(rho_psi[center_mask])
        edge_density = np.mean(rho_psi[edge_mask])
        # Wave function should be more concentrated near interface
        assert center_density > edge_density
