# Tests for Poisson solver with spatially varying epsilon
import numpy as np
from src import solver

# Manufactured solution helpers
def phi_exact_from_X(X):
    # X has shape (3, N)
    return np.sin(np.pi * X[0, :]) * np.sin(np.pi * X[1, :]) * np.sin(np.pi * X[2, :])

def rhs_for_phi_exact(phi_exact):
    # For phi = sin(pi x) sin(pi y) sin(pi z), Laplacian = -3*pi^2 * phi
    # PDE is -div(eps grad phi) = rho. For eps=1 => rho = 3*pi^2 * phi_exact
    return 3.0 * (np.pi ** 2) * phi_exact

def l2_error(phi_num, phi_exact, M):
    # M is the mass matrix assembled by solver. Compute sqrt((e)^T M (e))
    e = phi_num - phi_exact
    try:
        Me = M.dot(e)
    except Exception:
        Me = M @ e
    return np.sqrt(np.abs(e @ Me))

def test_poisson_default_epsilon():
    """Test that default epsilon=None works like the original implementation."""
    mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), char_length=0.4, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)

    # Create a simple source term
    rho = np.ones(basis.N)

    # Solve with default (epsilon=None)
    phi_default = solver.solve_poisson(mesh, basis, rho, bc_value=0.0)

    # Solve with explicit epsilon=1.0
    phi_explicit = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=1.0)

    # Should be very close (tighter tolerance)
    assert np.allclose(phi_default, phi_explicit, rtol=1e-12, atol=1e-12)
    assert np.all(np.isfinite(phi_default))

def test_poisson_constant_scalar_epsilon():
    """Test Poisson solver with constant scalar epsilon."""
    mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), char_length=0.4, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)

    rho = np.ones(basis.N)

    # Solve with epsilon=1.0 and epsilon=2.0
    phi1 = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=1.0)
    phi2 = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=2.0)

    # With doubled epsilon, the solution should scale differently
    # For -∇·(ε∇φ) = rho, doubling ε should approximately halve φ
    assert not np.allclose(phi1, phi2)
    assert np.all(np.isfinite(phi1))
    assert np.all(np.isfinite(phi2))

    # Check that phi2 is smaller than phi1 (roughly half)
    # This is approximate due to boundary conditions
    interior_mask = np.abs(phi1) > 1e-10
    ratio = np.abs(phi2[interior_mask]) / np.abs(phi1[interior_mask])
    assert np.mean(ratio) < 0.8  # Should be roughly 0.5, but allow margin

def test_poisson_spatially_varying_scalar_epsilon_array():
    """Test Poisson solver with spatially varying scalar epsilon as array."""
    mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), char_length=0.4, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)

    rho = np.ones(basis.N)
    X = basis.doflocs

    # Create spatially varying epsilon: higher on one side
    epsilon = 1.0 + X[0, :]  # varies from 1 to 2 along x-axis

    phi = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=epsilon)

    assert np.all(np.isfinite(phi))
    assert phi.shape[0] == basis.N

def test_poisson_spatially_varying_scalar_epsilon_callable():
    """Test Poisson solver with spatially varying scalar epsilon as callable."""
    mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), char_length=0.4, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)

    rho = np.ones(basis.N)

    # Define epsilon as a callable
    def epsilon_func(X):
        # X has shape (3, npts)
        return 1.0 + 0.5 * X[0, :]  # varies along x-axis

    phi = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=epsilon_func)

    assert np.all(np.isfinite(phi))
    assert phi.shape[0] == basis.N

def test_poisson_tensor_epsilon_callable_diagonal():
    """Test Poisson solver with diagonal tensor epsilon as callable."""
    mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), char_length=0.4, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)

    rho = np.ones(basis.N)

    # Define anisotropic epsilon (diagonal tensor)
    def epsilon_tensor(X):
        # X has shape (3, npts)
        npts = X.shape[1]
        eps = np.zeros((3, 3, npts))
        eps[0, 0, :] = 2.0  # x-direction
        eps[1, 1, :] = 1.0  # y-direction
        eps[2, 2, :] = 0.5  # z-direction
        return eps

    phi = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=epsilon_tensor)

    assert np.all(np.isfinite(phi))
    assert phi.shape[0] == basis.N

def test_poisson_tensor_epsilon_callable_anisotropic():
    """Test Poisson solver with full anisotropic tensor epsilon."""
    mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), char_length=0.4, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)

    rho = np.ones(basis.N)

    # Define anisotropic epsilon with off-diagonal terms
    def epsilon_tensor(X):
        # X has shape (3, npts)
        npts = X.shape[1]
        eps = np.zeros((3, 3, npts))
        # Diagonal
        eps[0, 0, :] = 2.0
        eps[1, 1, :] = 1.5
        eps[2, 2, :] = 1.0
        # Off-diagonal (symmetric for SPD matrix)
        eps[0, 1, :] = 0.3
        eps[1, 0, :] = 0.3
        eps[1, 2, :] = 0.2
        eps[2, 1, :] = 0.2
        return eps

    phi = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=epsilon_tensor)

    assert np.all(np.isfinite(phi))
    assert phi.shape[0] == basis.N

def test_poisson_tensor_epsilon_array_at_dofs():
    """Test Poisson solver with tensor epsilon given as array at DOFs."""
    mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), char_length=0.4, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)

    rho = np.ones(basis.N)
    X = basis.doflocs

    # Create spatially varying tensor epsilon at DOFs
    epsilon = np.zeros((basis.N, 3, 3))
    for i in range(basis.N):
        # Vary epsilon based on position
        epsilon[i, 0, 0] = 1.0 + 0.5 * X[0, i]  # x-direction varies with x
        epsilon[i, 1, 1] = 1.0 + 0.5 * X[1, i]  # y-direction varies with y
        epsilon[i, 2, 2] = 1.0 + 0.5 * X[2, i]  # z-direction varies with z

    phi = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=epsilon)

    assert np.all(np.isfinite(phi))
    assert phi.shape[0] == basis.N

def test_poisson_comparison_scalar_vs_isotropic_tensor():
    """Compare scalar epsilon with isotropic tensor epsilon - should give same result."""
    mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), char_length=0.4, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)

    rho = np.ones(basis.N)

    # Solve with scalar epsilon
    eps_scalar = 2.5
    phi_scalar = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=eps_scalar)

    # Solve with isotropic tensor epsilon
    def epsilon_tensor(X):
        npts = X.shape[1]
        eps = np.zeros((3, 3, npts))
        eps[0, 0, :] = eps_scalar;
        eps[1, 1, :] = eps_scalar;
        eps[2, 2, :] = eps_scalar;
        return eps

    phi_tensor = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=epsilon_tensor)

    # Should be very close
    assert np.allclose(phi_scalar, phi_tensor, rtol=1e-8, atol=1e-10)

def test_poisson_with_scf_integration():
    """Test that the modified solve_poisson still works in the SCF loop."""
    mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), char_length=0.45, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    Vext = lambda X: np.zeros(X.shape[1])

    # Run a short SCF loop with default epsilon
    E, modes, phi, Vfinal = solver.scf_loop(mesh, basis, K, M, Vext,
                                           coupling=1.0, maxiter=10, tol=1e-5,
                                           mix=0.4, nev=2, verbose=False, use_diis=False)

    assert np.isfinite(E[0])
    assert modes.shape[1] == 2;
    assert np.all(np.isfinite(phi))


# Manufactured-solution tests

def test_poisson_manufactured_solution():
    """Manufactured solution test: check numerical phi against analytic phi_exact."""
    mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), char_length=0.30, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)

    X = basis.doflocs;
    phi_ex = phi_exact_from_X(X);
    rho = rhs_for_phi_exact(phi_ex);

    phi_num = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=1.0);

    err = l2_error(phi_num, phi_ex, M);

    # Tolerance depends on mesh; with char_length ~0.30 this should be reasonably small.
    assert err < 2e-3

def test_poisson_manufactured_convergence():
    """Convergence test for manufactured solution. Expect ~O(h^2) L2 error for linear FE."""
    char_lengths = [0.6, 0.45, 0.30, 0.20]   # coarse -> fine
    errors = [];
    hs = [];

    for h in char_lengths:
        mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), char_length=h, verbose=False);
        mesh, basis, K, M = solver.assemble_operators(mesh);

        X = basis.doflocs;
        phi_ex = phi_exact_from_X(X);
        rho = rhs_for_phi_exact(phi_ex);

        phi_num = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=1.0);

        err = l2_error(phi_num, phi_ex, M);
        errors.append(err);
        hs.append(h);

    # compute empirical convergence rate from log-log slope
    logh = np.log(hs);
    loge = np.log(errors);
    slope, intercept = np.polyfit(logh, loge, 1);
    rate = -slope;

    # For linear basis, L2 rate should approach 2.0. Use conservative threshold 1.5 to avoid false failures.
    assert rate > 1.5, f"Observed L2 convergence rate too low: {rate:.2f}. errors={errors}",
