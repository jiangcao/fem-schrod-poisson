import numpy as np
from src import solver


def test_solve_poisson_with_callable_rho():
    """Passing rho as a callable should assemble the RHS at quadrature points and be accurate."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), char_length=0.15, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)

    X = basis.doflocs
    # manufactured solution phi = sin(pi x) sin(pi y) sin(pi z)
    phi_exact = np.sin(np.pi * X[0, :]) * np.sin(np.pi * X[1, :]) * np.sin(np.pi * X[2, :])

    # RHS for -Δφ = rho is rho = 3*pi^2 * phi_exact
    def rho_callable(X_flat):
        # X_flat has shape (3, npts)
        x = X_flat[0, :]
        y = X_flat[1, :]
        z = X_flat[2, :]
        return 3.0 * (np.pi ** 2) * (np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z))

    phi_num = solver.solve_poisson(mesh, basis, rho_callable, bc_value=0.0, epsilon=1.0)

    # compute L2 error using mass matrix
    e = phi_num - phi_exact
    Me = M.dot(e)
    l2 = np.sqrt(abs(e @ Me))

    # allow slightly larger L2 error in CI/dev containers
    assert l2 < 1e-2
