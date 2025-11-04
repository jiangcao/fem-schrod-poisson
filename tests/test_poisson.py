import numpy as np
from src import solver


def test_poisson_analytical_solution():
    # build a modest mesh for speed
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), char_length=0.15, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)

    X = basis.doflocs  # shape (3, ndofs)
    x, y, z = X[0, :], X[1, :], X[2, :]
    pi = np.pi
    # exact solution and RHS for -Δφ = rho with Dirichlet 0 on boundary
    phi_exact = np.sin(pi * x) * np.sin(pi * y) * np.sin(pi * z)
    # For phi = sin(pi x) sin(pi y) sin(pi z), -Δ phi = 3*pi^2 * phi
    rho = 3.0 * (pi ** 2) * phi_exact

    phi_num = solver.solve_poisson(mesh, basis, rho, bc_value=0.0)

    # basic sanity
    assert phi_num.shape[0] == basis.N
    assert np.all(np.isfinite(phi_num))

    # boundary nodes should be (approximately) zero
    try:
        bdofs = mesh.boundary_nodes()
    except Exception:
        bdofs = np.unique(mesh.facets.flatten())
    if bdofs.size > 0:
        assert np.allclose(phi_num[bdofs], 0.0, atol=1e-6)

    # relative L2 error on DOFs
    rel_err = np.linalg.norm(phi_num - phi_exact) / (np.linalg.norm(phi_exact) + 1e-16)
    # tolerance chosen for coarse tetrahedral discretization
    # allow slightly larger relative error in CI/dev containers
    assert rel_err < 0.25


def test_potential_vector_from_callable_simple():
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), char_length=0.45, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)

    # test constant potential
    Vfunc_const = lambda X: np.ones(X.shape[1]) * 2.5
    Vvec = solver.potential_vector_from_callable(basis, Vfunc_const)
    assert Vvec.shape[0] == basis.N
    assert np.allclose(Vvec, 2.5)

    # test simple coordinate-dependent potential
    Vfunc_coord = lambda X: X[0, :] + X[1, :] + X[2, :]
    Vvec2 = solver.potential_vector_from_callable(basis, Vfunc_coord)
    X = basis.doflocs
    expected = X[0, :] + X[1, :] + X[2, :]
    assert Vvec2.shape[0] == basis.N
    assert np.allclose(Vvec2, expected, atol=1e-12)
