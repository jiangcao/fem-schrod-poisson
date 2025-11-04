import numpy as np
from src import solver


def test_harmonic_ground_energy_dimensionless():
    """
    In dimensionless units (ħ=1, m=1), for 3D harmonic oscillator with ω=1,
    the ground state energy is E0 = (3/2) * ω = 1.5.
    We solve with coupling=0 (no Hartree) and check the lowest eigenvalue.
    """
    omega = 1.0
    # Use larger box to better approximate infinite domain
    # Center at origin, extend to ±3 (6 oscillator lengths)
    mesh = solver.make_mesh_box(x0=(-3, -3, -3), lengths=(6.0, 6.0, 6.0), char_length=0.4, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh, element_order=2)

    def Vext(X):
        x, y, z = X[0], X[1], X[2]
        r2 = x**2 + y**2 + z**2  # Center at origin now
        return 0.5 * (omega ** 2) * r2

    # Run a single-shot eigen-solve using scf_loop with coupling=0
    E, modes, phi, Vfinal = solver.scf_loop(mesh, basis, K, M, Vext,
                                            coupling=0.0, maxiter=1, tol=1e-12,
                                            mix=0.0, nev=3, verbose=False)
    E0 = float(E[0])
    assert abs(E0 - 1.5 * omega) < 0.15  # allow some FE/boundary error


def test_harmonic_density_center_peak():
    """
    The ground state density should peak at the trap center (0,0,0)
    and be smaller near the edges.
    """
    omega = 1.0
    mesh = solver.make_mesh_box(x0=(-3, -3, -3), lengths=(6.0, 6.0, 6.0), char_length=0.4, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh, element_order=2)

    def Vext(X):
        x, y, z = X[0], X[1], X[2]
        r2 = x**2 + y**2 + z**2  # Center at origin
        return 0.5 * (omega ** 2) * r2

    E, modes, phi, Vfinal = solver.scf_loop(mesh, basis, K, M, Vext,
                                            coupling=0.0, maxiter=1, tol=1e-12,
                                            mix=0.0, nev=1, verbose=False)
    psi0 = modes[:, 0]
    rho = np.abs(psi0) ** 2

    X = basis.doflocs
    r_center = (X[0]**2 + X[1]**2 + X[2]**2) ** 0.5

    center_mask = r_center < 0.5
    edge_mask = r_center > 2.0
    if np.sum(center_mask) == 0 or np.sum(edge_mask) == 0:
        # If mesh doesn't provide enough points in regions, relax thresholds
        center_mask = r_center < 0.8
        edge_mask = r_center > 1.5

    rho_center = float(np.mean(rho[center_mask]))
    rho_edge = float(np.mean(rho[edge_mask]))

    assert rho_center > 2.0 * rho_edge
