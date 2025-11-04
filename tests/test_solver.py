# simple pytest to run a tiny SCF and check outputs
import numpy as np
from src import solver

def test_scf_tiny_run_converges():
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), char_length=0.45, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    Vext = lambda X: np.zeros(X.shape[1])
    E, modes, phi, Vfinal = solver.scf_loop(
        mesh,
        basis,
        K,
        M,
        Vext,
        coupling=1.0,
        maxiter=20,
        tol=1e-5,
        mix=0.4,
        nev=2,
        verbose=False,
        use_diis=True,
        diis_max=4,
    )
    # basic sanity checks
    assert np.isfinite(E[0])
    assert modes.shape[1] == 2
    # check mode normalization: psi^T M psi ~= 1
    psi0 = modes[:, 0]
    norm = float(psi0 @ (M.dot(psi0)))
    assert abs(norm - 1.0) < 1e-6
    # phi should be finite and same length as dofs
    assert phi.shape[0] == basis.N
    assert np.all(np.isfinite(phi))