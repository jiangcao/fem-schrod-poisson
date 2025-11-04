import numpy as np
import matplotlib.pyplot as plt
from src import solver
from skfem import asm
from skfem.models.poisson import laplace


def solve_with_old_rhs(mesh, basis, rho, bc_value=0.0):
    """Solve using old RHS assembly b = M.dot(rho) (nodal mass projection)."""
    # assemble stiffness and mass
    A = asm(laplace, basis).tocsr()
    def mass(u, v, w):
        return u * v
    M = asm(mass, basis).tocsr()

    # build RHS using old approach
    b_old = M.dot(rho)

    # apply Dirichlet BC as in solver
    try:
        bdofs = mesh.boundary_nodes()
    except Exception:
        bdofs = np.unique(mesh.facets.flatten())
    ndofs = basis.N
    all_idx = np.arange(ndofs)
    interior = np.setdiff1d(all_idx, bdofs)

    Aii = A[interior][:, interior]
    bi = b_old[interior] - A[interior][:, bdofs].dot(np.full(len(bdofs), bc_value))
    phi = np.full(ndofs, bc_value, dtype=float)
    from scipy.sparse.linalg import spsolve
    phi_i = spsolve(Aii, bi)
    phi[interior] = phi_i
    return phi, M


def run(char_lengths=[0.6, 0.45, 0.30, 0.20]):
    errors_old = []
    errors_new = []
    hs = []

    for h in char_lengths:
        mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), char_length=h, verbose=False)
        mesh, basis, K, M = solver.assemble_operators(mesh)
        X = basis.doflocs
        phi_exact = np.sin(np.pi * X[0, :]) * np.sin(np.pi * X[1, :]) * np.sin(np.pi * X[2, :])
        rho = 3.0 * (np.pi ** 2) * phi_exact

        # old RHS
        phi_old, Mmat = solve_with_old_rhs(mesh, basis, rho, bc_value=0.0)
        e_old = phi_old - phi_exact
        l2_old = np.sqrt(abs(e_old @ (Mmat.dot(e_old))))

        # new RHS (current solver)
        phi_new = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=1.0)
        e_new = phi_new - phi_exact
        l2_new = np.sqrt(abs(e_new @ (Mmat.dot(e_new))))

        errors_old.append(l2_old)
        errors_new.append(l2_new)
        hs.append(h)
        print(f"h={h:.3f}  L2_old={l2_old:.6e}  L2_new={l2_new:.6e}")

    # compute rates
    logh = np.log(hs)
    loge_old = np.log(errors_old)
    loge_new = np.log(errors_new)
    slope_old, _ = np.polyfit(logh, loge_old, 1)
    slope_new, _ = np.polyfit(logh, loge_new, 1)
    rate_old = -slope_old
    rate_new = -slope_new

    print(f"Empirical L2 rates -> old: {rate_old:.3f}, new: {rate_new:.3f}")

    # plot
    plt.figure()
    plt.loglog(hs, errors_old, 'o-', label=f'old b=M*rho (rate={rate_old:.2f})')
    plt.loglog(hs, errors_new, 's-', label=f'new assemble (rate={rate_new:.2f})')
    plt.gca().invert_xaxis()
    plt.xlabel('h (char_length)')
    plt.ylabel('L2 error')
    plt.legend()
    plt.grid(True, which='both')
    out = 'results/convergence.png'
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150)
    print('Saved convergence plot to', out)


if __name__ == '__main__':
    run()
