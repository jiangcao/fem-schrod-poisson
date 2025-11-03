# ...existing code...
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skfem import Basis, ElementTetP1, asm
from skfem.mesh import MeshTet
from skfem.models.poisson import laplace
from skfem.io import from_meshio
import meshio
import pygmsh
import os

# Mesh generator using pygmsh with configurable char_length / bounding box
def make_mesh_box(x0=(0.0, 0.0, 0.0), lengths=(1.0, 1.0, 1.0), char_length=0.1, verbose=False):
    """
    Create a tetrahedral box mesh with pygmsh.
    - x0: lower corner (tuple of 3 floats)
    - lengths: (lx, ly, lz)
    - char_length: target cell size (smaller -> finer mesh)
    Returns a skfem MeshTet.
    """
    geom = pygmsh.occ.Geometry()
    model3D = geom.__enter__()    
    
    box = model3D.add_box(x0=x0, extents=lengths, mesh_size=char_length)
    smaller_lengths = tuple(np.array(lengths)*0.9)
    print(smaller_lengths)    
    
    model3D.synchronize()    
    model3D.add_physical(box, "box")    

    geom.generate_mesh()
    pygmsh.write("mesh.msh")
    # write/read via meshio to ensure compatibility, but avoid persisting large files
    meshio_mesh = meshio.read("mesh.msh")
    sk_mesh = from_meshio(meshio_mesh)
    
    if verbose:
        print(sk_mesh)        
    return sk_mesh

def assemble_operators(mesh):
    """
    Assemble stiffness (K) and mass (M) matrices on P1 tetrahedra.
    """
    element = ElementTetP1()
    basis = Basis(mesh, element)
    K = asm(laplace, basis)                # stiffness for -Δ
    # simple mass assembly via element routine (P1 mass)
    def mass(u, v, w):
        return u * v
    M = asm(mass, basis)
    return mesh, basis, K.tocsr(), M.tocsr()

def potential_vector_from_callable(basis, V_func):
    """
    Build potential vector evaluated at DOFs.
    V_func: callable(X) -> array(len=ndofs), X shape (3, ndofs)
    returns 1D numpy array of length ndofs
    """
    X = basis.doflocs  # shape (3, ndofs)
    return np.asarray(V_func(X))

def solve_generalized_eig(K, M, Vvec, nev=4, which='SM'):
    """
    Solve H psi = E M psi where H = -0.5*K + M*diag(Vvec)
    Vvec: 1d array length ndofs
    """
    Vdiag = sp.diags(Vvec)
    H = -0.5 * K + M.dot(Vdiag)
    E, psi = spla.eigsh(H, k=nev, M=M, which=which)
    idx = np.argsort(E)
    return E[idx], psi[:, idx]

def normalize_modes(psi, M):
    norms = np.sqrt(np.abs(np.sum(psi * (M.dot(psi)), axis=0)))
    return psi / norms

def solve_poisson(mesh, basis, rho, bc_value=0.0):
    """
    Solve -Δφ = rho with Dirichlet bc=bc_value on boundary nodes.
    rho: array length ndofs (density given at DOFs)
    Returns phi at DOFs.
    """
    A = asm(laplace, basis).tocsr()
    def mass(u, v, w):
        return u * v
    M = asm(mass, basis).tocsr()
    b = M.dot(rho)

    # identify boundary nodes (skfem Mesh has .boundary_nodes when created from meshio)
    try:
        bdofs = mesh.boundary_nodes()        
    except Exception:
        # fallback: mark nodes with any boundary facet
        bdofs = np.unique(mesh.facets.flatten())        
    
    ndofs = basis.N
    all_idx = np.arange(ndofs)
    interior = np.setdiff1d(all_idx, bdofs)

    Aii = A[interior][:, interior]
    bi = b[interior] - A[interior][:, bdofs].dot(np.full(len(bdofs), bc_value))
    phi = np.full(ndofs, bc_value, dtype=float)
    phi_i = spla.spsolve(Aii, bi)
    phi[interior] = phi_i
    return phi

def scf_loop(mesh, basis, K, M, Vext_func, coupling=1.0, maxiter=50, tol=1e-6, mix=0.3, nev=4, verbose=True):
    """
    Self-consistent loop with simple linear mixing of Hartree potential.
    - Vext_func(X) -> external potential at DOFs
    - mix: mixing parameter for new_potential = (1-mix)*old + mix*new
    Diagnostics printed each iteration when verbose=True.
    """
    ndofs = basis.N
    X = basis.doflocs
    Vext_vec = Vext_func(X)
    V_old = Vext_vec.copy()
    prev_energy = None

    for it in range(1, maxiter + 1):
        # assemble and solve eigenproblem with current potential
        E, modes = solve_generalized_eig(K, M, V_old, nev=nev, which='SM')
        modes = normalize_modes(modes, M)
        psi0 = modes[:, 0]
        rho = np.abs(psi0)**2

        # solve Poisson for Hartree potential (phi)
        phi = solve_poisson(mesh, basis, rho, bc_value=0.0)

        # build new potential vector and mix
        V_new = Vext_vec + coupling * phi
        V_mixed = (1.0 - mix) * V_old + mix * V_new
        # diagnostics
        total_energy = E[0]
        dens_change = np.linalg.norm(rho - (M.dot(psi0**2) if False else rho))  # placeholder for density metric
        pot_change = np.linalg.norm(V_mixed - V_old) / (1.0 + np.linalg.norm(V_old))
        if verbose:
            print(f"SCF it {it:3d}: E0 = {total_energy:.8e}, |ΔV|={pot_change:.3e}, mix={mix:.3f}")
        if prev_energy is not None and abs(total_energy - prev_energy) < tol and pot_change < tol:
            if verbose:
                print("SCF converged")
            V_old = V_mixed
            break
        V_old = V_mixed
        prev_energy = total_energy
    return E, modes, phi, V_old



# DIIS (Pulay) mixer
class DIIS:
    def __init__(self, max_vec=6):
        self.max_vec = max_vec
        self.Vs = []   # stored potentials (vectors)
        self.rs = []   # stored residuals (vectors)

    def add(self, V_old, V_new):
        r = V_new - V_old
        self.Vs.append(V_new.copy())
        self.rs.append(r.copy())
        if len(self.Vs) > self.max_vec:
            self.Vs.pop(0)
            self.rs.pop(0)

    def extrapolate(self):
        m = len(self.rs)
        if m == 0:
            return None
        if m == 1:
            return self.Vs[-1].copy()
        # build K matrix of inner products <r_i, r_j>
        K = np.empty((m, m), dtype=float)
        for i in range(m):
            for j in range(m):
                K[i, j] = np.dot(self.rs[i], self.rs[j])
        # augmented system [[K,1],[1^T,0]] [c; mu] = [0;1]
        A = np.empty((m + 1, m + 1), dtype=float)
        A[:m, :m] = K
        A[:m, m] = 1.0
        A[m, :m] = 1.0
        A[m, m] = 0.0
        b = np.zeros(m + 1, dtype=float)
        b[m] = 1.0
        try:
            sol = np.linalg.solve(A, b)
            c = sol[:m]
        except np.linalg.LinAlgError:
            # fallback to simple average
            c = np.ones(m) / m
        V_diis = np.zeros_like(self.Vs[0])
        for ci, Vi in zip(c, self.Vs):
            V_diis += ci * Vi
        return V_diis

# ...existing code (mesh, assemble_operators, potential builders) ...

def scf_loop(mesh, basis, K, M, Vext_func, coupling=1.0, maxiter=50, tol=1e-6,
             mix=0.3, nev=4, verbose=True, use_diis=False, diis_max=6):
    """
    Self-consistent loop with optional DIIS (use_diis=True).
    If DIIS is enabled, simple linear mixing is used for first iterations while DIIS accumulates.
    Returns E, modes, phi, V_final
    """
    ndofs = basis.N
    X = basis.doflocs
    Vext_vec = Vext_func(X)
    V_old = Vext_vec.copy()
    prev_energy = None

    diis = DIIS(max_vec=diis_max) if use_diis else None

    for it in range(1, maxiter + 1):
        E, modes = solve_generalized_eig(K, M, V_old, nev=nev, which='SM')
        modes = normalize_modes(modes, M)
        psi0 = modes[:, 0]
        rho = np.abs(psi0)**2

        phi = solve_poisson(mesh, basis, rho, bc_value=0.0)

        V_new = Vext_vec + coupling * phi

        if use_diis:
            # add pair to DIIS (use V_old->V_new residual)
            diis.add(V_old, V_new)
            # try extrapolate (requires at least 2 vectors)
            V_diis = diis.extrapolate()
            if V_diis is not None:
                V_mixed = V_diis
            else:
                # fallback to linear mixing until DIIS ready
                V_mixed = (1.0 - mix) * V_old + mix * V_new
        else:
            V_mixed = (1.0 - mix) * V_old + mix * V_new

        total_energy = E[0]
        pot_change = np.linalg.norm(V_mixed - V_old) / (1.0 + np.linalg.norm(V_old))
        if verbose:
            print(f"SCF it {it:3d}: E0 = {total_energy:.8e}, |ΔV|={pot_change:.3e}")

        if prev_energy is not None and abs(total_energy - prev_energy) < tol and pot_change < tol:
            if verbose:
                print("SCF converged")
            V_old = V_mixed
            break
        V_old = V_mixed
        prev_energy = total_energy

    return E, modes, phi, V_old

if __name__ == "__main__":
    # example run (small mesh for quick testing)
    mesh = make_mesh_box(x0=(0, 0, 0), lengths=(2.0, 2.0, 2.0), char_length=0.25, verbose=True)
    mesh, basis, K, M = assemble_operators(mesh)
    Vext = lambda X: np.zeros(X.shape[1])   # zero external potential
    E, modes, phi, Vfinal = scf_loop(mesh, basis, K, M, Vext, coupling=1.0,
                                     maxiter=30, tol=1e-6, mix=0.4, nev=4, use_diis=True)
    print("Lowest 5 eigenvalues:", E[0:5])
# ...existing code...