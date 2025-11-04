# ...existing code...
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skfem import Basis, ElementTetP1, ElementTetP2, asm, BilinearForm
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

def assemble_operators(mesh, element_order: int = 1):
    """Assemble stiffness and mass for the given mesh.

    element_order: 1 -> P1 (linear), 2 -> P2 (quadratic)
    """
    if element_order == 1:
        element = ElementTetP1()
    elif element_order == 2:
        element = ElementTetP2()
    else:
        raise ValueError(f"Unsupported element_order: {element_order}")

    basis = Basis(mesh, element)

    @BilinearForm
    def mass(u, v, w):
        return u * v

    K = asm(laplace, basis)
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

def solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=None):
    """
    Solve -∇·(ε∇φ) = rho with Dirichlet bc=bc_value on boundary nodes.
    
    Parameters:
    -----------
    mesh : skfem.Mesh
        The mesh
    basis : skfem.Basis
        The basis functions
    rho : array length ndofs
        Source term (density given at DOFs)
    bc_value : float, optional
        Dirichlet boundary condition value (default 0.0)
    epsilon : None, float, array, or callable, optional
        Spatially varying permittivity/diffusion coefficient:
        - None (default): constant epsilon=1.0 (standard Laplace)
        - float: constant scalar epsilon
        - array of shape (ndofs,): spatially varying scalar epsilon at DOFs
        - array of shape (ndofs, 3, 3): tensor epsilon at DOFs
        - callable(X) -> scalar array or tensor array: function of coordinates
          where X has shape (3, npts) and returns either:
            * 1D array of shape (npts,) for scalar epsilon
            * 3D array of shape (3, 3, npts) for tensor epsilon
    
    Returns:
    --------
    phi : array length ndofs
        Solution at DOFs
    """
    def mass(u, v, w):
        return u * v
    M = asm(mass, basis).tocsr()
    
    # Assemble stiffness matrix based on epsilon
    if epsilon is None:
        # Standard Laplace operator: -Δφ
        A = asm(laplace, basis).tocsr()
    elif np.isscalar(epsilon):
        # Constant scalar epsilon: -ε∇²φ
        A = epsilon * asm(laplace, basis).tocsr()
    elif callable(epsilon):
        # Epsilon is a callable function
        # Evaluate once to check dimensionality (returns scalar or tensor)
        # Note: We must evaluate epsilon again inside the form at quadrature points,
        # so this initial call is just for determining the return type
        eps_check = epsilon(basis.doflocs)
        if eps_check.ndim == 1:
            # Scalar field: -∇·(ε∇φ)
            def weighted_laplace(u, v, w):
                # w.x has shape (3, nelems, nqp)
                # Reshape to (3, nelems*nqp) for epsilon function
                x_flat = w.x.reshape(3, -1)
                eps_qp = epsilon(x_flat)  # Returns shape (nelems*nqp,)
                # Reshape back to (nelems, nqp)
                eps_qp = eps_qp.reshape(u.grad[0].shape)
                return eps_qp * (u.grad[0] * v.grad[0] + 
                                 u.grad[1] * v.grad[1] + 
                                 u.grad[2] * v.grad[2])
            A = asm(weighted_laplace, basis).tocsr()
        else:
            # Tensor field: -∇·(ε·∇φ)
            def tensor_laplace(u, v, w):
                # w.x has shape (3, nelems, nqp)
                # Reshape to (3, nelems*nqp) for epsilon function
                x_flat = w.x.reshape(3, -1)
                eps_tensor = epsilon(x_flat)  # Returns shape (3, 3, nelems*nqp)
                # Reshape to (3, 3, nelems, nqp)
                eps_tensor = eps_tensor.reshape(3, 3, *u.grad[0].shape)
                # Compute ε·∇u
                eps_grad_u = np.zeros((3,) + u.grad[0].shape)
                for i in range(3):
                    for j in range(3):
                        eps_grad_u[i] += eps_tensor[i, j] * u.grad[j]
                # Compute ∇v · (ε·∇u)
                return (eps_grad_u[0] * v.grad[0] + 
                        eps_grad_u[1] * v.grad[1] + 
                        eps_grad_u[2] * v.grad[2])
            A = asm(tensor_laplace, basis).tocsr()
    elif isinstance(epsilon, np.ndarray):
        if epsilon.ndim == 1:
            # Array of scalar values at DOFs
            if len(epsilon) != basis.N:
                raise ValueError(f"Epsilon array length {len(epsilon)} must match ndofs {basis.N}")
            # Interpolate to quadrature points
            eps_qp = basis.interpolate(epsilon)  # shape (nelems, nqp)
            
            def weighted_laplace(u, v, w):
                return eps_qp * (u.grad[0] * v.grad[0] + 
                                 u.grad[1] * v.grad[1] + 
                                 u.grad[2] * v.grad[2])
            A = asm(weighted_laplace, basis).tocsr()
        elif epsilon.ndim == 3:
            # Tensor epsilon at DOFs
            if epsilon.shape[0] == basis.N and epsilon.shape[1:] == (3, 3):
                # Epsilon given at DOFs: shape (ndofs, 3, 3)
                # Interpolate each component separately
                # Interpolate one component to get the shape, then allocate array
                eps_00 = basis.interpolate(epsilon[:, 0, 0])
                eps_tensor_qp = np.zeros((3, 3) + eps_00.shape)
                eps_tensor_qp[0, 0] = eps_00
                # Interpolate remaining components
                for i in range(3):
                    for j in range(3):
                        if i == 0 and j == 0:
                            continue  # Already done
                        eps_tensor_qp[i, j] = basis.interpolate(epsilon[:, i, j])
                
                def tensor_laplace(u, v, w):
                    # eps_tensor_qp has shape (3, 3, nelems, nqp)
                    # Compute ε·∇u
                    eps_grad_u = np.zeros((3,) + u.grad[0].shape)
                    for i in range(3):
                        for j in range(3):
                            eps_grad_u[i] += eps_tensor_qp[i, j] * u.grad[j]
                    # Compute ∇v · (ε·∇u)
                    return (eps_grad_u[0] * v.grad[0] + 
                            eps_grad_u[1] * v.grad[1] + 
                            eps_grad_u[2] * v.grad[2])
                A = asm(tensor_laplace, basis).tocsr()
            else:
                raise ValueError(f"Tensor epsilon shape {epsilon.shape} not recognized. "
                                 f"Expected (ndofs, 3, 3), got {epsilon.shape}")
        else:
            raise ValueError(f"Epsilon array must be 1D (scalar field) or 3D (tensor field), "
                             f"got {epsilon.ndim}D")
    else:
        raise TypeError(f"Epsilon must be None, scalar, array, or callable, got {type(epsilon)}")
    
    # Build RHS vector b = (v, rho).
    # Accept either:
    # - rho as a 1D array of nodal values (DOF values) => assemble using interpolated
    #   values at quadrature points for a more accurate load vector,
    # - rho as a callable(X) that returns values at quadrature points => assemble directly.
    if callable(rho):
        def rhs_form(v, w):
            # w.x has shape (3, nelems, nqp); reshape to (3, nelems*nqp) for callable
            x_flat = w.x.reshape(3, -1)
            rq = rho(x_flat)
            rq = rq.reshape(v.shape)
            return v * rq
        b = asm(rhs_form, basis)
    else:
        # rho is array of DOF values: interpolate to quadrature points and assemble
        if len(rho) != basis.N:
            raise ValueError(f"rho array length {len(rho)} must match ndofs {basis.N}")
        rho_qp = basis.interpolate(rho)
        def rhs_form(v, w):
            return v * rho_qp
        b = asm(rhs_form, basis)

    # identify boundary DOFs robustly for both linear and higher-order bases.
    # Prefer basis.boundary() which returns a FacetBasis containing the DOF indices
    # on boundary facets (works for P1, P2, etc.). Fall back to mesh.boundary_nodes()
    # (vertex-only) or the facet flatten fallback.
    try:
        fb = basis.boundary()
        try:
            bdofs = np.asarray(fb.dofs)
            # If it's an object array or nested, try to flatten
            try:
                bdofs = np.unique(bdofs.ravel().astype(int))
            except Exception:
                # handle nested sequences
                import numpy as _np
                try:
                    bdofs = _np.unique(_np.hstack([_np.asarray(x).ravel() for x in bdofs]))
                except Exception:
                    bdofs = _np.unique(_np.asarray(mesh.boundary_nodes()))
        except Exception:
            # fb.dofs may be unavailable; try get_dofs()
            try:
                gd = fb.get_dofs()
                all_dofs = []
                for v in gd.values():
                    all_dofs.extend(list(v))
                bdofs = np.unique(np.array(all_dofs, dtype=int))
            except Exception:
                # Last resort: try converting facet_dofs from basis
                try:
                    bdofs = np.unique(np.asarray(basis.facet_dofs).ravel())
                except Exception:
                    bdofs = np.unique(mesh.facets.flatten())
    except Exception:
        try:
            bdofs = mesh.boundary_nodes()
        except Exception:
            # fallback: mark nodes with any boundary facet
            bdofs = np.unique(mesh.facets.flatten())

        # If bdofs is empty (some skfem versions or mesh conversions), fall back
        # to geometric detection from DOF coordinates (works for P1 and P2).
        try:
            bdofs_arr = np.asarray(bdofs)
            if bdofs_arr.size == 0:
                # detect DOFs lying on the bounding box of the mesh
                X = basis.doflocs
                # bounding box from original mesh vertices
                p = mesh.p
                xmin, xmax = p[0, :].min(), p[0, :].max()
                ymin, ymax = p[1, :].min(), p[1, :].max()
                zmin, zmax = p[2, :].min(), p[2, :].max()
                tol = 1e-8
                mask = (
                    (np.abs(X[0, :] - xmin) < tol) | (np.abs(X[0, :] - xmax) < tol) |
                    (np.abs(X[1, :] - ymin) < tol) | (np.abs(X[1, :] - ymax) < tol) |
                    (np.abs(X[2, :] - zmin) < tol) | (np.abs(X[2, :] - zmax) < tol)
                )
                bdofs = np.nonzero(mask)[0]
        except Exception:
            pass
    
    ndofs = basis.N
    all_idx = np.arange(ndofs)
    interior = np.setdiff1d(all_idx, bdofs)

    Aii = A[interior][:, interior]
    bi = b[interior] - A[interior][:, bdofs].dot(np.full(len(bdofs), bc_value))
    phi = np.full(ndofs, bc_value, dtype=float)
    phi_i = spla.spsolve(Aii, bi)
    phi[interior] = phi_i
    return phi




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
