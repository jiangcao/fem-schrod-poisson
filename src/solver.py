# ...existing code...
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skfem import Basis, ElementTetP1, ElementTetP2, asm, BilinearForm, LinearForm
from skfem.mesh import MeshTet
from skfem.models.poisson import laplace
from skfem.io import from_meshio
import meshio
import pygmsh
import os
from dataclasses import dataclass


@dataclass
class PhysicalParams:
    """
    Container for physical constants. Defaults mimic the existing
    dimensionless setup (ħ = 1, m = 1, q = -1, ε0 = 1, one particle).

    - hbar: Planck's reduced constant
    - m_eff: effective mass
    - q: particle charge (electron: -1 in dimensionless units)
    - epsilon0: permittivity of free space (or unit system baseline)
    - n_particles: number of particles represented by |psi|^2
    """
    hbar: float = 1.0
    m_eff: float = 1.0
    q: float = -1.0
    epsilon0: float = 1.0
    n_particles: float = 1.0

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
    K = asm(laplace, basis).tocsr()

    @BilinearForm
    def mass(u, v, w):
        return u * v
    M = asm(mass, basis).tocsr()
    return mesh, basis, K, M

    # (no-op)

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
             mix=0.3, nev=4, verbose=True, use_diis=False, diis_max=6,
             phys: PhysicalParams | None = None, epsilon=None, mass_eff=None):
    """
    Self-consistent loop with optional DIIS (use_diis=True).
    If DIIS is enabled, simple linear mixing is used for first iterations while DIIS accumulates.
    
    Parameters:
    -----------
    epsilon : None, scalar, array, or callable
        Dielectric constant for Poisson equation (passed to solve_poisson)
    mass_eff : None, scalar, array, or callable
        Effective mass for Schrödinger equation (overrides phys.m_eff if provided)
        - scalar: constant effective mass
        - 1D array: spatially varying scalar mass at DOFs
        - callable(X): returns scalar array (npts,) or tensor array (3,3,npts)
    
    Returns E, modes, phi, V_final
    """
    ndofs = basis.N
    X = basis.doflocs
    Vext_vec = Vext_func(X)
    V_old = Vext_vec.copy()
    prev_energy = None

    diis = DIIS(max_vec=diis_max) if use_diis else None
    
    # Determine kinetic coefficient - note mass_eff parameter overrides phys.m_eff
    if mass_eff is None:
        kinetic_coeff = (phys.hbar ** 2) / (2.0 * phys.m_eff) if phys is not None else 0.5
        mass_eff_param = None
    else:
        # mass_eff is provided, use it
        kinetic_coeff = None  # Will be handled in solve_generalized_eig
        mass_eff_param = mass_eff

    for it in range(1, maxiter + 1):
        E, modes = solve_generalized_eig(K, M, V_old, nev=nev, which='SM', 
                                        kinetic_coeff=kinetic_coeff, 
                                        mass_eff=mass_eff_param,
                                        phys=phys,
                                        basis=basis, mesh=mesh, dirichlet_bc=False, Vfunc=Vext_func)
        modes = normalize_modes(modes, M)
        psi0 = modes[:, 0]
        # Convert to charge density if physical constants are provided
        if phys is not None:
            rho = phys.n_particles * phys.q * (np.abs(psi0) ** 2)
        else:
            rho = np.abs(psi0) ** 2

        phi = solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=epsilon)

        if phys is not None:
            V_new = Vext_vec + coupling * (phys.q * phi)
        else:
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

# --- Helpers and solvers ---

def potential_vector_from_callable(basis: Basis, Vfunc):
    """
    Evaluate a potential function Vfunc(X) at the DOF locations and return
    a vector of length ndofs.
    """
    X = basis.doflocs
    V = Vfunc(X)
    V = np.asarray(V).reshape(-1)
    if V.shape[0] != basis.N:
        raise ValueError("Vfunc must return an array of length equal to number of DOFs")
    return V


def normalize_modes(modes: np.ndarray, M: sp.spmatrix) -> np.ndarray:
    """
    Normalize eigenmodes such that psi^T M psi = 1 for each column.
    """
    modes = np.asarray(modes, dtype=float)
    for i in range(modes.shape[1]):
        v = modes[:, i]
        n = float(v @ (M.dot(v)))
        if n <= 0:
            continue
        modes[:, i] = v / np.sqrt(n)
    return modes


def _invert_mass_tensor_field(mass_tensor):
    """
    Invert a field of 3×3 mass tensors efficiently.
    
    Parameters:
    -----------
    mass_tensor : ndarray of shape (3, 3, npts)
        Mass tensor at each point
        
    Returns:
    --------
    mass_inv : ndarray of shape (3, 3, npts)
        Inverse mass tensor at each point
    """
    npts = mass_tensor.shape[2]
    mass_inv = np.zeros_like(mass_tensor)
    
    # Check if tensor is diagonal at all points (common case)
    is_diagonal = True
    for i in range(3):
        for j in range(3):
            if i != j and np.max(np.abs(mass_tensor[i, j, :])) > 1e-12:
                is_diagonal = False
                break
        if not is_diagonal:
            break
    
    if is_diagonal:
        # Optimized path for diagonal tensors - element-wise inversion
        for i in range(3):
            mass_inv[i, i, :] = 1.0 / mass_tensor[i, i, :]
    else:
        # General case - invert each matrix
        for i in range(npts):
            mass_inv[:, :, i] = np.linalg.inv(mass_tensor[:, :, i])
    
    return mass_inv


def solve_generalized_eig(
    K: sp.spmatrix,
    M: sp.spmatrix,
    V: np.ndarray,
    nev=4,
    which='SM',
    kinetic_coeff: float | None = 0.5,
    basis: Basis | None = None,
    mesh: MeshTet | None = None,
    dirichlet_bc: bool = True,
    Vfunc=None,
    mass_eff=None,
    phys: PhysicalParams | None = None,
):
    """
    Solve the generalized eigenproblem for Schrödinger equation with spatially varying mass.
    
    The equation is:
        -∇·(1/m_eff ∇ψ) + V ψ = E ψ   (with proper scaling by ħ²/2)
    
    or in tensor form:
        -∇·(1/m_eff_tensor ∇ψ) + V ψ = E ψ
    
    Parameters:
    -----------
    kinetic_coeff : float or None
        If mass_eff is None, this is used as the kinetic coefficient (ħ²/2m)
    mass_eff : None, scalar, array, or callable
        Effective mass specification:
        - None: use kinetic_coeff
        - scalar: constant effective mass
        - 1D array of length ndofs: scalar mass at DOFs
        - callable(X): returns scalar array (npts,) or tensor array (3,3,npts)
        - 3D array of shape (ndofs,3,3): tensor mass at DOFs
    phys : PhysicalParams or None
        Physical parameters (used for hbar when mass_eff is provided)
    """
    if sp.issparse(K):
        K = K.tocsr()
    if sp.issparse(M):
        M = M.tocsr()
    V = np.asarray(V).reshape(-1)
    n = M.shape[0]
    if V.shape[0] != n:
        raise ValueError("Dimension mismatch for potential vector V")

    # Assemble potential operator
    if Vfunc is not None and basis is not None:
        @BilinearForm
        def v_mass(u, v, w):
            X_flat = w.x.reshape(3, -1)
            V_qp = np.asarray(Vfunc(X_flat)).reshape(u.grad[0].shape)
            return V_qp * u * v
        V_M = asm(v_mass, basis).tocsr()
    else:
        # Fallback to diagonal potential
        V_M = sp.diags(V)

    # Assemble kinetic operator with spatially varying mass
    if mass_eff is None:
        # Use standard laplacian with constant kinetic coefficient
        K_eff = (kinetic_coeff if kinetic_coeff is not None else 0.5) * K
    else:
        # Assemble kinetic operator with spatially varying effective mass
        # The operator is -∇·(c/m_eff ∇ψ) where c = ħ²/2
        hbar = phys.hbar if phys is not None else 1.0
        c_coeff = (hbar ** 2) / 2.0
        
        if basis is None:
            raise ValueError("basis is required when mass_eff is specified")
        
        if callable(mass_eff):
            # Evaluate at quadrature points
            def is_tensor(m_samp):
                arr = np.asarray(m_samp)
                return arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[1] == 3
            
            @BilinearForm
            def kinetic_form(u, v, w):
                X_flat = w.x.reshape(3, -1)
                m_val = mass_eff(X_flat)
                m_arr = np.asarray(m_val)
                shape_qp = u.grad[0].shape
                
                if is_tensor(m_arr):
                    # Tensor mass: use helper function to invert efficiently
                    # m_arr has shape (3, 3, npts)
                    m_inv = _invert_mass_tensor_field(m_arr)
                    
                    # Assemble -c ∇·(m_inv ∇ψ)
                    res = 0.0
                    for i in range(3):
                        for j in range(3):
                            m_inv_qp = m_inv[i, j, :].reshape(shape_qp)
                            res = res + c_coeff * m_inv_qp * u.grad[j] * v.grad[i]
                    return res
                else:
                    # Scalar mass
                    m_qp = m_arr.reshape(shape_qp)
                    return c_coeff * (1.0 / m_qp) * (
                        u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1] + u.grad[2] * v.grad[2]
                    )
            
            K_eff = asm(kinetic_form, basis).tocsr()
            
        elif isinstance(mass_eff, np.ndarray):
            m_arr = np.asarray(mass_eff)
            if m_arr.ndim == 1:
                # Scalar mass at DOFs
                if m_arr.shape[0] != basis.N:
                    raise ValueError("mass_eff array length must match number of DOFs")
                m_qp = basis.interpolate(m_arr)
                
                @BilinearForm
                def kinetic_form(u, v, w):
                    return c_coeff * (1.0 / m_qp) * (
                        u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1] + u.grad[2] * v.grad[2]
                    )
                
                K_eff = asm(kinetic_form, basis).tocsr()
                
            elif m_arr.ndim == 3 and m_arr.shape[0] == basis.N and m_arr.shape[1:] == (3, 3):
                # Tensor mass at DOFs - interpolate and invert
                m_qp = np.zeros((3, 3) + basis.interpolate(m_arr[:, 0, 0]).shape)
                for i in range(3):
                    for j in range(3):
                        m_qp[i, j] = basis.interpolate(m_arr[:, i, j])
                
                # Use helper function to invert efficiently
                shape_qp = m_qp[0, 0].shape
                npts = np.prod(shape_qp)
                m_qp_flat = m_qp.reshape(3, 3, npts)
                m_inv_flat = _invert_mass_tensor_field(m_qp_flat)
                m_inv_qp = m_inv_flat.reshape((3, 3) + shape_qp)
                
                @BilinearForm
                def kinetic_form(u, v, w):
                    res = 0.0
                    for i in range(3):
                        for j in range(3):
                            res = res + c_coeff * m_inv_qp[i, j] * u.grad[j] * v.grad[i]
                    return res
                
                K_eff = asm(kinetic_form, basis).tocsr()
            else:
                raise ValueError("Unsupported mass_eff array shape")
        elif np.isscalar(mass_eff):
            # Constant scalar mass
            K_eff = c_coeff * (1.0 / float(mass_eff)) * K
        else:
            raise TypeError("mass_eff must be None, scalar, array, or callable")

    H = K_eff + V_M

    # Apply Dirichlet boundary conditions by restricting to interior DOFs if requested
    if dirichlet_bc and mesh is not None:
        try:
            bdofs = mesh.boundary_nodes()
        except Exception:
            try:
                bdofs = np.unique(mesh.facets.flatten())
            except Exception:
                bdofs = np.array([], dtype=int)
        bdofs = np.asarray(bdofs, dtype=int)
        all_idx = np.arange(M.shape[0])
        interior = np.setdiff1d(all_idx, bdofs)
        k_eff = min(nev, max(1, interior.size - 1))
        Hi = H[interior][:, interior]
        Mi = M[interior][:, interior]
        try:
            E, modes_i = spla.eigsh(Hi, k=k_eff, M=Mi, which=which)
        except Exception:
            E, modes_i = spla.eigsh(Hi, k=k_eff, M=Mi, sigma=0.0, which='LM')
        idx = np.argsort(E)
        E = E[idx]
        modes_i = modes_i[:, idx]
        modes = np.zeros((M.shape[0], modes_i.shape[1]))
        modes[interior, :] = modes_i
        return E, modes

    try:
        E, modes = spla.eigsh(H, k=nev, M=M, which=which)
    except Exception:
        # Fallback: use shift-invert near zero
        E, modes = spla.eigsh(H, k=nev, M=M, sigma=0.0, which='LM')
    idx = np.argsort(E)
    return E[idx], modes[:, idx]


def solve_poisson(mesh, basis: Basis, rho, bc_value: float = 0.0, epsilon=None):
    """

    epsilon can be:
      - None: treated as 1.0
      - scalar: constant
      - 1D array of length ndofs: scalar epsilon at DOFs (interpolated to quadrature)
      - callable(X): returns scalar array (npts,) or tensor array (3,3,npts)
      - 3D array of shape (ndofs,3,3): tensor epsilon at DOFs (interpolated per component)
    rho can be:
      - array of length ndofs
      - callable(X): returns array (npts,)
    """
    # Assemble stiffness matrix A depending on epsilon
    if epsilon is None:
        A = asm(laplace, basis).tocsr()
    elif np.isscalar(epsilon):
        A = float(epsilon) * asm(laplace, basis).tocsr()
    elif callable(epsilon):
        # Evaluate at quadrature points inside the bilinear form
        def is_tensor(eps_samp):
            arr = np.asarray(eps_samp)
            return arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[1] == 3

        @BilinearForm
        def weighted(u, v, w):
            X_flat = w.x.reshape(3, -1)
            eps_val = epsilon(X_flat)
            eps_arr = np.asarray(eps_val)
            shape_qp = u.grad[0].shape
            if is_tensor(eps_arr):
                # tensor case: eps[i,j,npts] -> reshape to (nelems, nqp)
                res = 0.0
                for i in range(3):
                    for j in range(3):
                        eps_qp = eps_arr[i, j, :].reshape(shape_qp)
                        res = res + eps_qp * u.grad[j] * v.grad[i]
                return res
            else:
                # scalar case: shape (npts,) -> reshape to quadrature shape
                eps_qp = eps_arr.reshape(shape_qp)
                return eps_qp * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1] + u.grad[2] * v.grad[2])

        A = asm(weighted, basis).tocsr()
    elif isinstance(epsilon, np.ndarray):
        eps_arr = np.asarray(epsilon)
        if eps_arr.ndim == 1:
            if eps_arr.shape[0] != basis.N:
                raise ValueError("epsilon array length must match number of DOFs")
            eps_qp = basis.interpolate(eps_arr)

            @BilinearForm
            def weighted(u, v, w):
                return eps_qp * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1] + u.grad[2] * v.grad[2])

            A = asm(weighted, basis).tocsr()
        elif eps_arr.ndim == 3 and eps_arr.shape[0] == basis.N and eps_arr.shape[1:] == (3, 3):
            # interpolate each component to quadrature
            eps_qp = np.zeros((3, 3) + basis.interpolate(eps_arr[:, 0, 0]).shape)
            for i in range(3):
                for j in range(3):
                    eps_qp[i, j] = basis.interpolate(eps_arr[:, i, j])

            @BilinearForm
            def weighted(u, v, w):
                res = 0.0
                for i in range(3):
                    for j in range(3):
                        res = res + eps_qp[i, j] * u.grad[j] * v.grad[i]
                return res

            A = asm(weighted, basis).tocsr()
        else:
            raise ValueError("Unsupported epsilon array shape")
    else:
        raise TypeError("epsilon must be None, scalar, array, or callable")

    # Assemble RHS b
    if callable(rho):
        @LinearForm
        def rhsform(v, w):
            Xq = w.x.reshape(3, -1)
            rho_qp = np.asarray(rho(Xq)).reshape(v.shape)
            return v * rho_qp
        b = asm(rhsform, basis)
    else:
        rho_arr = np.asarray(rho).reshape(-1)
        if rho_arr.shape[0] != basis.N:
            raise ValueError("rho array length must match number of DOFs")
        rho_qp = basis.interpolate(rho_arr)

        @LinearForm
        def rhsform(v, w):
            return v * rho_qp
        b = asm(rhsform, basis)

    # Boundary DOFs
    try:
        bdofs = mesh.boundary_nodes()
    except Exception:
        try:
            bdofs = np.unique(mesh.facets.flatten())
        except Exception:
            bdofs = np.array([], dtype=int)
    bdofs = np.asarray(bdofs, dtype=int)
    all_idx = np.arange(basis.N)
    interior = np.setdiff1d(all_idx, bdofs)

    # Solve interior system with Dirichlet boundary conditions
    Aii = A[interior][:, interior]
    bi = b[interior] - A[interior][:, bdofs].dot(np.full(bdofs.shape[0], bc_value))
    phi = np.full(basis.N, bc_value, dtype=float)
    phi_i = spla.spsolve(Aii, bi)
    phi[interior] = phi_i
    return phi

if __name__ == "__main__":
    # example run (small mesh for quick testing)
    mesh = make_mesh_box(x0=(0, 0, 0), lengths=(2.0, 2.0, 2.0), char_length=0.25, verbose=True)
    mesh, basis, K, M = assemble_operators(mesh)
    Vext = lambda X: np.zeros(X.shape[1])   # zero external potential
    E, modes, phi, Vfinal = scf_loop(mesh, basis, K, M, Vext, coupling=1.0,
                                     maxiter=30, tol=1e-6, mix=0.4, nev=4, use_diis=True)
    print("Lowest 5 eigenvalues:", E[0:5])
