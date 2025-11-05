"""
Demonstration of electromagnetic field treatment in the Schrödinger equation.

This is a PROTOTYPE implementation showing how to incorporate magnetic vector
potential A(r) into the electron Hamiltonian. This extends the existing solver
to handle the minimal coupling (Peierls substitution) p → p - qA.

NOT YET INTEGRATED into main solver - this is a demonstration/proof of concept.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skfem import Basis, ElementTetP1, asm, BilinearForm
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import solver


def assemble_paramagnetic_operator(basis, A_func, q, hbar, mass_eff=None):
    """
    Assemble the paramagnetic operator (linear in A).
    
    The paramagnetic term in the weak form is:
        (iqℏ/2m) ∫ [A·∇ψ φ - ψ A·∇φ] dV
    
    This is complex-valued and anti-Hermitian.
    
    Parameters:
    -----------
    basis : Basis
        Finite element basis
    A_func : callable
        Vector potential function A(X) returning (3, npts) array
    q : float
        Particle charge
    hbar : float
        Reduced Planck constant
    mass_eff : float, array, or callable
        Effective mass (default: 1.0)
    
    Returns:
    --------
    K_para : complex sparse matrix
        Paramagnetic contribution to Hamiltonian
    """
    if mass_eff is None:
        m_inv = 1.0
    elif np.isscalar(mass_eff):
        m_inv = 1.0 / float(mass_eff)
    elif callable(mass_eff):
        # Will evaluate inside form
        m_inv = None
    else:
        # Array at DOFs - interpolate
        m_inv = None
    
    coeff = 1j * q * hbar / 2.0
    
    @BilinearForm(dtype=np.complex128)
    def paramagnetic_form(u, v, w):
        """
        Computes: (iqℏ/2m)[A·∇u v - u A·∇v]
        """
        X_flat = w.x.reshape(3, -1)
        A_val = A_func(X_flat)  # (3, npts)
        
        # Reshape to match gradient shape
        shape_qp = u.grad[0].shape
        Ax = A_val[0, :].reshape(shape_qp)
        Ay = A_val[1, :].reshape(shape_qp)
        Az = A_val[2, :].reshape(shape_qp)
        
        # Evaluate mass if needed
        if m_inv is None:
            if callable(mass_eff):
                m_qp = mass_eff(X_flat).reshape(shape_qp)
                m_inv_qp = 1.0 / m_qp
            elif isinstance(mass_eff, np.ndarray):
                m_qp = basis.interpolate(mass_eff)
                m_inv_qp = 1.0 / m_qp
            else:
                m_inv_qp = 1.0
        else:
            m_inv_qp = m_inv
        
        # A·∇u
        A_dot_grad_u = Ax * u.grad[0] + Ay * u.grad[1] + Az * u.grad[2]
        # A·∇v
        A_dot_grad_v = Ax * v.grad[0] + Ay * v.grad[1] + Az * v.grad[2]
        
        # (iqℏ/2m)[A·∇u v - u A·∇v]
        return coeff * m_inv_qp * (A_dot_grad_u * v - u * A_dot_grad_v)
    
    K_para = asm(paramagnetic_form, basis).tocsr()
    return K_para


def compute_diamagnetic_potential(basis, A_func, q, mass_eff=None):
    """
    Compute the diamagnetic potential (quadratic in A).
    
    V_dia = (q²/2m) |A|²
    
    Parameters:
    -----------
    basis : Basis
        Finite element basis
    A_func : callable
        Vector potential function
    q : float
        Particle charge
    mass_eff : float, array, or callable
        Effective mass
    
    Returns:
    --------
    V_dia : array of length ndofs
        Diamagnetic potential at DOFs
    """
    X_dofs = basis.doflocs
    A_dofs = A_func(X_dofs)  # (3, ndofs)
    
    # |A|²
    A_squared = A_dofs[0]**2 + A_dofs[1]**2 + A_dofs[2]**2
    
    if mass_eff is None:
        m_val = 1.0
    elif np.isscalar(mass_eff):
        m_val = float(mass_eff)
    elif callable(mass_eff):
        m_val = mass_eff(X_dofs)
    else:
        m_val = mass_eff
    
    V_dia = (q**2 / (2.0 * m_val)) * A_squared
    return V_dia


def solve_schrodinger_em(
    mesh, basis, K, M, V_ext, 
    A_func, q, hbar=1.0, mass_eff=None,
    nev=4, which='SR'
):
    """
    Solve Schrödinger equation with electromagnetic fields.
    
    The Hamiltonian is:
        H = (ℏ²/2m)∇² + (iqℏ/m)A·∇ + (q²/2m)|A|² + V
    
    with proper treatment of spatially varying mass if provided.
    
    Parameters:
    -----------
    mesh : MeshTet
        Tetrahedral mesh
    basis : Basis
        Finite element basis
    K : sparse matrix
        Stiffness matrix (Laplacian)
    M : sparse matrix
        Mass matrix
    V_ext : array or callable
        External potential
    A_func : callable
        Vector potential A(X) returning (3, npts)
    q : float
        Particle charge
    hbar : float
        Reduced Planck constant
    mass_eff : None, float, array, or callable
        Effective mass
    nev : int
        Number of eigenvalues
    which : str
        Which eigenvalues ('SR' = smallest real part)
    
    Returns:
    --------
    E : array
        Eigenvalues (energies)
    modes : complex array
        Eigenvectors (wavefunctions)
    """
    # Get potential vector
    if callable(V_ext):
        V_vec = solver.potential_vector_from_callable(basis, V_ext)
    else:
        V_vec = np.asarray(V_ext).reshape(-1)
    
    # Assemble kinetic operator with mass
    c_coeff = (hbar ** 2) / 2.0
    if mass_eff is None:
        K_eff = c_coeff * K
    elif np.isscalar(mass_eff):
        K_eff = c_coeff * (1.0 / float(mass_eff)) * K
    else:
        # Need to assemble with spatially varying mass
        # (Reuse logic from solver.py)
        if callable(mass_eff):
            @BilinearForm
            def kinetic_form(u, v, w):
                X_flat = w.x.reshape(3, -1)
                m_qp = mass_eff(X_flat).reshape(u.grad[0].shape)
                return c_coeff * (1.0 / m_qp) * (
                    u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1] + u.grad[2] * v.grad[2]
                )
            K_eff = asm(kinetic_form, basis).tocsr()
        elif isinstance(mass_eff, np.ndarray):
            m_qp = basis.interpolate(mass_eff)
            @BilinearForm
            def kinetic_form(u, v, w):
                return c_coeff * (1.0 / m_qp) * (
                    u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1] + u.grad[2] * v.grad[2]
                )
            K_eff = asm(kinetic_form, basis).tocsr()
        else:
            raise ValueError("Unsupported mass_eff type")
    
    # Assemble EM operators
    print("Assembling paramagnetic operator...")
    K_para = assemble_paramagnetic_operator(basis, A_func, q, hbar, mass_eff)
    
    print("Computing diamagnetic potential...")
    V_dia = compute_diamagnetic_potential(basis, A_func, q, mass_eff)
    
    # Total potential
    V_total = V_vec + V_dia
    
    # Potential operator (diagonal)
    V_M = sp.diags(V_total, dtype=np.complex128)
    
    # Full Hamiltonian (complex)
    H = K_eff.astype(np.complex128) + K_para + V_M
    M_complex = M.astype(np.complex128)
    
    print(f"Hamiltonian: {H.shape}, nnz={H.nnz}, dtype={H.dtype}")
    print(f"Mass matrix: {M_complex.shape}, dtype={M_complex.dtype}")
    
    # Solve generalized eigenvalue problem
    print(f"Solving for {nev} eigenvalues...")
    try:
        E, modes = spla.eigs(H, k=nev, M=M_complex, which=which, tol=1e-9)
    except Exception as e:
        print(f"eigs failed: {e}, trying eigsh with real part...")
        # Fallback: if Hamiltonian is Hermitian, use eigsh on real part
        # (only valid if paramagnetic term is properly symmetrized)
        H_herm = 0.5 * (H + H.conj().T)
        E, modes = spla.eigsh(H_herm.real, k=nev, M=M, which='SA')
        modes = modes.astype(np.complex128)
    
    # Sort by real part of energy
    idx = np.argsort(E.real)
    E = E[idx]
    modes = modes[:, idx]
    
    # Normalize
    for i in range(modes.shape[1]):
        v = modes[:, i]
        n = np.sqrt(np.abs(np.conj(v) @ M_complex.dot(v)))
        if n > 1e-10:
            modes[:, i] = v / n
    
    return E, modes


def vector_potential_uniform_field(B_z, gauge='symmetric'):
    """
    Vector potential for uniform magnetic field B = B_z ẑ.
    
    Parameters:
    -----------
    B_z : float
        Magnetic field strength in z-direction
    gauge : str
        'symmetric': A = B × r / 2 = (-B_z y/2, B_z x/2, 0)
        'landau_x': A = (-B_z y, 0, 0)
        'landau_y': A = (0, B_z x, 0)
    
    Returns:
    --------
    A_func : callable
        Vector potential function A(X)
    """
    if gauge == 'symmetric':
        def A_func(X):
            Ax = -0.5 * B_z * X[1, :]
            Ay =  0.5 * B_z * X[0, :]
            Az = np.zeros_like(X[0, :])
            return np.array([Ax, Ay, Az])
    elif gauge == 'landau_x':
        def A_func(X):
            Ax = -B_z * X[1, :]
            Ay = np.zeros_like(X[0, :])
            Az = np.zeros_like(X[0, :])
            return np.array([Ax, Ay, Az])
    elif gauge == 'landau_y':
        def A_func(X):
            Ax = np.zeros_like(X[0, :])
            Ay = B_z * X[0, :]
            Az = np.zeros_like(X[0, :])
            return np.array([Ax, Ay, Az])
    else:
        raise ValueError(f"Unknown gauge: {gauge}")
    
    return A_func


def demo_landau_levels():
    """
    Demonstrate Landau levels for a 2D electron gas in a uniform magnetic field.
    
    Expected energy levels: E_n = ℏω_c(n + 1/2) where ω_c = |q|B/m
    """
    print("=" * 70)
    print("DEMO: Landau Levels in Uniform Magnetic Field")
    print("=" * 70)
    
    # Physical parameters
    hbar = 1.0
    m_eff = 1.0
    q = -1.0  # electron charge
    B_z = 1.0  # magnetic field
    
    omega_c = abs(q) * B_z / m_eff
    print(f"\nPhysical parameters:")
    print(f"  ℏ = {hbar}")
    print(f"  m_eff = {m_eff}")
    print(f"  q = {q}")
    print(f"  B_z = {B_z}")
    print(f"  Cyclotron frequency ω_c = |q|B/m = {omega_c}")
    print(f"  Expected ground state E_0 ≈ {0.5 * hbar * omega_c}")
    
    # Create mesh (thin slab in z to approximate 2D)
    print("\nCreating mesh...")
    Lxy = 6.0  # large enough to contain several Landau orbitals
    Lz = 1.0   # thin in z
    mesh = solver.make_mesh_box(
        x0=(-Lxy/2, -Lxy/2, -Lz/2), 
        lengths=(Lxy, Lxy, Lz), 
        char_length=0.5
    )
    mesh, basis, K, M = solver.assemble_operators(mesh)
    print(f"Mesh: {mesh.p.shape[1]} nodes, {basis.N} DOFs")
    
    # Vector potential (symmetric gauge)
    A_func = vector_potential_uniform_field(B_z, gauge='symmetric')
    
    # External potential: weak harmonic confinement to localize states
    omega_conf = 0.1 * omega_c  # much weaker than cyclotron
    def V_ext(X):
        return 0.5 * m_eff * omega_conf**2 * (X[0]**2 + X[1]**2)
    
    print(f"\nAdding weak harmonic confinement ω_conf = {omega_conf}")
    
    # Solve
    nev = 8
    E, modes = solve_schrodinger_em(
        mesh, basis, K, M, V_ext,
        A_func=A_func, q=q, hbar=hbar, mass_eff=m_eff,
        nev=nev
    )
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'n':<4} {'E_n':<15} {'E_analytic':<15} {'Error':<12}")
    print("-" * 70)
    for n in range(len(E)):
        E_n = E[n].real
        # For pure Landau levels (no confinement): E = ℏω_c(n + 1/2)
        # With confinement: E ≈ ℏω_c(n + 1/2) + corrections
        E_analytic = hbar * omega_c * (n + 0.5)
        error = E_n - E_analytic
        print(f"{n:<4} {E_n:<15.6f} {E_analytic:<15.6f} {error:<12.6f}")
    
    print("\nNote: Exact agreement with Landau levels requires:")
    print("  1. 2D geometry (thin Lz limit)")
    print("  2. No confinement potential or very weak confinement")
    print("  3. Sufficient mesh resolution")
    print("  4. Proper boundary conditions")
    
    return E, modes


def demo_aharonov_bohm():
    """
    Demonstrate Aharonov-Bohm effect: magnetic flux confined to a region,
    but vector potential nonzero outside where B=0.
    
    The wavefunction acquires a phase shift depending on the enclosed flux,
    even in regions where the magnetic field is zero.
    """
    print("\n\n" + "=" * 70)
    print("DEMO: Aharonov-Bohm Effect")
    print("=" * 70)
    print("\nThis demonstrates that the vector potential A has physical effects")
    print("even in regions where the magnetic field B = ∇×A = 0.")
    
    # Parameters
    hbar = 1.0
    m_eff = 1.0
    q = -1.0
    flux = 2.0 * np.pi * hbar / abs(q)  # One flux quantum
    r_solenoid = 0.3  # solenoid radius
    
    print(f"\nConfiguration:")
    print(f"  Magnetic flux Φ = {flux:.4f} (one flux quantum)")
    print(f"  Solenoid radius = {r_solenoid}")
    
    # Vector potential: A_θ = Φ/(2πr) for r > r_solenoid
    def A_solenoid(X):
        r = np.sqrt(X[0]**2 + X[1]**2)
        theta = np.arctan2(X[1], X[0])
        
        A = np.zeros((3, X.shape[1]))
        mask = r > r_solenoid
        
        # A_θ = Φ/(2πr) in cylindrical coords
        # Convert to Cartesian: A_x = -A_θ sin(θ), A_y = A_θ cos(θ)
        A_theta = np.zeros_like(r)
        A_theta[mask] = flux / (2 * np.pi * r[mask])
        
        A[0, :] = -A_theta * np.sin(theta)
        A[1, :] =  A_theta * np.cos(theta)
        A[2, :] = 0.0
        
        return A
    
    # Mesh
    print("\nCreating mesh...")
    L = 4.0
    mesh = solver.make_mesh_box(
        x0=(-L/2, -L/2, -0.5), 
        lengths=(L, L, 1.0), 
        char_length=0.3
    )
    mesh, basis, K, M = solver.assemble_operators(mesh)
    print(f"Mesh: {basis.N} DOFs")
    
    # External potential: radial confinement to keep particle in annulus
    def V_ext(X):
        r = np.sqrt(X[0]**2 + X[1]**2)
        # Soft walls at inner and outer radii
        V = np.zeros_like(r)
        V += 10.0 * np.exp(-10.0 * (r - r_solenoid))  # repel from solenoid
        V += 0.5 * (r - 1.0)**2  # harmonic around r=1
        return V
    
    # Solve
    nev = 6
    E, modes = solve_schrodinger_em(
        mesh, basis, K, M, V_ext,
        A_func=A_solenoid, q=q, hbar=hbar, mass_eff=m_eff,
        nev=nev
    )
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'State':<6} {'Energy':<15} {'Phase (deg)':<15}")
    print("-" * 70)
    for n in range(len(E)):
        E_n = E[n].real
        # Extract phase at a sample point (if wavefunction has definite phase)
        psi_sample = modes[basis.N // 2, n]
        phase = np.angle(psi_sample) * 180 / np.pi
        print(f"{n:<6} {E_n:<15.6f} {phase:<15.2f}")
    
    print("\nNote: The Aharonov-Bohm phase shift δφ = qΦ/(ℏ) manifests as")
    print("energy level shifts and splitting in a confined geometry.")
    print(f"For one flux quantum, δφ = 2π, giving gauge-invariant observable effects.")
    
    return E, modes


if __name__ == "__main__":
    print("Electromagnetic Field Treatment in Schrödinger Equation")
    print("=" * 70)
    print("\nThis script demonstrates how to incorporate magnetic vector potentials")
    print("into the electron Hamiltonian using the minimal coupling prescription.")
    print("\nImplements: H = (1/2m)|p - qA|² + V")
    print("            = (ℏ²/2m)∇² + (iqℏ/m)A·∇ + (q²/2m)|A|² + V")
    
    # Run demonstrations
    try:
        E_landau, modes_landau = demo_landau_levels()
        E_ab, modes_ab = demo_aharonov_bohm()
        
        print("\n" + "=" * 70)
        print("DEMOS COMPLETED")
        print("=" * 70)
        print("\nFor integration into main solver, see docs/ELECTROMAGNETIC_FIELDS.md")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
