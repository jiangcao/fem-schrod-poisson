# Electromagnetic Fields in the Electron Hamiltonian

## Overview

This document describes how to incorporate electromagnetic fields into the electron Hamiltonian in the Schrödinger equation, specifically treating the magnetic vector potential **A** and electric scalar potential φ.

## Theory

### Minimal Coupling (Gauge-Covariant Derivative)

In the presence of electromagnetic fields, the momentum operator is modified through **minimal coupling** (also called the Peierls substitution):

**p̂** → **p̂** - q**A**

where:
- **p̂** = -iℏ∇ is the canonical momentum operator
- q is the particle charge (q = -e for electrons, where e > 0)
- **A**(**r**) is the magnetic vector potential

### Hamiltonian with Electromagnetic Fields

The full single-particle Hamiltonian becomes:

```
Ĥ = (1/2m) |p̂ - q𝐀|² + qφ + V_ext
```

Expanding the kinetic term:

```
Ĥ = (1/2m)(-iℏ∇ - q𝐀)·(-iℏ∇ - q𝐀) + qφ + V_ext
  = (ℏ²/2m)∇² + (iqℏ/m)(𝐀·∇ + ∇·𝐀) + (q²/2m)|𝐀|² + qφ + V_ext
```

In the **Coulomb gauge** (∇·**A** = 0), this simplifies to:

```
Ĥ = (ℏ²/2m)∇² + (iqℏ/m)𝐀·∇ + (q²/2m)|𝐀|² + qφ + V_ext
```

### With Spatially Varying Effective Mass

For semiconductors with spatially varying effective mass m_eff(**r**), the kinetic operator becomes:

```
T̂ = -∇·[(ℏ²/2m_eff(𝐫))(∇ - (iq/ℏ)𝐀)]
```

This can be expanded to:

```
T̂ = -∇·[(ℏ²/2m_eff)∇ψ] + ∇·[(iqℏ/2m_eff)𝐀ψ]
```

More explicitly:

```
T̂ = -(ℏ²/2)∇·[(1/m_eff)∇ψ] 
    + (iqℏ/2)∇·[(1/m_eff)𝐀ψ]
```

Expanding the second term:

```
∇·[(1/m_eff)𝐀ψ] = (1/m_eff)[(∇·𝐀)ψ + 𝐀·∇ψ] + ∇(1/m_eff)·𝐀ψ
```

In Coulomb gauge (∇·**A** = 0):

```
T̂ = -(ℏ²/2)∇·[(1/m_eff)∇ψ] 
    + (iqℏ/2m_eff)𝐀·∇ψ 
    + (iqℏ/2)∇(1/m_eff)·𝐀ψ
```

The **diamagnetic term** (quadratic in **A**) is typically negligible for weak fields but becomes:

```
V_dia = (q²/2m_eff)|𝐀|²
```

### Physical Examples

1. **Uniform magnetic field B = B_z ẑ**: Use symmetric gauge
   - **A** = (-B_z y/2, B_z x/2, 0)
   - Leads to Landau levels

2. **Solenoid/Aharonov-Bohm**: Magnetic flux confined to a region
   - **A** ≠ 0 even where **B** = 0
   - Demonstrates gauge-dependent quantum phase

3. **Spin-orbit coupling**: Can be represented as an effective vector potential
   - Rashba coupling, Dresselhaus coupling in semiconductors

## Implementation Strategy

### 1. Data Structure for Vector Potential

Define **A**(**r**) as a callable or array:

```python
# Option 1: Callable returning (3, npts) array
def vector_potential(X):
    """
    X: (3, npts) array of coordinates
    Returns: (3, npts) array [Ax, Ay, Az] at each point
    """
    Ax = -0.5 * B_z * X[1, :]  # -B_z * y / 2
    Ay =  0.5 * B_z * X[0, :]  #  B_z * x / 2
    Az = np.zeros_like(X[0, :])
    return np.array([Ax, Ay, Az])

# Option 2: Array at DOFs - shape (ndofs, 3)
A_dofs = np.zeros((basis.N, 3))
A_dofs[:, 0] = -0.5 * B_z * basis.doflocs[1, :]
A_dofs[:, 1] =  0.5 * B_z * basis.doflocs[0, :]
```

### 2. Modify the Kinetic Operator

The weak form of the Schrödinger equation becomes:

```
∫ (ℏ²/2m_eff)(∇ψ)·(∇φ) dV 
  + ∫ (iqℏ/2m_eff)[𝐀·∇ψ φ - ψ 𝐀·∇φ] dV 
  + ∫ [(q²/2m_eff)|𝐀|² + V]ψφ dV 
  = E ∫ ψφ dV
```

This requires:
1. **Paramagnetic term** (linear in **A**): complex-valued, breaks Hermitian symmetry
2. **Diamagnetic term** (quadratic in **A**): real-valued, adds to potential

### 3. Complex Wavefunctions

The vector potential introduces **complex phases** in the wavefunction. The solver must be generalized to handle complex arithmetic:

- Use `np.complex128` for wavefunctions
- Use complex sparse matrices
- Eigenvalue solver: `scipy.sparse.linalg.eigs` (for non-Hermitian) or ensure proper symmetrization

### 4. Gauge Considerations

- **Coulomb gauge** (∇·**A** = 0): Simplifies divergence terms
- **Symmetric gauge** (for uniform B): **A** = **B** × **r**/2
- **Landau gauge**: **A** = (0, B_z x, 0) or **A** = (-B_z y, 0, 0)

The physical results must be gauge-invariant, but numerical implementations may differ in efficiency.

## Proposed Code Structure

### Addition to PhysicalParams

```python
@dataclass
class PhysicalParams:
    hbar: float = 1.0
    m_eff: float = 1.0
    q: float = -1.0
    epsilon0: float = 1.0
    n_particles: float = 1.0
    # NEW:
    vector_potential: callable | np.ndarray | None = None  # A(r)
    use_complex: bool = False  # Enable complex wavefunctions
```

### New Function: Assemble EM Operators

```python
def assemble_em_operators(basis, phys: PhysicalParams, mass_eff=None):
    """
    Assemble paramagnetic and diamagnetic operators for EM fields.
    
    Returns:
    --------
    K_para : complex sparse matrix
        Paramagnetic term (linear in A, imaginary)
    V_dia : real array
        Diamagnetic potential (quadratic in A)
    """
    if phys.vector_potential is None:
        return None, np.zeros(basis.N)
    
    # Evaluate A at quadrature points or DOFs
    # ... (implementation details)
    
    # Paramagnetic: (iqℏ/2m_eff)[A·∇ψ φ - ψ A·∇φ]
    # Diamagnetic: (q²/2m_eff)|A|²
    
    return K_para, V_dia
```

### Modified solve_generalized_eig

The function needs to accept electromagnetic parameters and handle complex matrices:

```python
def solve_generalized_eig_em(
    K, M, V, nev=4, 
    phys: PhysicalParams | None = None,
    mass_eff=None,
    ...
):
    """
    Extended to handle electromagnetic fields with complex wavefunctions.
    """
    if phys and phys.vector_potential is not None:
        # Build EM operators
        K_para, V_dia = assemble_em_operators(basis, phys, mass_eff)
        
        # Full Hamiltonian: H = K_eff + K_para + V_M + V_dia_M
        H = K_eff + K_para + sp.diags(V + V_dia)
        
        # Use complex eigenvalue solver
        E, modes = spla.eigs(H, k=nev, M=M, which='SR')
        # Sort by real part
        idx = np.argsort(E.real)
        return E[idx].real, modes[:, idx]
    else:
        # Existing real-valued path
        ...
```

## Example Usage

### Uniform Magnetic Field (Landau Levels)

```python
import numpy as np
from src import solver

# Physical parameters
B_z = 0.5  # Magnetic field in z-direction
phys = solver.PhysicalParams(
    hbar=1.0,
    m_eff=1.0,
    q=-1.0,
    use_complex=True
)

# Symmetric gauge: A = B × r / 2
def A_symmetric(X):
    Ax = -0.5 * B_z * X[1, :]
    Ay =  0.5 * B_z * X[0, :]
    Az = np.zeros_like(X[0, :])
    return np.array([Ax, Ay, Az])

phys.vector_potential = A_symmetric

# Solve
mesh = solver.make_mesh_box(x0=(-2,-2,-2), lengths=(4,4,4), char_length=0.3)
mesh, basis, K, M = solver.assemble_operators(mesh)
Vext = lambda X: 0.5 * (X[0]**2 + X[1]**2)  # Harmonic confinement in x-y

E, modes = solver.solve_generalized_eig_em(
    K, M, Vext, nev=10, phys=phys, basis=basis
)

# Expected: Landau levels E_n = ℏω_c(n + 1/2) where ω_c = |q|B/m
omega_c = abs(phys.q) * B_z / phys.m_eff
print(f"Cyclotron frequency: {omega_c}")
print(f"Ground state energy: {E[0]} (expected ~{0.5*omega_c*phys.hbar})")
```

### Aharonov-Bohm Effect

```python
def A_solenoid(X, radius=0.5, flux=1.0):
    """
    Vector potential for flux Φ confined to cylinder of given radius.
    Outside: A_θ = Φ/(2πr)
    """
    r = np.sqrt(X[0]**2 + X[1]**2)
    theta = np.arctan2(X[1], X[0])
    
    A = np.zeros((3, X.shape[1]))
    mask = r > radius
    
    # A_θ = Φ/(2πr) → A_x = -A_θ sin(θ), A_y = A_θ cos(θ)
    A_theta = flux / (2 * np.pi * r[mask])
    A[0, mask] = -A_theta * np.sin(theta[mask])
    A[1, mask] =  A_theta * np.cos(theta[mask])
    
    return A

phys.vector_potential = lambda X: A_solenoid(X, radius=0.5, flux=2*np.pi)
```

## References

1. **Gauge-invariant formulation**: See Nenciu (1991), "Dynamics of band electrons in electric and magnetic fields"
2. **Finite element implementation**: Bao et al. (2013), "Numerical methods for the nonlinear Schrödinger equation with nonzero far-field conditions"
3. **Landau levels in FEM**: See quantum Hall effect simulations
4. **Effective mass + EM**: BenDaniel-Duke boundary conditions extended to magnetic fields

## Next Steps

1. Implement `assemble_em_operators` function
2. Add complex number support throughout solver
3. Create test case for uniform magnetic field (Landau levels)
4. Validate against analytical solutions
5. Extend visualization to handle complex wavefunctions (plot |ψ|², phase)
6. Consider time-dependent extensions (TDSE with EM fields)

## Additional Considerations

### Numerical Stability

- Complex arithmetic can amplify numerical errors
- May need higher precision near field singularities
- Gauge singularities (e.g., along solenoid axis) require careful mesh design

### Performance

- Complex matrices double memory requirements
- Consider using symmetries (e.g., angular momentum for cylindrical symmetry)
- For weak fields, perturbation theory may be more efficient

### Self-Consistent EM

For very strong currents, include magnetic field from the electron current:
- **B** = ∇ × **A** determined by **j** = Re[ψ*(ℏ∇ - q**A**)ψ/m]
- Requires solving coupled Maxwell-Schrödinger equations
