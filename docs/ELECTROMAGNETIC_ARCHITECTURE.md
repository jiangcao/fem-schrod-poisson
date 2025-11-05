# Electromagnetic Fields Implementation Architecture

## Conceptual Flow

```
User Input
    │
    ├─ Magnetic Field B
    ├─ Physical Parameters (ℏ, m, q)
    └─ Geometry/Mesh
    │
    ▼
Vector Potential A(r)
    │
    ├─ Uniform Field: A = B×r/2 (symmetric gauge)
    ├─ Solenoid: A_θ = Φ/(2πr)
    └─ Custom: User-defined function
    │
    ▼
Operator Assembly
    │
    ├─────────────────┬─────────────────┬─────────────────┐
    │                 │                 │                 │
    ▼                 ▼                 ▼                 ▼
Kinetic K      Paramagnetic K_para  Diamagnetic V_dia  Potential V
(ℏ²/2m)∇²      (iqℏ/m)A·∇           (q²/2m)|A|²        External
REAL           COMPLEX              REAL               REAL
Laplacian      Anti-Hermitian       Positive           Diagonal
    │                 │                 │                 │
    └─────────────────┴─────────────────┴─────────────────┘
                            │
                            ▼
                Full Hamiltonian H (complex)
                    H = K + K_para + diag(V + V_dia)
                            │
                            ▼
                Complex Eigenvalue Problem
                    H|ψ⟩ = E M|ψ⟩
                scipy.sparse.linalg.eigs
                            │
                            ▼
                Complex Wavefunctions ψ
                    │
                    ├─ Amplitude: |ψ|²
                    └─ Phase: arg(ψ)
                            │
                            ▼
                Physical Observables
                    ├─ Energy: E (real)
                    ├─ Density: ρ = |ψ|²
                    └─ Current: j = Re[ψ*(p-qA)ψ/m]
```

## Code Structure

```
src/solver.py (existing)
    │
    ├─ PhysicalParams (dataclass)
    │   ├─ hbar, m_eff, q, epsilon0
    │   └─ [NEW] vector_potential, use_complex
    │
    ├─ make_mesh_box()
    ├─ assemble_operators(mesh) → K, M
    ├─ solve_generalized_eig() → E, modes
    └─ scf_loop() → E, modes, phi, V

examples/demo_electromagnetic.py (new)
    │
    ├─ assemble_paramagnetic_operator(basis, A, q, ℏ, m)
    │   └─ Returns: K_para (complex sparse matrix)
    │
    ├─ compute_diamagnetic_potential(basis, A, q, m)
    │   └─ Returns: V_dia (real array at DOFs)
    │
    ├─ solve_schrodinger_em(mesh, basis, K, M, V, A, ...)
    │   ├─ Assembles: K_eff + K_para + diag(V + V_dia)
    │   └─ Solves: Complex eigenvalue problem
    │
    ├─ vector_potential_uniform_field(B, gauge)
    │   └─ Returns: A(X) callable
    │
    ├─ demo_landau_levels()
    │   └─ Shows: Landau quantization E_n = ℏω_c(n+1/2)
    │
    └─ demo_aharonov_bohm()
        └─ Shows: Gauge-dependent quantum phases

tests/test_electromagnetic.py (new)
    │
    ├─ test_paramagnetic_hermiticity()
    ├─ test_diamagnetic_positive()
    ├─ test_zero_field_recovery()
    ├─ test_cyclotron_frequency()
    └─ test_energy_gauge_invariance()
```

## Mathematical Structure

### Weak Form (Bilinear Form)

```
a(ψ, φ) = Kinetic + Paramagnetic + Potential

Kinetic:
    ∫ (ℏ²/2m) ∇ψ · ∇φ dV
    → Assembled as: c * K where K is Laplacian

Paramagnetic:
    ∫ (iqℏ/2m)[A·∇ψ φ - ψ A·∇φ] dV
    → Assembled using BilinearForm with complex dtype
    → Anti-Hermitian: K_para† = -K_para

Potential:
    ∫ (V + V_dia)ψφ dV
    where V_dia = (q²/2m)|A|²
    → Assembled as: diag(V + V_dia)
```

### Gauge Transformations

```
Under gauge transformation: A → A + ∇χ

Wavefunction transforms: ψ → ψ exp[i(q/ℏ)χ]

Physical observables unchanged:
    - Energy E (real part)
    - Density ρ = |ψ|²
    - Current j = Re[ψ*(p-qA)ψ/m]

Phase changes: arg(ψ) → arg(ψ) + qχ/ℏ
```

## Data Flow Example: Landau Levels

```
Input:
    B_z = 1.0 (uniform field in z)
    m_eff = 1.0
    q = -1.0
    ω_c = |q|B/m = 1.0

    │
    ▼

Vector Potential (symmetric gauge):
    A_x(x,y,z) = -B_z y/2 = -0.5y
    A_y(x,y,z) = +B_z x/2 = +0.5x
    A_z(x,y,z) = 0

    │
    ▼

Mesh: 531 DOFs (6×6×1 box)
    X = basis.doflocs  (3, 531)
    A_dofs = A_func(X)  (3, 531)

    │
    ▼

Operators:
    K: 531×531 sparse (ℏ²/2m)∇²
    K_para: 531×531 complex sparse
        K_para[i,j] = ∫ (iqℏ/2m)[A·∇φ_j φ_i - φ_j A·∇φ_i] dV
    V_dia: 531-vector
        V_dia[i] = (q²/2m)|A(x_i)|²

    │
    ▼

Hamiltonian: H = K + K_para + diag(V + V_dia)
    531×531 complex matrix

    │
    ▼

Eigensolve: scipy.sparse.linalg.eigs(H, k=8, M=M)
    E: [1.78, 3.15, 4.15, ...] (complex, but Im ≈ 0)
    modes: 531×8 complex array

    │
    ▼

Analyze:
    Expected: E_n = ℏω_c(n+1/2) = [0.5, 1.5, 2.5, ...]
    Actual: [1.78, 3.15, 4.15, ...] (shifted by confinement)
    Structure: ~ω_c spacing visible ✓
```

## Integration Points

To integrate into main `src/solver.py`:

### 1. Extend PhysicalParams
```python
@dataclass
class PhysicalParams:
    hbar: float = 1.0
    m_eff: float = 1.0
    q: float = -1.0
    epsilon0: float = 1.0
    n_particles: float = 1.0
    # NEW:
    vector_potential: callable | None = None
    use_complex: bool = False
```

### 2. New Function in solver.py
```python
def assemble_em_operators(basis, A_func, phys, mass_eff=None):
    """Assemble paramagnetic and diamagnetic operators."""
    K_para = assemble_paramagnetic_operator(...)
    V_dia = compute_diamagnetic_potential(...)
    return K_para, V_dia
```

### 3. Extend solve_generalized_eig
```python
def solve_generalized_eig(..., phys=None):
    if phys and phys.vector_potential:
        # EM path
        K_para, V_dia = assemble_em_operators(basis, phys.vector_potential, phys)
        H = K_eff + K_para + sp.diags(V + V_dia, dtype=np.complex128)
        E, modes = spla.eigs(H, k=nev, M=M.astype(complex), which='SR')
    else:
        # Standard path (existing code)
        H = K_eff + sp.diags(V)
        E, modes = spla.eigsh(H, k=nev, M=M, which='SA')
    return E, modes
```

### 4. Update scf_loop
```python
def scf_loop(..., phys=None):
    for iteration in range(maxiter):
        E, modes = solve_generalized_eig(..., phys=phys)
        # Handle complex modes:
        rho = phys.n_particles * abs(phys.q) * np.abs(modes[:, 0])**2
        phi = solve_poisson(mesh, basis, rho, ...)
        # ... convergence check ...
```

### 5. Update Visualization
```python
# visualization.py
def plot_complex_wavefunction(basis, mode):
    """Plot |ψ|² and phase separately."""
    density = np.abs(mode)**2
    phase = np.angle(mode)
    # ... plot both ...
```

## Performance Considerations

```
Operation              Real        Complex     Ratio
───────────────────────────────────────────────────
Memory (matrix)        8 bytes     16 bytes    2×
Memory (vector)        8 bytes     16 bytes    2×
Matrix-vector mult     ~O(nnz)     ~2×O(nnz)   2×
Eigensolve (eigsh)     Fast        N/A         -
Eigensolve (eigs)      N/A         Slower      ~1.5-3×
Total overhead                                  ~2-3×
```

**Optimization strategies:**
1. Use symmetric gauge when possible (better conditioning)
2. Sparse matrix operations (already used)
3. Limit to weak fields if diamagnetic negligible
4. Consider perturbation theory for small corrections

## Physical Parameter Ranges

```
Typical Condensed Matter (SI units):

ℏ = 1.055×10⁻³⁴ J·s
e = 1.602×10⁻¹⁹ C
m_e = 9.109×10⁻³¹ kg
m_eff ≈ 0.1-10 × m_e (semiconductors)

B: 0.01-10 Tesla (lab fields)
ω_c = eB/m_eff: 10⁹-10¹² rad/s
ℓ_B = √(ℏ/eB): 10-100 nm
```

```
Dimensionless Units (typical in code):

ℏ = 1
m = 1
q = -1
B: 0.1-10 (scaled)
ω_c = |q|B/m = B
ℓ_B = 1/√B

Length scale: ~10-100 nm → L = 1 (unit length)
Energy scale: ~1-100 meV → E = 1 (unit energy)
```

## Error Sources & Mitigation

```
Error Source               Impact              Mitigation
─────────────────────────────────────────────────────────────
Finite mesh resolution     Discretization      Refine mesh
Boundary effects           Level shifts        Larger domain
3D vs 2D (Landau)         Extra kinetic       Thin slab (Lz→0)
Confinement potential      Level shifts        Weak confinement
Gauge singularities        Numerical instability   Avoid in mesh
Complex arithmetic         Rounding errors     Higher precision
Non-Hermitian solver       Convergence         More eigenvectors
```

## Validation Checklist

✓ Paramagnetic operator anti-Hermitian  
✓ Diamagnetic potential positive  
✓ A=0 recovers field-free case  
✓ Landau level structure visible  
✓ Ground state energy scales with B  
✓ Gauge transformations preserve energy  
✓ Aharonov-Bohm phase effects present  
✓ Complex wavefunctions normalized  
✓ Current conservation (advanced)  

## Summary

The implementation follows a clear architecture:

1. **Input**: Magnetic field B → Vector potential A(r)
2. **Assembly**: Build complex operators K_para, V_dia
3. **Solve**: Complex eigenvalue problem H|ψ⟩ = E|ψ⟩
4. **Output**: Complex wavefunctions with physical phases

Key innovation: Proper treatment of gauge-dependent phases while maintaining gauge-invariant observables.
