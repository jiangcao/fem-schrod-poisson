# Electromagnetic Fields: Physical Applications & Examples

## Overview

This document provides practical examples and physical applications of the electromagnetic field implementation in the Schrödinger-Poisson solver.

## Table of Contents

1. [Landau Levels and Quantum Hall Effect](#landau-levels)
2. [Aharonov-Bohm Effect](#aharonov-bohm)
3. [Quantum Dots in Magnetic Fields](#quantum-dots)
4. [Flux Quantization in Rings](#flux-quantization)
5. [Spin-Orbit Coupling (Effective Vector Potential)](#spin-orbit)
6. [Superconducting Vortices](#superconducting)

---

## 1. Landau Levels and Quantum Hall Effect {#landau-levels}

### Physical Setup

- **System**: 2D electron gas in perpendicular magnetic field B = B_z ẑ
- **Hamiltonian**: H = (1/2m)(p - qA)² + V_conf
- **Key Physics**: Cyclotron motion → discrete energy levels

### Theory

**Cyclotron frequency**: ω_c = |q|B/m  
**Landau levels**: E_n = ℏω_c(n + 1/2), n = 0, 1, 2, ...  
**Degeneracy**: Each level has degeneracy ~ Φ/Φ₀ where Φ₀ = h/|q|

**Magnetic length**: ℓ_B = √(ℏ/|q|B) (size of Landau orbital)

### Implementation

```python
import numpy as np
from src import solver
from examples.demo_electromagnetic import (
    solve_schrodinger_em,
    vector_potential_uniform_field
)

# Physical parameters
hbar = 1.0
m_eff = 0.067  # GaAs effective mass (in units of m_e)
q = -1.0
B_z = 5.0  # Tesla (scaled)

omega_c = abs(q) * B_z / m_eff
l_B = np.sqrt(hbar / (abs(q) * B_z))

print(f"Cyclotron frequency: {omega_c:.3f}")
print(f"Magnetic length: {l_B:.3f} nm")

# Create 2D-like mesh (thin slab)
Lxy = 10 * l_B  # Large enough for several orbitals
Lz = 0.5 * l_B  # Thin in z direction
mesh = solver.make_mesh_box(
    x0=(-Lxy/2, -Lxy/2, -Lz/2),
    lengths=(Lxy, Lxy, Lz),
    char_length=l_B/3  # Resolve magnetic length
)
mesh, basis, K, M = solver.assemble_operators(mesh)

# Vector potential (symmetric gauge)
A_func = vector_potential_uniform_field(B_z, gauge='symmetric')

# Weak harmonic confinement (optional, to localize states)
omega_conf = 0.05 * omega_c
def V_conf(X):
    return 0.5 * m_eff * omega_conf**2 * (X[0]**2 + X[1]**2)

# Solve
E, modes = solve_schrodinger_em(
    mesh, basis, K, M, V_conf,
    A_func=A_func, q=q, hbar=hbar, mass_eff=m_eff,
    nev=10
)

# Analyze Landau level structure
print("\nLandau Levels:")
print("n   E_numerical    E_theory      Error")
for n in range(len(E)):
    E_theory = hbar * omega_c * (n + 0.5)
    error = E[n].real - E_theory
    print(f"{n}   {E[n].real:12.6f}   {E_theory:12.6f}   {error:+.6f}")

# Quantum Hall conductivity
# σ_xy = (n_filled * q²) / h
n_filled = 3  # Number of filled Landau levels
sigma_xy = n_filled * q**2  # In dimensionless units
print(f"\nHall conductivity (n={n_filled}): σ_xy = {sigma_xy}")
```

### Expected Results

- Energy spacing ΔE ≈ ℏω_c between levels
- Ground state E₀ ≈ ℏω_c/2
- Wavefunction localized on scale ℓ_B
- Quantum Hall plateaus at integer filling

### Experimental Connection

- **GaAs/AlGaAs**: m* ≈ 0.067 m_e, B = 1-10 T → ω_c ~ 1-10 meV
- **Graphene**: Linear dispersion → relativistic Landau levels E_n ∝ √(nB)
- **Quantum Hall effect**: Integer and fractional (many-body) plateaus

---

## 2. Aharonov-Bohm Effect {#aharonov-bohm}

### Physical Setup

- **System**: Particle encircling region with magnetic flux Φ
- **Key Physics**: Vector potential A ≠ 0 even where B = 0
- **Observable**: Interference pattern shift by phase δφ = qΦ/ℏ

### Theory

**Phase shift**: Wavefunction acquires phase  
```
ψ → ψ exp[i(q/ℏ)∮A·dl] = ψ exp[iqΦ/ℏ]
```

**Flux quantum**: Φ₀ = 2πℏ/|q| (one period of phase)

**Gauge invariance**: Phase change is observable in interference

### Implementation

```python
def vector_potential_solenoid(Phi, r_sol=0.5):
    """
    Vector potential for flux Φ confined to solenoid radius r_sol.
    
    Outside solenoid (r > r_sol):
        A_θ = Φ/(2πr)
        B = 0
    
    Inside solenoid (r < r_sol):
        A_θ = Φr/(2πr_sol²)
        B = Φ/(πr_sol²) ẑ (uniform)
    """
    def A_func(X):
        r = np.sqrt(X[0]**2 + X[1]**2)
        theta = np.arctan2(X[1], X[0])
        
        A = np.zeros((3, X.shape[1]))
        
        # Outside solenoid
        mask_out = r > r_sol
        A_theta_out = Phi / (2 * np.pi * r[mask_out])
        
        # Inside solenoid
        mask_in = r <= r_sol
        A_theta_in = Phi * r[mask_in] / (2 * np.pi * r_sol**2)
        
        # Convert to Cartesian
        A_theta = np.zeros_like(r)
        A_theta[mask_out] = A_theta_out
        A_theta[mask_in] = A_theta_in
        
        A[0, :] = -A_theta * np.sin(theta)  # A_x
        A[1, :] =  A_theta * np.cos(theta)  # A_y
        A[2, :] = 0.0
        
        return A
    
    return A_func

# Setup
hbar = 1.0
m_eff = 1.0
q = -1.0
Phi = 2 * np.pi * hbar / abs(q)  # One flux quantum
r_sol = 0.3

# Mesh: annular region around solenoid
mesh = solver.make_mesh_box(
    x0=(-3, -3, -0.5), lengths=(6, 6, 1), char_length=0.3
)
mesh, basis, K, M = solver.assemble_operators(mesh)

A_func = vector_potential_solenoid(Phi, r_sol)

# Potential: keep particle in annulus
def V_guide(X):
    r = np.sqrt(X[0]**2 + X[1]**2)
    V = 10.0 * np.exp(-5.0*(r - r_sol))  # Repel from solenoid
    V += 0.1 * (r - 1.5)**2  # Confine around r=1.5
    return V

# Solve
E, modes = solve_schrodinger_em(
    mesh, basis, K, M, V_guide,
    A_func=A_func, q=q, hbar=hbar, mass_eff=m_eff,
    nev=8
)

# Analyze phase
print("\nAharonov-Bohm Effect:")
print("State   Energy        Phase (rad)   Phase/2π")
for n in range(len(E)):
    # Sample wavefunction at a reference point
    idx_sample = basis.N // 2
    psi_sample = modes[idx_sample, n]
    phase = np.angle(psi_sample)
    print(f"{n}      {E[n].real:10.6f}   {phase:+8.4f}      {phase/(2*np.pi):+6.3f}")

# Expected phase shift per flux quantum
delta_phi = q * Phi / hbar
print(f"\nExpected phase shift: δφ = qΦ/ℏ = {delta_phi:.4f} rad = {delta_phi/(2*np.pi):.4f} × 2π")
```

### Expected Results

- Energy levels split/shift with flux
- Phase of ψ changes by 2π per flux quantum
- Observable in interference experiments (ring geometry)

### Experimental Connection

- **Metallic rings**: Persistent currents, conductance oscillations
- **Superconducting loops**: SQUID magnetometry
- **Topological phases**: Berry phase, Aharonov-Casher effect

---

## 3. Quantum Dots in Magnetic Fields {#quantum-dots}

### Physical Setup

- **System**: Electron confined in 2D parabolic potential + magnetic field
- **Hamiltonian**: Fock-Darwin Hamiltonian
- **Key Physics**: Competition between confinement and magnetic field

### Theory

**Fock-Darwin spectrum**:
```
E_{n,l} = ℏΩ(2n + |l| + 1) + (ℏω₀²l)/(2Ω)
```
where Ω = √(ω₀² + ω_c²/4) and l is angular momentum quantum number

**Crossover**: 
- Low B: ω_c < ω₀ → confinement dominates
- High B: ω_c > ω₀ → magnetic field dominates (Landau-like)

### Implementation

```python
# Physical parameters
hbar = 1.0
m_eff = 1.0
q = -1.0
omega_0 = 1.0  # Confinement frequency
B_z = 2.0

omega_c = abs(q) * B_z / m_eff
Omega = np.sqrt(omega_0**2 + omega_c**2 / 4)

print(f"Confinement ω₀ = {omega_0:.3f}")
print(f"Cyclotron ω_c = {omega_c:.3f}")
print(f"Effective Ω = {Omega:.3f}")

# Mesh
mesh = solver.make_mesh_box(
    x0=(-3, -3, -0.5), lengths=(6, 6, 1), char_length=0.4
)
mesh, basis, K, M = solver.assemble_operators(mesh)

# Vector potential
A_func = vector_potential_uniform_field(B_z, gauge='symmetric')

# Parabolic confinement
def V_dot(X):
    r2 = X[0]**2 + X[1]**2
    return 0.5 * m_eff * omega_0**2 * r2

# Solve
E, modes = solve_schrodinger_em(
    mesh, basis, K, M, V_dot,
    A_func=A_func, q=q, hbar=hbar, mass_eff=m_eff,
    nev=12
)

# Fock-Darwin levels (analytical)
def fock_darwin(n, l, omega_0, omega_c, hbar):
    Omega = np.sqrt(omega_0**2 + omega_c**2 / 4)
    return hbar * Omega * (2*n + abs(l) + 1) + (hbar * omega_0**2 * l) / (2 * Omega)

print("\nFock-Darwin Spectrum:")
print("State   E_num      (n,l)   E_theory   Error")
# Assign quantum numbers (requires angular momentum analysis)
quantum_numbers = [(0,0), (0,1), (0,-1), (1,0), (0,2), (0,-2)]
for i, (n, l) in enumerate(quantum_numbers[:len(E)]):
    E_theory = fock_darwin(n, l, omega_0, omega_c, hbar)
    print(f"{i}      {E[i].real:8.4f}   ({n},{l:+2d})   {E_theory:8.4f}   {E[i].real-E_theory:+.4f}")

# Visualize density (|ψ|²) for ground state
density = np.abs(modes[:, 0])**2
# ... plot using visualization tools ...
```

### Expected Results

- Ground state: (n=0, l=0), E ≈ ℏΩ
- Excited states: Shell structure with angular momentum
- B→0: Harmonic oscillator levels ℏω₀(2n + |l| + 1)
- B→∞: Landau levels ℏω_c(n + 1/2)

### Experimental Connection

- **GaAs quantum dots**: Few-electron systems, Coulomb interactions
- **Carbon nanotubes**: 1D confinement + orbital magnetic moment
- **Graphene quantum dots**: Valley degeneracy, edge states

---

## 4. Flux Quantization in Rings {#flux-quantization}

### Physical Setup

- **System**: Particle on a ring (1D) threaded by magnetic flux
- **Boundary condition**: ψ(θ + 2π) = ψ(θ)
- **Key Physics**: Flux changes angular momentum quantization

### Theory

**Eigenstates**: ψ_l(θ) = exp(ilθ)/√(2π)

**Energy levels**:
```
E_l = (ℏ²/2mR²)(l - Φ/Φ₀)²
```
where l = 0, ±1, ±2, ... and Φ₀ = 2πℏ/|q|

**Persistent current**: I = -∂E₀/∂Φ (oscillates with period Φ₀)

### Implementation (3D approximation)

```python
def vector_potential_axial_flux(Phi, R_ring=1.0):
    """
    Vector potential for flux through ring (approximate in 3D).
    For ring at r=R_ring, flux Φ along z-axis.
    """
    def A_func(X):
        r = np.sqrt(X[0]**2 + X[1]**2)
        theta = np.arctan2(X[1], X[0])
        
        # A_θ = Φ/(2πr) (outside flux tube)
        A_theta = Phi / (2 * np.pi * r + 1e-10)
        
        A = np.zeros((3, X.shape[1]))
        A[0, :] = -A_theta * np.sin(theta)
        A[1, :] =  A_theta * np.cos(theta)
        
        return A
    
    return A_func

# Parameters
hbar = 1.0
m_eff = 1.0
q = -1.0
R_ring = 1.0
Phi_values = np.linspace(0, 2*np.pi*hbar/abs(q), 11)  # 0 to 1 flux quantum

E_ground = []

for Phi in Phi_values:
    # Create ring geometry (torus-like mesh would be better)
    mesh = solver.make_mesh_box(
        x0=(-2, -2, -0.5), lengths=(4, 4, 1), char_length=0.3
    )
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    A_func = vector_potential_axial_flux(Phi, R_ring)
    
    # Confine to ring
    def V_ring(X):
        r = np.sqrt(X[0]**2 + X[1]**2)
        return 10.0 * (r - R_ring)**2  # Harmonic around r=R_ring
    
    E, modes = solve_schrodinger_em(
        mesh, basis, K, M, V_ring,
        A_func=A_func, q=q, hbar=hbar, mass_eff=m_eff,
        nev=3
    )
    
    E_ground.append(E[0].real)

# Plot E_0(Φ) - should be parabolic with min at Φ=0, period Φ₀
import matplotlib.pyplot as plt
plt.figure()
plt.plot(Phi_values / (2*np.pi*hbar/abs(q)), E_ground, 'o-')
plt.xlabel('Φ/Φ₀')
plt.ylabel('E₀')
plt.title('Ground State Energy vs Flux')
plt.grid()
plt.savefig('flux_quantization.png')

# Persistent current
I_persistent = -np.gradient(E_ground, Phi_values)
plt.figure()
plt.plot(Phi_values / (2*np.pi*hbar/abs(q)), I_persistent, 'o-')
plt.xlabel('Φ/Φ₀')
plt.ylabel('I (persistent current)')
plt.title('Persistent Current vs Flux')
plt.grid()
plt.savefig('persistent_current.png')
```

### Expected Results

- E₀(Φ) oscillates with period Φ₀
- Minimum at Φ = nΦ₀ (integer n)
- Persistent current I ∝ sin(2πΦ/Φ₀)

---

## 5. Spin-Orbit Coupling (Effective A) {#spin-orbit}

### Physical Setup

- **System**: Electron in 2D with Rashba spin-orbit coupling
- **Effective Hamiltonian**: Can be mapped to vector potential
- **Key Physics**: Spin-momentum locking

### Theory

**Rashba Hamiltonian**:
```
H_R = α(σ_x p_y - σ_y p_x)
```
where α is Rashba coupling strength, σ are Pauli matrices

**Effective vector potential** (for one spin component):
```
A_eff,x = -mα σ_y / q
A_eff,y = +mα σ_x / q
```

### Implementation (Single Spin)

```python
# Rashba parameter
alpha = 0.5  # Coupling strength

# Effective vector potential (for spin-up state, σ_z = +1)
def A_rashba_up(X):
    """
    Effective A for spin-up with Rashba SOC.
    This is an approximation treating one spin sector.
    """
    A = np.zeros((3, X.shape[1]))
    # For Rashba: A_eff ~ α(y, -x, 0) × constant
    A[0, :] = -m_eff * alpha  # Constant in simple model
    A[1, :] = 0.0
    return A

# Note: Full treatment requires 2-component spinor wavefunction
# This is a simplified single-spin demonstration
```

**Better approach**: Solve coupled 2×2 Schrödinger equation (requires extension)

---

## 6. Superconducting Vortices {#superconducting}

### Physical Setup

- **System**: Superconductor in type-II phase with vortex
- **Order parameter**: ψ(r) → 0 at vortex core
- **Magnetic field**: B = Φ₀ δ(r) (flux quantum per vortex)

### Theory

**Ginzburg-Landau equations** (beyond current implementation):
```
(-iℏ∇ - q*A)²ψ = α|ψ|²ψ + β|ψ|⁴ψ
∇×B = μ₀j_s where j_s ∝ Re[ψ*(p-q*A)ψ]
```

**Single vortex vector potential**:
```
A_θ(r) = Φ₀/(2πr) for r > ξ (coherence length)
```

### Implementation (Linear Schrödinger approximation)

```python
def A_vortex(Phi0=2*np.pi, r_core=0.2):
    """
    Single vortex vector potential.
    """
    def A_func(X):
        r = np.sqrt(X[0]**2 + X[1]**2)
        theta = np.arctan2(X[1], X[0])
        
        # Regularize at core
        r_reg = np.maximum(r, r_core)
        A_theta = Phi0 / (2 * np.pi * r_reg)
        
        A = np.zeros((3, X.shape[1]))
        A[0, :] = -A_theta * np.sin(theta)
        A[1, :] =  A_theta * np.cos(theta)
        
        return A
    
    return A_func

# Solve with vortex potential
A_func = A_vortex(Phi0=2*np.pi*hbar/abs(q), r_core=0.3)

# Add potential to suppress |ψ| at vortex core
def V_core(X):
    r = np.sqrt(X[0]**2 + X[1]**2)
    return 100.0 * np.exp(-r**2 / (2*0.3**2))  # Large barrier at core

# Solve...
```

**Note**: Full treatment requires nonlinear Ginzburg-Landau (future work)

---

## Summary

| Application | Key Physics | Observable | Difficulty |
|------------|-------------|-----------|-----------|
| Landau Levels | Cyclotron motion | E_n = ℏω_c(n+1/2) | Medium |
| Aharonov-Bohm | Gauge phase | Phase shift δφ=qΦ/ℏ | Medium |
| Quantum Dots | Confinement + B | Fock-Darwin levels | Easy |
| Flux Rings | Periodic BC | E(Φ) oscillations | Hard |
| Spin-Orbit | Effective A | Spin texture | Hard* |
| Vortices | Nonlinear | Core structure | Hard* |

*Requires extensions beyond current linear implementation

---

## References

1. **Landau & Lifshitz**: Quantum Mechanics (Non-Relativistic Theory), Ch. XV
2. **Prange & Girvin**: The Quantum Hall Effect (comprehensive review)
3. **Aharonov & Bohm (1959)**: Significance of Electromagnetic Potentials in QM
4. **Fock (1928), Darwin (1930)**: Parabolic quantum dot in magnetic field
5. **Winkler (2003)**: Spin-orbit coupling effects in 2D electron systems
6. **Tinkham (2004)**: Introduction to Superconductivity (Ginzburg-Landau)

---

## Further Reading

- **Numerical Methods**: Bao & Cai (2013) on spectral methods for rotating BEC
- **Topology**: Nakahara (2003), Geometry, Topology and Physics
- **Experiments**: Reviews on quantum Hall, persistent currents, AB oscillations
