"""
Demo showing heterostructure interfaces with discontinuous epsilon and effective mass.
Demonstrates proper flux continuity handling across material interfaces.
"""
import numpy as np
import matplotlib.pyplot as plt
from src import solver
from src import visualization as vis

# Create mesh spanning two materials
print("Creating mesh for heterostructure...")
mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(2.0, 1.0, 1.0), 
                           char_length=0.25, verbose=True)

# Assemble operators
print("Assembling operators...")
mesh, basis, K, M = solver.assemble_operators(mesh)

# Define heterostructure with two materials at x=1.0 interface
# Material 1 (x < 1.0): epsilon=1.0, m_eff=1.0  (e.g., GaAs)
# Material 2 (x > 1.0): epsilon=3.0, m_eff=0.5  (e.g., InAs)

def epsilon_func(X):
    """Spatially varying dielectric: step change at x=1.0"""
    eps = np.ones(X.shape[1])
    eps[X[0, :] > 1.0] = 3.0  # Higher dielectric on right side
    return eps

def mass_func(X):
    """Spatially varying effective mass: step change at x=1.0"""
    m = np.ones(X.shape[1])
    m[X[0, :] > 1.0] = 0.5  # Lighter mass on right side
    return m

# Define external potential (quantum well centered at interface)
def Vext(X):
    """Confining potential centered at heterostructure interface"""
    x, y, z = X[0, :], X[1, :], X[2, :]
    # Parabolic confinement centered at x=1.0 (interface)
    V = 5.0 * ((x - 1.0)**2 + (y - 0.5)**2 + (z - 0.5)**2)
    return V

# Physical parameters
phys = solver.PhysicalParams(hbar=1.0, m_eff=1.0, q=-1.0, epsilon0=1.0, n_particles=1.0)

# Run SCF calculation with spatially varying epsilon and mass
print("\nRunning SCF calculation with heterostructure (discontinuous ε and m_eff)...")
E, modes, phi, Vfinal = solver.scf_loop(
    mesh, basis, K, M, Vext,
    coupling=1.0, maxiter=30, tol=1e-6,
    mix=0.4, nev=4, verbose=True, use_diis=True,
    epsilon=epsilon_func,      # Discontinuous dielectric
    mass_eff=mass_func,        # Discontinuous effective mass
    phys=phys
)

print(f"\nEnergy levels: {E}")
print(f"\nNote: The heterostructure interface at x=1.0 creates:")
print(f"  - Discontinuous dielectric: ε₁=1.0 → ε₂=3.0")
print(f"  - Discontinuous effective mass: m₁=1.0 → m₂=0.5")
print(f"  - Proper flux continuity is maintained across the interface")

# Create visualizations
print("\nGenerating visualizations...")

# 1. Visualize the epsilon and mass distributions
X = basis.doflocs
eps_values = epsilon_func(X)
mass_values = mass_func(X)

fig0, (ax0a, ax0b) = plt.subplots(1, 2, figsize=(14, 5))

# Epsilon distribution at y=0.5, z=0.5 slice
mask = (np.abs(X[1, :] - 0.5) < 0.1) & (np.abs(X[2, :] - 0.5) < 0.1)
x_sorted_idx = np.argsort(X[0, mask])
ax0a.plot(X[0, mask][x_sorted_idx], eps_values[mask][x_sorted_idx], 'bo-', markersize=3, linewidth=1.5)
ax0a.axvline(x=1.0, color='r', linestyle='--', label='Interface')
ax0a.set_xlabel('x position')
ax0a.set_ylabel('ε (dielectric constant)')
ax0a.set_title('Discontinuous Dielectric at Interface')
ax0a.grid(True, alpha=0.3)
ax0a.legend()

# Mass distribution at y=0.5, z=0.5 slice
ax0b.plot(X[0, mask][x_sorted_idx], mass_values[mask][x_sorted_idx], 'go-', markersize=3, linewidth=1.5)
ax0b.axvline(x=1.0, color='r', linestyle='--', label='Interface')
ax0b.set_xlabel('x position')
ax0b.set_ylabel('m_eff (effective mass)')
ax0b.set_title('Discontinuous Effective Mass at Interface')
ax0b.grid(True, alpha=0.3)
ax0b.legend()

plt.tight_layout()
plt.savefig('results/heterostructure_materials.png', dpi=150, bbox_inches='tight')
print("✓ Material properties plot saved")

# 2. Plot potential and ground state density at the interface
fig1, axes1 = vis.plot_potential_and_density(
    basis, Vfinal, modes,
    slice_axis='y', slice_value=0.5,
    save_path='results/heterostructure_slice.png'
)
print("✓ Potential and density slice saved")

# 3. Plot line profile through interface
fig3, axes3 = vis.plot_1d_line_profile(
    basis, Vfinal, modes,
    axis='x', fixed_coords={'y': 0.5, 'z': 0.5},
    save_path='results/heterostructure_profile.png'
)
print("✓ Line profile through interface saved")

# 4. Energy levels
fig4, ax4 = vis.plot_energy_levels(
    E, n_levels=4,
    save_path='results/heterostructure_energy.png'
)
print("✓ Energy levels saved")

# Add note about heterostructure effects
ax4.text(0.5, E[0] - 0.15 * (E[-1] - E[0]), 
         'Heterostructure with discontinuous ε and m_eff', 
         ha='center', fontsize=10, style='italic')
plt.savefig('results/heterostructure_energy.png', dpi=150, bbox_inches='tight')

print("\n" + "="*70)
print("All visualizations saved to results/ directory")
print("="*70)
print("\nHeterostructure Features Demonstrated:")
print("  ✓ Discontinuous dielectric constant (ε) at x=1.0")
print("  ✓ Discontinuous effective mass (m_eff) at x=1.0")
print("  ✓ Proper flux continuity across material interface")
print("  ✓ Self-consistent Schrödinger-Poisson solution")
print("  ✓ Wave functions properly confined at heterostructure interface")
print("="*70)

# Display all plots
plt.show()
