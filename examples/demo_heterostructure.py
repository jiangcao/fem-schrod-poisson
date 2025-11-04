"""
Demo showing visualization of results with spatially varying epsilon.
"""
import numpy as np
import matplotlib.pyplot as plt
from src import solver
from src import visualization as vis

# Create mesh
print("Creating mesh...")
mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                           char_length=0.3, verbose=True)

# Assemble operators
print("Assembling operators...")
mesh, basis, K, M = solver.assemble_operators(mesh)

# Define spatially varying epsilon (dielectric constant)
# Higher epsilon on one side, simulating a heterostructure
def epsilon_func(X):
    """Spatially varying dielectric: higher on x > 0.5"""
    eps = np.ones(X.shape[1])
    eps[X[0, :] > 0.5] = 3.0  # Higher dielectric on right side
    return eps

# Define external potential (well structure)
def Vext(X):
    """Confining potential"""
    x, y, z = X[0, :], X[1, :], X[2, :]
    # Parabolic confinement
    V = 5.0 * ((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2)
    return V

# Run SCF calculation with spatially varying epsilon
print("Running SCF calculation with spatially varying epsilon...")
E, modes, phi, Vfinal = solver.scf_loop(
    mesh, basis, K, M, Vext,
    coupling=1.0, maxiter=30, tol=1e-6,
    mix=0.4, nev=4, verbose=True, use_diis=True,
    epsilon=epsilon_func  # Pass epsilon to SCF loop
)

print(f"\nEnergy levels: {E}")

# Create visualizations
print("\nGenerating visualizations...")

# 1. Visualize the epsilon distribution
X = basis.doflocs
eps_values = epsilon_func(X)

fig0, ax0 = plt.subplots(figsize=(10, 4))
mask = np.abs(X[2, :] - 0.5) < 0.1  # z=0.5 slice
sc = ax0.scatter(X[0, mask], X[1, mask], c=eps_values[mask], 
                cmap='RdYlBu_r', s=30, edgecolors='none')
ax0.set_xlabel('x')
ax0.set_ylabel('y')
ax0.set_title('Spatially Varying Epsilon (dielectric constant) at z=0.5')
ax0.set_aspect('equal')
plt.colorbar(sc, ax=ax0, label='ε')
plt.tight_layout()
plt.savefig('results/epsilon_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Epsilon distribution plot saved")

# 2. Plot potential and ground state density
fig1, axes1 = vis.plot_potential_and_density(
    basis, Vfinal, modes,
    slice_axis='z', slice_value=0.5,
    save_path='results/heterostructure_slice.png'
)
print("✓ Potential and density slice saved")

# 3. Multiple slices to see 3D structure
fig2, axes2 = vis.plot_multiple_slices(
    basis, Vfinal, modes,
    slice_axis='x', slice_positions=[0.3, 0.5, 0.7],
    save_path='results/heterostructure_x_slices.png'
)
print("✓ Multiple x-slices saved")

# 4. Line profiles through center
fig3, axes3 = vis.plot_1d_line_profile(
    basis, Vfinal, modes,
    axis='x', fixed_coords={'y': 0.5, 'z': 0.5},
    save_path='results/heterostructure_profile.png'
)
print("✓ Line profile saved")

# 5. Energy levels
fig4, ax4 = vis.plot_energy_levels(
    E, n_levels=4,
    save_path='results/heterostructure_energy.png'
)
print("✓ Energy levels saved")

# Add note about epsilon effect
ax4.text(0.5, E[0] - 0.15 * (E[-1] - E[0]), 
         'With spatially varying ε', 
         ha='center', fontsize=10, style='italic')
plt.savefig('results/heterostructure_energy.png', dpi=150, bbox_inches='tight')

print("\nAll visualizations saved to results/ directory")
print("\nNote: The spatially varying epsilon affects the effective potential")
print("and confines the wavefunction differently in different regions.")

# Display all plots
plt.show()
