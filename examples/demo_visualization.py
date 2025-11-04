"""
Demo script showing how to use the visualization utilities.
"""
import numpy as np
import matplotlib.pyplot as plt
from src import solver
from src import visualization as vis

# Create a mesh
print("Creating mesh...")
mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                           char_length=0.2, verbose=True)

# Assemble operators
print("Assembling operators...")
mesh, basis, K, M = solver.assemble_operators(mesh)

# Define external potential (e.g., harmonic oscillator)
def Vext(X):
    """3D harmonic potential centered at (0.5, 0.5, 0.5)"""
    x, y, z = X[0, :], X[1, :], X[2, :]
    r2 = (x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2
    return 10.0 * r2

# Run SCF calculation
print("Running SCF calculation...")
E, modes, phi, Vfinal = solver.scf_loop(
    mesh, basis, K, M, Vext,
    coupling=1.0, maxiter=30, tol=1e-6,
    mix=0.6, nev=4, verbose=True, use_diis=True
)

print(f"\nEnergy levels: {E}")

# Visualizations
print("\nGenerating visualizations...")

# 1. Plot potential and ground state density on a single slice
fig1, axes1 = vis.plot_potential_and_density(
    basis, Vfinal, modes,
    slice_axis='z', slice_value=0.5,
    save_path='results/potential_and_density.png'
)

# 2. Plot probability density on multiple slices
fig2, axes2 = vis.plot_multiple_slices(
    basis, Vfinal, modes,
    slice_axis='z', slice_positions=[0.3, 0.5, 0.7],
    save_path='results/density_slices.png'
)

# 3. Plot 1D line profiles through the center
fig3, axes3 = vis.plot_1d_line_profile(
    basis, Vfinal, modes,
    axis='z', fixed_coords={'x': 0.5, 'y': 0.5},
    save_path='results/line_profiles.png'
)

# 4. Plot 3D isosurface of probability density
fig4, ax4 = vis.plot_3d_isosurface(
    basis, modes,
    iso_level=0.3,
    save_path='results/density_3d.png'
)

# 5. Plot energy level diagram
fig5, ax5 = vis.plot_energy_levels(
    E, n_levels=4,
    save_path='results/energy_levels.png'
)

print("\nAll visualizations saved to results/ directory")
print("Displaying plots...")
plt.show()
