"""
Demo script showing how to use the visualization utilities.
"""
import numpy as np
import matplotlib.pyplot as plt
from src import solver
from src import visualization as vis

# Create a mesh - use larger domain to better approximate infinite system
print("Creating mesh...")
mesh = solver.make_mesh_box(x0=(-3, -3, -3), lengths=(6.0, 6.0, 6.0), 
                           char_length=0.4, verbose=True)

# Assemble operators with P2 elements for better accuracy
print("Assembling operators...")
mesh, basis, K, M = solver.assemble_operators(mesh, element_order=2)

# Define external potential (e.g., harmonic oscillator)
def Vext(X):
    """3D harmonic potential centered at origin"""
    x, y, z = X[0, :], X[1, :], X[2, :]
    r2 = x**2 + y**2 + z**2
    return 0.5 * r2  # ω=1 harmonic oscillator

# Run SCF calculation - no coupling for pure harmonic oscillator
print("Running SCF calculation...")
E, modes, phi, Vfinal = solver.scf_loop(
    mesh, basis, K, M, Vext,
    coupling=0.0, maxiter=1, tol=1e-6,
    mix=0.0, nev=4, verbose=True
)

print(f"\nEnergy levels: {E}")

# Visualizations
print("\nGenerating visualizations...")

# 1. Plot potential and ground state density on a single slice through center
fig1, axes1 = vis.plot_potential_and_density(
    basis, Vfinal, modes,
    slice_axis='z', slice_value=0.0,
    save_path='results/potential_and_density.png'
)

# 2. Plot probability density on multiple slices
fig2, axes2 = vis.plot_multiple_slices(
    basis, Vfinal, modes,
    slice_axis='z', slice_positions=[-1.0, 0.0, 1.0],
    save_path='results/density_slices.png'
)

# 3. Plot 1D line profiles through the center
fig3, axes3 = vis.plot_1d_line_profile(
    basis, Vfinal, modes,
    axis='z', fixed_coords={'x': 0.0, 'y': 0.0},
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
