"""
Quick test of visualization functions.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from src import solver, visualization as vis

def test_visualization_functions():
    """Test that all visualization functions can be called without errors."""
    print("Creating test mesh...")
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                               char_length=0.4, verbose=False)
    
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    # Create simple test data
    print("Creating test data...")
    Vext = lambda X: np.zeros(X.shape[1])
    E, modes, phi, Vfinal = solver.scf_loop(
        mesh, basis, K, M, Vext,
        coupling=1.0, maxiter=5, tol=1e-5,
        mix=0.4, nev=2, verbose=False, use_diis=False
    )
    
    print("Testing plot_potential_and_density...")
    fig1, _ = vis.plot_potential_and_density(basis, Vfinal, modes, slice_axis='z')
    plt.close(fig1)
    print("✓ plot_potential_and_density works")
    
    print("Testing plot_multiple_slices...")
    fig2, _ = vis.plot_multiple_slices(basis, Vfinal, modes, slice_positions=[0.3, 0.5, 0.7])
    plt.close(fig2)
    print("✓ plot_multiple_slices works")
    
    print("Testing plot_1d_line_profile...")
    fig3, _ = vis.plot_1d_line_profile(basis, Vfinal, modes, axis='z')
    plt.close(fig3)
    print("✓ plot_1d_line_profile works")
    
    print("Testing plot_3d_isosurface...")
    fig4, _ = vis.plot_3d_isosurface(basis, modes, iso_level=0.3)
    plt.close(fig4)
    print("✓ plot_3d_isosurface works")
    
    print("Testing plot_energy_levels...")
    fig5, _ = vis.plot_energy_levels(E, n_levels=2)
    plt.close(fig5)
    print("✓ plot_energy_levels works")
    
    print("\n✓ All visualization functions work correctly!")

if __name__ == "__main__":
    test_visualization_functions()
