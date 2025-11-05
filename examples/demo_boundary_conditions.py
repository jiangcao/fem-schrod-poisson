#!/usr/bin/env python3
"""
Example demonstrating boundary condition utilities for the Poisson solver.

This script shows different ways to specify boundary conditions:
1. Simple constant value on all boundaries
2. Different values on different faces
3. Position-dependent boundary values
4. Complex boundary conditions using custom functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src import solver, boundary_conditions as bc_utils

def main():
    print("=" * 70)
    print("Boundary Condition Utilities - Examples")
    print("=" * 70)
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    # Create a test mesh
    print("\nCreating mesh...")
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                                char_length=0.25, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    print(f"  Mesh created: {basis.N} DOFs, {basis.nelems} elements")
    
    # Create a simple source term
    rho = np.ones(basis.N)
    
    # Example 1: Simple constant boundary value (backward compatible)
    print("\n" + "-" * 70)
    print("Example 1: Constant boundary value (bc_value=0.0)")
    phi1 = solver.solve_poisson(mesh, basis, rho, bc_value=0.0)
    print(f"  Solution range: [{phi1.min():.6f}, {phi1.max():.6f}]")
    
    # Example 2: Different values on left and right boundaries
    print("\n" + "-" * 70)
    print("Example 2: Different values on opposite faces")
    print("  phi=0 on x=0 (left), phi=1 on x=1 (right)")
    
    # Get boundary nodes on left and right faces
    nodes_left = bc_utils.get_boundary_nodes_by_plane(basis, 'x', 0.0)
    nodes_right = bc_utils.get_boundary_nodes_by_plane(basis, 'x', 1.0)
    
    print(f"  Found {len(nodes_left)} nodes on left face")
    print(f"  Found {len(nodes_right)} nodes on right face")
    
    # Create boundary conditions for each face
    bc_left = bc_utils.create_dirichlet_bc(basis, value=0.0, nodes=nodes_left)
    bc_right = bc_utils.create_dirichlet_bc(basis, value=1.0, nodes=nodes_right)
    
    # Combine boundary conditions
    bc = bc_utils.combine_boundary_conditions([bc_left, bc_right])
    
    # Solve with combined BC
    phi2 = solver.solve_poisson(mesh, basis, rho, bc=bc)
    print(f"  Solution range: [{phi2.min():.6f}, {phi2.max():.6f}]")
    
    # Example 3: Position-dependent boundary values
    print("\n" + "-" * 70)
    print("Example 3: Position-dependent boundary values")
    print("  phi = x + y on all boundaries")
    
    def bc_func(X):
        """Boundary value depends on position"""
        return X[0] + X[1]
    
    bc3 = bc_utils.create_dirichlet_bc(basis, value=bc_func)
    phi3 = solver.solve_poisson(mesh, basis, np.zeros(basis.N), bc=bc3)
    print(f"  Solution range: [{phi3.min():.6f}, {phi3.max():.6f}]")
    
    # Example 4: Complex boundary conditions using custom functions
    print("\n" + "-" * 70)
    print("Example 4: Custom boundary region")
    print("  phi=1 where x+y+z > 2, phi=0 elsewhere on boundary")
    
    # Define custom condition for high-value region
    def high_region(X):
        return X[0] + X[1] + X[2] > 2.0
    
    nodes_high = bc_utils.get_boundary_nodes_by_function(basis, high_region)
    nodes_all = bc_utils.get_boundary_nodes(mesh)
    
    print(f"  Found {len(nodes_high)} nodes in high-value region")
    print(f"  Total boundary nodes: {len(nodes_all)}")
    
    # Create BCs
    bc_all = bc_utils.create_dirichlet_bc(basis, value=0.0, nodes=nodes_all)
    bc_high = bc_utils.create_dirichlet_bc(basis, value=1.0, nodes=nodes_high)
    bc4 = bc_utils.combine_boundary_conditions([bc_all, bc_high])
    
    phi4 = solver.solve_poisson(mesh, basis, rho, bc=bc4)
    print(f"  Solution range: [{phi4.min():.6f}, {phi4.max():.6f}]")
    
    # Example 5: Using bounding boxes
    print("\n" + "-" * 70)
    print("Example 5: Boundary conditions using bounding boxes")
    print("  phi=1 in corner region (x,y,z > 0.8), phi=0 elsewhere")
    
    nodes_corner = bc_utils.get_boundary_nodes_by_box(
        basis, xmin=0.8, ymin=0.8, zmin=0.8
    )
    
    print(f"  Found {len(nodes_corner)} nodes in corner region")
    
    bc_all = bc_utils.create_dirichlet_bc(basis, value=0.0)
    bc_corner = bc_utils.create_dirichlet_bc(basis, value=1.0, nodes=nodes_corner)
    bc5 = bc_utils.combine_boundary_conditions([bc_all, bc_corner])
    
    phi5 = solver.solve_poisson(mesh, basis, rho, bc=bc5)
    print(f"  Solution range: [{phi5.min():.6f}, {phi5.max():.6f}]")
    
    # Example 6: Using domain bounds utility
    print("\n" + "-" * 70)
    print("Example 6: Using get_domain_bounds and get_boundary_faces")
    
    bounds = bc_utils.get_domain_bounds(basis)
    print(f"  Domain bounds:")
    print(f"    x: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}]")
    print(f"    y: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}]")
    print(f"    z: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}]")
    
    # Get faces using the convenience function
    x_faces = bc_utils.get_boundary_faces(basis, axis='x')
    y_faces = bc_utils.get_boundary_faces(basis, axis='y')
    z_faces = bc_utils.get_boundary_faces(basis, axis='z')
    
    print(f"  X-faces: {len(x_faces['min'])} nodes (min), {len(x_faces['max'])} nodes (max)")
    print(f"  Y-faces: {len(y_faces['min'])} nodes (min), {len(y_faces['max'])} nodes (max)")
    print(f"  Z-faces: {len(z_faces['min'])} nodes (min), {len(z_faces['max'])} nodes (max)")
    
    # Create a potential with gradients in all directions
    bc_xmin = bc_utils.create_dirichlet_bc(basis, value=0.0, nodes=x_faces['min'])
    bc_xmax = bc_utils.create_dirichlet_bc(basis, value=1.0, nodes=x_faces['max'])
    bc_ymin = bc_utils.create_dirichlet_bc(basis, value=0.0, nodes=y_faces['min'])
    bc_ymax = bc_utils.create_dirichlet_bc(basis, value=0.5, nodes=y_faces['max'])
    bc6 = bc_utils.combine_boundary_conditions([bc_xmin, bc_xmax, bc_ymin, bc_ymax])
    
    phi6 = solver.solve_poisson(mesh, basis, rho, bc=bc6)
    print(f"  Solution with mixed BCs: [{phi6.min():.6f}, {phi6.max():.6f}]")
    
    # Visualize Example 2 (gradient from left to right)
    print("\n" + "-" * 70)
    print("Creating visualization for Example 2...")
    
    X = basis.doflocs
    
    # Extract a slice at z=0.5
    tol = 0.1
    mask = np.abs(X[2, :] - 0.5) < tol
    
    x_slice = X[0, mask]
    y_slice = X[1, mask]
    phi_slice = phi2[mask]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x_slice, y_slice, c=phi_slice, cmap='viridis', 
                   s=50, alpha=0.8, edgecolors='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Poisson Solution with phi=0 (left) and phi=1 (right)\nSlice at z=0.5')
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='φ')
    plt.tight_layout()
    plt.savefig('results/boundary_conditions_example.png', dpi=150, bbox_inches='tight')
    print("  ✓ Visualization saved to results/boundary_conditions_example.png")
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Simple constant boundary values")
    print("  ✓ Different values on different faces")
    print("  ✓ Position-dependent boundary values")
    print("  ✓ Custom boundary regions using functions")
    print("  ✓ Bounding box selection")
    print("  ✓ Domain bounds and face utilities")
    print("  ✓ Combining multiple boundary conditions")
    print("=" * 70)

if __name__ == "__main__":
    main()
