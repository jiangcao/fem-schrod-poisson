#!/usr/bin/env python3
"""
Example demonstrating the Poisson solver with spatially varying epsilon.

This script shows different use cases:
1. Constant scalar epsilon
2. Spatially varying scalar epsilon (as array)
3. Spatially varying scalar epsilon (as callable)
4. Diagonal tensor epsilon
5. Anisotropic tensor epsilon
"""

import numpy as np
from src import solver

def main():
    print("=" * 70)
    print("Poisson Solver with Spatially Varying Epsilon - Examples")
    print("=" * 70)
    
    # Create a test mesh
    print("\nCreating mesh...")
    mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), 
                                char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    print(f"  Mesh created: {basis.N} DOFs, {basis.nelems} elements")
    
    # Create a simple source term
    rho = np.ones(basis.N)
    
    # Example 1: Constant scalar epsilon
    print("\n" + "-" * 70)
    print("Example 1: Constant scalar epsilon = 2.0")
    phi1 = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=2.0)
    print(f"  Solution range: [{phi1.min():.6f}, {phi1.max():.6f}]")
    print(f"  Max interior value: {phi1.max():.6f}")
    
    # Example 2: Spatially varying scalar epsilon (array at DOFs)
    print("\n" + "-" * 70)
    print("Example 2: Spatially varying scalar epsilon (array)")
    X = basis.doflocs
    epsilon_scalar = 1.0 + 0.5 * X[0, :]  # varies from 1.0 to 1.5 along x
    phi2 = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=epsilon_scalar)
    print(f"  Epsilon range: [{epsilon_scalar.min():.6f}, {epsilon_scalar.max():.6f}]")
    print(f"  Solution range: [{phi2.min():.6f}, {phi2.max():.6f}]")
    print(f"  Max interior value: {phi2.max():.6f}")
    
    # Example 3: Spatially varying scalar epsilon (callable)
    print("\n" + "-" * 70)
    print("Example 3: Spatially varying scalar epsilon (callable)")
    def epsilon_func(X):
        # X has shape (3, npts)
        r2 = X[0, :]**2 + X[1, :]**2 + X[2, :]**2
        return 1.0 + 2.0 * np.exp(-5.0 * r2)  # Higher near origin
    phi3 = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=epsilon_func)
    print(f"  Solution range: [{phi3.min():.6f}, {phi3.max():.6f}]")
    print(f"  Max interior value: {phi3.max():.6f}")
    
    # Example 4: Diagonal tensor epsilon (anisotropic but axis-aligned)
    print("\n" + "-" * 70)
    print("Example 4: Diagonal tensor epsilon")
    def epsilon_diagonal(X):
        # Different diffusion in each direction
        npts = X.shape[1]
        eps = np.zeros((3, 3, npts))
        eps[0, 0, :] = 3.0  # High diffusion in x
        eps[1, 1, :] = 1.0  # Medium diffusion in y
        eps[2, 2, :] = 0.5  # Low diffusion in z
        return eps
    phi4 = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=epsilon_diagonal)
    print(f"  Epsilon_xx = 3.0, Epsilon_yy = 1.0, Epsilon_zz = 0.5")
    print(f"  Solution range: [{phi4.min():.6f}, {phi4.max():.6f}]")
    print(f"  Max interior value: {phi4.max():.6f}")
    
    # Example 5: Full anisotropic tensor epsilon with off-diagonal terms
    print("\n" + "-" * 70)
    print("Example 5: Anisotropic tensor epsilon with off-diagonal terms")
    def epsilon_anisotropic(X):
        npts = X.shape[1]
        eps = np.zeros((3, 3, npts))
        # Diagonal
        eps[0, 0, :] = 2.0
        eps[1, 1, :] = 1.5
        eps[2, 2, :] = 1.0
        # Off-diagonal (symmetric for SPD matrix)
        eps[0, 1, :] = 0.5
        eps[1, 0, :] = 0.5
        eps[0, 2, :] = 0.3
        eps[2, 0, :] = 0.3
        return eps
    phi5 = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=epsilon_anisotropic)
    print(f"  Tensor with off-diagonal coupling")
    print(f"  Solution range: [{phi5.min():.6f}, {phi5.max():.6f}]")
    print(f"  Max interior value: {phi5.max():.6f}")
    
    # Example 6: Tensor epsilon as array at DOFs
    print("\n" + "-" * 70)
    print("Example 6: Tensor epsilon as array at DOFs")
    epsilon_array = np.zeros((basis.N, 3, 3))
    for i in range(basis.N):
        # Vary epsilon based on position
        x, y, z = X[:, i]
        epsilon_array[i, 0, 0] = 1.0 + 0.5 * x
        epsilon_array[i, 1, 1] = 1.0 + 0.5 * y
        epsilon_array[i, 2, 2] = 1.0 + 0.5 * z
    phi6 = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=epsilon_array)
    print(f"  Position-dependent diagonal tensor")
    print(f"  Solution range: [{phi6.min():.6f}, {phi6.max():.6f}]")
    print(f"  Max interior value: {phi6.max():.6f}")
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
