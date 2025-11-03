# fem-schrod-poisson

Finite element solver for the Schrödinger-Poisson system with support for spatially varying permittivity.

## Features

- **Schrödinger-Poisson solver**: Self-consistent solution of coupled quantum-classical system
- **Poisson solver with spatially varying epsilon**: Solves `-∇·(ε∇φ) = ρ` with flexible epsilon specification:
  - Constant scalar epsilon
  - Spatially varying scalar epsilon (array or callable)
  - Spatially varying tensor epsilon (anisotropic materials)
- **DIIS acceleration**: Pulay mixing for faster SCF convergence
- **3D tetrahedral meshes**: Using pygmsh/gmsh mesh generation

## Installation

Create venv and install deps:

    python -m venv .venv && source .venv/bin/activate
    pip install -U pip
    pip install scikit-fem scipy numpy meshio pygmsh pytest

## Usage

### Basic Poisson Solver

```python
from src import solver
import numpy as np

# Create mesh
mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), char_length=0.3)
mesh, basis, K, M = solver.assemble_operators(mesh)

# Source term
rho = np.ones(basis.N)

# Solve with constant epsilon
phi = solver.solve_poisson(mesh, basis, rho, bc_value=0.0, epsilon=2.0)
```

### Spatially Varying Scalar Epsilon

```python
# As array at DOFs
X = basis.doflocs
epsilon = 1.0 + 0.5 * X[0, :]  # varies along x-axis
phi = solver.solve_poisson(mesh, basis, rho, epsilon=epsilon)

# As callable function
def epsilon_func(X):
    r2 = X[0,:]**2 + X[1,:]**2 + X[2,:]**2
    return 1.0 + 2.0 * np.exp(-5.0 * r2)

phi = solver.solve_poisson(mesh, basis, rho, epsilon=epsilon_func)
```

### Tensor Epsilon (Anisotropic Materials)

```python
# Diagonal tensor (different diffusion in each direction)
def epsilon_diagonal(X):
    npts = X.shape[1]
    eps = np.zeros((3, 3, npts))
    eps[0, 0, :] = 3.0  # x-direction
    eps[1, 1, :] = 1.0  # y-direction
    eps[2, 2, :] = 0.5  # z-direction
    return eps

phi = solver.solve_poisson(mesh, basis, rho, epsilon=epsilon_diagonal)

# Full anisotropic tensor with off-diagonal terms
def epsilon_anisotropic(X):
    npts = X.shape[1]
    eps = np.zeros((3, 3, npts))
    eps[0, 0, :] = 2.0
    eps[1, 1, :] = 1.5
    eps[2, 2, :] = 1.0
    eps[0, 1, :] = eps[1, 0, :] = 0.5  # coupling terms
    return eps

phi = solver.solve_poisson(mesh, basis, rho, epsilon=epsilon_anisotropic)
```

### Schrödinger-Poisson SCF Loop

```python
Vext = lambda X: np.zeros(X.shape[1])  # external potential
E, modes, phi, Vfinal = solver.scf_loop(
    mesh, basis, K, M, Vext,
    coupling=1.0, maxiter=30, tol=1e-6,
    mix=0.4, nev=4, use_diis=True
)
```

## Examples

See `examples/demo_epsilon.py` for complete examples of all epsilon variations.

Run examples:

    PYTHONPATH=. python examples/demo_epsilon.py

## Testing

Run tests:

    pytest tests/ -v
