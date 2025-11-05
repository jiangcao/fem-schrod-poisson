# fem-schrod-poisson

Finite element solver for the Schrödinger-Poisson system with support for heterostructure interfaces, spatially varying permittivity, and anisotropic effective mass.

## Features

- **Schrödinger-Poisson solver**: Self-consistent solution of coupled quantum-classical system
- **Heterostructure interface support**: Handles discontinuous ε and effective mass with proper flux continuity
- **Poisson solver with spatially varying epsilon**: Solves `-∇·(ε∇φ) = ρ` with flexible epsilon specification:
  - Constant scalar epsilon
  - Spatially varying scalar epsilon (array or callable)
  - Spatially varying tensor epsilon (anisotropic materials)
- **Schrödinger solver with spatially varying effective mass**: Solves `-∇·(1/m_eff ∇ψ) + Vψ = Eψ`:
  - Constant scalar mass
  - Spatially varying scalar mass (array or callable)
  - Anisotropic effective mass tensor (3×3 tensor at each point)
  - Support for discontinuous mass fields at material interfaces
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

### Spatially Varying Effective Mass

```python
# Constant scalar mass
mass_eff = 0.5
E, modes, phi, Vfinal = solver.scf_loop(
    mesh, basis, K, M, Vext,
    mass_eff=mass_eff, phys=phys
)

# Spatially varying scalar mass (as array at DOFs)
X = basis.doflocs
mass_eff = 0.5 + 0.5 * X[0, :]  # varies along x-axis
E, modes, phi, Vfinal = solver.scf_loop(
    mesh, basis, K, M, Vext,
    mass_eff=mass_eff, phys=phys
)

# Spatially varying scalar mass (as callable)
def mass_func(X):
    r2 = X[0]**2 + X[1]**2 + X[2]**2
    return 0.5 + 0.5 * r2

E, modes, phi, Vfinal = solver.scf_loop(
    mesh, basis, K, M, Vext,
    mass_eff=mass_func, phys=phys
)
```

### Anisotropic Effective Mass Tensor

```python
# Diagonal tensor (different mass in each direction)
def mass_tensor(X):
    npts = X.shape[1]
    m = np.zeros((3, 3, npts))
    m[0, 0, :] = 1.0  # mx
    m[1, 1, :] = 2.0  # my
    m[2, 2, :] = 0.5  # mz
    return m

E, modes, phi, Vfinal = solver.scf_loop(
    mesh, basis, K, M, Vext,
    mass_eff=mass_tensor, phys=phys
)
```

### Heterostructure Interfaces

Handle discontinuous material properties at interfaces:

```python
# Define heterostructure with step discontinuity at x=1.0
def epsilon_func(X):
    """Material 1: ε=1.0, Material 2: ε=3.0"""
    eps = np.ones(X.shape[1])
    eps[X[0, :] > 1.0] = 3.0
    return eps

def mass_func(X):
    """Material 1: m=1.0, Material 2: m=0.5"""
    m = np.ones(X.shape[1])
    m[X[0, :] > 1.0] = 0.5
    return m

# Solver handles flux continuity automatically
E, modes, phi, Vfinal = solver.scf_loop(
    mesh, basis, K, M, Vext,
    epsilon=epsilon_func,
    mass_eff=mass_func,
    phys=phys
)
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

    ### Visualization

    Visualize potential and wave function probability density:

    ```python
    from src import visualization as vis

    # Plot potential and density on a 2D slice
    fig, axes = vis.plot_potential_and_density(
        basis, Vfinal, modes,
        slice_axis='z', slice_value=0.5,
        save_path='results/slice.png'
    )

    # Plot multiple slices
    fig, axes = vis.plot_multiple_slices(
        basis, Vfinal, modes,
        slice_axis='z', slice_positions=[0.3, 0.5, 0.7]
    )

    # 1D line profile through the center
    fig, axes = vis.plot_1d_line_profile(
        basis, Vfinal, modes,
        axis='z', fixed_coords={'x': 0.5, 'y': 0.5}
    )

    # 3D isosurface of probability density
    fig, ax = vis.plot_3d_isosurface(basis, modes, iso_level=0.5)

    # Energy level diagram
    fig, ax = vis.plot_energy_levels(E, n_levels=4)
    ```

    See `docs/VISUALIZATION.md` for detailed documentation.

## Examples

- `examples/demo_epsilon.py` - Poisson solver with various epsilon configurations
- `examples/demo_visualization.py` - Visualization utilities showcase
- `examples/demo_heterostructure.py` - Heterostructure interfaces with discontinuous ε and m_eff

Run examples:

    PYTHONPATH=. python examples/demo_epsilon.py
    PYTHONPATH=. python examples/demo_heterostructure.py

## Testing

Run tests:

    pytest tests/ -v
