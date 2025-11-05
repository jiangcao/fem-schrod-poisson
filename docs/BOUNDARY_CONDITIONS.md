# Boundary Condition Utilities

This document describes the boundary condition utilities in `src/boundary_conditions.py`, which provide flexible ways to define boundary conditions for the Poisson solver.

## Overview

The boundary condition utilities make it easy to:
- Identify boundary nodes by position, region, or custom criteria
- Set different boundary values on different surfaces
- Create position-dependent boundary conditions
- Combine multiple boundary condition specifications

## Core Functions

### Getting Boundary Nodes

#### `get_boundary_nodes(mesh)`
Get all boundary nodes from the mesh.

```python
from src import boundary_conditions as bc_utils

bdofs = bc_utils.get_boundary_nodes(mesh)
print(f"Found {len(bdofs)} boundary nodes")
```

#### `get_boundary_nodes_by_plane(basis, axis, value, tol=1e-6)`
Get boundary nodes on a plane perpendicular to an axis.

```python
# Get nodes on the left boundary (x=0)
nodes_left = bc_utils.get_boundary_nodes_by_plane(basis, axis='x', value=0.0)

# Get nodes on the right boundary (x=Lx)
Lx = 1.0
nodes_right = bc_utils.get_boundary_nodes_by_plane(basis, axis='x', value=Lx)

# Works with any axis
nodes_top = bc_utils.get_boundary_nodes_by_plane(basis, axis='y', value=1.0)
nodes_front = bc_utils.get_boundary_nodes_by_plane(basis, axis='z', value=0.5)
```

**Parameters:**
- `basis`: The finite element basis
- `axis`: Axis perpendicular to the plane ('x', 'y', or 'z')
- `value`: Position along the axis
- `tol`: Tolerance for determining if a point is on the plane (default: 1e-6)

#### `get_boundary_nodes_by_box(basis, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, tol=1e-6)`
Get boundary nodes within a bounding box.

```python
# Get nodes on left face (x near 0)
nodes_left = bc_utils.get_boundary_nodes_by_box(basis, xmin=-0.01, xmax=0.01)

# Get nodes in a corner
nodes_corner = bc_utils.get_boundary_nodes_by_box(
    basis, xmin=0.9, ymin=0.9, zmin=0.9
)

# Get nodes on an edge (intersection of two faces)
nodes_edge = bc_utils.get_boundary_nodes_by_box(
    basis, xmin=-0.01, xmax=0.01, ymin=-0.01, ymax=0.01
)
```

**Parameters:**
- `basis`: The finite element basis
- `xmin, xmax, ymin, ymax, zmin, zmax`: Bounds of the box (None = no constraint)
- `tol`: Tolerance for boundary checks

#### `get_boundary_nodes_by_function(basis, condition)`
Get boundary nodes that satisfy a custom condition.

```python
# Get nodes on a sphere of radius 0.5
def on_sphere(X):
    r = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    return np.abs(r - 0.5) < 0.01

nodes_sphere = bc_utils.get_boundary_nodes_by_function(basis, on_sphere)

# Get nodes where x + y > 1.0
nodes_diagonal = bc_utils.get_boundary_nodes_by_function(
    basis, lambda X: X[0] + X[1] > 1.0
)

# Get nodes in cylindrical region
def in_cylinder(X):
    r2 = X[0]**2 + X[1]**2
    return (r2 < 0.25) & (X[2] > 0.3) & (X[2] < 0.7)

nodes_cyl = bc_utils.get_boundary_nodes_by_function(basis, in_cylinder)
```

**Parameters:**
- `basis`: The finite element basis
- `condition`: Function that takes coordinates X (shape 3 x npts) and returns boolean array

### Creating Boundary Conditions

#### `create_dirichlet_bc(basis, value=0.0, nodes=None)`
Create a Dirichlet boundary condition specification.

```python
# Constant zero on all boundaries
bc = bc_utils.create_dirichlet_bc(basis, value=0.0)

# Constant value on specific nodes
left_nodes = bc_utils.get_boundary_nodes_by_plane(basis, 'x', 0.0)
bc = bc_utils.create_dirichlet_bc(basis, value=1.0, nodes=left_nodes)

# Position-dependent value
def bc_func(X):
    return X[0] + X[1]  # phi = x + y on boundary

bc = bc_utils.create_dirichlet_bc(basis, value=bc_func)
```

**Parameters:**
- `basis`: The finite element basis
- `value`: Boundary value (float or callable)
  - `float`: constant value on all specified nodes
  - `callable(X)`: function that takes coordinates and returns values
- `nodes`: Specific nodes where BC applies (None = all boundary nodes)

**Returns:** Dictionary with keys:
- `'type'`: `'dirichlet'`
- `'nodes'`: Array of node indices
- `'values'`: Boundary values (scalar or array)

#### `combine_boundary_conditions(bc_list)`
Combine multiple boundary condition specifications.

When nodes overlap, later conditions override earlier ones.

```python
# Set phi=0 on all boundaries except left where phi=1
bc_all = bc_utils.create_dirichlet_bc(basis, value=0.0)
left_nodes = bc_utils.get_boundary_nodes_by_plane(basis, 'x', 0.0)
bc_left = bc_utils.create_dirichlet_bc(basis, value=1.0, nodes=left_nodes)
bc_combined = bc_utils.combine_boundary_conditions([bc_all, bc_left])

# Different values on each face
x_faces = bc_utils.get_boundary_faces(basis, 'x')
y_faces = bc_utils.get_boundary_faces(basis, 'y')

bc_list = [
    bc_utils.create_dirichlet_bc(basis, value=0.0, nodes=x_faces['min']),
    bc_utils.create_dirichlet_bc(basis, value=1.0, nodes=x_faces['max']),
    bc_utils.create_dirichlet_bc(basis, value=0.5, nodes=y_faces['max'])
]
bc = bc_utils.combine_boundary_conditions(bc_list)
```

**Parameters:**
- `bc_list`: List of boundary condition dictionaries

**Returns:** Combined boundary condition dictionary

### Convenience Functions

#### `get_domain_bounds(basis)`
Get the bounding box of the domain.

```python
bounds = bc_utils.get_domain_bounds(basis)
print(f"x: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}]")
print(f"y: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}]")
print(f"z: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}]")

xmin, xmax = bounds['x']
nodes_right = bc_utils.get_boundary_nodes_by_plane(basis, 'x', xmax)
```

**Returns:** Dictionary with keys 'x', 'y', 'z' and values (min, max) tuples

#### `get_boundary_faces(basis, axis='x')`
Get nodes on the min and max faces perpendicular to an axis.

```python
# Get left and right boundary nodes
x_faces = bc_utils.get_boundary_faces(basis, axis='x')
nodes_left = x_faces['min']
nodes_right = x_faces['max']

# Set different BCs on each face
bc_left = bc_utils.create_dirichlet_bc(basis, value=0.0, nodes=nodes_left)
bc_right = bc_utils.create_dirichlet_bc(basis, value=1.0, nodes=nodes_right)
bc = bc_utils.combine_boundary_conditions([bc_left, bc_right])
```

**Parameters:**
- `basis`: The finite element basis
- `axis`: Axis perpendicular to the faces ('x', 'y', or 'z')

**Returns:** Dictionary with 'min' and 'max' keys containing node arrays

## Usage with solve_poisson

The `solve_poisson` function accepts an optional `bc` parameter:

```python
# Old way (still works)
phi = solver.solve_poisson(mesh, basis, rho, bc_value=0.0)

# New way with flexible BCs
bc = bc_utils.create_dirichlet_bc(basis, value=0.0)
phi = solver.solve_poisson(mesh, basis, rho, bc=bc)

# Different values on different faces
nodes_left = bc_utils.get_boundary_nodes_by_plane(basis, 'x', 0.0)
nodes_right = bc_utils.get_boundary_nodes_by_plane(basis, 'x', 1.0)
bc_left = bc_utils.create_dirichlet_bc(basis, value=0.0, nodes=nodes_left)
bc_right = bc_utils.create_dirichlet_bc(basis, value=1.0, nodes=nodes_right)
bc = bc_utils.combine_boundary_conditions([bc_left, bc_right])
phi = solver.solve_poisson(mesh, basis, rho, bc=bc)
```

## Complete Examples

### Example 1: Simple Gradient

Set up a potential with phi=0 on the left and phi=1 on the right:

```python
from src import solver, boundary_conditions as bc_utils
import numpy as np

# Create mesh
mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), char_length=0.2)
mesh, basis, K, M = solver.assemble_operators(mesh)

# Constant source
rho = np.ones(basis.N)

# Get boundary nodes
nodes_left = bc_utils.get_boundary_nodes_by_plane(basis, 'x', 0.0)
nodes_right = bc_utils.get_boundary_nodes_by_plane(basis, 'x', 1.0)

# Create BCs
bc_left = bc_utils.create_dirichlet_bc(basis, value=0.0, nodes=nodes_left)
bc_right = bc_utils.create_dirichlet_bc(basis, value=1.0, nodes=nodes_right)
bc = bc_utils.combine_boundary_conditions([bc_left, bc_right])

# Solve
phi = solver.solve_poisson(mesh, basis, rho, bc=bc)
```

### Example 2: Position-Dependent BCs

Set boundary values that depend on position:

```python
# BC: phi = sin(2π x) * sin(2π y) on all boundaries
def bc_func(X):
    return np.sin(2 * np.pi * X[0]) * np.sin(2 * np.pi * X[1])

bc = bc_utils.create_dirichlet_bc(basis, value=bc_func)
phi = solver.solve_poisson(mesh, basis, rho, bc=bc)
```

### Example 3: Complex Boundary Regions

Set different values in different regions:

```python
# Get all boundary nodes
all_nodes = bc_utils.get_boundary_nodes(mesh)

# Define regions
def high_region(X):
    return X[2] > 0.7  # Top region

def low_region(X):
    return X[2] < 0.3  # Bottom region

nodes_high = bc_utils.get_boundary_nodes_by_function(basis, high_region)
nodes_low = bc_utils.get_boundary_nodes_by_function(basis, low_region)

# Create BCs: high=1.0, low=-1.0, middle=0.0
bc_all = bc_utils.create_dirichlet_bc(basis, value=0.0, nodes=all_nodes)
bc_high = bc_utils.create_dirichlet_bc(basis, value=1.0, nodes=nodes_high)
bc_low = bc_utils.create_dirichlet_bc(basis, value=-1.0, nodes=nodes_low)

bc = bc_utils.combine_boundary_conditions([bc_all, bc_high, bc_low])
phi = solver.solve_poisson(mesh, basis, rho, bc=bc)
```

### Example 4: Heterostructure with BCs

Combine spatially varying epsilon with custom boundary conditions:

```python
# Define heterostructure
def epsilon_func(X):
    eps = np.ones(X.shape[1])
    eps[X[0, :] > 0.5] = 3.0  # Higher dielectric on right
    return eps

# Different BCs on each side
x_faces = bc_utils.get_boundary_faces(basis, 'x')
bc_left = bc_utils.create_dirichlet_bc(basis, value=0.0, nodes=x_faces['min'])
bc_right = bc_utils.create_dirichlet_bc(basis, value=2.0, nodes=x_faces['max'])
bc = bc_utils.combine_boundary_conditions([bc_left, bc_right])

# Solve
phi = solver.solve_poisson(mesh, basis, rho, epsilon=epsilon_func, bc=bc)
```

## Best Practices

1. **Start simple**: Use `get_boundary_nodes_by_plane` for straightforward face-based BCs
2. **Combine BCs**: Always set a default BC on all boundaries first, then override specific regions
3. **Check node counts**: Verify you're getting the expected number of nodes with `len(nodes)`
4. **Tolerance matters**: Adjust `tol` parameter if nodes are missing near boundaries
5. **Position-dependent values**: Use callable functions for complex boundary value distributions
6. **Visualize**: Always plot the solution to verify your boundary conditions are correct

## Testing

The module includes comprehensive tests in `tests/test_boundary_conditions.py`:
- Test all node selection functions
- Test BC creation with constants and callables
- Test combining BCs
- Test integration with `solve_poisson`
- Test backward compatibility

Run tests with:
```bash
pytest tests/test_boundary_conditions.py -v
```

## Future Enhancements

Possible extensions:
- Neumann boundary conditions (flux/gradient BCs)
- Robin/mixed boundary conditions
- Support for GMSH physical groups
- Time-dependent boundary conditions
- Periodic boundary conditions
