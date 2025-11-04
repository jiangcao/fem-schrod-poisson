# Visualization Utilities

This module provides visualization functions for plotting potential and wave function probability densities from the Schrödinger-Poisson solver.

## Features

### 1. **2D Slice Plots**
Plot potential and probability density on 2D slices through the 3D domain:
```python
from src import visualization as vis

fig, axes = vis.plot_potential_and_density(
    basis, potential, psi,
    slice_axis='z', slice_value=0.5
)
```

### 2. **Multiple Slice Views**
View probability density on multiple parallel slices:
```python
fig, axes = vis.plot_multiple_slices(
    basis, potential, psi,
    slice_axis='z', 
    slice_positions=[0.25, 0.5, 0.75]
)
```

### 3. **1D Line Profiles**
Plot potential and density along a line through the domain:
```python
fig, axes = vis.plot_1d_line_profile(
    basis, potential, psi,
    axis='z',
    fixed_coords={'x': 0.5, 'y': 0.5}
)
```

### 4. **3D Isosurface**
Create 3D scatter plots of probability density:
```python
fig, ax = vis.plot_3d_isosurface(
    basis, psi,
    iso_level=0.5  # Show regions above 50% of max density
)
```

### 5. **Energy Level Diagrams**
Visualize computed energy eigenvalues:
```python
fig, ax = vis.plot_energy_levels(
    energies, n_levels=10
)
```

## Example Usage

See `examples/demo_visualization.py` for a complete example:

```python
import numpy as np
from src import solver, visualization as vis

# Create mesh and run calculation
mesh = solver.make_mesh_box(x0=(0,0,0), lengths=(1.0,1.0,1.0), char_length=0.2)
mesh, basis, K, M = solver.assemble_operators(mesh)

# Define potential
Vext = lambda X: 10.0 * ((X[0]-0.5)**2 + (X[1]-0.5)**2 + (X[2]-0.5)**2)

# Run SCF
E, modes, phi, Vfinal = solver.scf_loop(
    mesh, basis, K, M, Vext,
    coupling=1.0, maxiter=30, nev=4
)

# Visualize results
vis.plot_potential_and_density(basis, Vfinal, modes, save_path='results/plot.png')
vis.plot_energy_levels(E, save_path='results/energy.png')
```

## Function Reference

### `plot_potential_and_density(basis, potential, psi, ...)`
Creates side-by-side plots of potential and probability density on a 2D slice.

**Parameters:**
- `basis`: skfem.Basis object
- `potential`: array (ndofs,) - potential values at DOFs
- `psi`: array (ndofs,) or (ndofs, nmodes) - wave function(s)
- `slice_axis`: 'x', 'y', or 'z' - axis perpendicular to slice
- `slice_value`: float - position along slice_axis
- `save_path`: str (optional) - path to save figure

### `plot_multiple_slices(basis, potential, psi, ...)`
Shows probability density on multiple parallel slices.

**Parameters:**
- `slice_positions`: list of float - positions for slices
- Other parameters same as above

### `plot_1d_line_profile(basis, potential, psi, ...)`
Plots 1D profiles along a coordinate axis.

**Parameters:**
- `axis`: 'x', 'y', or 'z' - axis to plot along
- `fixed_coords`: dict - fixed coordinates for other axes, e.g., `{'x': 0.5, 'y': 0.5}`

### `plot_3d_isosurface(basis, psi, ...)`
Creates 3D scatter plot of probability density.

**Parameters:**
- `iso_level`: float - threshold level (fraction of max density)
- `alpha`: float - transparency (0-1)

### `plot_energy_levels(energies, ...)`
Plots energy level diagram.

**Parameters:**
- `energies`: array - eigenvalues
- `n_levels`: int (optional) - number of levels to plot

## Notes

- All functions support `save_path` parameter to save figures
- Colormaps can be customized via `cmap` or `cmap_potential`/`cmap_density` parameters
- Figure sizes can be adjusted via `figsize` parameter
- Functions return matplotlib figure and axes objects for further customization

## Requirements

- matplotlib >= 3.5.0
- numpy
- scipy
- scikit-fem

Install with:
```bash
pip install matplotlib
```
