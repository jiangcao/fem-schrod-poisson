# Visualization Utilities - Summary

## What Was Added

I've added comprehensive visualization utilities to the fem-schrod-poisson package for plotting potential and wave function probability densities.

## New Files

1. **`src/visualization.py`** - Main visualization module with 5 key functions:
   - `plot_potential_and_density()` - Side-by-side plots of potential and |ψ|² on 2D slices
   - `plot_multiple_slices()` - Multiple parallel slices through the domain
   - `plot_1d_line_profile()` - 1D line profiles along coordinate axes
   - `plot_3d_isosurface()` - 3D scatter plots of probability density
   - `plot_energy_levels()` - Energy level diagrams

2. **`docs/VISUALIZATION.md`** - Complete documentation with examples and API reference

3. **`examples/demo_visualization.py`** - Demo script showing all visualization functions

4. **`examples/demo_heterostructure.py`** - Advanced example with spatially varying epsilon

5. **`tests/test_visualization.py`** - Unit tests for all visualization functions

## Features

### Easy to Use
```python
from src import visualization as vis

# Quick 2D slice view
vis.plot_potential_and_density(basis, potential, psi, 
                               slice_axis='z', slice_value=0.5)
```

### Flexible
- Works with any mesh and basis from solver
- Supports multi-mode wave functions (automatically uses ground state)
- Customizable colormaps, figure sizes, slice positions
- Optional save to file

### Comprehensive
- 2D slices through 3D domain
- Multiple parallel slices
- 1D line profiles
- 3D isosurfaces
- Energy level diagrams

## Dependencies

Added `matplotlib>=3.5.0` to requirements.txt

## Testing

All tests pass:
- 10 epsilon tests (spatially varying, tensor)
- 3 solver tests (backward compatibility)
- 2 Poisson tests
- 1 visualization test

**Total: 15/15 tests passing ✓**

## Usage Examples

### Basic Usage
```python
# Run calculation
E, modes, phi, Vfinal = solver.scf_loop(...)

# Visualize
from src import visualization as vis
vis.plot_potential_and_density(basis, Vfinal, modes)
vis.plot_energy_levels(E)
```

### With Spatially Varying Epsilon
```python
# Define heterostructure
epsilon_func = lambda X: 1.0 + 2.0 * (X[0,:] > 0.5)

# Run with epsilon
E, modes, phi, Vfinal = solver.scf_loop(..., epsilon=epsilon_func)

# Visualize results
vis.plot_multiple_slices(basis, Vfinal, modes, 
                        slice_positions=[0.3, 0.5, 0.7])
```

## Documentation

- README.md updated with visualization section
- VISUALIZATION.md created with detailed API reference
- Example scripts demonstrate all features
- All functions have comprehensive docstrings

## Integration

- Seamlessly integrates with existing solver
- Non-breaking changes (all existing tests pass)
- Clean API via `src.visualization` module
- Works with both regular and tensor epsilon calculations
