"""
Utility functions for defining boundary conditions on finite element meshes.

This module provides convenient functions for:
- Identifying boundary nodes by position (e.g., x=0, y=max)
- Setting boundary conditions by region
- Working with GMSH physical groups
- Handling mixed boundary conditions (Dirichlet and Neumann)
"""
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from skfem import Basis
from skfem.mesh import MeshTet


def get_boundary_nodes(mesh: MeshTet) -> np.ndarray:
    """
    Get all boundary node indices from the mesh.
    
    Parameters:
    -----------
    mesh : MeshTet
        The finite element mesh
        
    Returns:
    --------
    bdofs : ndarray
        Array of boundary node indices
    """
    try:
        bdofs = mesh.boundary_nodes()
    except Exception:
        try:
            bdofs = np.unique(mesh.facets.flatten())
        except Exception:
            bdofs = np.array([], dtype=int)
    return np.asarray(bdofs, dtype=int)


def get_boundary_nodes_by_plane(
    basis: Basis,
    axis: str = 'x',
    value: float = 0.0,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Get boundary nodes on a plane perpendicular to a given axis.
    
    Parameters:
    -----------
    basis : Basis
        The finite element basis
    axis : str
        Axis perpendicular to the plane ('x', 'y', or 'z')
    value : float
        Position along the axis
    tol : float
        Tolerance for determining if a point is on the plane
        
    Returns:
    --------
    nodes : ndarray
        Array of node indices on the specified plane
        
    Examples:
    ---------
    >>> # Get nodes on the x=0 plane
    >>> nodes_x0 = get_boundary_nodes_by_plane(basis, axis='x', value=0.0)
    >>> 
    >>> # Get nodes on the right boundary (x=Lx)
    >>> Lx = 1.0
    >>> nodes_xmax = get_boundary_nodes_by_plane(basis, axis='x', value=Lx)
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis.lower() not in axis_map:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis}")
    
    axis_idx = axis_map[axis.lower()]
    X = basis.doflocs
    
    # Get all boundary nodes first
    bdofs = get_boundary_nodes(basis.mesh)
    
    # Filter by position
    mask = np.abs(X[axis_idx, bdofs] - value) < tol
    return bdofs[mask]


def get_boundary_nodes_by_box(
    basis: Basis,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Get boundary nodes within a bounding box.
    
    Parameters:
    -----------
    basis : Basis
        The finite element basis
    xmin, xmax, ymin, ymax, zmin, zmax : float, optional
        Bounds of the box. If None, no constraint on that bound.
    tol : float
        Tolerance for boundary checks
        
    Returns:
    --------
    nodes : ndarray
        Array of node indices within the specified box
        
    Examples:
    ---------
    >>> # Get nodes on left boundary (x=0)
    >>> nodes_left = get_boundary_nodes_by_box(basis, xmin=-tol, xmax=tol)
    >>> 
    >>> # Get nodes on top-right corner
    >>> nodes_corner = get_boundary_nodes_by_box(
    ...     basis, xmin=0.9, ymin=0.9, zmin=0.9
    ... )
    """
    X = basis.doflocs
    bdofs = get_boundary_nodes(basis.mesh)
    
    mask = np.ones(len(bdofs), dtype=bool)
    
    if xmin is not None:
        mask &= X[0, bdofs] >= xmin - tol
    if xmax is not None:
        mask &= X[0, bdofs] <= xmax + tol
    if ymin is not None:
        mask &= X[1, bdofs] >= ymin - tol
    if ymax is not None:
        mask &= X[1, bdofs] <= ymax + tol
    if zmin is not None:
        mask &= X[2, bdofs] >= zmin - tol
    if zmax is not None:
        mask &= X[2, bdofs] <= zmax + tol
    
    return bdofs[mask]


def get_boundary_nodes_by_function(
    basis: Basis,
    condition: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Get boundary nodes that satisfy a custom condition.
    
    Parameters:
    -----------
    basis : Basis
        The finite element basis
    condition : callable
        Function that takes coordinates X (shape 3 x npts) and returns
        a boolean array indicating which points satisfy the condition
        
    Returns:
    --------
    nodes : ndarray
        Array of node indices satisfying the condition
        
    Examples:
    ---------
    >>> # Get nodes on a sphere of radius 0.5 centered at origin
    >>> def on_sphere(X):
    ...     r = np.sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    ...     return np.abs(r - 0.5) < 0.01
    >>> nodes_sphere = get_boundary_nodes_by_function(basis, on_sphere)
    >>> 
    >>> # Get nodes where x + y > 1.0
    >>> nodes_diag = get_boundary_nodes_by_function(
    ...     basis, lambda X: X[0] + X[1] > 1.0
    ... )
    """
    X = basis.doflocs
    bdofs = get_boundary_nodes(basis.mesh)
    
    # Evaluate condition only on boundary nodes
    X_boundary = X[:, bdofs]
    mask = condition(X_boundary)
    
    return bdofs[mask]


def create_dirichlet_bc(
    basis: Basis,
    value: Union[float, Callable[[np.ndarray], np.ndarray]] = 0.0,
    nodes: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, float, Callable]]:
    """
    Create a Dirichlet boundary condition specification.
    
    Parameters:
    -----------
    basis : Basis
        The finite element basis
    value : float or callable
        Boundary value(s). Can be:
        - float: constant value on all boundary nodes
        - callable: function that takes coordinates X and returns values
    nodes : ndarray, optional
        Specific nodes where BC applies. If None, applies to all boundary nodes.
        
    Returns:
    --------
    bc_dict : dict
        Dictionary with 'nodes', 'values', and 'type' keys
        
    Examples:
    ---------
    >>> # Constant zero on all boundaries
    >>> bc = create_dirichlet_bc(basis, value=0.0)
    >>> 
    >>> # Constant value on specific nodes
    >>> left_nodes = get_boundary_nodes_by_plane(basis, 'x', 0.0)
    >>> bc = create_dirichlet_bc(basis, value=1.0, nodes=left_nodes)
    >>> 
    >>> # Position-dependent value
    >>> def bc_func(X):
    ...     return X[0] + X[1]  # phi = x + y on boundary
    >>> bc = create_dirichlet_bc(basis, value=bc_func)
    """
    if nodes is None:
        nodes = get_boundary_nodes(basis.mesh)
    
    nodes = np.asarray(nodes, dtype=int)
    
    # Evaluate callable boundary values
    if callable(value):
        X = basis.doflocs[:, nodes]
        bc_values = np.asarray(value(X))
        # Validate shape
        if bc_values.ndim != 1:
            bc_values = bc_values.reshape(-1)
        if len(bc_values) != len(nodes):
            raise ValueError(
                f"Callable boundary value function must return array of length {len(nodes)}, "
                f"got {len(bc_values)}"
            )
    else:
        bc_values = float(value)
    
    return {
        'type': 'dirichlet',
        'nodes': nodes,
        'values': bc_values
    }


def combine_boundary_conditions(
    bc_list: List[Dict]
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Combine multiple boundary condition specifications.
    
    When nodes overlap, later conditions override earlier ones.
    
    Parameters:
    -----------
    bc_list : list of dict
        List of boundary condition dictionaries
        
    Returns:
    --------
    combined_bc : dict
        Combined boundary condition with 'nodes' and 'values'
        
    Examples:
    ---------
    >>> # Set phi=0 on all boundaries except left where phi=1
    >>> bc_all = create_dirichlet_bc(basis, value=0.0)
    >>> left_nodes = get_boundary_nodes_by_plane(basis, 'x', 0.0)
    >>> bc_left = create_dirichlet_bc(basis, value=1.0, nodes=left_nodes)
    >>> bc_combined = combine_boundary_conditions([bc_all, bc_left])
    """
    if not bc_list:
        return {'nodes': np.array([], dtype=int), 'values': np.array([])}
    
    # Remove duplicates (keep last value for each node)
    # Use dictionary for O(n) performance instead of O(n²) loop
    node_to_value = {}
    
    for bc in bc_list:
        if bc['type'] != 'dirichlet':
            raise NotImplementedError("Only Dirichlet BCs supported in combine")
        
        nodes = bc['nodes']
        values = bc['values']
        
        # Handle scalar values
        if np.isscalar(values):
            values = np.full(len(nodes), values)
        
        # Update dictionary (later values override earlier ones)
        for node, val in zip(nodes, values):
            node_to_value[int(node)] = val
    
    # Convert back to arrays
    unique_nodes = np.array(list(node_to_value.keys()), dtype=int)
    final_values = np.array(list(node_to_value.values()))
    
    # Sort by node index for consistency
    sort_idx = np.argsort(unique_nodes)
    unique_nodes = unique_nodes[sort_idx]
    final_values = final_values[sort_idx]
    
    return {
        'type': 'dirichlet',
        'nodes': unique_nodes,
        'values': final_values
    }

def get_domain_bounds(basis: Basis) -> Dict[str, Tuple[float, float]]:
    """
    Get the bounding box of the domain.
    
    Parameters:
    -----------
    basis : Basis
        The finite element basis
        
    Returns:
    --------
    bounds : dict
        Dictionary with keys 'x', 'y', 'z' and values (min, max) tuples
        
    Examples:
    ---------
    >>> bounds = get_domain_bounds(basis)
    >>> xmin, xmax = bounds['x']
    >>> # Get nodes on the right boundary
    >>> nodes_right = get_boundary_nodes_by_plane(basis, 'x', xmax)
    """
    X = basis.doflocs
    return {
        'x': (float(X[0].min()), float(X[0].max())),
        'y': (float(X[1].min()), float(X[1].max())),
        'z': (float(X[2].min()), float(X[2].max()))
    }


def get_boundary_faces(basis: Basis, axis: str = 'x') -> Dict[str, np.ndarray]:
    """
    Get nodes on the min and max faces perpendicular to an axis.
    
    Parameters:
    -----------
    basis : Basis
        The finite element basis
    axis : str
        Axis perpendicular to the faces ('x', 'y', or 'z')
        
    Returns:
    --------
    faces : dict
        Dictionary with 'min' and 'max' keys containing node arrays
        
    Examples:
    ---------
    >>> # Get left and right boundary nodes
    >>> x_faces = get_boundary_faces(basis, axis='x')
    >>> nodes_left = x_faces['min']
    >>> nodes_right = x_faces['max']
    >>> 
    >>> # Set different BCs on each face
    >>> bc_left = create_dirichlet_bc(basis, value=0.0, nodes=nodes_left)
    >>> bc_right = create_dirichlet_bc(basis, value=1.0, nodes=nodes_right)
    """
    bounds = get_domain_bounds(basis)
    axis_bounds = bounds[axis.lower()]
    
    nodes_min = get_boundary_nodes_by_plane(basis, axis, axis_bounds[0])
    nodes_max = get_boundary_nodes_by_plane(basis, axis, axis_bounds[1])
    
    return {
        'min': nodes_min,
        'max': nodes_max
    }
