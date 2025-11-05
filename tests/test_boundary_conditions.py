"""
Tests for boundary condition utilities.
"""
import numpy as np
import pytest
from src import solver, boundary_conditions as bc_utils


def test_get_boundary_nodes():
    """Test basic boundary node extraction."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                                char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    bdofs = bc_utils.get_boundary_nodes(mesh)
    
    assert len(bdofs) > 0
    assert bdofs.dtype == np.int64 or bdofs.dtype == np.int32
    assert np.all(bdofs >= 0)
    assert np.all(bdofs < basis.N)


def test_get_boundary_nodes_by_plane():
    """Test getting boundary nodes on a specific plane."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                                char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    # Get nodes on x=0 plane
    nodes_x0 = bc_utils.get_boundary_nodes_by_plane(basis, axis='x', value=0.0)
    
    assert len(nodes_x0) > 0
    
    # Verify that all nodes are indeed on x=0
    X = basis.doflocs
    assert np.allclose(X[0, nodes_x0], 0.0, atol=1e-6)
    
    # Get nodes on x=1 plane
    nodes_x1 = bc_utils.get_boundary_nodes_by_plane(basis, axis='x', value=1.0)
    assert len(nodes_x1) > 0
    assert np.allclose(X[0, nodes_x1], 1.0, atol=1e-6)
    
    # Test other axes
    nodes_y0 = bc_utils.get_boundary_nodes_by_plane(basis, axis='y', value=0.0)
    nodes_z1 = bc_utils.get_boundary_nodes_by_plane(basis, axis='z', value=1.0)
    assert len(nodes_y0) > 0
    assert len(nodes_z1) > 0


def test_get_boundary_nodes_by_box():
    """Test getting boundary nodes within a bounding box."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                                char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    X = basis.doflocs
    
    # Get nodes on left face (x near 0)
    nodes_left = bc_utils.get_boundary_nodes_by_box(basis, xmin=-0.01, xmax=0.01)
    assert len(nodes_left) > 0
    assert np.all(X[0, nodes_left] < 0.02)
    
    # Get nodes in a corner
    nodes_corner = bc_utils.get_boundary_nodes_by_box(
        basis, xmin=0.9, ymin=0.9, zmin=0.9
    )
    assert len(nodes_corner) > 0
    assert np.all(X[0, nodes_corner] > 0.89)
    assert np.all(X[1, nodes_corner] > 0.89)
    assert np.all(X[2, nodes_corner] > 0.89)


def test_get_boundary_nodes_by_function():
    """Test getting boundary nodes using a custom function."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                                char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    X = basis.doflocs
    
    # Get nodes where x + y > 1.5
    def condition(X):
        return X[0] + X[1] > 1.5
    
    nodes = bc_utils.get_boundary_nodes_by_function(basis, condition)
    assert len(nodes) > 0
    
    # Verify condition
    assert np.all(X[0, nodes] + X[1, nodes] > 1.49)


def test_create_dirichlet_bc_constant():
    """Test creating Dirichlet BC with constant value."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                                char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    # Create BC with constant value
    bc = bc_utils.create_dirichlet_bc(basis, value=1.5)
    
    assert bc['type'] == 'dirichlet'
    assert len(bc['nodes']) > 0
    assert np.isscalar(bc['values']) or len(bc['values']) == len(bc['nodes'])


def test_create_dirichlet_bc_callable():
    """Test creating Dirichlet BC with position-dependent value."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                                char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    X = basis.doflocs
    
    # Create BC with callable value
    def bc_func(X):
        return X[0] + X[1]  # phi = x + y on boundary
    
    bc = bc_utils.create_dirichlet_bc(basis, value=bc_func)
    
    assert bc['type'] == 'dirichlet'
    assert len(bc['nodes']) > 0
    assert len(bc['values']) == len(bc['nodes'])
    
    # Verify values
    expected = X[0, bc['nodes']] + X[1, bc['nodes']]
    assert np.allclose(bc['values'], expected, atol=1e-12)


def test_create_dirichlet_bc_specific_nodes():
    """Test creating BC on specific nodes."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                                char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    # Get specific nodes
    nodes_x0 = bc_utils.get_boundary_nodes_by_plane(basis, axis='x', value=0.0)
    
    # Create BC only on these nodes
    bc = bc_utils.create_dirichlet_bc(basis, value=2.0, nodes=nodes_x0)
    
    assert bc['type'] == 'dirichlet'
    assert len(bc['nodes']) == len(nodes_x0)
    assert np.array_equal(bc['nodes'], nodes_x0)


def test_combine_boundary_conditions():
    """Test combining multiple boundary conditions."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                                char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    # Create BCs for different faces
    nodes_left = bc_utils.get_boundary_nodes_by_plane(basis, 'x', 0.0)
    nodes_right = bc_utils.get_boundary_nodes_by_plane(basis, 'x', 1.0)
    
    bc_left = bc_utils.create_dirichlet_bc(basis, value=0.0, nodes=nodes_left)
    bc_right = bc_utils.create_dirichlet_bc(basis, value=1.0, nodes=nodes_right)
    
    # Combine
    bc_combined = bc_utils.combine_boundary_conditions([bc_left, bc_right])
    
    assert 'nodes' in bc_combined
    assert 'values' in bc_combined
    assert len(bc_combined['nodes']) > 0
    assert len(bc_combined['values']) == len(bc_combined['nodes'])


def test_get_domain_bounds():
    """Test getting domain bounding box."""
    mesh = solver.make_mesh_box(x0=(0.5, 0.5, 0.5), lengths=(2.0, 1.0, 1.5), 
                                char_length=0.4, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    bounds = bc_utils.get_domain_bounds(basis)
    
    assert 'x' in bounds
    assert 'y' in bounds
    assert 'z' in bounds
    
    # Check approximate bounds
    xmin, xmax = bounds['x']
    ymin, ymax = bounds['y']
    zmin, zmax = bounds['z']
    
    assert xmin < 0.6  # Should be close to 0.5
    assert xmax > 2.4  # Should be close to 2.5
    assert ymin < 0.6
    assert ymax > 1.4
    assert zmin < 0.6
    assert zmax > 1.9


def test_get_boundary_faces():
    """Test getting min and max boundary faces."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                                char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    X = basis.doflocs
    
    # Get x-faces
    x_faces = bc_utils.get_boundary_faces(basis, axis='x')
    
    assert 'min' in x_faces
    assert 'max' in x_faces
    assert len(x_faces['min']) > 0
    assert len(x_faces['max']) > 0
    
    # Verify positions
    assert np.allclose(X[0, x_faces['min']], 0.0, atol=1e-6)
    assert np.allclose(X[0, x_faces['max']], 1.0, atol=1e-6)
    
    # Test other axes
    y_faces = bc_utils.get_boundary_faces(basis, axis='y')
    z_faces = bc_utils.get_boundary_faces(basis, axis='z')
    
    assert len(y_faces['min']) > 0
    assert len(z_faces['max']) > 0


def test_solve_poisson_with_bc_dict():
    """Test solving Poisson equation with boundary condition dictionary."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                                char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    # Create source term
    rho = np.ones(basis.N)
    
    # Create boundary condition: phi=0 on left (x=0), phi=1 on right (x=1)
    nodes_left = bc_utils.get_boundary_nodes_by_plane(basis, 'x', 0.0)
    nodes_right = bc_utils.get_boundary_nodes_by_plane(basis, 'x', 1.0)
    
    bc_left = bc_utils.create_dirichlet_bc(basis, value=0.0, nodes=nodes_left)
    bc_right = bc_utils.create_dirichlet_bc(basis, value=1.0, nodes=nodes_right)
    bc = bc_utils.combine_boundary_conditions([bc_left, bc_right])
    
    # Solve with BC dictionary
    phi = solver.solve_poisson(mesh, basis, rho, bc=bc)
    
    assert phi.shape[0] == basis.N
    assert np.all(np.isfinite(phi))
    
    # Check boundary values
    assert np.allclose(phi[nodes_left], 0.0, atol=1e-6)
    assert np.allclose(phi[nodes_right], 1.0, atol=1e-6)


def test_solve_poisson_backward_compatibility():
    """Test that old bc_value parameter still works."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                                char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    
    rho = np.ones(basis.N)
    
    # Old way should still work
    phi = solver.solve_poisson(mesh, basis, rho, bc_value=0.5)
    
    assert phi.shape[0] == basis.N
    assert np.all(np.isfinite(phi))
    
    # Check that boundary values are approximately 0.5
    bdofs = bc_utils.get_boundary_nodes(mesh)
    assert np.allclose(phi[bdofs], 0.5, atol=1e-6)


def test_solve_poisson_with_callable_bc():
    """Test solving with position-dependent boundary values."""
    mesh = solver.make_mesh_box(x0=(0, 0, 0), lengths=(1.0, 1.0, 1.0), 
                                char_length=0.3, verbose=False)
    mesh, basis, K, M = solver.assemble_operators(mesh)
    X = basis.doflocs
    
    # Zero source
    rho = np.zeros(basis.N)
    
    # Boundary condition: phi = x on all boundaries (should give phi = x everywhere)
    def bc_func(X):
        return X[0]
    
    bc = bc_utils.create_dirichlet_bc(basis, value=bc_func)
    phi = solver.solve_poisson(mesh, basis, rho, bc=bc)
    
    # Solution should be approximately phi = x
    # (exact in the limit of fine mesh for this simple problem)
    expected = X[0, :]
    rel_error = np.linalg.norm(phi - expected) / (np.linalg.norm(expected) + 1e-16)
    
    # Allow some error due to discretization
    assert rel_error < 0.1
