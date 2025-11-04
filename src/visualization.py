"""
Visualization utilities for plotting potential and wave function probability densities.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_potential_and_density(basis, potential, psi, figsize=(15, 5), 
                                slice_axis='z', slice_value=0.5,
                                cmap_potential='viridis', cmap_density='hot',
                                save_path=None):
    """
    Plot potential and wave function probability density on a 2D slice through the 3D domain.
    
    Parameters:
    -----------
    basis : skfem.Basis
        The finite element basis
    potential : array of shape (ndofs,)
        Potential values at DOFs
    psi : array of shape (ndofs,) or (ndofs, nmodes)
        Wave function(s) at DOFs. If 2D, plots the first mode (psi[:, 0])
    figsize : tuple, optional
        Figure size (default: (15, 5))
    slice_axis : str, optional
        Which axis to slice through: 'x', 'y', or 'z' (default: 'z')
    slice_value : float, optional
        Position along slice_axis to take the slice (default: 0.5)
    cmap_potential : str, optional
        Colormap for potential (default: 'viridis')
    cmap_density : str, optional
        Colormap for probability density (default: 'hot')
    save_path : str, optional
        If provided, save figure to this path
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    # Extract coordinates
    X = basis.doflocs  # shape (3, ndofs)
    
    # Handle multi-mode psi
    if psi.ndim == 2:
        psi = psi[:, 0]
    
    # Compute probability density
    prob_density = np.abs(psi)**2
    
    # Determine slice indices
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if slice_axis.lower() not in axis_map:
        raise ValueError(f"slice_axis must be 'x', 'y', or 'z', got {slice_axis}")
    
    slice_idx = axis_map[slice_axis.lower()]
    other_axes = [i for i in range(3) if i != slice_idx]
    
    # Find points near the slice
    tolerance = 0.05  # Slice thickness
    mask = np.abs(X[slice_idx, :] - slice_value) < tolerance
    
    if np.sum(mask) < 10:
        print(f"Warning: Only {np.sum(mask)} points found near slice. Expanding tolerance.")
        tolerance = 0.1
        mask = np.abs(X[slice_idx, :] - slice_value) < tolerance
    
    # Extract slice data
    x_slice = X[other_axes[0], mask]
    y_slice = X[other_axes[1], mask]
    V_slice = potential[mask]
    rho_slice = prob_density[mask]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot potential
    sc1 = axes[0].scatter(x_slice, y_slice, c=V_slice, cmap=cmap_potential, 
                          s=20, alpha=0.8, edgecolors='none')
    axes[0].set_xlabel(['x', 'y', 'z'][other_axes[0]])
    axes[0].set_ylabel(['x', 'y', 'z'][other_axes[1]])
    axes[0].set_title(f'Potential (slice at {slice_axis}={slice_value:.2f})')
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0], label='Potential')
    
    # Plot probability density
    sc2 = axes[1].scatter(x_slice, y_slice, c=rho_slice, cmap=cmap_density,
                          s=20, alpha=0.8, edgecolors='none')
    axes[1].set_xlabel(['x', 'y', 'z'][other_axes[0]])
    axes[1].set_ylabel(['x', 'y', 'z'][other_axes[1]])
    axes[1].set_title(f'|ψ|² (slice at {slice_axis}={slice_value:.2f})')
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1], label='Probability Density')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def plot_multiple_slices(basis, potential, psi, slice_axis='z', 
                         slice_positions=[0.25, 0.5, 0.75],
                         figsize=(15, 10), cmap='viridis', save_path=None):
    """
    Plot probability density on multiple slices through the domain.
    
    Parameters:
    -----------
    basis : skfem.Basis
        The finite element basis
    potential : array of shape (ndofs,)
        Potential values at DOFs (used for reference, not plotted)
    psi : array of shape (ndofs,) or (ndofs, nmodes)
        Wave function(s) at DOFs. If 2D, plots the first mode
    slice_axis : str, optional
        Which axis to slice through: 'x', 'y', or 'z' (default: 'z')
    slice_positions : list of float, optional
        Positions along slice_axis for multiple slices
    figsize : tuple, optional
        Figure size
    cmap : str, optional
        Colormap for probability density
    save_path : str, optional
        If provided, save figure to this path
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    # Extract coordinates
    X = basis.doflocs
    
    # Handle multi-mode psi
    if psi.ndim == 2:
        psi = psi[:, 0]
    
    # Compute probability density
    prob_density = np.abs(psi)**2
    
    # Determine slice indices
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    slice_idx = axis_map[slice_axis.lower()]
    other_axes = [i for i in range(3) if i != slice_idx]
    
    # Create subplots
    n_slices = len(slice_positions)
    ncols = min(3, n_slices)
    nrows = (n_slices + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_slices == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, pos in enumerate(slice_positions):
        # Find points near this slice
        tolerance = 0.05
        mask = np.abs(X[slice_idx, :] - pos) < tolerance
        
        if np.sum(mask) < 10:
            tolerance = 0.1
            mask = np.abs(X[slice_idx, :] - pos) < tolerance
        
        # Extract slice data
        x_slice = X[other_axes[0], mask]
        y_slice = X[other_axes[1], mask]
        rho_slice = prob_density[mask]
        
        # Plot
        sc = axes[i].scatter(x_slice, y_slice, c=rho_slice, cmap=cmap,
                            s=20, alpha=0.8, edgecolors='none')
        axes[i].set_xlabel(['x', 'y', 'z'][other_axes[0]])
        axes[i].set_ylabel(['x', 'y', 'z'][other_axes[1]])
        axes[i].set_title(f'{slice_axis}={pos:.2f}')
        axes[i].set_aspect('equal')
        plt.colorbar(sc, ax=axes[i], label='|ψ|²')
    
    # Hide unused subplots
    for i in range(n_slices, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Probability Density Slices (along {slice_axis}-axis)', 
                 fontsize=14, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def plot_3d_isosurface(basis, psi, iso_level=0.5, figsize=(10, 8),
                       cmap='coolwarm', alpha=0.6, save_path=None):
    """
    Plot 3D isosurface of probability density.
    
    Note: This creates a simple scatter plot. For true isosurfaces, 
    consider using pyvista or mayavi.
    
    Parameters:
    -----------
    basis : skfem.Basis
        The finite element basis
    psi : array of shape (ndofs,) or (ndofs, nmodes)
        Wave function(s) at DOFs
    iso_level : float, optional
        Threshold level for isosurface (as fraction of max density)
    figsize : tuple, optional
        Figure size
    cmap : str, optional
        Colormap
    alpha : float, optional
        Transparency
    save_path : str, optional
        If provided, save figure to this path
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    # Extract coordinates
    X = basis.doflocs
    
    # Handle multi-mode psi
    if psi.ndim == 2:
        psi = psi[:, 0]
    
    # Compute probability density
    prob_density = np.abs(psi)**2
    
    # Select points above threshold
    threshold = iso_level * np.max(prob_density)
    mask = prob_density > threshold
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(X[0, mask], X[1, mask], X[2, mask], 
                   c=prob_density[mask], cmap=cmap, 
                   s=10, alpha=alpha, edgecolors='none')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'Probability Density (|ψ|² > {iso_level:.1%} of max)')
    
    plt.colorbar(sc, ax=ax, label='|ψ|²', shrink=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax


def plot_1d_line_profile(basis, potential, psi, axis='z', fixed_coords=None,
                         figsize=(12, 5), save_path=None):
    """
    Plot 1D line profiles of potential and probability density along a coordinate axis.
    
    Parameters:
    -----------
    basis : skfem.Basis
        The finite element basis
    potential : array of shape (ndofs,)
        Potential values at DOFs
    psi : array of shape (ndofs,) or (ndofs, nmodes)
        Wave function(s) at DOFs
    axis : str, optional
        Axis along which to plot: 'x', 'y', or 'z' (default: 'z')
    fixed_coords : dict, optional
        Fixed coordinates for other axes, e.g., {'x': 0.5, 'y': 0.5}
        If None, uses center of domain
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    # Extract coordinates
    X = basis.doflocs
    
    # Handle multi-mode psi
    if psi.ndim == 2:
        psi = psi[:, 0]
    
    # Compute probability density
    prob_density = np.abs(psi)**2
    
    # Determine axis
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[axis.lower()]
    
    # Set fixed coordinates
    if fixed_coords is None:
        fixed_coords = {}
        for ax_name, idx in axis_map.items():
            if idx != axis_idx:
                fixed_coords[ax_name] = (np.min(X[idx, :]) + np.max(X[idx, :])) / 2
    
    # Build mask for points near the line
    mask = np.ones(X.shape[1], dtype=bool)
    tolerance = 0.05
    
    for ax_name, value in fixed_coords.items():
        idx = axis_map[ax_name]
        mask &= np.abs(X[idx, :] - value) < tolerance
    
    if np.sum(mask) < 10:
        print(f"Warning: Only {np.sum(mask)} points found. Expanding tolerance.")
        tolerance = 0.1
        mask = np.ones(X.shape[1], dtype=bool)
        for ax_name, value in fixed_coords.items():
            idx = axis_map[ax_name]
            mask &= np.abs(X[idx, :] - value) < tolerance
    
    # Extract line data
    coord = X[axis_idx, mask]
    V_line = potential[mask]
    rho_line = prob_density[mask]
    
    # Sort by coordinate
    sort_idx = np.argsort(coord)
    coord = coord[sort_idx]
    V_line = V_line[sort_idx]
    rho_line = rho_line[sort_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot potential
    axes[0].plot(coord, V_line, 'b.-', linewidth=2, markersize=4)
    axes[0].set_xlabel(axis)
    axes[0].set_ylabel('Potential')
    axes[0].set_title(f'Potential along {axis}-axis')
    axes[0].grid(True, alpha=0.3)
    
    # Plot probability density
    axes[1].plot(coord, rho_line, 'r.-', linewidth=2, markersize=4)
    axes[1].set_xlabel(axis)
    axes[1].set_ylabel('|ψ|²')
    axes[1].set_title(f'Probability Density along {axis}-axis')
    axes[1].grid(True, alpha=0.3)
    
    # Add fixed coordinates to title
    fixed_str = ', '.join([f'{k}={v:.2f}' for k, v in fixed_coords.items()])
    fig.suptitle(f'Line profiles ({fixed_str})', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def plot_energy_levels(energies, n_levels=None, figsize=(8, 6), save_path=None):
    """
    Plot energy level diagram.
    
    Parameters:
    -----------
    energies : array
        Eigenvalues (energy levels)
    n_levels : int, optional
        Number of levels to plot (default: all)
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        If provided, save figure to this path
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    if n_levels is None:
        n_levels = len(energies)
    else:
        n_levels = min(n_levels, len(energies))
    
    E = energies[:n_levels]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, energy in enumerate(E):
        ax.hlines(energy, 0, 1, colors='blue', linewidth=2)
        ax.text(1.05, energy, f'E_{i} = {energy:.6f}', 
                va='center', fontsize=10)
    
    ax.set_xlim(-0.1, 1.5)
    ax.set_ylim(E[0] - 0.1 * (E[-1] - E[0]), 
                E[-1] + 0.1 * (E[-1] - E[0]))
    ax.set_ylabel('Energy')
    ax.set_title(f'Energy Levels (first {n_levels} states)')
    ax.set_xticks([])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, ax
