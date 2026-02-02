"""
Forward Wave Propagation for Surfzone Model

Propagates wave height forward along a traced path from boundary to mesh point
using shoaling transformation.

The path is obtained by reversing the backward-traced ray path.
At each step, we compute local wave properties and apply shoaling to update
the wave height.

Note: This module does NOT include breaking detection - that is deferred
to future work.
"""

import numpy as np
from numba import njit
from typing import Tuple

from .wave_physics import (
    deep_water_properties,
    local_wave_properties,
    shoaling_coefficient,
)
from .ray_tracer import interpolate_depth_indexed


@njit(cache=True)
def interpolate_depth_along_path(
    path_x: np.ndarray,
    path_y: np.ndarray,
    # Mesh arrays
    points_x: np.ndarray,
    points_y: np.ndarray,
    depth: np.ndarray,
    triangles: np.ndarray,
    # Spatial index
    grid_x_min: float,
    grid_y_min: float,
    grid_cell_size: float,
    grid_n_cells_x: int,
    grid_n_cells_y: int,
    grid_cell_starts: np.ndarray,
    grid_cell_counts: np.ndarray,
    grid_triangles: np.ndarray,
) -> np.ndarray:
    """
    Interpolate depth at each point along the path.

    Uses the mesh spatial index for fast triangle lookup.

    Args:
        path_x, path_y: Path coordinates (N points)
        points_x, points_y, depth, triangles: Mesh arrays
        grid_*: Spatial index arrays

    Returns:
        depth_along_path: Depth at each path point (N,)
            Positive values indicate water depth.
            NaN indicates point outside mesh domain.
    """
    n_points = len(path_x)
    depth_along_path = np.empty(n_points, dtype=np.float64)

    for i in range(n_points):
        depth_along_path[i] = interpolate_depth_indexed(
            path_x[i], path_y[i],
            points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles,
        )

    return depth_along_path


@njit(cache=True)
def propagate_wave_forward(
    path_depth: np.ndarray,
    H0: float,
    T: float,
) -> Tuple[float, float, np.ndarray]:
    """
    Propagate wave height forward along path using shoaling.

    Steps through the path from boundary (deep water) toward mesh point (shallow),
    computing the shoaling coefficient at each step.

    The shoaling coefficient Ks relates wave height to group velocity:
        H = H0 * Ks
        Ks = sqrt(Cg0 / Cg)

    where Cg0 is the group velocity at the boundary (first point in path)
    and Cg is the local group velocity.

    Args:
        path_depth: Depth at each point along path (N,)
                    Path should go from boundary (deep) toward mesh (shallow)
        H0: Wave height at boundary (start of path) (m)
        T: Wave period (s)

    Returns:
        H_final: Wave height at end of path (mesh point) (m)
        K_total: Total shoaling coefficient at mesh point
        H_along_path: Wave height at each step (N,) - for debugging/visualization
    """
    n_points = len(path_depth)

    if n_points == 0:
        return H0, 1.0, np.array([H0])

    # Get deep water reference properties
    L0, C0, Cg0 = deep_water_properties(T)

    # Allocate output array
    H_along_path = np.empty(n_points, dtype=np.float64)

    # Get group velocity at boundary (first point)
    h_boundary = path_depth[0]
    if h_boundary > 0 and not np.isnan(h_boundary):
        _, _, _, _, Cg_boundary = local_wave_properties(L0, T, h_boundary)
    else:
        # Use deep water value if boundary depth is invalid
        Cg_boundary = Cg0

    # Propagate forward along path
    for i in range(n_points):
        h = path_depth[i]

        if np.isnan(h) or h <= 0:
            # Invalid depth - use previous value or H0
            if i > 0:
                H_along_path[i] = H_along_path[i - 1]
            else:
                H_along_path[i] = H0
            continue

        # Compute local wave properties
        _, _, _, _, Cg_local = local_wave_properties(L0, T, h)

        # Shoaling coefficient relative to boundary
        # Ks = sqrt(Cg_boundary / Cg_local)
        if Cg_local > 0:
            Ks = np.sqrt(Cg_boundary / Cg_local)
        else:
            Ks = 1.0

        # Apply shoaling
        H_along_path[i] = H0 * Ks

    # Final values at mesh point (last point in path)
    H_final = H_along_path[-1]

    # Total shoaling coefficient
    K_total = H_final / H0 if H0 > 0 else 1.0

    return H_final, K_total, H_along_path


@njit(cache=True)
def propagate_wave_forward_simple(
    depth_at_boundary: float,
    depth_at_mesh: float,
    H0: float,
    T: float,
) -> Tuple[float, float]:
    """
    Simple shoaling calculation between two depths.

    This is a simplified version that only considers start and end depths,
    without stepping through the full path. Useful when the path is not
    needed for visualization.

    The result is identical to stepping through the path if depth
    varies smoothly (which it does for most surfzone applications).

    Args:
        depth_at_boundary: Water depth at boundary (m)
        depth_at_mesh: Water depth at mesh point (m)
        H0: Wave height at boundary (m)
        T: Wave period (s)

    Returns:
        H_final: Wave height at mesh point (m)
        K_total: Shoaling coefficient
    """
    if H0 <= 0:
        return 0.0, 1.0

    # Get deep water reference
    L0, C0, Cg0 = deep_water_properties(T)

    # Group velocity at boundary
    if depth_at_boundary > 0 and not np.isnan(depth_at_boundary):
        _, _, _, _, Cg_boundary = local_wave_properties(L0, T, depth_at_boundary)
    else:
        Cg_boundary = Cg0

    # Group velocity at mesh point
    if depth_at_mesh > 0 and not np.isnan(depth_at_mesh):
        _, _, _, _, Cg_mesh = local_wave_properties(L0, T, depth_at_mesh)
    else:
        Cg_mesh = Cg_boundary  # No shoaling if depth invalid

    # Shoaling coefficient
    if Cg_mesh > 0:
        K_total = np.sqrt(Cg_boundary / Cg_mesh)
    else:
        K_total = 1.0

    H_final = H0 * K_total

    return H_final, K_total


def forward_propagate_from_backward_path(
    path_x: np.ndarray,
    path_y: np.ndarray,
    H_boundary: float,
    T: float,
    mesh_arrays: dict,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Forward propagate wave height using a backward-traced path.

    The backward path goes from mesh point to boundary, so we reverse it
    to get the forward propagation direction (boundary to mesh).

    Args:
        path_x, path_y: Path from backward tracing (mesh -> boundary)
        H_boundary: Wave height at boundary (m)
        T: Wave period (s)
        mesh_arrays: Dictionary of Numba-compatible mesh arrays

    Returns:
        H_at_mesh: Wave height at mesh point (m)
        K_shoaling: Total shoaling coefficient
        forward_x: Forward path x coordinates (boundary -> mesh)
        forward_y: Forward path y coordinates (boundary -> mesh)
    """
    # Reverse the path (mesh->boundary becomes boundary->mesh)
    forward_x = path_x[::-1].copy()
    forward_y = path_y[::-1].copy()

    # Interpolate depth along forward path
    path_depth = interpolate_depth_along_path(
        forward_x, forward_y,
        mesh_arrays['points_x'],
        mesh_arrays['points_y'],
        mesh_arrays['depth'],
        mesh_arrays['triangles'],
        mesh_arrays['grid_x_min'],
        mesh_arrays['grid_y_min'],
        mesh_arrays['grid_cell_size'],
        mesh_arrays['grid_n_cells_x'],
        mesh_arrays['grid_n_cells_y'],
        mesh_arrays['grid_cell_starts'],
        mesh_arrays['grid_cell_counts'],
        mesh_arrays['grid_triangles'],
    )

    # Propagate wave forward
    H_at_mesh, K_shoaling, _ = propagate_wave_forward(path_depth, H_boundary, T)

    return H_at_mesh, K_shoaling, forward_x, forward_y
