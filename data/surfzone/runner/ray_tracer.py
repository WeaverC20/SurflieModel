"""
Ray Tracer for Surfzone Wave Propagation (LEGACY - DO NOT USE)

DEPRECATED: This is the old FORWARD ray tracer. Use backward_ray_tracer.py instead.

The forward approach traces rays from offshore boundary toward shore, which is
inefficient because many rays miss areas of interest. The backward approach
(backward_ray_tracer.py) traces from mesh points toward the boundary, ensuring
every computation is useful.

This file is kept only for the utility functions (interpolate_depth_indexed,
celerity_gradient_indexed) which are used by the backward tracer.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
from numba import njit, prange

from .wave_physics import (
    G,
    deep_water_properties,
    local_wave_properties,
    local_wavelength_fenton_mckee,
    local_celerity,
    shoaling_coefficient,
    refraction_snell,
    refraction_coefficient,
    wind_modification,
    wave_height,
    breaker_index_rattanapitikon,
    breaker_index_wind_modified,
    check_breaking,
    iribarren_number,
    classify_breaker_type,
    calculate_slope,
    nautical_to_math,
    update_ray_direction,
    BREAKER_TYPE_LABELS,
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_STEP_SIZE = 10.0  # meters (base step, adapts to 3m in shallow water)
DEFAULT_MAX_STEPS = 1000  # Increased for smaller steps
DEFAULT_MIN_DEPTH = 0.15  # meters (stop tracing when depth < this)


@dataclass
class RayResult:
    """
    Result of tracing a single wave ray.

    Attributes:
        start_x: Starting UTM x coordinate (m)
        start_y: Starting UTM y coordinate (m)
        break_x: Breaking UTM x coordinate (m), NaN if no breaking
        break_y: Breaking UTM y coordinate (m), NaN if no breaking
        break_depth: Water depth at breaking (m)
        break_height: Wave height at breaking (m)
        break_period: Wave period at breaking (s)
        breaker_type: Breaker type code (0=spilling, 1=plunging, 2=collapsing, 3=surging)
        iribarren: Iribarren number at breaking
        beach_slope: Local beach slope at breaking
        did_break: Whether the wave broke within the domain
        termination_reason: Why ray tracing stopped
        path_x: Array of x coordinates along ray path (if stored)
        path_y: Array of y coordinates along ray path (if stored)
        path_h: Array of heights along ray path (if stored)
        partition_id: Source wave partition ID
    """
    start_x: float
    start_y: float
    break_x: float
    break_y: float
    break_depth: float
    break_height: float
    break_period: float
    breaker_type: int
    iribarren: float
    beach_slope: float
    did_break: bool
    termination_reason: str
    path_x: Optional[np.ndarray] = None
    path_y: Optional[np.ndarray] = None
    path_h: Optional[np.ndarray] = None
    partition_id: int = 0

    @property
    def breaker_type_label(self) -> str:
        return BREAKER_TYPE_LABELS.get(self.breaker_type, "Unknown")


# =============================================================================
# Spatial Index for Fast Triangle Lookup
# =============================================================================

def build_triangle_grid(
    points_x: np.ndarray,
    points_y: np.ndarray,
    triangles: np.ndarray,
    cell_size: float = 100.0,
) -> Dict[str, np.ndarray]:
    """
    Build a grid-based spatial index for fast triangle lookup.

    Creates a regular grid where each cell stores the indices of triangles
    that overlap it. This enables O(1) lookup instead of O(n) linear search.

    Args:
        points_x, points_y: Mesh vertex coordinates
        triangles: Triangle vertex indices (n_triangles, 3)
        cell_size: Size of grid cells in meters

    Returns:
        Dictionary with grid arrays for use in Numba functions:
        - grid_x_min, grid_y_min: Grid origin
        - grid_cell_size: Cell size
        - grid_n_cells_x, grid_n_cells_y: Number of cells
        - grid_cell_starts: Start index in grid_triangles for each cell
        - grid_cell_counts: Number of triangles in each cell
        - grid_triangles: Flattened array of triangle indices by cell
    """
    # Compute bounding box with padding
    x_min = points_x.min() - cell_size
    x_max = points_x.max() + cell_size
    y_min = points_y.min() - cell_size
    y_max = points_y.max() + cell_size

    n_cells_x = int(np.ceil((x_max - x_min) / cell_size))
    n_cells_y = int(np.ceil((y_max - y_min) / cell_size))
    n_cells = n_cells_x * n_cells_y

    logger.info(f"Building spatial index: {n_cells_x}x{n_cells_y} = {n_cells:,} cells, cell_size={cell_size}m")

    # For each triangle, find which cells it overlaps
    # Use triangle bounding box to find candidate cells
    n_triangles = triangles.shape[0]

    # First pass: count triangles per cell
    cell_counts = np.zeros(n_cells, dtype=np.int32)

    for tri_idx in range(n_triangles):
        i0, i1, i2 = triangles[tri_idx]

        # Triangle bounding box
        tri_x_min = min(points_x[i0], points_x[i1], points_x[i2])
        tri_x_max = max(points_x[i0], points_x[i1], points_x[i2])
        tri_y_min = min(points_y[i0], points_y[i1], points_y[i2])
        tri_y_max = max(points_y[i0], points_y[i1], points_y[i2])

        # Cell range this triangle overlaps
        cx_min = max(0, int((tri_x_min - x_min) / cell_size))
        cx_max = min(n_cells_x - 1, int((tri_x_max - x_min) / cell_size))
        cy_min = max(0, int((tri_y_min - y_min) / cell_size))
        cy_max = min(n_cells_y - 1, int((tri_y_max - y_min) / cell_size))

        for cy in range(cy_min, cy_max + 1):
            for cx in range(cx_min, cx_max + 1):
                cell_idx = cy * n_cells_x + cx
                cell_counts[cell_idx] += 1

    # Compute start indices (prefix sum)
    cell_starts = np.zeros(n_cells + 1, dtype=np.int32)
    cell_starts[1:] = np.cumsum(cell_counts)
    total_entries = cell_starts[-1]

    logger.info(f"  Total cell entries: {total_entries:,} (avg {total_entries/n_cells:.1f} per cell)")

    # Second pass: fill triangle indices
    grid_triangles = np.empty(total_entries, dtype=np.int32)
    cell_fill = np.zeros(n_cells, dtype=np.int32)  # Current fill position per cell

    for tri_idx in range(n_triangles):
        i0, i1, i2 = triangles[tri_idx]

        tri_x_min = min(points_x[i0], points_x[i1], points_x[i2])
        tri_x_max = max(points_x[i0], points_x[i1], points_x[i2])
        tri_y_min = min(points_y[i0], points_y[i1], points_y[i2])
        tri_y_max = max(points_y[i0], points_y[i1], points_y[i2])

        cx_min = max(0, int((tri_x_min - x_min) / cell_size))
        cx_max = min(n_cells_x - 1, int((tri_x_max - x_min) / cell_size))
        cy_min = max(0, int((tri_y_min - y_min) / cell_size))
        cy_max = min(n_cells_y - 1, int((tri_y_max - y_min) / cell_size))

        for cy in range(cy_min, cy_max + 1):
            for cx in range(cx_min, cx_max + 1):
                cell_idx = cy * n_cells_x + cx
                pos = cell_starts[cell_idx] + cell_fill[cell_idx]
                grid_triangles[pos] = tri_idx
                cell_fill[cell_idx] += 1

    return {
        'grid_x_min': np.float64(x_min),
        'grid_y_min': np.float64(y_min),
        'grid_cell_size': np.float64(cell_size),
        'grid_n_cells_x': np.int32(n_cells_x),
        'grid_n_cells_y': np.int32(n_cells_y),
        'grid_cell_starts': cell_starts.astype(np.int32),
        'grid_cell_counts': cell_counts.astype(np.int32),
        'grid_triangles': grid_triangles.astype(np.int32),
    }


# =============================================================================
# Core Ray Tracing (Numba-accelerated)
# =============================================================================

@njit(cache=True)
def interpolate_depth_indexed(
    x: float,
    y: float,
    points_x: np.ndarray,
    points_y: np.ndarray,
    depth: np.ndarray,
    triangles: np.ndarray,
    grid_x_min: float,
    grid_y_min: float,
    grid_cell_size: float,
    grid_n_cells_x: int,
    grid_n_cells_y: int,
    grid_cell_starts: np.ndarray,
    grid_cell_counts: np.ndarray,
    grid_triangles: np.ndarray,
) -> float:
    """
    Interpolate depth at a point using grid-indexed triangle lookup.

    Uses the spatial index for O(1) cell lookup instead of O(n) linear search.

    Args:
        x, y: Query point in UTM
        points_x, points_y: Mesh vertex coordinates
        depth: Depth values at vertices
        triangles: Triangle vertex indices
        grid_*: Spatial index arrays

    Returns:
        Interpolated depth (positive = below water), NaN if outside mesh
    """
    # Find grid cell
    cx = int((x - grid_x_min) / grid_cell_size)
    cy = int((y - grid_y_min) / grid_cell_size)

    # Check bounds
    if cx < 0 or cx >= grid_n_cells_x or cy < 0 or cy >= grid_n_cells_y:
        return np.nan

    cell_idx = cy * grid_n_cells_x + cx
    start = grid_cell_starts[cell_idx]
    count = grid_cell_counts[cell_idx]

    # Search only triangles in this cell
    for j in range(count):
        tri_idx = grid_triangles[start + j]
        i0, i1, i2 = triangles[tri_idx]

        x0, y0 = points_x[i0], points_y[i0]
        x1, y1 = points_x[i1], points_y[i1]
        x2, y2 = points_x[i2], points_y[i2]

        # Barycentric coordinates
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)

        if abs(denom) < 1e-10:
            continue

        a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
        b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
        c = 1.0 - a - b

        # Check if inside triangle
        eps = -1e-6
        if a >= eps and b >= eps and c >= eps:
            d0, d1, d2 = depth[i0], depth[i1], depth[i2]
            return a * d0 + b * d1 + c * d2

    return np.nan


@njit(cache=True)
def interpolate_coast_distance_indexed(
    x: float,
    y: float,
    points_x: np.ndarray,
    points_y: np.ndarray,
    coast_distance: np.ndarray,
    triangles: np.ndarray,
    grid_x_min: float,
    grid_y_min: float,
    grid_cell_size: float,
    grid_n_cells_x: int,
    grid_n_cells_y: int,
    grid_cell_starts: np.ndarray,
    grid_cell_counts: np.ndarray,
    grid_triangles: np.ndarray,
) -> float:
    """
    Interpolate distance-from-coastline at a point using grid-indexed triangle lookup.

    Uses the spatial index for O(1) cell lookup instead of O(n) linear search.

    Args:
        x, y: Query point in UTM
        points_x, points_y: Mesh vertex coordinates
        coast_distance: Distance from coastline values at vertices (m)
        triangles: Triangle vertex indices
        grid_*: Spatial index arrays

    Returns:
        Interpolated distance from coastline (m), NaN if outside mesh
    """
    # Find grid cell
    cx = int((x - grid_x_min) / grid_cell_size)
    cy = int((y - grid_y_min) / grid_cell_size)

    # Check bounds
    if cx < 0 or cx >= grid_n_cells_x or cy < 0 or cy >= grid_n_cells_y:
        return np.nan

    cell_idx = cy * grid_n_cells_x + cx
    start = grid_cell_starts[cell_idx]
    count = grid_cell_counts[cell_idx]

    # Search only triangles in this cell
    for j in range(count):
        tri_idx = grid_triangles[start + j]
        i0, i1, i2 = triangles[tri_idx]

        x0, y0 = points_x[i0], points_y[i0]
        x1, y1 = points_x[i1], points_y[i1]
        x2, y2 = points_x[i2], points_y[i2]

        # Barycentric coordinates
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)

        if abs(denom) < 1e-10:
            continue

        a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
        b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
        c = 1.0 - a - b

        # Check if inside triangle
        eps = -1e-6
        if a >= eps and b >= eps and c >= eps:
            d0, d1, d2 = coast_distance[i0], coast_distance[i1], coast_distance[i2]
            return a * d0 + b * d1 + c * d2

    return np.nan


@njit(cache=True)
def celerity_gradient_indexed(
    x: float,
    y: float,
    T: float,
    L0: float,
    points_x: np.ndarray,
    points_y: np.ndarray,
    depth: np.ndarray,
    triangles: np.ndarray,
    grid_x_min: float,
    grid_y_min: float,
    grid_cell_size: float,
    grid_n_cells_x: int,
    grid_n_cells_y: int,
    grid_cell_starts: np.ndarray,
    grid_cell_counts: np.ndarray,
    grid_triangles: np.ndarray,
    h_step: float = 10.0,
) -> Tuple[float, float]:
    """
    Calculate celerity gradient at a point using finite differences with indexed lookup.
    """
    # Get depths at offset points using indexed lookup
    d_c = interpolate_depth_indexed(
        x, y, points_x, points_y, depth, triangles,
        grid_x_min, grid_y_min, grid_cell_size, grid_n_cells_x, grid_n_cells_y,
        grid_cell_starts, grid_cell_counts, grid_triangles
    )
    d_xp = interpolate_depth_indexed(
        x + h_step, y, points_x, points_y, depth, triangles,
        grid_x_min, grid_y_min, grid_cell_size, grid_n_cells_x, grid_n_cells_y,
        grid_cell_starts, grid_cell_counts, grid_triangles
    )
    d_xm = interpolate_depth_indexed(
        x - h_step, y, points_x, points_y, depth, triangles,
        grid_x_min, grid_y_min, grid_cell_size, grid_n_cells_x, grid_n_cells_y,
        grid_cell_starts, grid_cell_counts, grid_triangles
    )
    d_yp = interpolate_depth_indexed(
        x, y + h_step, points_x, points_y, depth, triangles,
        grid_x_min, grid_y_min, grid_cell_size, grid_n_cells_x, grid_n_cells_y,
        grid_cell_starts, grid_cell_counts, grid_triangles
    )
    d_ym = interpolate_depth_indexed(
        x, y - h_step, points_x, points_y, depth, triangles,
        grid_x_min, grid_y_min, grid_cell_size, grid_n_cells_x, grid_n_cells_y,
        grid_cell_starts, grid_cell_counts, grid_triangles
    )

    if np.isnan(d_c) or d_c <= 0:
        return 0.0, 0.0

    L_c = local_wavelength_fenton_mckee(L0, d_c)
    C_c = local_celerity(L_c, T)

    # dC/dx
    if not np.isnan(d_xp) and not np.isnan(d_xm) and d_xp > 0 and d_xm > 0:
        L_xp = local_wavelength_fenton_mckee(L0, d_xp)
        L_xm = local_wavelength_fenton_mckee(L0, d_xm)
        C_xp = local_celerity(L_xp, T)
        C_xm = local_celerity(L_xm, T)
        dC_dx = (C_xp - C_xm) / (2.0 * h_step)
    else:
        dC_dx = 0.0

    # dC/dy
    if not np.isnan(d_yp) and not np.isnan(d_ym) and d_yp > 0 and d_ym > 0:
        L_yp = local_wavelength_fenton_mckee(L0, d_yp)
        L_ym = local_wavelength_fenton_mckee(L0, d_ym)
        C_yp = local_celerity(L_yp, T)
        C_ym = local_celerity(L_ym, T)
        dC_dy = (C_yp - C_ym) / (2.0 * h_step)
    else:
        dC_dy = 0.0

    return dC_dx, dC_dy


@njit(cache=True)
def trace_single_ray(
    start_x: float,
    start_y: float,
    H0: float,
    T: float,
    theta0_nautical: float,
    U_wind: float,
    theta_wind_nautical: float,
    # Mesh data (triangulated)
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
    # Configuration
    step_size: float = DEFAULT_STEP_SIZE,
    max_steps: int = DEFAULT_MAX_STEPS,
    min_depth: float = DEFAULT_MIN_DEPTH,
    alpha_wind: float = 0.03,
    Cw: float = 0.15,
    gamma_w: float = 0.15,
) -> Tuple[
    float, float, float, float, float,  # break_x, break_y, break_depth, break_height, break_period
    int, float, float,  # breaker_type, iribarren, beach_slope
    bool, int,  # did_break, termination_code
    np.ndarray, np.ndarray, np.ndarray,  # path_x, path_y, path_h
]:
    """
    Trace a single wave ray from offshore to shore.

    This is the core Numba-accelerated function with grid-indexed triangle lookup.

    Args:
        start_x, start_y: Starting position in UTM (m)
        H0: Deep water wave height (m)
        T: Wave period (s)
        theta0_nautical: Deep water wave direction (degrees, nautical FROM)
        U_wind: Wind speed (m/s)
        theta_wind_nautical: Wind direction (degrees, nautical FROM)
        points_x, points_y, depth, triangles: Mesh data
        grid_*: Spatial index arrays
        step_size: Step distance for marching (m)
        max_steps: Maximum number of steps
        min_depth: Minimum depth before stopping (m)
        alpha_wind: Wind energy coefficient
        Cw: Wind modification coefficient for breaking
        gamma_w: Wind modification for steepness

    Returns:
        Tuple of results (see RayResult for descriptions)
        termination_code: 0=broke, 1=reached shore, 2=left domain, 3=max steps
    """
    # Initialize path storage
    path_x = np.empty(max_steps, dtype=np.float64)
    path_y = np.empty(max_steps, dtype=np.float64)
    path_h = np.empty(max_steps, dtype=np.float64)

    # Calculate deep water reference properties
    L0, C0, Cg0 = deep_water_properties(T)

    # Convert directions to math convention (radians)
    theta0 = nautical_to_math(theta0_nautical)
    theta_wind = nautical_to_math(theta_wind_nautical)

    # Initialize ray position and direction
    x = start_x
    y = start_y

    # Direction: wave travels TOWARD shore (opposite of FROM direction)
    dx = np.cos(theta0)
    dy = np.sin(theta0)

    # Initialize coefficients
    Kw = 1.0  # Wind modification factor

    # Get initial depth
    h_prev = interpolate_depth_indexed(
        x, y, points_x, points_y, depth, triangles,
        grid_x_min, grid_y_min, grid_cell_size, grid_n_cells_x, grid_n_cells_y,
        grid_cell_starts, grid_cell_counts, grid_triangles
    )
    if np.isnan(h_prev) or h_prev <= 0:
        # Invalid starting point - return empty arrays (not uninitialized garbage)
        return (
            np.nan, np.nan, np.nan, np.nan, T,
            0, 0.0, 0.0,
            False, 2,  # Left domain
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    # Track for slope calculation
    h_history = np.empty(10, dtype=np.float64)
    h_history_idx = 0
    distance_traveled = 0.0

    # Result variables
    break_x = np.nan
    break_y = np.nan
    break_depth = np.nan
    break_height = np.nan
    breaker_type = 0
    xi = 0.0
    beach_slope = 0.0
    did_break = False
    termination_code = 3  # Default: max steps

    # Ray marching loop
    step = 0
    for step in range(max_steps):
        # Store path point
        path_x[step] = x
        path_y[step] = y

        # Get current depth
        h = interpolate_depth_indexed(
            x, y, points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size, grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles
        )

        if np.isnan(h):
            termination_code = 2  # Left domain
            break

        # Adaptive step size: smaller steps in shallow water for accuracy
        if h < 2.0:
            current_step = step_size * 0.3  # ~3m steps in very shallow water
        elif h < 5.0:
            current_step = step_size * 0.5  # ~5m steps in shallow water
        else:
            current_step = step_size  # Full step in deeper water

        # Calculate local wave properties
        L, k, C, n, Cg = local_wave_properties(L0, T, h)

        # Calculate shoaling coefficient
        Ks = shoaling_coefficient(Cg0, Cg)

        # Calculate refraction
        theta = refraction_snell(C, C0, theta0)
        Kr = refraction_coefficient(theta0, theta)

        # Calculate wind modification
        phi = theta_wind - theta
        Kw = wind_modification(Kw, U_wind, phi, C, L, current_step, alpha_wind)

        # Calculate current wave height
        H = wave_height(H0, Ks, Kr, Kw)
        path_h[step] = H

        # Update slope history
        h_history[h_history_idx % 10] = h
        h_history_idx += 1

        # Calculate beach slope from recent history
        if h_history_idx >= 3:
            n_hist = min(h_history_idx, 10)
            h_recent = h_history[:n_hist]
            dh = h_recent[0] - h_recent[n_hist - 1]
            slope_distance = (n_hist - 1) * current_step
            if slope_distance > 0:
                beach_slope = abs(dh) / slope_distance
            else:
                beach_slope = 0.02

        # Ensure reasonable slope
        beach_slope = max(0.005, min(0.2, beach_slope))

        # Calculate breaker index
        H0_L0 = H0 / L0 if L0 > 0 else 0.01
        gamma_b0 = breaker_index_rattanapitikon(H0_L0, beach_slope)
        gamma_b = breaker_index_wind_modified(gamma_b0, U_wind, phi, C, Cw)

        # Check breaking criterion BEFORE checking min_depth
        # This ensures we catch breaking at shallow depths
        if check_breaking(H, h, gamma_b):
            did_break = True
            break_x = x
            break_y = y
            break_depth = h
            break_height = H

            # Calculate Iribarren number
            xi = iribarren_number(beach_slope, H, L0)
            breaker_type = classify_breaker_type(xi)

            termination_code = 0  # Broke
            # Note: Don't increment step here - path_x[step] already has the
            # breaking position from the start of this iteration
            break

        # Now check if we've reached minimum depth (after breaking check)
        if h <= min_depth:
            termination_code = 1  # Reached shore without breaking
            break

        # Update ray direction (refraction)
        dC_dx, dC_dy = celerity_gradient_indexed(
            x, y, T, L0, points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size, grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles
        )
        dx, dy = update_ray_direction(dx, dy, C, dC_dx, dC_dy, current_step)

        # Advance ray position with adaptive step
        x += dx * current_step
        y += dy * current_step

        h_prev = h
        distance_traveled += current_step

    # Trim path arrays to actual length
    actual_steps = step + 1 if step < max_steps else max_steps
    path_x = path_x[:actual_steps].copy()
    path_y = path_y[:actual_steps].copy()
    path_h = path_h[:actual_steps].copy()

    return (
        break_x, break_y, break_depth, break_height, T,
        breaker_type, xi, beach_slope,
        did_break, termination_code,
        path_x, path_y, path_h,
    )


# =============================================================================
# High-Level Ray Tracer Class
# =============================================================================

class RayTracer:
    """
    High-level interface for wave ray tracing.

    Uses a grid-based spatial index for fast triangle lookup.

    Example usage:
        tracer = RayTracer(mesh)
        results = tracer.trace_from_boundary(boundary_conditions, wind_data)
    """

    TERMINATION_REASONS = {
        0: "broke",
        1: "reached_shore",
        2: "left_domain",
        3: "max_steps",
    }

    def __init__(
        self,
        mesh: 'SurfZoneMesh',
        step_size: float = DEFAULT_STEP_SIZE,
        max_steps: int = DEFAULT_MAX_STEPS,
        min_depth: float = DEFAULT_MIN_DEPTH,
    ):
        """
        Initialize ray tracer with mesh data and spatial index.

        Args:
            mesh: SurfZoneMesh object with bathymetry (should have spatial index)
            step_size: Step distance for ray marching (m)
            max_steps: Maximum number of steps per ray
            min_depth: Minimum depth before stopping (m)
        """
        self.mesh = mesh
        self.step_size = step_size
        self.max_steps = max_steps
        self.min_depth = min_depth

        # Get Numba-compatible arrays from mesh (includes spatial index if available)
        arrays = mesh.get_numba_arrays()
        self.points_x = arrays['points_x']
        self.points_y = arrays['points_y']
        self.depth = arrays['depth']
        self.triangles = arrays['triangles']

        # Check if spatial index is included from mesh
        if 'grid_x_min' in arrays:
            logger.info("Using pre-built spatial index from mesh")
            self.grid_x_min = arrays['grid_x_min']
            self.grid_y_min = arrays['grid_y_min']
            self.grid_cell_size = arrays['grid_cell_size']
            self.grid_n_cells_x = arrays['grid_n_cells_x']
            self.grid_n_cells_y = arrays['grid_n_cells_y']
            self.grid_cell_starts = arrays['grid_cell_starts']
            self.grid_cell_counts = arrays['grid_cell_counts']
            self.grid_triangles = arrays['grid_triangles']
        else:
            # Build spatial index if not pre-built (backwards compatibility)
            logger.info("Building spatial index (not found in mesh, will be slow)...")
            logger.info("Tip: Regenerate mesh to include pre-built spatial index")
            grid = build_triangle_grid(
                self.points_x, self.points_y, self.triangles,
                cell_size=500.0
            )
            self.grid_x_min = grid['grid_x_min']
            self.grid_y_min = grid['grid_y_min']
            self.grid_cell_size = grid['grid_cell_size']
            self.grid_n_cells_x = grid['grid_n_cells_x']
            self.grid_n_cells_y = grid['grid_n_cells_y']
            self.grid_cell_starts = grid['grid_cell_starts']
            self.grid_cell_counts = grid['grid_cell_counts']
            self.grid_triangles = grid['grid_triangles']

        logger.info(
            f"RayTracer initialized: {len(self.points_x)} mesh points, "
            f"step={step_size}m, max_steps={max_steps}"
        )

    def trace_single(
        self,
        x: float,
        y: float,
        H0: float,
        T: float,
        direction: float,
        U_wind: float = 0.0,
        wind_direction: float = 0.0,
        partition_id: int = 0,
        store_path: bool = False,
    ) -> RayResult:
        """
        Trace a single wave ray.

        Args:
            x, y: Starting position in UTM (m)
            H0: Deep water wave height (m)
            T: Wave period (s)
            direction: Wave direction (degrees, nautical FROM)
            U_wind: Wind speed (m/s)
            wind_direction: Wind direction (degrees, nautical FROM)
            partition_id: Source partition ID (for tracking)
            store_path: Whether to store the ray path

        Returns:
            RayResult with breaking information
        """
        (
            break_x, break_y, break_depth, break_height, break_period,
            breaker_type, xi, beach_slope,
            did_break, termination_code,
            path_x, path_y, path_h,
        ) = trace_single_ray(
            x, y, H0, T, direction, U_wind, wind_direction,
            self.points_x, self.points_y, self.depth, self.triangles,
            self.grid_x_min, self.grid_y_min, self.grid_cell_size,
            self.grid_n_cells_x, self.grid_n_cells_y,
            self.grid_cell_starts, self.grid_cell_counts, self.grid_triangles,
            self.step_size, self.max_steps, self.min_depth,
        )

        return RayResult(
            start_x=x,
            start_y=y,
            break_x=break_x,
            break_y=break_y,
            break_depth=break_depth,
            break_height=break_height,
            break_period=break_period,
            breaker_type=breaker_type,
            iribarren=xi,
            beach_slope=beach_slope,
            did_break=did_break,
            termination_reason=self.TERMINATION_REASONS.get(termination_code, "unknown"),
            path_x=path_x if store_path else None,
            path_y=path_y if store_path else None,
            path_h=path_h if store_path else None,
            partition_id=partition_id,
        )

    def trace_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        H0: np.ndarray,
        T: np.ndarray,
        direction: np.ndarray,
        U_wind: np.ndarray,
        wind_direction: np.ndarray,
        partition_ids: Optional[np.ndarray] = None,
        store_paths: bool = False,
    ) -> List[RayResult]:
        """
        Trace multiple rays in batch.

        Args:
            x, y: Starting positions (m)
            H0: Wave heights (m)
            T: Wave periods (s)
            direction: Wave directions (degrees, nautical)
            U_wind: Wind speeds (m/s)
            wind_direction: Wind directions (degrees, nautical)
            partition_ids: Source partition IDs
            store_paths: Whether to store ray paths

        Returns:
            List of RayResult objects
        """
        n_rays = len(x)
        if partition_ids is None:
            partition_ids = np.zeros(n_rays, dtype=np.int32)

        results = []

        for i in range(n_rays):
            result = self.trace_single(
                x[i], y[i], H0[i], T[i], direction[i],
                U_wind[i], wind_direction[i],
                partition_id=int(partition_ids[i]),
                store_path=store_paths,
            )
            results.append(result)

        return results

    def trace_from_boundary(
        self,
        boundary: 'BoundaryConditions',
        U_wind: float = 0.0,
        wind_direction: float = 0.0,
        store_paths: bool = False,
    ) -> List[RayResult]:
        """
        Trace rays from boundary conditions.

        Traces one ray per valid partition at each boundary point.

        Args:
            boundary: BoundaryConditions from SwanInputProvider
            U_wind: Wind speed (m/s), constant for now
            wind_direction: Wind direction (degrees, nautical FROM)
            store_paths: Whether to store ray paths

        Returns:
            List of RayResult objects
        """
        results = []

        for partition in boundary.partitions:
            valid_mask = partition.is_valid

            n_valid = np.sum(valid_mask)
            if n_valid == 0:
                continue

            x = boundary.x[valid_mask]
            y = boundary.y[valid_mask]
            H0 = partition.hs[valid_mask]
            T = partition.tp[valid_mask]
            direction = partition.direction[valid_mask]

            logger.info(
                f"Tracing {n_valid} rays for {partition.label} "
                f"(Hs={H0.min():.2f}-{H0.max():.2f}m)"
            )

            U_wind_arr = np.full(n_valid, U_wind, dtype=np.float64)
            wind_dir_arr = np.full(n_valid, wind_direction, dtype=np.float64)
            partition_ids = np.full(n_valid, partition.partition_id, dtype=np.int32)

            batch_results = self.trace_batch(
                x, y, H0, T, direction,
                U_wind_arr, wind_dir_arr,
                partition_ids, store_paths,
            )

            results.extend(batch_results)

        return results

    def summary(self, results: List[RayResult]) -> str:
        """Generate summary of ray tracing results."""
        n_total = len(results)
        n_broke = sum(1 for r in results if r.did_break)
        n_shore = sum(1 for r in results if r.termination_reason == "reached_shore")
        n_domain = sum(1 for r in results if r.termination_reason == "left_domain")

        breaking_heights = [r.break_height for r in results if r.did_break]
        breaking_depths = [r.break_depth for r in results if r.did_break]

        type_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for r in results:
            if r.did_break:
                type_counts[r.breaker_type] += 1

        lines = [
            f"Ray Tracing Results: {n_total} rays",
        ]

        if n_total > 0:
            lines.extend([
                f"  Broke: {n_broke} ({100*n_broke/n_total:.1f}%)",
                f"  Reached shore: {n_shore}",
                f"  Left domain: {n_domain}",
            ])
        else:
            lines.append("  No rays traced")

        if breaking_heights:
            lines.extend([
                f"Breaking Statistics:",
                f"  Height: {min(breaking_heights):.2f} - {max(breaking_heights):.2f} m "
                f"(mean: {np.mean(breaking_heights):.2f})",
                f"  Depth: {min(breaking_depths):.2f} - {max(breaking_depths):.2f} m",
                f"Breaker Types:",
            ])
            for code, label in BREAKER_TYPE_LABELS.items():
                if type_counts[code] > 0:
                    lines.append(f"  {label}: {type_counts[code]}")

        return '\n'.join(lines)
