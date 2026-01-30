"""
Backward Ray Tracer for Surfzone Wave Simulation

Traces rays BACKWARD from near-shore mesh points toward the SWAN boundary
(deep water) to determine which swell partitions contribute to each location.

This is more efficient than forward tracing because we only compute
wave properties at locations where we actually need them.

KEY PHYSICS (time-reversed from forward tracing):
1. Direction: Rays point AWAY from shore (opposite of wave travel direction)
2. Refraction: Rays bend toward FASTER celerity (deeper water)
   - Forward: dθ/ds = -(1/C) · ∂C/∂n (bends toward slower C / shallow)
   - Backward: dθ/ds = +(1/C) · ∂C/∂n (bends toward faster C / deep)
   - Achieved by NEGATING the celerity gradients
3. Shoaling: K_s = sqrt(Cg_boundary / Cg_mesh) - applied in reverse

The refraction equation uses the same update_ray_direction() function as
forward tracing, but with negated gradients to reverse the bending direction.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads

from .wave_physics import (
    deep_water_properties,
    local_wave_properties,
    shoaling_coefficient,
    update_ray_direction,
    nautical_to_math,
    math_to_nautical,
)
from .ray_tracer import (
    interpolate_depth_indexed,
    celerity_gradient_indexed,
)

logger = logging.getLogger(__name__)

# Configuration defaults
DEFAULT_STEP_SIZE = 10.0  # meters
DEFAULT_MAX_STEPS = 500
DEFAULT_MAX_ITERATIONS = 15
DEFAULT_CONVERGENCE_TOLERANCE = 0.10  # 10% of directional spread
DEFAULT_ALPHA = 0.6  # Gradient descent relaxation factor


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PartitionContribution:
    """
    Contribution of a single wave partition to a mesh point.

    Attributes:
        partition_id: ID of the source partition (0=wind sea, 1-3=swells)
        H: Wave height at mesh point after shoaling (m)
        T: Wave period (s), unchanged from boundary
        direction: Wave direction at mesh point (degrees, nautical)
        K_shoaling: Cumulative shoaling coefficient along path
        converged: Whether the iteration converged
        n_iterations: Number of iterations to converge
        path_x: Optional ray path x coordinates (for debugging)
        path_y: Optional ray path y coordinates (for debugging)
    """
    partition_id: int
    H: float
    T: float
    direction: float
    K_shoaling: float
    converged: bool
    n_iterations: int = 0
    path_x: Optional[np.ndarray] = None
    path_y: Optional[np.ndarray] = None


@dataclass
class MeshPointResult:
    """
    Complete wave information at a single mesh point.

    Attributes:
        mesh_x: UTM x coordinate (m)
        mesh_y: UTM y coordinate (m)
        mesh_depth: Water depth at point (m, positive below water)
        contributions: List of wave contributions from different partitions
    """
    mesh_x: float
    mesh_y: float
    mesh_depth: float
    contributions: List[PartitionContribution]

    @property
    def total_Hs(self) -> float:
        """Combined significant wave height (RMS of contributions)."""
        if not self.contributions:
            return 0.0
        return np.sqrt(sum(c.H**2 for c in self.contributions if c.converged))

    @property
    def dominant_direction(self) -> float:
        """Energy-weighted mean direction."""
        if not self.contributions:
            return np.nan

        converged = [c for c in self.contributions if c.converged]
        if not converged:
            return np.nan

        # Energy-weighted circular mean
        total_energy = sum(c.H**2 for c in converged)
        if total_energy == 0:
            return np.nan

        x = sum(c.H**2 * np.cos(np.radians(c.direction)) for c in converged)
        y = sum(c.H**2 * np.sin(np.radians(c.direction)) for c in converged)

        return np.degrees(np.arctan2(y, x)) % 360


@dataclass
class BoundaryPartition:
    """
    Wave partition data at a boundary segment.

    Attributes:
        partition_id: Partition ID (0=wind sea, 1-3=swells)
        Hs: Significant wave height (m)
        Tp: Peak period (s)
        direction: Wave direction (degrees, nautical FROM)
        directional_spread: Directional spread (degrees)
    """
    partition_id: int
    Hs: float
    Tp: float
    direction: float
    directional_spread: float

    @property
    def is_valid(self) -> bool:
        """Check if partition has valid data."""
        return (
            self.Hs > 0 and
            self.Tp > 0 and
            not np.isnan(self.direction) and
            self.directional_spread > 0
        )


@dataclass
class BoundarySegment:
    """
    A segment of the offshore boundary with wave partitions.

    Attributes:
        x: UTM x coordinate of segment center
        y: UTM y coordinate of segment center
        partitions: List of wave partitions at this segment
    """
    x: float
    y: float
    partitions: List[BoundaryPartition]


# =============================================================================
# Initial Guess Computation
# =============================================================================

@njit(cache=True)
def compute_initial_direction_blended(
    x: float,
    y: float,
    theta_partition_nautical: float,
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
    deep_weight: float = 0.8,
    h_step: float = 20.0,
) -> float:
    """
    Compute initial wave direction as weighted blend of depth gradient and partition.

    Blends two directions:
    - Direction toward deeper water (from local depth gradient)
    - Partition direction from boundary

    For backward tracing, the ray travels in direction (-cos(θ_M), -sin(θ_M)).
    The depth gradient component aligns the ray with deepening water.
    The partition component biases toward the expected swell direction.

    Args:
        x, y: Mesh point location
        theta_partition_nautical: Partition direction at boundary (nautical degrees)
        points_x, points_y, depth, triangles: Mesh arrays
        grid_*: Spatial index arrays
        deep_weight: Weight for depth gradient direction (0-1), default 0.8
                     Partition gets weight (1 - deep_weight)
        h_step: Finite difference step size (m)

    Returns:
        Blended initial wave direction θ_M in nautical convention (degrees),
        or partition direction if depth gradient cannot be computed.
    """
    # Compute depth gradient using finite differences
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

    # Compute gradients (handle edge cases)
    if np.isnan(d_xp) or np.isnan(d_xm) or d_xp <= 0 or d_xm <= 0:
        dh_dx = 0.0
    else:
        dh_dx = (d_xp - d_xm) / (2.0 * h_step)

    if np.isnan(d_yp) or np.isnan(d_ym) or d_yp <= 0 or d_ym <= 0:
        dh_dy = 0.0
    else:
        dh_dy = (d_yp - d_ym) / (2.0 * h_step)

    # Check if we have a valid gradient
    grad_mag = np.sqrt(dh_dx**2 + dh_dy**2)
    if grad_mag < 1e-8:
        # Flat bottom or invalid - fall back to partition direction
        return theta_partition_nautical

    # Depth gradient (dh_dx, dh_dy) points toward deeper water.
    # For backward tracing, ray direction is (-cos(θ_M), -sin(θ_M)).
    # We want ray direction to align with depth gradient:
    #   (-cos(θ_M), -sin(θ_M)) ∝ (dh_dx, dh_dy)
    # So: (cos(θ_M), sin(θ_M)) ∝ (-dh_dx, -dh_dy)
    # Therefore: θ_deep (math) = atan2(-dh_dy, -dh_dx)
    theta_deep_math = np.arctan2(-dh_dy, -dh_dx)

    # Convert partition direction to math convention (radians)
    theta_partition_math = nautical_to_math(theta_partition_nautical)

    # Blend directions using vector interpolation (proper for circular quantities)
    # Convert both to unit vectors, blend, then back to angle
    partition_weight = 1.0 - deep_weight

    # Unit vectors for each direction
    deep_x = np.cos(theta_deep_math)
    deep_y = np.sin(theta_deep_math)
    partition_x = np.cos(theta_partition_math)
    partition_y = np.sin(theta_partition_math)

    # Weighted blend
    blend_x = deep_weight * deep_x + partition_weight * partition_x
    blend_y = deep_weight * deep_y + partition_weight * partition_y

    # Normalize (in case weights don't sum to 1 or vectors partially cancel)
    blend_mag = np.sqrt(blend_x**2 + blend_y**2)
    if blend_mag < 1e-8:
        # Vectors cancelled out (opposite directions) - use partition
        return theta_partition_nautical

    # Convert back to angle
    theta_blend_math = np.arctan2(blend_y, blend_x)

    # Convert to nautical
    return math_to_nautical(theta_blend_math)


# =============================================================================
# Core Backward Tracing (Numba-accelerated)
# =============================================================================

@njit(cache=False)  # TEMP: force recompile
def trace_backward_single(
    # Start point (surfzone mesh point)
    start_x: float,
    start_y: float,
    # Wave parameters
    T: float,
    theta_M_nautical: float,  # Direction at mesh point (wave travel direction)
    # Mesh data arrays
    points_x: np.ndarray,
    points_y: np.ndarray,
    depth: np.ndarray,
    triangles: np.ndarray,
    # Spatial index arrays
    grid_x_min: float,
    grid_y_min: float,
    grid_cell_size: float,
    grid_n_cells_x: int,
    grid_n_cells_y: int,
    grid_cell_starts: np.ndarray,
    grid_cell_counts: np.ndarray,
    grid_triangles: np.ndarray,
    # Boundary definition (depth threshold for "reached offshore boundary")
    boundary_depth_threshold: float,
    # Configuration
    step_size: float = DEFAULT_STEP_SIZE,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Tuple[float, float, float, float, float, bool, np.ndarray, np.ndarray]:
    """
    Trace a single ray BACKWARD from near-shore mesh point toward deep water.

    The ray direction is NEGATED (pointing away from shore toward ocean).
    Celerity gradients are NEGATED so rays bend toward FASTER celerity
    (deeper water) instead of slower celerity as in forward tracing.

    Args:
        start_x, start_y: Starting position (mesh point) in UTM
        T: Wave period (s)
        theta_M_nautical: Direction at mesh point (degrees, nautical)
        points_x, points_y, depth, triangles: Mesh arrays
        grid_*: Spatial index arrays
        boundary_depth_threshold: Depth threshold (m) - ray has "reached boundary"
                                   when depth exceeds this value (offshore deep water)
        step_size: Ray marching step size (m)
        max_steps: Maximum steps before giving up

    Returns:
        end_x, end_y: Where ray reached boundary (or stopped)
        theta_arrival_nautical: Direction ray had when arriving at boundary
        Cg_start: Group velocity at start (mesh point)
        Cg_end: Group velocity at end (boundary)
        reached_boundary: True if ray reached the boundary
        path_x, path_y: Arrays of ray path coordinates
    """
    # Initialize path storage
    path_x = np.empty(max_steps, dtype=np.float64)
    path_y = np.empty(max_steps, dtype=np.float64)

    # Get deep water reference properties
    L0, C0, Cg0 = deep_water_properties(T)

    # Convert direction to math convention (radians)
    theta_M = nautical_to_math(theta_M_nautical)

    # BACKWARD: Negate direction to point AWAY from shore (toward ocean)
    # Combined with negated gradients below, this makes rays bend toward
    # FASTER celerity (deeper water) - the correct backward behavior
    dx = -np.cos(theta_M)  # NEGATED for backward tracing
    dy = -np.sin(theta_M)  # NEGATED for backward tracing

    x, y = start_x, start_y

    # Get starting depth and Cg
    h_start = interpolate_depth_indexed(
        x, y, points_x, points_y, depth, triangles,
        grid_x_min, grid_y_min, grid_cell_size,
        grid_n_cells_x, grid_n_cells_y,
        grid_cell_starts, grid_cell_counts, grid_triangles
    )

    if np.isnan(h_start) or h_start <= 0:
        # Invalid starting point
        return (
            np.nan, np.nan, np.nan, np.nan, np.nan, False,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    # Calculate Cg at start
    _, _, _, _, Cg_start = local_wave_properties(L0, T, h_start)

    # Track end Cg (will be updated as we trace)
    Cg_end = Cg_start

    step = 0
    for step in range(max_steps):
        # Store path point
        path_x[step] = x
        path_y[step] = y

        # Get current depth
        h = interpolate_depth_indexed(
            x, y, points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles
        )

        if np.isnan(h) or h <= 0:
            # Left valid domain (hit land or outside mesh)
            path_x = path_x[:step + 1].copy()
            path_y = path_y[:step + 1].copy()
            return (
                x, y, np.nan, Cg_start, Cg_end, False,
                path_x, path_y,
            )

        # Check if reached boundary (deep water)
        if h > boundary_depth_threshold:
            # Ray has reached boundary
            # Wave direction is OPPOSITE of ray direction (we're tracing backward)
            # Negate to get wave travel direction
            theta_current = np.arctan2(-dy, -dx)
            theta_arrival_nautical = math_to_nautical(theta_current)

            # Trim path arrays
            path_x = path_x[:step + 1].copy()
            path_y = path_y[:step + 1].copy()

            return (
                x, y, theta_arrival_nautical, Cg_start, Cg_end, True,
                path_x, path_y,
            )

        # Local wave properties (h was already computed above)
        L, k, C, n, Cg = local_wave_properties(L0, T, h)
        Cg_end = Cg  # Update end Cg

        # Get celerity gradient for refraction
        dC_dx, dC_dy = celerity_gradient_indexed(
            x, y, T, L0, points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles
        )

        # BACKWARD: NEGATE gradients so rays bend toward FASTER celerity (deeper water)
        # Forward: dθ/ds = -(1/C) · ∂C/∂n (bends toward slower C / shallow)
        # Backward: dθ/ds = +(1/C) · ∂C/∂n (bends toward faster C / deep)
        # We achieve this by passing negated gradients to the standard formula
        dx, dy = update_ray_direction(dx, dy, C, -dC_dx, -dC_dy, step_size)

        # Normal positive step (direction already points toward ocean)
        x += dx * step_size
        y += dy * step_size

    # Max steps reached without hitting boundary
    path_x = path_x[:step + 1].copy()
    path_y = path_y[:step + 1].copy()

    return (
        x, y, np.nan, Cg_start, Cg_end, False,
        path_x, path_y,
    )


# =============================================================================
# Parallel Tracing (Numba prange)
# =============================================================================

@njit(cache=True)
def _trace_single_partition_converge(
    mesh_x: float,
    mesh_y: float,
    # Partition data
    T: float,
    theta_partition: float,
    delta_theta: float,
    H_partition: float,
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
    # Boundary
    boundary_depth_threshold: float,
    # Config
    alpha: float,
    max_iterations: int,
    tolerance: float,
    step_size: float,
    max_steps: int,
) -> Tuple[float, float, float, float, bool, int]:
    """
    Trace single partition with convergence - pure Numba version.

    Initial guess is a blend of:
    - 80% direction toward deeper water (from depth gradient)
    - 20% partition direction from boundary

    Returns:
        H_mesh, T, direction, K_shoaling, converged, n_iterations
    """
    # Compute initial guess: blend of depth gradient (80%) and partition (20%)
    theta_M = compute_initial_direction_blended(
        mesh_x, mesh_y, theta_partition,
        points_x, points_y, depth, triangles,
        grid_x_min, grid_y_min, grid_cell_size,
        grid_n_cells_x, grid_n_cells_y,
        grid_cell_starts, grid_cell_counts, grid_triangles,
        deep_weight=0.8,
    )

    for iteration in range(max_iterations):
        # Trace backward
        (
            end_x, end_y, theta_arrival, Cg_start, Cg_end, reached,
            path_x, path_y,
        ) = trace_backward_single(
            mesh_x, mesh_y, T, theta_M,
            points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles,
            boundary_depth_threshold, step_size, max_steps,
        )

        if not reached:
            return np.nan, T, np.nan, np.nan, False, iteration + 1

        # Check convergence
        angle_diff = theta_arrival - theta_partition
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360

        error = abs(angle_diff) / delta_theta if delta_theta > 0 else abs(angle_diff)

        if error < tolerance:
            # Converged
            if Cg_start > 0:
                K_s = np.sqrt(Cg_end / Cg_start)
            else:
                K_s = 1.0
            H_mesh = H_partition * K_s
            return H_mesh, T, theta_M, K_s, True, iteration + 1

        # Gradient descent update
        theta_M = theta_M - alpha * angle_diff
        theta_M = theta_M % 360

    # Failed to converge - return best guess
    if Cg_start > 0:
        K_s = np.sqrt(Cg_end / Cg_start)
    else:
        K_s = 1.0
    H_mesh = H_partition * K_s
    return H_mesh, T, theta_M, K_s, False, max_iterations


@njit(parallel=True, cache=True)
def trace_all_parallel(
    # Mesh points to trace (N points)
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    # Partition data for each point (N x max_partitions)
    # Each row is: [T, direction, spread, Hs] for up to max_partitions
    partition_T: np.ndarray,
    partition_direction: np.ndarray,
    partition_spread: np.ndarray,
    partition_Hs: np.ndarray,
    partition_valid: np.ndarray,  # Boolean mask
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
    # Boundary
    boundary_depth_threshold: float,
    # Config
    alpha: float,
    max_iterations: int,
    tolerance: float,
    step_size: float,
    max_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Trace all mesh points in parallel using Numba prange.

    Args:
        mesh_x, mesh_y: Arrays of mesh point coordinates (N,)
        partition_*: Partition data arrays (N, max_partitions)
        partition_valid: Boolean mask for valid partitions (N, max_partitions)
        Other args: Mesh arrays and configuration

    Returns:
        result_H: Wave heights at each point for each partition (N, max_partitions)
        result_T: Periods (N, max_partitions)
        result_direction: Directions at mesh point (N, max_partitions)
        result_K: Shoaling coefficients (N, max_partitions)
        result_converged: Convergence flags (N, max_partitions)
        result_iterations: Iteration counts (N, max_partitions)
    """
    n_points = len(mesh_x)
    max_partitions = partition_T.shape[1]

    # Output arrays
    result_H = np.full((n_points, max_partitions), np.nan, dtype=np.float64)
    result_T = np.full((n_points, max_partitions), np.nan, dtype=np.float64)
    result_direction = np.full((n_points, max_partitions), np.nan, dtype=np.float64)
    result_K = np.full((n_points, max_partitions), np.nan, dtype=np.float64)
    result_converged = np.zeros((n_points, max_partitions), dtype=np.bool_)
    result_iterations = np.zeros((n_points, max_partitions), dtype=np.int32)

    # Parallel loop over mesh points
    for i in prange(n_points):
        mx = mesh_x[i]
        my = mesh_y[i]

        for p in range(max_partitions):
            if not partition_valid[i, p]:
                continue

            T = partition_T[i, p]
            theta_partition = partition_direction[i, p]
            delta_theta = partition_spread[i, p]
            H_partition = partition_Hs[i, p]

            H, T_out, direction, K_s, converged, n_iter = _trace_single_partition_converge(
                mx, my,
                T, theta_partition, delta_theta, H_partition,
                points_x, points_y, depth, triangles,
                grid_x_min, grid_y_min, grid_cell_size,
                grid_n_cells_x, grid_n_cells_y,
                grid_cell_starts, grid_cell_counts, grid_triangles,
                boundary_depth_threshold,
                alpha, max_iterations, tolerance, step_size, max_steps,
            )

            result_H[i, p] = H
            result_T[i, p] = T_out
            result_direction[i, p] = direction
            result_K[i, p] = K_s
            result_converged[i, p] = converged
            result_iterations[i, p] = n_iter

    return result_H, result_T, result_direction, result_K, result_converged, result_iterations


# =============================================================================
# Convergence Iteration
# =============================================================================

def backward_trace_with_convergence(
    mesh_x: float,
    mesh_y: float,
    partition: BoundaryPartition,
    # Mesh arrays (from BackwardRayTracer)
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
    # Boundary
    boundary_depth_threshold: float,
    # Configuration
    alpha: float = DEFAULT_ALPHA,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE,
    step_size: float = DEFAULT_STEP_SIZE,
    max_steps: int = DEFAULT_MAX_STEPS,
    store_path: bool = False,
) -> PartitionContribution:
    """
    Trace backward from mesh point with iteration to converge on correct direction.

    The goal is to find theta_M (direction at mesh point) such that when we
    trace backward to the boundary, the ray arrives with direction matching
    the partition's direction (within tolerance of directional spread).

    Args:
        mesh_x, mesh_y: Mesh point coordinates
        partition: Wave partition from boundary
        *_arrays: Mesh and spatial index arrays
        boundary_depth_threshold: Y coordinate of boundary
        alpha: Gradient descent relaxation factor (0.5-0.7)
        max_iterations: Maximum convergence iterations
        tolerance: Convergence tolerance (fraction of directional spread)
        step_size: Ray marching step size
        max_steps: Maximum steps per trace
        store_path: Whether to store ray path in result

    Returns:
        PartitionContribution with wave parameters at mesh point
    """
    T = partition.Tp
    theta_partition = partition.direction  # Target direction at boundary
    delta_theta = partition.directional_spread
    H_partition = partition.Hs

    # Initial guess: blend of depth gradient (80%) and partition (20%)
    theta_M = compute_initial_direction_blended(
        mesh_x, mesh_y, theta_partition,
        points_x, points_y, depth, triangles,
        grid_x_min, grid_y_min, grid_cell_size,
        grid_n_cells_x, grid_n_cells_y,
        grid_cell_starts, grid_cell_counts, grid_triangles,
        deep_weight=0.8,
    )

    path_x_final = None
    path_y_final = None

    for iteration in range(max_iterations):
        # Trace backward
        (
            end_x, end_y, theta_arrival, Cg_start, Cg_end, reached,
            path_x, path_y,
        ) = trace_backward_single(
            mesh_x, mesh_y, T, theta_M,
            points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles,
            boundary_depth_threshold, step_size, max_steps,
        )

        if not reached:
            # Ray didn't reach boundary - return with path data for debugging
            return PartitionContribution(
                partition_id=partition.partition_id,
                H=np.nan,
                T=T,
                direction=np.nan,
                K_shoaling=np.nan,
                converged=False,
                n_iterations=iteration + 1,
                path_x=path_x if store_path else None,
                path_y=path_y if store_path else None,
            )

        # Store path from last iteration
        if store_path:
            path_x_final = path_x
            path_y_final = path_y

        # Check convergence
        # Normalize angle difference to [-180, 180]
        angle_diff = theta_arrival - theta_partition
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360

        error = abs(angle_diff) / delta_theta if delta_theta > 0 else abs(angle_diff)

        if error < tolerance:
            # Converged - compute final wave height
            # K_s = sqrt(Cg_boundary / Cg_mesh)
            if Cg_start > 0:
                K_s = np.sqrt(Cg_end / Cg_start)
            else:
                K_s = 1.0

            H_mesh = H_partition * K_s

            return PartitionContribution(
                partition_id=partition.partition_id,
                H=H_mesh,
                T=T,
                direction=theta_M,
                K_shoaling=K_s,
                converged=True,
                n_iterations=iteration + 1,
                path_x=path_x_final,
                path_y=path_y_final,
            )

        # Gradient descent update
        # Move theta_M in the direction that would make theta_arrival closer to theta_partition
        theta_M = theta_M - alpha * angle_diff

        # Keep in valid range
        theta_M = theta_M % 360

    # Failed to converge after max iterations
    # Return best guess anyway
    if Cg_start > 0:
        K_s = np.sqrt(Cg_end / Cg_start)
    else:
        K_s = 1.0

    H_mesh = H_partition * K_s

    return PartitionContribution(
        partition_id=partition.partition_id,
        H=H_mesh,
        T=T,
        direction=theta_M,
        K_shoaling=K_s,
        converged=False,
        n_iterations=max_iterations,
        path_x=path_x_final if store_path else None,
        path_y=path_y_final if store_path else None,
    )


# =============================================================================
# High-Level Ray Tracer Class
# =============================================================================

class BackwardRayTracer:
    """
    High-level interface for backward wave ray tracing.

    Traces rays backward from surfzone mesh points to the offshore boundary
    to determine which wave partitions contribute to each location.

    Example usage:
        tracer = BackwardRayTracer(mesh, boundary_depth_threshold=50.0)  # 50m depth
        result = tracer.trace_mesh_point(x, y, partitions)

        # Or trace all mesh points
        results = tracer.trace_all_mesh_points(mesh_x, mesh_y, boundary_segments)
    """

    def __init__(
        self,
        mesh: 'SurfZoneMesh',
        boundary_depth_threshold: float,
        step_size: float = DEFAULT_STEP_SIZE,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        convergence_tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE,
        alpha: float = DEFAULT_ALPHA,
    ):
        """
        Initialize backward ray tracer.

        Args:
            mesh: SurfZoneMesh with bathymetry and spatial index
            boundary_depth_threshold: Depth threshold (m) - rays considered to have
                                      "reached boundary" when depth exceeds this value
                                      (e.g., 50m for offshore deep water)
            step_size: Ray marching step size (m)
            max_steps: Maximum steps per ray trace
            max_iterations: Maximum convergence iterations
            convergence_tolerance: Convergence criterion (fraction of spread)
            alpha: Gradient descent relaxation factor
        """
        self.mesh = mesh
        self.boundary_depth_threshold = boundary_depth_threshold
        self.step_size = step_size
        self.max_steps = max_steps
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.alpha = alpha

        # Get Numba-compatible arrays from mesh
        arrays = mesh.get_numba_arrays()
        self.points_x = arrays['points_x']
        self.points_y = arrays['points_y']
        self.depth = arrays['depth']
        self.triangles = arrays['triangles']

        # Spatial index (should be included from mesh)
        if 'grid_x_min' in arrays:
            self.grid_x_min = arrays['grid_x_min']
            self.grid_y_min = arrays['grid_y_min']
            self.grid_cell_size = arrays['grid_cell_size']
            self.grid_n_cells_x = arrays['grid_n_cells_x']
            self.grid_n_cells_y = arrays['grid_n_cells_y']
            self.grid_cell_starts = arrays['grid_cell_starts']
            self.grid_cell_counts = arrays['grid_cell_counts']
            self.grid_triangles = arrays['grid_triangles']
        else:
            raise ValueError(
                "Mesh does not have pre-built spatial index. "
                "Regenerate mesh with build_spatial_index=True"
            )

        logger.info(
            f"BackwardRayTracer initialized: {len(self.points_x)} mesh points, "
            f"boundary_depth={boundary_depth_threshold}m, step={step_size}m"
        )

    def trace_mesh_point(
        self,
        mesh_x: float,
        mesh_y: float,
        partitions: List[BoundaryPartition],
        store_paths: bool = False,
    ) -> MeshPointResult:
        """
        Trace all partitions for a single mesh point.

        Args:
            mesh_x, mesh_y: Mesh point coordinates (UTM)
            partitions: List of wave partitions to trace
            store_paths: Whether to store ray paths in results

        Returns:
            MeshPointResult with all contributing partitions
        """
        contributions = []

        for partition in partitions:
            if not partition.is_valid:
                continue

            result = backward_trace_with_convergence(
                mesh_x, mesh_y, partition,
                self.points_x, self.points_y, self.depth, self.triangles,
                self.grid_x_min, self.grid_y_min, self.grid_cell_size,
                self.grid_n_cells_x, self.grid_n_cells_y,
                self.grid_cell_starts, self.grid_cell_counts, self.grid_triangles,
                self.boundary_depth_threshold,
                self.alpha, self.max_iterations, self.convergence_tolerance,
                self.step_size, self.max_steps, store_paths,
            )

            contributions.append(result)

        # Get depth at mesh point
        mesh_depth = interpolate_depth_indexed(
            mesh_x, mesh_y,
            self.points_x, self.points_y, self.depth, self.triangles,
            self.grid_x_min, self.grid_y_min, self.grid_cell_size,
            self.grid_n_cells_x, self.grid_n_cells_y,
            self.grid_cell_starts, self.grid_cell_counts, self.grid_triangles,
        )

        return MeshPointResult(
            mesh_x=mesh_x,
            mesh_y=mesh_y,
            mesh_depth=mesh_depth,
            contributions=contributions,
        )

    def trace_all_mesh_points(
        self,
        mesh_points_x: np.ndarray,
        mesh_points_y: np.ndarray,
        boundary_segments: List[BoundarySegment],
        store_paths: bool = False,
    ) -> List[MeshPointResult]:
        """
        Trace all mesh points using "to the left" boundary selection.

        For Southern California (coast runs N-S, swell from W/NW):
        Mesh points are mapped to boundary segments based on their
        alongshore (y) position.

        Args:
            mesh_points_x, mesh_points_y: Arrays of mesh point coordinates
            boundary_segments: List of boundary segments with partitions
            store_paths: Whether to store ray paths

        Returns:
            List of MeshPointResult for each mesh point
        """
        results = []
        n_points = len(mesh_points_x)
        n_boundary = len(boundary_segments)

        if n_boundary == 0:
            logger.warning("No boundary segments provided")
            return results

        # Sort boundary segments by y coordinate for mapping
        sorted_segments = sorted(boundary_segments, key=lambda s: s.y)
        boundary_y_coords = np.array([s.y for s in sorted_segments])

        logger.info(f"Tracing {n_points} mesh points with {n_boundary} boundary segments")

        for i, (mx, my) in enumerate(zip(mesh_points_x, mesh_points_y)):
            # "To the left" boundary selection
            # Find nearest boundary segment by y coordinate
            idx = np.searchsorted(boundary_y_coords, my)
            idx = min(idx, n_boundary - 1)

            partitions = sorted_segments[idx].partitions

            result = self.trace_mesh_point(mx, my, partitions, store_paths)
            results.append(result)

            # Progress logging
            if (i + 1) % 1000 == 0:
                logger.info(f"  Traced {i + 1}/{n_points} points")

        # Summary statistics
        n_converged = sum(
            1 for r in results
            for c in r.contributions
            if c.converged
        )
        n_total = sum(len(r.contributions) for r in results)

        logger.info(
            f"Backward tracing complete: {n_converged}/{n_total} "
            f"({100*n_converged/n_total:.1f}%) converged"
        )

        return results

    def trace_all_parallel(
        self,
        mesh_points_x: np.ndarray,
        mesh_points_y: np.ndarray,
        boundary_segments: List[BoundarySegment],
        max_partitions: int = 4,
        n_threads: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Trace all mesh points in parallel using Numba prange.

        This is significantly faster than trace_all_mesh_points for large meshes.
        Returns raw numpy arrays instead of Python objects for maximum speed.

        Args:
            mesh_points_x, mesh_points_y: Arrays of mesh point coordinates
            boundary_segments: List of boundary segments with partitions
            max_partitions: Maximum partitions per point (default 4)
            n_threads: Number of threads (None = use all available)

        Returns:
            Tuple of arrays (all shape N x max_partitions):
            - result_H: Wave heights
            - result_T: Periods
            - result_direction: Directions at mesh point
            - result_K: Shoaling coefficients
            - result_converged: Convergence flags (boolean)
            - result_iterations: Iteration counts
        """
        n_points = len(mesh_points_x)
        n_boundary = len(boundary_segments)

        if n_boundary == 0:
            logger.warning("No boundary segments provided")
            empty = np.full((n_points, max_partitions), np.nan)
            return empty, empty, empty, empty, np.zeros((n_points, max_partitions), dtype=bool), np.zeros((n_points, max_partitions), dtype=np.int32)

        # Set thread count if specified
        if n_threads is not None:
            set_num_threads(n_threads)

        actual_threads = get_num_threads()
        logger.info(f"Parallel tracing {n_points:,} points with {actual_threads} threads")

        # Sort boundary segments by y coordinate
        sorted_segments = sorted(boundary_segments, key=lambda s: s.y)
        boundary_y_coords = np.array([s.y for s in sorted_segments])

        # Pre-allocate partition arrays (N x max_partitions)
        partition_T = np.zeros((n_points, max_partitions), dtype=np.float64)
        partition_direction = np.zeros((n_points, max_partitions), dtype=np.float64)
        partition_spread = np.zeros((n_points, max_partitions), dtype=np.float64)
        partition_Hs = np.zeros((n_points, max_partitions), dtype=np.float64)
        partition_valid = np.zeros((n_points, max_partitions), dtype=np.bool_)

        # Fill partition arrays - map each point to its boundary segment
        logger.info("  Preparing partition data...")
        t_prep = time.perf_counter()

        for i in range(n_points):
            my = mesh_points_y[i]
            # "To the left" boundary selection
            idx = np.searchsorted(boundary_y_coords, my)
            idx = min(idx, n_boundary - 1)

            partitions = sorted_segments[idx].partitions
            for p, partition in enumerate(partitions[:max_partitions]):
                if partition.is_valid:
                    partition_T[i, p] = partition.Tp
                    partition_direction[i, p] = partition.direction
                    partition_spread[i, p] = partition.directional_spread
                    partition_Hs[i, p] = partition.Hs
                    partition_valid[i, p] = True

        prep_time = time.perf_counter() - t_prep
        logger.info(f"  Prep time: {prep_time:.2f}s")

        # Run parallel tracing
        logger.info("  Running parallel ray tracing...")
        t_trace = time.perf_counter()

        result_H, result_T, result_direction, result_K, result_converged, result_iterations = trace_all_parallel(
            np.ascontiguousarray(mesh_points_x),
            np.ascontiguousarray(mesh_points_y),
            partition_T, partition_direction, partition_spread, partition_Hs, partition_valid,
            self.points_x, self.points_y, self.depth, self.triangles,
            self.grid_x_min, self.grid_y_min, self.grid_cell_size,
            self.grid_n_cells_x, self.grid_n_cells_y,
            self.grid_cell_starts, self.grid_cell_counts, self.grid_triangles,
            self.boundary_depth_threshold,
            self.alpha, self.max_iterations, self.convergence_tolerance,
            self.step_size, self.max_steps,
        )

        trace_time = time.perf_counter() - t_trace
        logger.info(f"  Trace time: {trace_time:.2f}s")

        # Summary stats
        n_valid = np.sum(partition_valid)
        n_converged = np.sum(result_converged)
        convergence_rate = 100 * n_converged / n_valid if n_valid > 0 else 0
        throughput = n_points / trace_time if trace_time > 0 else 0

        logger.info(
            f"Parallel tracing complete:\n"
            f"  Points: {n_points:,}\n"
            f"  Converged: {n_converged:,}/{n_valid:,} ({convergence_rate:.1f}%)\n"
            f"  Throughput: {throughput:,.0f} points/sec"
        )

        return result_H, result_T, result_direction, result_K, result_converged, result_iterations

    def compute_total_Hs(self, result_H: np.ndarray, result_converged: np.ndarray) -> np.ndarray:
        """
        Compute total Hs from parallel results (RMS of converged contributions).

        Args:
            result_H: Wave heights array (N x max_partitions)
            result_converged: Convergence flags (N x max_partitions)

        Returns:
            total_Hs: Combined Hs for each point (N,)
        """
        # Mask non-converged values
        H_masked = np.where(result_converged, result_H, 0.0)
        # RMS combination
        return np.sqrt(np.nansum(H_masked**2, axis=1))

    def summary(self, results: List[MeshPointResult]) -> str:
        """Generate summary of backward ray tracing results."""
        n_points = len(results)

        total_contributions = sum(len(r.contributions) for r in results)
        converged_contributions = sum(
            1 for r in results
            for c in r.contributions
            if c.converged
        )

        # Collect Hs values
        hs_values = [r.total_Hs for r in results if r.total_Hs > 0]

        # Iteration statistics
        iterations = [
            c.n_iterations
            for r in results
            for c in r.contributions
            if c.converged
        ]

        lines = [
            f"Backward Ray Tracing Results",
            f"  Mesh points: {n_points}",
            f"  Total partition traces: {total_contributions}",
            f"  Converged: {converged_contributions} ({100*converged_contributions/total_contributions:.1f}%)" if total_contributions > 0 else "  No traces",
        ]

        if hs_values:
            lines.extend([
                f"Wave Heights:",
                f"  Hs range: {min(hs_values):.2f} - {max(hs_values):.2f} m",
                f"  Hs mean: {np.mean(hs_values):.2f} m",
            ])

        if iterations:
            lines.extend([
                f"Convergence:",
                f"  Iterations: {min(iterations)} - {max(iterations)} (mean: {np.mean(iterations):.1f})",
            ])

        return '\n'.join(lines)
