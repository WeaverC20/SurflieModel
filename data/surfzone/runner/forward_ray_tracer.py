"""
Forward Ray Tracing with Energy Deposition for Surfzone Model

This module implements forward ray tracing from the SWAN boundary into the surfzone,
depositing energy as rays propagate. This approach naturally captures wave focusing
effects where bathymetry causes ray convergence.

Key features:
- Rays initialized at constant spatial density along offshore boundary
- Directional Gaussian kernel for energy deposition (only deposits forward)
- Tube width tracking via geometric formula (dW/ds = W × dθ/ds)
- Parallelized across rays using Numba prange

References:
- Ray theory: Dean & Dalrymple (1991)
- Energy flux conservation: Longuet-Higgins & Stewart (1964)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from numba import njit, prange

from .wave_physics import (
    deep_water_properties,
    local_wave_properties,
    update_ray_direction,
    nautical_to_math,
)
from .ray_tracer import (
    interpolate_depth_indexed,
    celerity_gradient_indexed,
    celerity_gradient_smoothed,
)

if TYPE_CHECKING:
    from data.surfzone.mesh import SurfZoneMesh
    from .swan_input_provider import BoundaryConditions, WavePartition


# Physical constants
RHO = 1025.0  # Seawater density (kg/m³)
G = 9.81  # Gravitational acceleration (m/s²)
TWO_PI = 2.0 * np.pi


@dataclass
class ForwardTracerConfig:
    """Configuration for forward ray tracing."""

    # Ray initialization
    boundary_spacing_m: float = 50.0  # Spacing between boundary sample points

    # Ray propagation - adaptive step sizing based on depth
    max_distance_m: float = 6000.0  # Maximum distance a ray can travel (6km)
    step_deep_m: float = 50.0       # Step size in deep water (>10m depth)
    step_10m_m: float = 20.0        # Step size at 10m depth
    step_5m_m: float = 10.0         # Step size at 5m depth
    step_3m_m: float = 5.0          # Step size at ≤3m depth

    # Energy deposition
    kernel_sigma_m: float = 25.0  # Gaussian kernel width

    # Tube width
    min_tube_width_m: float = 1.0  # Minimum tube width (prevent collapse)

    # Physics-based refraction limits (prevents ray "parking" on steep bathymetry)
    # Minimum curvature radius = wavelength (ray theory breaks down below this)
    # CFL: step size = fraction of curvature radius for numerical stability
    cfl_curvature_fraction: float = 0.1  # Step ≤ 10% of curvature radius
    min_curvature_step_m: float = 1.0    # Floor on step size (meters)

    # Wavelength-dependent gradient smoothing
    # Long-period waves "see" smoother bathymetry than short-period waves
    # Based on WKB validity: bathymetry should vary slowly relative to wavelength
    gradient_min_smooth_m: float = 25.0    # Minimum smoothing radius (m)
    gradient_smooth_fraction: float = 0.25  # Fraction of wavelength (L/4)


@dataclass
class RayBatch:
    """
    Batch of rays for parallel propagation.

    All arrays have shape (n_rays,).
    """
    # Initial positions (UTM)
    x: np.ndarray
    y: np.ndarray

    # Initial directions (math convention: 0=east, counter-clockwise)
    theta: np.ndarray

    # Wave parameters
    period: np.ndarray  # Wave period (s)
    power: np.ndarray   # Wave power P = E × Cg × W (Watts, conserved)

    # Tube width (perpendicular to ray)
    width: np.ndarray  # Initial tube width (m)

    # Partition ID for each ray
    partition_id: np.ndarray

    @property
    def n_rays(self) -> int:
        return len(self.x)


class BoundaryRayInitializer:
    """
    Initialize rays from SWAN boundary at constant spatial density.

    For each partition:
    1. Sample offshore boundary at regular intervals (boundary_spacing_m)
    2. At each sample, create a single ray at the exact SWAN direction
    3. Compute initial power P = E × Cg × W (conserved during propagation)
    """

    def __init__(
        self,
        mesh: 'SurfZoneMesh',
        boundary_conditions: 'BoundaryConditions',
        config: ForwardTracerConfig,
    ):
        self.mesh = mesh
        self.boundary = boundary_conditions
        self.config = config

        # Build boundary lookup for SWAN data interpolation
        self._boundary_tree = None
        self._build_boundary_lookup()

    def _build_boundary_lookup(self) -> None:
        """Build KD-tree for fast boundary point lookup."""
        from scipy.spatial import cKDTree

        boundary_points = np.column_stack([self.boundary.x, self.boundary.y])
        self._boundary_tree = cKDTree(boundary_points)

    def _sample_offshore_boundary(self) -> List[Tuple[float, float, float]]:
        """
        Sample the offshore boundary at regular intervals.

        Returns list of (x, y, tangent_angle_rad) tuples in UTM.
        The tangent angle is in math convention (radians, 0=East, CCW positive).
        """
        samples = []

        if self.mesh.offshore_boundary is None:
            raise ValueError("Mesh has no offshore boundary defined")

        for boundary_segment in self.mesh.offshore_boundary:
            # Calculate cumulative distance along segment
            if len(boundary_segment) < 2:
                continue

            dx = np.diff(boundary_segment[:, 0])
            dy = np.diff(boundary_segment[:, 1])
            segment_lengths = np.sqrt(dx**2 + dy**2)
            cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
            total_length = cumulative_dist[-1]

            # Sample at regular intervals
            n_samples = max(2, int(total_length / self.config.boundary_spacing_m))
            sample_distances = np.linspace(0, total_length, n_samples)

            for d in sample_distances:
                # Interpolate position
                idx = np.searchsorted(cumulative_dist, d) - 1
                idx = max(0, min(idx, len(boundary_segment) - 2))

                t = (d - cumulative_dist[idx]) / (cumulative_dist[idx + 1] - cumulative_dist[idx] + 1e-10)
                x = boundary_segment[idx, 0] + t * (boundary_segment[idx + 1, 0] - boundary_segment[idx, 0])
                y = boundary_segment[idx, 1] + t * (boundary_segment[idx + 1, 1] - boundary_segment[idx, 1])

                # Tangent direction from the local segment
                dx_seg = boundary_segment[idx + 1, 0] - boundary_segment[idx, 0]
                dy_seg = boundary_segment[idx + 1, 1] - boundary_segment[idx, 1]
                tangent_angle = np.arctan2(dy_seg, dx_seg)

                samples.append((x, y, tangent_angle))

        return samples

    def _get_partition_at_boundary(
        self,
        x: float,
        y: float,
        partition: 'WavePartition',
    ) -> Tuple[float, float, float, bool]:
        """
        Get wave parameters at a boundary point via nearest neighbor lookup.

        Returns (Hs, Tp, direction_nautical, is_valid).
        """
        # Find nearest SWAN boundary point
        dist, idx = self._boundary_tree.query([x, y])

        if not partition.is_valid[idx]:
            return 0.0, 0.0, 0.0, False

        return (
            float(partition.hs[idx]),
            float(partition.tp[idx]),
            float(partition.direction[idx]),
            True,
        )

    def create_rays_for_partition(self, partition_idx: int) -> RayBatch:
        """
        Create rays for a single wave partition.

        Creates 1 ray per boundary point at the exact SWAN direction.
        Each ray's power P = E × Cg × W is computed and conserved during propagation.

        Args:
            partition_idx: Index into boundary_conditions.partitions

        Returns:
            RayBatch with initialized rays
        """
        partition = self.boundary.partitions[partition_idx]
        boundary_samples = self._sample_offshore_boundary()

        # Pre-allocate arrays (1 ray per boundary point)
        max_rays = len(boundary_samples)

        ray_x = np.empty(max_rays, dtype=np.float64)
        ray_y = np.empty(max_rays, dtype=np.float64)
        ray_theta = np.empty(max_rays, dtype=np.float64)
        ray_period = np.empty(max_rays, dtype=np.float64)
        ray_power = np.empty(max_rays, dtype=np.float64)
        ray_width = np.empty(max_rays, dtype=np.float64)
        ray_partition = np.empty(max_rays, dtype=np.int32)

        ray_count = 0

        for bx, by, tangent_angle in boundary_samples:
            hs, tp, dir_nautical, is_valid = self._get_partition_at_boundary(bx, by, partition)

            if not is_valid or hs < 0.01 or tp < 1.0:
                continue

            # Convert direction to math convention (wave travel direction)
            dir_math = nautical_to_math(dir_nautical)

            # Correct tube width for boundary-swell angle
            # W should be the perpendicular distance between adjacent rays,
            # not the along-contour spacing. When boundary tangent aligns with
            # swell direction, rays pile up and W must shrink accordingly.
            angle_diff = dir_math - tangent_angle
            sin_factor = abs(np.sin(angle_diff))
            W_corrected = max(
                self.config.min_tube_width_m,
                self.config.boundary_spacing_m * sin_factor,
            )

            # Deep water wavelength and group velocity at boundary
            L0 = G * tp * tp / TWO_PI
            Cg0 = L0 / (2.0 * tp)  # Deep water group velocity

            # Calculate wave energy density from Hs: E = (1/8) * ρ * g * Hs²
            energy = (1.0 / 8.0) * RHO * G * (hs ** 2)

            # Initial power P = E × Cg × W (conserved during propagation)
            power = energy * Cg0 * W_corrected

            # Single ray at exact SWAN direction
            ray_x[ray_count] = bx
            ray_y[ray_count] = by
            ray_theta[ray_count] = dir_math
            ray_period[ray_count] = tp
            ray_power[ray_count] = power
            ray_width[ray_count] = W_corrected
            ray_partition[ray_count] = partition_idx
            ray_count += 1

        # Trim to actual count
        return RayBatch(
            x=ray_x[:ray_count].copy(),
            y=ray_y[:ray_count].copy(),
            theta=ray_theta[:ray_count].copy(),
            period=ray_period[:ray_count].copy(),
            power=ray_power[:ray_count].copy(),
            width=ray_width[:ray_count].copy(),
            partition_id=ray_partition[:ray_count].copy(),
        )

    def create_all_rays(self) -> RayBatch:
        """
        Create rays for all partitions.

        Returns:
            Combined RayBatch with rays from all partitions
        """
        batches = []

        for i in range(self.boundary.n_partitions):
            batch = self.create_rays_for_partition(i)
            if batch.n_rays > 0:
                batches.append(batch)

        if not batches:
            # Return empty batch
            return RayBatch(
                x=np.array([], dtype=np.float64),
                y=np.array([], dtype=np.float64),
                theta=np.array([], dtype=np.float64),
                period=np.array([], dtype=np.float64),
                power=np.array([], dtype=np.float64),
                width=np.array([], dtype=np.float64),
                partition_id=np.array([], dtype=np.int32),
            )

        # Concatenate all batches
        return RayBatch(
            x=np.concatenate([b.x for b in batches]),
            y=np.concatenate([b.y for b in batches]),
            theta=np.concatenate([b.theta for b in batches]),
            period=np.concatenate([b.period for b in batches]),
            power=np.concatenate([b.power for b in batches]),
            width=np.concatenate([b.width for b in batches]),
            partition_id=np.concatenate([b.partition_id for b in batches]),
        )


# =============================================================================
# Numba-accelerated Helper Functions
# =============================================================================

@njit(cache=True)
def compute_step_size(
    depth: float,
    step_deep: float,
    step_10m: float,
    step_5m: float,
    step_3m: float,
) -> float:
    """
    Compute adaptive step size based on water depth.

    Uses larger steps in deep water (coarser mesh) and smaller steps
    in shallow water (finer mesh near coastline).

    Args:
        depth: Water depth in meters
        step_deep: Step size for deep water (>10m)
        step_10m: Step size at 10m depth
        step_5m: Step size at 5m depth
        step_3m: Step size at ≤3m depth

    Returns:
        Step size in meters
    """
    if depth <= 3.0:
        return step_3m
    elif depth <= 5.0:
        return step_5m
    elif depth <= 10.0:
        return step_10m
    else:
        return step_deep


# =============================================================================
# Numba-accelerated Energy Deposition
# =============================================================================

@njit(cache=True)
def _find_nearby_points(
    ray_x: float,
    ray_y: float,
    cutoff: float,
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    grid_x_min: float,
    grid_y_min: float,
    grid_cell_size: float,
    grid_n_cells_x: int,
    grid_n_cells_y: int,
    point_grid_starts: np.ndarray,
    point_grid_counts: np.ndarray,
    point_grid_indices: np.ndarray,
) -> np.ndarray:
    """
    Find mesh points within cutoff distance of ray position.

    Uses proper spatial index for O(k) lookup where k is the number of
    points in nearby cells (typically 50-200), instead of O(N) where
    N is the total number of mesh points (325K+).

    Args:
        ray_x, ray_y: Ray position
        cutoff: Maximum distance to search (typically 3*sigma)
        mesh_x, mesh_y: All mesh point coordinates
        grid_*: Spatial grid parameters
        point_grid_starts: Start index in point_grid_indices for each cell
        point_grid_counts: Number of points in each cell
        point_grid_indices: Flat array of point indices sorted by cell

    Returns:
        Array of mesh point indices within cutoff distance
    """
    # Find the cell range to search
    cx_center = int((ray_x - grid_x_min) / grid_cell_size)
    cy_center = int((ray_y - grid_y_min) / grid_cell_size)

    cells_to_search = int(np.ceil(cutoff / grid_cell_size))

    cx_min = max(0, cx_center - cells_to_search)
    cx_max = min(grid_n_cells_x - 1, cx_center + cells_to_search)
    cy_min = max(0, cy_center - cells_to_search)
    cy_max = min(grid_n_cells_y - 1, cy_center + cells_to_search)

    cutoff_sq = cutoff * cutoff

    # First pass: count nearby points (O(k) where k = points in nearby cells)
    n_nearby = 0
    for cy in range(cy_min, cy_max + 1):
        for cx in range(cx_min, cx_max + 1):
            cell_idx = cy * grid_n_cells_x + cx
            start = point_grid_starts[cell_idx]
            count = point_grid_counts[cell_idx]

            for j in range(count):
                i = point_grid_indices[start + j]
                dx = mesh_x[i] - ray_x
                dy = mesh_y[i] - ray_y
                if dx*dx + dy*dy < cutoff_sq:
                    n_nearby += 1

    # Second pass: collect indices
    nearby = np.empty(n_nearby, dtype=np.int64)
    idx = 0
    for cy in range(cy_min, cy_max + 1):
        for cx in range(cx_min, cx_max + 1):
            cell_idx = cy * grid_n_cells_x + cx
            start = point_grid_starts[cell_idx]
            count = point_grid_counts[cell_idx]

            for j in range(count):
                i = point_grid_indices[start + j]
                dx = mesh_x[i] - ray_x
                dy = mesh_y[i] - ray_y
                if dx*dx + dy*dy < cutoff_sq:
                    nearby[idx] = i
                    idx += 1

    return nearby


@njit(cache=True)
def deposit_ray_energy_directional(
    ray_x: float,
    ray_y: float,
    ray_dx: float,
    ray_dy: float,
    E_local: float,
    area: float,
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    energy_grid: np.ndarray,
    dir_x_grid: np.ndarray,
    dir_y_grid: np.ndarray,
    ray_count: np.ndarray,
    sigma: float,
    nearby_indices: np.ndarray,
):
    """
    Deposit energy density to nearby mesh points using directional Gaussian kernel.

    Only deposits in the ray's forward hemisphere (cos(angle) > 0).
    This prevents energy leakage into shielded areas like crevices.

    Also accumulates direction components weighted by energy for computing
    energy-weighted average direction at each point.

    Args:
        ray_x, ray_y: Ray position
        ray_dx, ray_dy: Ray direction (unit vector, math convention)
        E_local: Local energy density P/(Cg×W) (J/m²)
        area: Tube area swept per step W × ds (m²)
        mesh_x, mesh_y: Mesh point coordinates
        energy_grid: Energy density accumulator (modified in place)
        dir_x_grid: Direction x-component accumulator Σ(energy × cos(θ)) (modified in place)
        dir_y_grid: Direction y-component accumulator Σ(energy × sin(θ)) (modified in place)
        ray_count: Ray counter per point (modified in place)
        sigma: Gaussian kernel width
        nearby_indices: Indices of nearby mesh points to check

    Units:
        E_local (J/m²) × area (m²) × normalized_weight (1/m²) = J/m² (energy density)
    """
    sigma_sq = sigma * sigma

    for idx in nearby_indices:
        dx = mesh_x[idx] - ray_x
        dy = mesh_y[idx] - ray_y
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(dist_sq) + 1e-10

        # Direction factor: dot product gives cos(angle)
        cos_angle = (dx * ray_dx + dy * ray_dy) / dist

        # Only deposit in forward hemisphere
        if cos_angle <= 0:
            continue

        # Gaussian weight
        weight = np.exp(-0.5 * dist_sq / sigma_sq)

        # Normalize by cos-weighted hemisphere integral = 2σ²
        # (NOT πσ² - the integral of exp(-r²/2σ²) × cos(θ) over forward hemisphere = 2σ²)
        normalized_weight = weight * cos_angle / (2.0 * sigma_sq)

        # Deposit energy density
        energy_contribution = E_local * area * normalized_weight
        energy_grid[idx] += energy_contribution

        # Accumulate direction weighted by energy (for energy-weighted average)
        dir_x_grid[idx] += energy_contribution * ray_dx
        dir_y_grid[idx] += energy_contribution * ray_dy

        ray_count[idx] += 1


@njit(cache=True)
def propagate_single_ray(
    start_x: float,
    start_y: float,
    start_theta: float,
    start_power: float,
    start_width: float,
    period: float,
    # Mesh arrays
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    points_x: np.ndarray,
    points_y: np.ndarray,
    depth: np.ndarray,
    triangles: np.ndarray,
    # Spatial index for depth interpolation
    grid_x_min: float,
    grid_y_min: float,
    grid_cell_size: float,
    grid_n_cells_x: int,
    grid_n_cells_y: int,
    grid_cell_starts: np.ndarray,
    grid_cell_counts: np.ndarray,
    grid_triangles: np.ndarray,
    # Point spatial index for energy deposition (O(k) lookups)
    point_grid_starts: np.ndarray,
    point_grid_counts: np.ndarray,
    point_grid_indices: np.ndarray,
    # Output
    energy_grid: np.ndarray,
    dir_x_grid: np.ndarray,
    dir_y_grid: np.ndarray,
    ray_count: np.ndarray,
    # Config - adaptive step sizing
    max_distance: float,
    step_deep: float,
    step_10m: float,
    step_5m: float,
    step_3m: float,
    kernel_sigma: float,
    min_width: float,
    # Physics-based refraction limits
    cfl_curvature_fraction: float,
    min_curvature_step: float,
    # Wavelength-dependent gradient smoothing
    gradient_min_smooth: float,
    gradient_smooth_fraction: float,
) -> Tuple[float, bool, float]:
    """
    Propagate a single ray and deposit energy density at each step.

    Uses power-based model where P = E × Cg × W is conserved.
    Local energy density E = P / (Cg × W) increases with focusing (smaller W).

    Step size adapts to water depth:
    - Deep water (>10m): step_deep (default 50m)
    - 10m depth: step_10m (default 20m)
    - 5m depth: step_5m (default 10m)
    - ≤3m depth: step_3m (default 5m)

    Also accumulates direction components (dir_x_grid, dir_y_grid) weighted by
    energy for computing energy-weighted average direction at each mesh point.

    Returns:
        Tuple of (distance_traveled, broke, final_depth)
    """
    x = start_x
    y = start_y
    theta = start_theta
    power = start_power  # Power is conserved (not energy)
    width = start_width

    # Deep water properties (computed once per ray)
    L0 = 9.81 * period * period / (2.0 * 3.141592653589793)

    cutoff = 3.0 * kernel_sigma
    broke = False
    final_depth = 0.0
    distance_traveled = 0.0

    while distance_traveled < max_distance:
        # 1. Interpolate depth at current position
        h = interpolate_depth_indexed(
            x, y, points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles
        )

        if np.isnan(h) or h <= 0:
            # Hit land or outside mesh
            final_depth = h if not np.isnan(h) else 0.0
            return distance_traveled, False, final_depth

        final_depth = h

        # 2. Compute adaptive step size based on depth
        current_step = compute_step_size(h, step_deep, step_10m, step_5m, step_3m)

        # 3. Compute local wave properties
        # Fenton-McKee approximation for wavelength
        x_fm = (2.0 * 3.141592653589793 * h / L0) ** 0.75
        L = L0 * (np.tanh(x_fm) ** (2.0 / 3.0))
        C = L / period

        # Group velocity
        k = 2.0 * 3.141592653589793 / L
        kh = k * h
        if kh > 10.0:
            n = 0.5
        elif kh < 0.01:
            n = 1.0
        else:
            n = 0.5 * (1.0 + 2.0 * kh / np.sinh(2.0 * kh))
        Cg = n * C

        # 4. Compute local energy density from power conservation
        # P = E × Cg × W is conserved, so E = P / (Cg × W)
        # Focusing (W smaller) → higher E; Spreading (W larger) → lower E
        E_local = power / (Cg * width)

        # 5. Ray direction components
        dx = np.cos(theta)
        dy = np.sin(theta)

        # 6. Compute tube area for this step
        area = width * current_step

        # 7. Find nearby points and deposit energy density (O(k) using spatial index)
        nearby = _find_nearby_points(
            x, y, cutoff,
            mesh_x, mesh_y,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            point_grid_starts, point_grid_counts, point_grid_indices,
        )

        if len(nearby) > 0:
            deposit_ray_energy_directional(
                x, y, dx, dy,
                E_local, area,
                mesh_x, mesh_y,
                energy_grid, dir_x_grid, dir_y_grid, ray_count,
                kernel_sigma, nearby,
            )

        # 8. Compute refraction (celerity gradients with wavelength-dependent smoothing)
        dC_dx, dC_dy = celerity_gradient_smoothed(
            x, y, period, L0, L,
            points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles,
            gradient_min_smooth, gradient_smooth_fraction,
        )

        # 9. Apply physics-based refraction limits
        # Compute perpendicular gradient magnitude
        dC_dn = -dy * dC_dx + dx * dC_dy
        dC_dn_mag = abs(dC_dn)

        if C > 0 and dC_dn_mag > 0:
            # Compute curvature radius: R = C / |dC/dn|
            curvature_radius = C / dC_dn_mag

            # Physical limit: minimum curvature radius = wavelength
            # Ray theory breaks down below R = L (waves diffract, don't refract sharply)
            if curvature_radius < L:
                curvature_radius = L
                # Clamp the gradient to enforce minimum radius
                dC_dn_mag = C / L
                dC_dn = np.sign(dC_dn) * dC_dn_mag if dC_dn != 0 else 0.0

            # CFL condition: step size should be fraction of curvature radius
            cfl_max_step = cfl_curvature_fraction * curvature_radius
            cfl_max_step = max(cfl_max_step, min_curvature_step)
            if current_step > cfl_max_step:
                current_step = cfl_max_step
                # Recompute area with adjusted step
                area = width * current_step

        # 10. Update direction using (possibly clamped) gradient
        if C > 0:
            dtheta = -(current_step / C) * dC_dn
            theta += dtheta

            # 11. Update tube width: dW/ds = W × dθ/ds
            width += width * dtheta
            width = max(width, min_width)

        # 12. Move ray forward
        x += dx * current_step
        y += dy * current_step
        distance_traveled += current_step

    return distance_traveled, broke, final_depth


@njit(cache=True)
def propagate_single_ray_with_path(
    start_x: float,
    start_y: float,
    start_theta: float,
    start_power: float,
    start_width: float,
    period: float,
    # Mesh arrays
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    points_x: np.ndarray,
    points_y: np.ndarray,
    depth: np.ndarray,
    triangles: np.ndarray,
    # Spatial index for depth interpolation
    grid_x_min: float,
    grid_y_min: float,
    grid_cell_size: float,
    grid_n_cells_x: int,
    grid_n_cells_y: int,
    grid_cell_starts: np.ndarray,
    grid_cell_counts: np.ndarray,
    grid_triangles: np.ndarray,
    # Point spatial index for energy deposition (O(k) lookups)
    point_grid_starts: np.ndarray,
    point_grid_counts: np.ndarray,
    point_grid_indices: np.ndarray,
    # Output
    energy_grid: np.ndarray,
    dir_x_grid: np.ndarray,
    dir_y_grid: np.ndarray,
    ray_count: np.ndarray,
    # Path output arrays (pre-allocated, max possible steps)
    path_x: np.ndarray,
    path_y: np.ndarray,
    path_depth: np.ndarray,
    path_direction: np.ndarray,
    path_tube_width: np.ndarray,
    path_Hs_local: np.ndarray,
    # Config - adaptive step sizing
    max_distance: float,
    step_deep: float,
    step_10m: float,
    step_5m: float,
    step_3m: float,
    kernel_sigma: float,
    min_width: float,
    # Physics-based refraction limits
    cfl_curvature_fraction: float,
    min_curvature_step: float,
    # Wavelength-dependent gradient smoothing
    gradient_min_smooth: float,
    gradient_smooth_fraction: float,
) -> int:
    """
    Propagate a single ray, deposit energy, and record path.

    Same as propagate_single_ray but also fills path arrays.
    Uses adaptive step sizing based on water depth.

    Returns:
        Number of steps taken (length of valid path data)
    """
    x = start_x
    y = start_y
    theta = start_theta
    power = start_power
    width = start_width

    L0 = 9.81 * period * period / (2.0 * 3.141592653589793)
    cutoff = 3.0 * kernel_sigma
    distance_traveled = 0.0
    step = 0
    max_steps = len(path_x)  # Use pre-allocated array length as safety limit

    while distance_traveled < max_distance and step < max_steps:
        # 1. Interpolate depth at current position
        h = interpolate_depth_indexed(
            x, y, points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles
        )

        if np.isnan(h) or h <= 0:
            return step

        # 2. Compute adaptive step size based on depth
        current_step = compute_step_size(h, step_deep, step_10m, step_5m, step_3m)

        # 3. Compute local wave properties
        x_fm = (2.0 * 3.141592653589793 * h / L0) ** 0.75
        L = L0 * (np.tanh(x_fm) ** (2.0 / 3.0))
        C = L / period

        k = 2.0 * 3.141592653589793 / L
        kh = k * h
        if kh > 10.0:
            n = 0.5
        elif kh < 0.01:
            n = 1.0
        else:
            n = 0.5 * (1.0 + 2.0 * kh / np.sinh(2.0 * kh))
        Cg = n * C

        # 4. Compute local energy density and Hs
        E_local = power / (Cg * width)
        Hs_local = np.sqrt(8.0 * E_local / (1025.0 * 9.81))

        # 5. Record path data
        path_x[step] = x
        path_y[step] = y
        path_depth[step] = h
        path_direction[step] = theta
        path_tube_width[step] = width
        path_Hs_local[step] = Hs_local

        # 6. Ray direction components
        dx = np.cos(theta)
        dy = np.sin(theta)

        # 7. Compute tube area and deposit energy
        area = width * current_step
        nearby = _find_nearby_points(
            x, y, cutoff,
            mesh_x, mesh_y,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            point_grid_starts, point_grid_counts, point_grid_indices,
        )

        if len(nearby) > 0:
            deposit_ray_energy_directional(
                x, y, dx, dy,
                E_local, area,
                mesh_x, mesh_y,
                energy_grid, dir_x_grid, dir_y_grid, ray_count,
                kernel_sigma, nearby,
            )

        # 8. Compute refraction (with wavelength-dependent smoothing)
        dC_dx, dC_dy = celerity_gradient_smoothed(
            x, y, period, L0, L,
            points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles,
            gradient_min_smooth, gradient_smooth_fraction,
        )

        # 9. Apply physics-based refraction limits
        dC_dn = -dy * dC_dx + dx * dC_dy
        dC_dn_mag = abs(dC_dn)

        if C > 0 and dC_dn_mag > 0:
            curvature_radius = C / dC_dn_mag

            # Physical limit: minimum curvature radius = wavelength
            if curvature_radius < L:
                curvature_radius = L
                dC_dn_mag = C / L
                dC_dn = np.sign(dC_dn) * dC_dn_mag if dC_dn != 0 else 0.0

            # CFL: step size should be fraction of curvature radius
            cfl_max_step = cfl_curvature_fraction * curvature_radius
            cfl_max_step = max(cfl_max_step, min_curvature_step)
            if current_step > cfl_max_step:
                current_step = cfl_max_step
                area = width * current_step

        # 10. Update direction and width
        if C > 0:
            dtheta = -(current_step / C) * dC_dn
            theta += dtheta
            width += width * dtheta
            width = max(width, min_width)

        # 11. Move ray forward
        x += dx * current_step
        y += dy * current_step
        distance_traveled += current_step
        step += 1

    return step


@njit(parallel=True, cache=True)
def propagate_all_rays(
    # Ray initial state
    rays_x: np.ndarray,
    rays_y: np.ndarray,
    rays_theta: np.ndarray,
    rays_power: np.ndarray,
    rays_width: np.ndarray,
    rays_period: np.ndarray,
    # Mesh arrays
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
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
    # Point spatial index for O(k) lookups
    point_grid_starts: np.ndarray,
    point_grid_counts: np.ndarray,
    point_grid_indices: np.ndarray,
    # Config - adaptive step sizing
    max_distance: float,
    step_deep: float,
    step_10m: float,
    step_5m: float,
    step_3m: float,
    kernel_sigma: float,
    min_width: float,
    # Physics-based refraction limits
    cfl_curvature_fraction: float,
    min_curvature_step: float,
    # Wavelength-dependent gradient smoothing
    gradient_min_smooth: float,
    gradient_smooth_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagate all rays in parallel and deposit energy.

    NOTE: This function creates a separate energy grid per thread to avoid
    race conditions, then combines them at the end.

    Returns:
        Tuple of (energy_grid, dir_x_grid, dir_y_grid, ray_counts, distances_traveled, broke_flags)
    """
    n_rays = len(rays_x)
    n_points = len(mesh_x)

    # Output arrays
    distances_traveled = np.empty(n_rays, dtype=np.float64)
    broke_flags = np.empty(n_rays, dtype=np.bool_)

    # NOTE: For true parallel execution, each thread needs its own energy grid.
    # This is a simplified version that uses sequential execution for energy deposition.
    # For production, consider using atomic operations or thread-local storage.

    energy_grid = np.zeros(n_points, dtype=np.float64)
    dir_x_grid = np.zeros(n_points, dtype=np.float64)
    dir_y_grid = np.zeros(n_points, dtype=np.float64)
    ray_counts = np.zeros(n_points, dtype=np.int32)

    # Sequential for correctness (parallel deposition needs atomic ops)
    for ray_idx in range(n_rays):
        dist, broke, _ = propagate_single_ray(
            rays_x[ray_idx],
            rays_y[ray_idx],
            rays_theta[ray_idx],
            rays_power[ray_idx],
            rays_width[ray_idx],
            rays_period[ray_idx],
            mesh_x, mesh_y,
            points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles,
            point_grid_starts, point_grid_counts, point_grid_indices,
            energy_grid, dir_x_grid, dir_y_grid, ray_counts,
            max_distance, step_deep, step_10m, step_5m, step_3m,
            kernel_sigma, min_width,
            cfl_curvature_fraction, min_curvature_step,
            gradient_min_smooth, gradient_smooth_fraction,
        )
        distances_traveled[ray_idx] = dist
        broke_flags[ray_idx] = broke

    return energy_grid, dir_x_grid, dir_y_grid, ray_counts, distances_traveled, broke_flags


# =============================================================================
# Main Forward Ray Tracer Class
# =============================================================================

class ForwardRayTracer:
    """
    Forward ray tracer with energy deposition.

    Usage:
        tracer = ForwardRayTracer(mesh, boundary_conditions, config)
        energy, ray_counts = tracer.trace_all_partitions()
        Hs = tracer.energy_to_wave_height(energy)

    With path tracking:
        tracer = ForwardRayTracer(mesh, boundary_conditions, config,
                                  track_paths=True, sample_fraction=0.1)
        Hs, energy, ray_counts, per_partition_data, ray_paths = tracer.run()
    """

    def __init__(
        self,
        mesh: 'SurfZoneMesh',
        boundary_conditions: 'BoundaryConditions',
        config: Optional[ForwardTracerConfig] = None,
        track_paths: bool = False,
        sample_fraction: float = 0.1,
    ):
        self.mesh = mesh
        self.boundary = boundary_conditions
        self.config = config or ForwardTracerConfig()

        # Path tracking options
        self.track_paths = track_paths
        self.sample_fraction = np.clip(sample_fraction, 0.0, 1.0)

        # Initialize ray generator
        self.initializer = BoundaryRayInitializer(mesh, boundary_conditions, self.config)

        # Get mesh arrays for Numba
        self._mesh_arrays = mesh.get_numba_arrays()

        # Build point spatial index for O(k) lookups (instead of O(N))
        self._point_grid_starts, self._point_grid_counts, self._point_grid_indices = \
            self._build_point_spatial_index()

        # Path collection storage (filled during tracing if track_paths=True)
        self._collected_paths = []
        self._total_rays_traced = 0

    def _build_point_spatial_index(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a spatial index mapping cells to point indices.

        This enables O(k) point lookup instead of O(N) by indexing
        points by their grid cell, similar to the triangle spatial index.

        Returns:
            point_grid_starts: Start index in point_grid_indices for each cell
            point_grid_counts: Number of points in each cell
            point_grid_indices: Flat array of point indices sorted by cell
        """
        mesh_x = self._mesh_arrays['points_x']
        mesh_y = self._mesh_arrays['points_y']
        grid_x_min = float(self._mesh_arrays['grid_x_min'])
        grid_y_min = float(self._mesh_arrays['grid_y_min'])
        cell_size = float(self._mesh_arrays['grid_cell_size'])
        n_cells_x = int(self._mesh_arrays['grid_n_cells_x'])
        n_cells_y = int(self._mesh_arrays['grid_n_cells_y'])

        n_points = len(mesh_x)
        n_cells = n_cells_x * n_cells_y

        # Compute cell index for each point
        cell_x = np.clip(((mesh_x - grid_x_min) / cell_size).astype(np.int32), 0, n_cells_x - 1)
        cell_y = np.clip(((mesh_y - grid_y_min) / cell_size).astype(np.int32), 0, n_cells_y - 1)
        cell_idx = cell_y * n_cells_x + cell_x

        # Count points per cell
        cell_counts = np.zeros(n_cells, dtype=np.int32)
        for i in range(n_points):
            cell_counts[cell_idx[i]] += 1

        # Compute start indices (cumulative sum)
        cell_starts = np.zeros(n_cells, dtype=np.int32)
        cell_starts[1:] = np.cumsum(cell_counts[:-1])

        # Fill point indices array
        point_indices = np.empty(n_points, dtype=np.int32)
        current_pos = cell_starts.copy()
        for i in range(n_points):
            c = cell_idx[i]
            point_indices[current_pos[c]] = i
            current_pos[c] += 1

        return cell_starts, cell_counts, point_indices

    def trace_partition(self, partition_idx: int, batch_size: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Trace rays for a single partition and accumulate energy and direction.

        If track_paths is enabled, sampled ray paths are collected into self._collected_paths.

        Args:
            partition_idx: Index of partition to trace
            batch_size: Number of rays to process per batch (for progress updates)

        Returns:
            Tuple of (energy_grid, dir_x_grid, dir_y_grid, ray_counts) arrays
        """
        import time
        import sys

        # Create rays for this partition
        rays = self.initializer.create_rays_for_partition(partition_idx)

        if rays.n_rays == 0:
            n_points = len(self._mesh_arrays['points_x'])
            return (
                np.zeros(n_points),
                np.zeros(n_points),
                np.zeros(n_points),
                np.zeros(n_points, dtype=np.int32),
            )

        print(f"  Partition {partition_idx}: {rays.n_rays:,} rays")

        # Get mesh arrays
        ma = self._mesh_arrays
        n_points = len(ma['points_x'])

        # Initialize accumulators
        energy_grid = np.zeros(n_points, dtype=np.float64)
        dir_x_grid = np.zeros(n_points, dtype=np.float64)
        dir_y_grid = np.zeros(n_points, dtype=np.float64)
        ray_counts = np.zeros(n_points, dtype=np.int32)
        all_steps = []
        all_broke = []

        # Determine which rays to sample for path tracking
        n_rays = rays.n_rays
        if self.track_paths and self.sample_fraction > 0:
            n_sample = max(1, int(n_rays * self.sample_fraction))
            sample_indices = set(np.random.choice(n_rays, size=n_sample, replace=False))
        else:
            sample_indices = set()

        # Pre-allocate path arrays for path-tracked rays
        # Max possible steps = max_distance / smallest step size + buffer
        max_possible_steps = int(self.config.max_distance_m / self.config.step_3m_m) + 100
        path_x = np.zeros(max_possible_steps, dtype=np.float64)
        path_y = np.zeros(max_possible_steps, dtype=np.float64)
        path_depth = np.zeros(max_possible_steps, dtype=np.float64)
        path_direction = np.zeros(max_possible_steps, dtype=np.float64)
        path_tube_width = np.zeros(max_possible_steps, dtype=np.float64)
        path_Hs_local = np.zeros(max_possible_steps, dtype=np.float64)

        # Track rays individually when path tracking is needed
        start_time = time.time()
        rays_done = 0
        all_distances = []

        for ray_idx in range(n_rays):
            original_ray_idx = self._total_rays_traced + ray_idx

            if ray_idx in sample_indices:
                # Use path-tracking version
                steps = propagate_single_ray_with_path(
                    rays.x[ray_idx], rays.y[ray_idx], rays.theta[ray_idx],
                    rays.power[ray_idx], rays.width[ray_idx], rays.period[ray_idx],
                    ma['points_x'], ma['points_y'],
                    ma['points_x'], ma['points_y'],
                    ma['depth'], ma['triangles'],
                    float(ma['grid_x_min']), float(ma['grid_y_min']),
                    float(ma['grid_cell_size']),
                    int(ma['grid_n_cells_x']), int(ma['grid_n_cells_y']),
                    ma['grid_cell_starts'], ma['grid_cell_counts'], ma['grid_triangles'],
                    self._point_grid_starts, self._point_grid_counts, self._point_grid_indices,
                    energy_grid, dir_x_grid, dir_y_grid, ray_counts,
                    path_x, path_y, path_depth, path_direction, path_tube_width, path_Hs_local,
                    self.config.max_distance_m,
                    self.config.step_deep_m, self.config.step_10m_m,
                    self.config.step_5m_m, self.config.step_3m_m,
                    self.config.kernel_sigma_m, self.config.min_tube_width_m,
                    self.config.cfl_curvature_fraction, self.config.min_curvature_step_m,
                    self.config.gradient_min_smooth_m, self.config.gradient_smooth_fraction,
                )

                # Store the path data (copy the valid portion)
                if steps > 0:
                    self._collected_paths.append({
                        'partition': partition_idx,
                        'original_idx': original_ray_idx,
                        'length': steps,
                        'x': path_x[:steps].copy(),
                        'y': path_y[:steps].copy(),
                        'depth': path_depth[:steps].copy(),
                        'direction': path_direction[:steps].copy(),
                        'tube_width': path_tube_width[:steps].copy(),
                        'Hs_local': path_Hs_local[:steps].copy(),
                    })

                all_steps.append(steps)
                all_broke.append(False)
            else:
                # Use standard version (no path tracking)
                dist, broke, _ = propagate_single_ray(
                    rays.x[ray_idx], rays.y[ray_idx], rays.theta[ray_idx],
                    rays.power[ray_idx], rays.width[ray_idx], rays.period[ray_idx],
                    ma['points_x'], ma['points_y'],
                    ma['points_x'], ma['points_y'],
                    ma['depth'], ma['triangles'],
                    float(ma['grid_x_min']), float(ma['grid_y_min']),
                    float(ma['grid_cell_size']),
                    int(ma['grid_n_cells_x']), int(ma['grid_n_cells_y']),
                    ma['grid_cell_starts'], ma['grid_cell_counts'], ma['grid_triangles'],
                    self._point_grid_starts, self._point_grid_counts, self._point_grid_indices,
                    energy_grid, dir_x_grid, dir_y_grid, ray_counts,
                    self.config.max_distance_m,
                    self.config.step_deep_m, self.config.step_10m_m,
                    self.config.step_5m_m, self.config.step_3m_m,
                    self.config.kernel_sigma_m, self.config.min_tube_width_m,
                    self.config.cfl_curvature_fraction, self.config.min_curvature_step_m,
                    self.config.gradient_min_smooth_m, self.config.gradient_smooth_fraction,
                )
                all_distances.append(dist)
                all_broke.append(broke)

            # Progress update (every batch_size rays)
            rays_done += 1
            if rays_done % batch_size == 0 or rays_done == n_rays:
                elapsed = time.time() - start_time
                rays_per_sec = rays_done / elapsed if elapsed > 0 else 0
                remaining = (n_rays - rays_done) / rays_per_sec if rays_per_sec > 0 else 0
                pct = 100 * rays_done / n_rays
                print(f"    Progress: {rays_done:,}/{n_rays:,} rays ({pct:.1f}%) - {rays_per_sec:.0f} rays/s - ETA: {remaining:.1f}s", end='\r')
                sys.stdout.flush()

        print()  # New line after progress

        # Update total rays counter
        self._total_rays_traced += n_rays

        # Report statistics
        n_broke = sum(all_broke)
        avg_dist = np.mean(all_distances) if all_distances else 0
        n_sampled = len(sample_indices)
        if self.track_paths and n_sampled > 0:
            print(f"    Avg distance: {avg_dist:.0f}m, Sampled paths: {n_sampled:,} ({100*n_sampled/n_rays:.1f}%)")
        else:
            print(f"    Avg distance: {avg_dist:.0f}m")

        return energy_grid, dir_x_grid, dir_y_grid, ray_counts

    def trace_all_partitions(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Trace rays for all partitions and collect per-partition data.

        Returns:
            Tuple of (total_energy, total_ray_counts, per_partition_data)
            where per_partition_data is a dict mapping partition_idx to
            (energy, dir_x, dir_y, ray_counts) tuples.
        """
        # Reset path collection for new run
        self._collected_paths = []
        self._total_rays_traced = 0

        n_points = len(self._mesh_arrays['points_x'])
        total_energy = np.zeros(n_points)
        total_ray_counts = np.zeros(n_points, dtype=np.int32)
        per_partition_data = {}

        for i in range(self.boundary.n_partitions):
            energy, dir_x, dir_y, counts = self.trace_partition(i)

            # Store per-partition data
            per_partition_data[i] = {
                'energy': energy.copy(),
                'dir_x': dir_x.copy(),
                'dir_y': dir_y.copy(),
                'ray_counts': counts.copy(),
            }

            # Accumulate totals
            total_energy += energy
            total_ray_counts += counts

        return total_energy, total_ray_counts, per_partition_data

    def direction_components_to_nautical(self, dir_x: np.ndarray, dir_y: np.ndarray) -> np.ndarray:
        """
        Convert accumulated direction components to nautical direction (degrees FROM).

        Args:
            dir_x: Accumulated x-direction components Σ(energy × cos(θ))
            dir_y: Accumulated y-direction components Σ(energy × sin(θ))

        Returns:
            Direction in degrees (nautical FROM convention: 0=N, 90=E, 180=S, 270=W)
        """
        # Math angle from accumulated components (radians)
        # atan2(y, x) gives angle in math convention (0=E, counter-clockwise)
        theta_math = np.arctan2(dir_y, dir_x)

        # Convert to degrees
        theta_deg = np.degrees(theta_math)

        # Convert from math convention (travel direction) to nautical FROM convention
        # Math: 0=E, 90=N, 180=W, 270=S (travel direction, counter-clockwise from E)
        # Nautical FROM: 0=N, 90=E, 180=S, 270=W (direction waves come FROM)
        # travel_deg = (90 - theta_deg) % 360  # Convert to nautical travel
        # from_deg = (travel_deg + 180) % 360  # Convert to FROM
        # Simplified:
        direction_nautical = (270.0 - theta_deg) % 360.0

        return direction_nautical

    def energy_to_wave_height(self, energy: np.ndarray) -> np.ndarray:
        """
        Convert accumulated energy to significant wave height.

        Uses: E = (1/8) × ρ × g × Hs²
        Therefore: Hs = sqrt(8 × E / (ρ × g))

        Args:
            energy: Energy array (J/m)

        Returns:
            Significant wave height array (m)
        """
        # Avoid division issues
        energy_safe = np.maximum(energy, 0)
        Hs = np.sqrt(8.0 * energy_safe / (RHO * G))
        return Hs

    def _build_ray_path_data(self) -> Optional['RayPathData']:
        """
        Build RayPathData from collected paths.

        Returns:
            RayPathData object, or None if no paths were collected
        """
        from .surfzone_result import RayPathData

        if not self._collected_paths:
            return None

        n_sampled = len(self._collected_paths)
        total_steps = sum(p['length'] for p in self._collected_paths)

        # Build per-ray metadata arrays
        ray_partition = np.empty(n_sampled, dtype=np.int32)
        ray_start_idx = np.empty(n_sampled, dtype=np.int64)
        ray_length = np.empty(n_sampled, dtype=np.int32)
        ray_original_idx = np.empty(n_sampled, dtype=np.int32)

        # Build concatenated path arrays
        path_x = np.empty(total_steps, dtype=np.float32)
        path_y = np.empty(total_steps, dtype=np.float32)
        path_depth = np.empty(total_steps, dtype=np.float32)
        path_direction = np.empty(total_steps, dtype=np.float32)
        path_tube_width = np.empty(total_steps, dtype=np.float32)
        path_Hs_local = np.empty(total_steps, dtype=np.float32)

        current_idx = 0
        for i, path in enumerate(self._collected_paths):
            length = path['length']
            ray_partition[i] = path['partition']
            ray_start_idx[i] = current_idx
            ray_length[i] = length
            ray_original_idx[i] = path['original_idx']

            path_x[current_idx:current_idx + length] = path['x']
            path_y[current_idx:current_idx + length] = path['y']
            path_depth[current_idx:current_idx + length] = path['depth']
            path_direction[current_idx:current_idx + length] = path['direction']
            path_tube_width[current_idx:current_idx + length] = path['tube_width']
            path_Hs_local[current_idx:current_idx + length] = path['Hs_local']

            current_idx += length

        return RayPathData(
            ray_partition=ray_partition,
            ray_start_idx=ray_start_idx,
            ray_length=ray_length,
            ray_original_idx=ray_original_idx,
            path_x=path_x,
            path_y=path_y,
            path_depth=path_depth,
            path_direction=path_direction,
            path_tube_width=path_tube_width,
            path_Hs_local=path_Hs_local,
            n_rays_total=self._total_rays_traced,
            sample_fraction=self.sample_fraction,
        )

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, Optional['RayPathData']]:
        """
        Run complete forward ray tracing simulation.

        Returns:
            Tuple of (Hs, energy, ray_counts, per_partition_data, ray_paths)
            where per_partition_data is a dict mapping partition_idx to
            dict with keys: 'energy', 'dir_x', 'dir_y', 'ray_counts', 'Hs', 'direction'
            and ray_paths is RayPathData if track_paths=True, else None
        """
        print(f"Forward ray tracing: {self.boundary.n_partitions} partitions")
        if self.track_paths:
            print(f"  Path tracking enabled: sampling {self.sample_fraction*100:.0f}% of rays")

        energy, ray_counts, per_partition_data = self.trace_all_partitions()
        Hs = self.energy_to_wave_height(energy)

        # Compute Hs and direction for each partition
        for part_idx, part_data in per_partition_data.items():
            part_data['Hs'] = self.energy_to_wave_height(part_data['energy'])
            part_data['direction'] = self.direction_components_to_nautical(
                part_data['dir_x'], part_data['dir_y']
            )

        # Report coverage
        n_covered = np.sum(ray_counts > 0)
        n_total = len(ray_counts)
        print(f"Coverage: {n_covered:,} / {n_total:,} points ({100*n_covered/n_total:.1f}%)")

        # Build ray path data if tracking was enabled
        ray_paths = None
        if self.track_paths:
            ray_paths = self._build_ray_path_data()
            if ray_paths:
                print(f"Ray paths: {ray_paths.summary()}")

        return Hs, energy, ray_counts, per_partition_data, ray_paths
