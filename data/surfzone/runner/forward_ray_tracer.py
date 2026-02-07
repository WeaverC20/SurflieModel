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
    breaker_index_mcowan,
)
from .ray_tracer import (
    interpolate_depth_indexed,
    celerity_gradient_indexed,
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

    # Ray propagation
    step_size_m: float = 10.0  # Step size for ray marching
    max_steps: int = 300  # Maximum steps before termination

    # Energy deposition
    kernel_sigma_m: float = 25.0  # Gaussian kernel width

    # Tube width
    min_tube_width_m: float = 1.0  # Minimum tube width (prevent collapse)

    # Breaking
    breaker_index: float = 0.78  # McCowan breaker index


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

    def _sample_offshore_boundary(self) -> List[Tuple[float, float]]:
        """
        Sample the offshore boundary at regular intervals.

        Returns list of (x, y) points in UTM.
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

                samples.append((x, y))

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

        for bx, by in boundary_samples:
            hs, tp, dir_nautical, is_valid = self._get_partition_at_boundary(bx, by, partition)

            if not is_valid or hs < 0.01 or tp < 1.0:
                continue

            # Convert direction to math convention (wave travel direction)
            dir_math = nautical_to_math(dir_nautical)

            # Deep water wavelength and group velocity at boundary
            L0 = G * tp * tp / TWO_PI
            Cg0 = L0 / (2.0 * tp)  # Deep water group velocity

            # Calculate wave energy density from Hs: E = (1/8) * ρ * g * Hs²
            energy = (1.0 / 8.0) * RHO * G * (hs ** 2)

            # Initial power P = E × Cg × W (conserved during propagation)
            power = energy * Cg0 * self.config.boundary_spacing_m

            # Single ray at exact SWAN direction
            ray_x[ray_count] = bx
            ray_y[ray_count] = by
            ray_theta[ray_count] = dir_math
            ray_period[ray_count] = tp
            ray_power[ray_count] = power
            ray_width[ray_count] = self.config.boundary_spacing_m
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
    ray_count: np.ndarray,
    sigma: float,
    nearby_indices: np.ndarray,
):
    """
    Deposit energy density to nearby mesh points using directional Gaussian kernel.

    Only deposits in the ray's forward hemisphere (cos(angle) > 0).
    This prevents energy leakage into shielded areas like crevices.

    Args:
        ray_x, ray_y: Ray position
        ray_dx, ray_dy: Ray direction (unit vector)
        E_local: Local energy density P/(Cg×W) (J/m²)
        area: Tube area swept per step W × ds (m²)
        mesh_x, mesh_y: Mesh point coordinates
        energy_grid: Energy density accumulator (modified in place)
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
        energy_grid[idx] += E_local * area * normalized_weight
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
    ray_count: np.ndarray,
    # Config
    step_size: float,
    max_steps: int,
    kernel_sigma: float,
    min_width: float,
    breaker_index: float,
) -> Tuple[int, bool, float]:
    """
    Propagate a single ray and deposit energy density at each step.

    Uses power-based model where P = E × Cg × W is conserved.
    Local energy density E = P / (Cg × W) increases with focusing (smaller W).

    Returns:
        Tuple of (steps_taken, broke, final_depth)
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

    for step in range(max_steps):
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
            return step, False, final_depth

        final_depth = h

        # 2. Compute local wave properties
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

        # 3. Compute local energy density from power conservation
        # P = E × Cg × W is conserved, so E = P / (Cg × W)
        # Focusing (W smaller) → higher E; Spreading (W larger) → lower E
        E_local = power / (Cg * width)

        # 4. Check breaking using local energy density
        H = np.sqrt(8.0 * E_local / (1025.0 * 9.81))
        H_break = breaker_index * h
        if H > H_break:
            broke = True
            return step, True, final_depth

        # 5. Ray direction components
        dx = np.cos(theta)
        dy = np.sin(theta)

        # 6. Compute tube area for this step
        area = width * step_size

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
                energy_grid, ray_count,
                kernel_sigma, nearby,
            )

        # 8. Compute refraction (celerity gradients)
        dC_dx, dC_dy = celerity_gradient_indexed(
            x, y, period, L0,
            points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles,
        )

        # 9. Update direction (forward refraction - no negation needed)
        if C > 0:
            dC_dn = -dy * dC_dx + dx * dC_dy
            dtheta = -(step_size / C) * dC_dn
            theta += dtheta

            # 10. Update tube width: dW/ds = W × dθ/ds
            width += width * dtheta
            width = max(width, min_width)

        # 11. Move ray forward
        x += dx * step_size
        y += dy * step_size

    return max_steps, broke, final_depth


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
    # Config
    step_size: float,
    max_steps: int,
    kernel_sigma: float,
    min_width: float,
    breaker_index: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagate all rays in parallel and deposit energy.

    NOTE: This function creates a separate energy grid per thread to avoid
    race conditions, then combines them at the end.

    Returns:
        Tuple of (energy_grid, ray_counts, steps_taken, broke_flags)
    """
    n_rays = len(rays_x)
    n_points = len(mesh_x)

    # Output arrays
    steps_taken = np.empty(n_rays, dtype=np.int32)
    broke_flags = np.empty(n_rays, dtype=np.bool_)

    # NOTE: For true parallel execution, each thread needs its own energy grid.
    # This is a simplified version that uses sequential execution for energy deposition.
    # For production, consider using atomic operations or thread-local storage.

    energy_grid = np.zeros(n_points, dtype=np.float64)
    ray_counts = np.zeros(n_points, dtype=np.int32)

    # Sequential for correctness (parallel deposition needs atomic ops)
    for ray_idx in range(n_rays):
        steps, broke, _ = propagate_single_ray(
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
            energy_grid, ray_counts,
            step_size, max_steps, kernel_sigma, min_width, breaker_index,
        )
        steps_taken[ray_idx] = steps
        broke_flags[ray_idx] = broke

    return energy_grid, ray_counts, steps_taken, broke_flags


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
    """

    def __init__(
        self,
        mesh: 'SurfZoneMesh',
        boundary_conditions: 'BoundaryConditions',
        config: Optional[ForwardTracerConfig] = None,
    ):
        self.mesh = mesh
        self.boundary = boundary_conditions
        self.config = config or ForwardTracerConfig()

        # Initialize ray generator
        self.initializer = BoundaryRayInitializer(mesh, boundary_conditions, self.config)

        # Get mesh arrays for Numba
        self._mesh_arrays = mesh.get_numba_arrays()

        # Build point spatial index for O(k) lookups (instead of O(N))
        self._point_grid_starts, self._point_grid_counts, self._point_grid_indices = \
            self._build_point_spatial_index()

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

    def trace_partition(self, partition_idx: int, batch_size: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trace rays for a single partition and accumulate energy.

        Args:
            partition_idx: Index of partition to trace
            batch_size: Number of rays to process per batch (for progress updates)

        Returns:
            Tuple of (energy_grid, ray_counts) arrays
        """
        import time
        import sys

        # Create rays for this partition
        rays = self.initializer.create_rays_for_partition(partition_idx)

        if rays.n_rays == 0:
            n_points = len(self._mesh_arrays['points_x'])
            return np.zeros(n_points), np.zeros(n_points, dtype=np.int32)

        print(f"  Partition {partition_idx}: {rays.n_rays:,} rays")

        # Get mesh arrays
        ma = self._mesh_arrays
        n_points = len(ma['points_x'])

        # Initialize accumulators
        energy_grid = np.zeros(n_points, dtype=np.float64)
        ray_counts = np.zeros(n_points, dtype=np.int32)
        all_steps = []
        all_broke = []

        # Process rays in batches for progress updates
        n_rays = rays.n_rays
        n_batches = (n_rays + batch_size - 1) // batch_size
        start_time = time.time()

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_rays)

            # Get batch slice
            batch_x = rays.x[batch_start:batch_end]
            batch_y = rays.y[batch_start:batch_end]
            batch_theta = rays.theta[batch_start:batch_end]
            batch_power = rays.power[batch_start:batch_end]
            batch_width = rays.width[batch_start:batch_end]
            batch_period = rays.period[batch_start:batch_end]

            # Propagate batch
            batch_energy, batch_counts, batch_steps, batch_broke = propagate_all_rays(
                batch_x, batch_y, batch_theta,
                batch_power, batch_width, batch_period,
                ma['points_x'], ma['points_y'],
                ma['points_x'], ma['points_y'],
                ma['depth'], ma['triangles'],
                float(ma['grid_x_min']), float(ma['grid_y_min']),
                float(ma['grid_cell_size']),
                int(ma['grid_n_cells_x']), int(ma['grid_n_cells_y']),
                ma['grid_cell_starts'], ma['grid_cell_counts'], ma['grid_triangles'],
                self._point_grid_starts, self._point_grid_counts, self._point_grid_indices,
                self.config.step_size_m,
                self.config.max_steps,
                self.config.kernel_sigma_m,
                self.config.min_tube_width_m,
                self.config.breaker_index,
            )

            # Accumulate results
            energy_grid += batch_energy
            ray_counts += batch_counts
            all_steps.extend(batch_steps)
            all_broke.extend(batch_broke)

            # Progress update
            elapsed = time.time() - start_time
            rays_done = batch_end
            rays_per_sec = rays_done / elapsed if elapsed > 0 else 0
            remaining = (n_rays - rays_done) / rays_per_sec if rays_per_sec > 0 else 0
            pct = 100 * rays_done / n_rays
            print(f"    Progress: {rays_done:,}/{n_rays:,} rays ({pct:.1f}%) - {rays_per_sec:.0f} rays/s - ETA: {remaining:.1f}s", end='\r')
            sys.stdout.flush()

        print()  # New line after progress

        # Report statistics
        n_broke = sum(all_broke)
        avg_steps = np.mean(all_steps) if all_steps else 0
        print(f"    Avg steps: {avg_steps:.1f}, Broke: {n_broke:,} ({100*n_broke/n_rays:.1f}%)")

        return energy_grid, ray_counts

    def trace_all_partitions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Trace rays for all partitions and combine energy.

        Returns:
            Tuple of (total_energy, total_ray_counts) arrays
        """
        n_points = len(self._mesh_arrays['points_x'])
        total_energy = np.zeros(n_points)
        total_ray_counts = np.zeros(n_points, dtype=np.int32)

        for i in range(self.boundary.n_partitions):
            energy, counts = self.trace_partition(i)
            total_energy += energy
            total_ray_counts += counts

        return total_energy, total_ray_counts

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

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run complete forward ray tracing simulation.

        Returns:
            Tuple of (Hs, energy, ray_counts)
        """
        print(f"Forward ray tracing: {self.boundary.n_partitions} partitions")

        energy, ray_counts = self.trace_all_partitions()
        Hs = self.energy_to_wave_height(energy)

        # Report coverage
        n_covered = np.sum(ray_counts > 0)
        n_total = len(ray_counts)
        print(f"Coverage: {n_covered:,} / {n_total:,} points ({100*n_covered/n_total:.1f}%)")

        return Hs, energy, ray_counts
