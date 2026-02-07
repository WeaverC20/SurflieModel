"""
Surfzone Simulation Result Data Structures

Dataclasses for storing wave propagation results from surfzone simulation.
Focus: Primary swell only, forward-propagated wave heights at mesh points.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np


@dataclass
class SurfzonePointResult:
    """
    Result for a single mesh point - primary swell only.

    Stores the forward-propagated wave height from SWAN boundary to mesh point.
    """
    # Mesh point location (UTM)
    mesh_x: float
    mesh_y: float
    mesh_depth: float  # Water depth at mesh point (m, positive)

    # Boundary conditions (from SWAN primary swell)
    boundary_x: float
    boundary_y: float
    boundary_Hs: float       # Significant wave height at boundary (m)
    boundary_Tp: float       # Peak period (s)
    boundary_direction: float  # Direction at boundary (nautical degrees)

    # Convergence info from backward ray tracing
    converged: bool
    n_iterations: int
    direction_at_mesh: float  # Wave direction at mesh point (nautical degrees)

    # Forward-propagated wave properties at mesh point
    H_at_mesh: float         # Wave height after shoaling (m)
    K_shoaling: float        # Total shoaling coefficient (H_mesh / H_boundary)


@dataclass
class SurfzoneSimulationResult:
    """
    Complete simulation results for all mesh points.

    Contains individual point results and quick-access numpy arrays
    for efficient analysis and visualization.

    When sampling is used, all filtered points are included but only
    sampled ones have computed values (non-sampled have NaN).
    """
    # Metadata
    region_name: str
    timestamp: str           # ISO format
    depth_range: tuple       # (min_depth, max_depth) filter used
    partition_id: int        # 1 = primary swell

    # Counts
    n_points: int            # Total points in depth range
    n_sampled: int           # Points actually processed
    n_converged: int         # Points that converged

    # Individual point results (only for sampled points)
    point_results: List[SurfzonePointResult]

    # Quick-access arrays (for visualization/analysis)
    # All arrays have shape (n_points,) - includes ALL filtered points
    mesh_x: np.ndarray
    mesh_y: np.ndarray
    mesh_depth: np.ndarray
    sampled: np.ndarray      # Boolean: True if point was processed
    H_at_mesh: np.ndarray    # Wave heights (NaN for non-sampled)
    converged: np.ndarray    # Boolean convergence flags (False for non-sampled)
    direction_at_mesh: np.ndarray  # Wave directions (NaN for non-sampled)
    K_shoaling: np.ndarray   # Shoaling coefficients (NaN for non-sampled)

    # Boundary arrays (corresponding to each mesh point)
    boundary_Hs: np.ndarray
    boundary_Tp: np.ndarray
    boundary_direction: np.ndarray

    @property
    def convergence_rate(self) -> float:
        """Fraction of sampled points that converged."""
        return self.n_converged / self.n_sampled if self.n_sampled > 0 else 0.0

    @property
    def sample_rate(self) -> float:
        """Fraction of points that were sampled."""
        return self.n_sampled / self.n_points if self.n_points > 0 else 0.0

    @property
    def mean_H_converged(self) -> float:
        """Mean wave height at converged points."""
        if self.n_converged == 0:
            return np.nan
        return float(np.mean(self.H_at_mesh[self.converged]))

    @property
    def mean_K_shoaling(self) -> float:
        """Mean shoaling coefficient at converged points."""
        if self.n_converged == 0:
            return np.nan
        return float(np.mean(self.K_shoaling[self.converged]))

    def summary(self) -> str:
        """Return summary string of results."""
        lines = [
            f"SurfzoneSimulationResult: {self.region_name}",
            f"  Timestamp: {self.timestamp}",
            f"  Depth range: {self.depth_range[0]:.1f} - {self.depth_range[1]:.1f} m",
            f"  Points: {self.n_points} total, {self.n_sampled} sampled ({100*self.sample_rate:.1f}%)",
            f"  Converged: {self.n_converged} ({100*self.convergence_rate:.1f}% of sampled)",
        ]

        if self.n_converged > 0:
            H_conv = self.H_at_mesh[self.converged]
            K_conv = self.K_shoaling[self.converged]
            lines.extend([
                f"  Wave height at mesh: {H_conv.min():.2f} - {H_conv.max():.2f} m "
                f"(mean: {H_conv.mean():.2f} m)",
                f"  Shoaling coefficient: {K_conv.min():.2f} - {K_conv.max():.2f} "
                f"(mean: {K_conv.mean():.2f})",
            ])

        return '\n'.join(lines)


@dataclass
class ForwardTracingResult:
    """
    Result from forward ray tracing with energy deposition.

    Unlike backward tracing, forward tracing accumulates energy from all
    rays (all partitions, all directions) at each mesh point.
    """
    # Metadata
    region_name: str
    timestamp: str
    n_partitions: int

    # Statistics
    n_points: int           # Total mesh points
    n_covered: int          # Points hit by at least one ray
    n_rays_total: int       # Total rays traced

    # Quick-access arrays (shape: n_points)
    mesh_x: np.ndarray
    mesh_y: np.ndarray
    mesh_depth: np.ndarray
    H_at_mesh: np.ndarray      # Significant wave height (m)
    energy: np.ndarray         # Accumulated energy (J/m)
    ray_count: np.ndarray      # Number of rays hitting each point

    @property
    def coverage_rate(self) -> float:
        """Fraction of points covered by at least one ray."""
        return self.n_covered / self.n_points if self.n_points > 0 else 0.0

    @property
    def mean_H(self) -> float:
        """Mean wave height at covered points."""
        if self.n_covered == 0:
            return np.nan
        covered = self.ray_count > 0
        return float(np.mean(self.H_at_mesh[covered]))

    @property
    def max_H(self) -> float:
        """Maximum wave height."""
        return float(np.nanmax(self.H_at_mesh))

    def summary(self) -> str:
        """Return summary string of results."""
        lines = [
            f"ForwardTracingResult: {self.region_name}",
            f"  Timestamp: {self.timestamp}",
            f"  Partitions: {self.n_partitions}",
            f"  Points: {self.n_points:,} total, {self.n_covered:,} covered ({100*self.coverage_rate:.1f}%)",
            f"  Rays traced: {self.n_rays_total:,}",
        ]

        if self.n_covered > 0:
            covered = self.ray_count > 0
            H_cov = self.H_at_mesh[covered]
            lines.extend([
                f"  Wave height: {H_cov.min():.2f} - {H_cov.max():.2f} m (mean: {H_cov.mean():.2f} m)",
                f"  Rays per point: {self.ray_count[covered].min()} - {self.ray_count[covered].max()} "
                f"(mean: {self.ray_count[covered].mean():.1f})",
            ])

        return '\n'.join(lines)


def create_simulation_result(
    region_name: str,
    depth_range: tuple,
    partition_id: int,
    point_results: List,
    all_x: Optional[np.ndarray] = None,
    all_y: Optional[np.ndarray] = None,
    all_depths: Optional[np.ndarray] = None,
    sampled_mask: Optional[np.ndarray] = None,
) -> SurfzoneSimulationResult:
    """
    Create a SurfzoneSimulationResult from point results.

    When sampling is used, point_results contains (index, result) tuples,
    and all_x/all_y/all_depths/sampled_mask provide info for all filtered points.

    Args:
        region_name: Name of the region (e.g., "Southern California")
        depth_range: (min_depth, max_depth) tuple
        partition_id: Partition ID (1 = primary swell)
        point_results: List of SurfzonePointResult objects, or (index, result) tuples
        all_x: X coordinates of all filtered points (optional, for sampling)
        all_y: Y coordinates of all filtered points (optional, for sampling)
        all_depths: Depths of all filtered points (optional, for sampling)
        sampled_mask: Boolean mask indicating which points were sampled (optional)

    Returns:
        SurfzoneSimulationResult with arrays extracted
    """
    # Handle empty case
    if len(point_results) == 0 and (all_x is None or len(all_x) == 0):
        return SurfzoneSimulationResult(
            region_name=region_name,
            timestamp=datetime.now().isoformat(),
            depth_range=depth_range,
            partition_id=partition_id,
            n_points=0,
            n_sampled=0,
            n_converged=0,
            point_results=[],
            mesh_x=np.array([]),
            mesh_y=np.array([]),
            mesh_depth=np.array([]),
            sampled=np.array([], dtype=bool),
            H_at_mesh=np.array([]),
            converged=np.array([], dtype=bool),
            direction_at_mesh=np.array([]),
            K_shoaling=np.array([]),
            boundary_Hs=np.array([]),
            boundary_Tp=np.array([]),
            boundary_direction=np.array([]),
        )

    # Check if point_results contains (index, result) tuples (sampling mode)
    # or just SurfzonePointResult objects (legacy mode)
    if len(point_results) > 0 and isinstance(point_results[0], tuple):
        # Sampling mode: point_results is [(index, result), ...]
        indexed_results = point_results
        results_only = [r for _, r in indexed_results]
    else:
        # Legacy mode: point_results is [result, ...] with no sampling
        results_only = point_results
        indexed_results = [(i, r) for i, r in enumerate(point_results)]

    # Determine total points and sampled mask
    if all_x is not None:
        n_points = len(all_x)
        mesh_x = all_x.copy()
        mesh_y = all_y.copy()
        mesh_depth = all_depths.copy()
        sampled = sampled_mask.copy() if sampled_mask is not None else np.ones(n_points, dtype=bool)
    else:
        # Legacy mode: all points were sampled
        n_points = len(results_only)
        mesh_x = np.array([r.mesh_x for r in results_only])
        mesh_y = np.array([r.mesh_y for r in results_only])
        mesh_depth = np.array([r.mesh_depth for r in results_only])
        sampled = np.ones(n_points, dtype=bool)

    # Initialize arrays with NaN (for non-sampled points)
    H_at_mesh = np.full(n_points, np.nan)
    converged = np.zeros(n_points, dtype=bool)
    direction_at_mesh = np.full(n_points, np.nan)
    K_shoaling = np.full(n_points, np.nan)
    boundary_Hs = np.full(n_points, np.nan)
    boundary_Tp = np.full(n_points, np.nan)
    boundary_direction = np.full(n_points, np.nan)

    # Fill in values from point results at their indices
    for idx, result in indexed_results:
        H_at_mesh[idx] = result.H_at_mesh
        converged[idx] = result.converged
        direction_at_mesh[idx] = result.direction_at_mesh
        K_shoaling[idx] = result.K_shoaling
        boundary_Hs[idx] = result.boundary_Hs
        boundary_Tp[idx] = result.boundary_Tp
        boundary_direction[idx] = result.boundary_direction

    n_sampled = int(np.sum(sampled))
    n_converged = int(np.sum(converged))

    return SurfzoneSimulationResult(
        region_name=region_name,
        timestamp=datetime.now().isoformat(),
        depth_range=depth_range,
        partition_id=partition_id,
        n_points=n_points,
        n_sampled=n_sampled,
        n_converged=n_converged,
        point_results=results_only,
        mesh_x=mesh_x,
        mesh_y=mesh_y,
        mesh_depth=mesh_depth,
        sampled=sampled,
        H_at_mesh=H_at_mesh,
        converged=converged,
        direction_at_mesh=direction_at_mesh,
        K_shoaling=K_shoaling,
        boundary_Hs=boundary_Hs,
        boundary_Tp=boundary_Tp,
        boundary_direction=boundary_direction,
    )
