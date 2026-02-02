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
    """
    # Metadata
    region_name: str
    timestamp: str           # ISO format
    depth_range: tuple       # (min_depth, max_depth) filter used
    partition_id: int        # 1 = primary swell

    # Counts
    n_points: int
    n_converged: int

    # Individual point results
    point_results: List[SurfzonePointResult]

    # Quick-access arrays (for visualization/analysis)
    # All arrays have shape (n_points,)
    mesh_x: np.ndarray
    mesh_y: np.ndarray
    mesh_depth: np.ndarray
    H_at_mesh: np.ndarray    # Wave heights at all mesh points
    converged: np.ndarray    # Boolean convergence flags
    direction_at_mesh: np.ndarray  # Wave directions at mesh points
    K_shoaling: np.ndarray   # Shoaling coefficients

    # Boundary arrays (corresponding to each mesh point)
    boundary_Hs: np.ndarray
    boundary_Tp: np.ndarray
    boundary_direction: np.ndarray

    @property
    def convergence_rate(self) -> float:
        """Fraction of points that converged."""
        return self.n_converged / self.n_points if self.n_points > 0 else 0.0

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
            f"  Points: {self.n_points} total, {self.n_converged} converged ({100*self.convergence_rate:.1f}%)",
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


def create_simulation_result(
    region_name: str,
    depth_range: tuple,
    partition_id: int,
    point_results: List[SurfzonePointResult],
) -> SurfzoneSimulationResult:
    """
    Create a SurfzoneSimulationResult from a list of point results.

    Extracts arrays from point results for efficient access.

    Args:
        region_name: Name of the region (e.g., "Southern California")
        depth_range: (min_depth, max_depth) tuple
        partition_id: Partition ID (1 = primary swell)
        point_results: List of SurfzonePointResult objects

    Returns:
        SurfzoneSimulationResult with arrays extracted
    """
    n_points = len(point_results)

    if n_points == 0:
        return SurfzoneSimulationResult(
            region_name=region_name,
            timestamp=datetime.now().isoformat(),
            depth_range=depth_range,
            partition_id=partition_id,
            n_points=0,
            n_converged=0,
            point_results=[],
            mesh_x=np.array([]),
            mesh_y=np.array([]),
            mesh_depth=np.array([]),
            H_at_mesh=np.array([]),
            converged=np.array([], dtype=bool),
            direction_at_mesh=np.array([]),
            K_shoaling=np.array([]),
            boundary_Hs=np.array([]),
            boundary_Tp=np.array([]),
            boundary_direction=np.array([]),
        )

    # Extract arrays from point results
    mesh_x = np.array([r.mesh_x for r in point_results])
    mesh_y = np.array([r.mesh_y for r in point_results])
    mesh_depth = np.array([r.mesh_depth for r in point_results])
    H_at_mesh = np.array([r.H_at_mesh for r in point_results])
    converged = np.array([r.converged for r in point_results], dtype=bool)
    direction_at_mesh = np.array([r.direction_at_mesh for r in point_results])
    K_shoaling = np.array([r.K_shoaling for r in point_results])
    boundary_Hs = np.array([r.boundary_Hs for r in point_results])
    boundary_Tp = np.array([r.boundary_Tp for r in point_results])
    boundary_direction = np.array([r.boundary_direction for r in point_results])

    n_converged = int(np.sum(converged))

    return SurfzoneSimulationResult(
        region_name=region_name,
        timestamp=datetime.now().isoformat(),
        depth_range=depth_range,
        partition_id=partition_id,
        n_points=n_points,
        n_converged=n_converged,
        point_results=point_results,
        mesh_x=mesh_x,
        mesh_y=mesh_y,
        mesh_depth=mesh_depth,
        H_at_mesh=H_at_mesh,
        converged=converged,
        direction_at_mesh=direction_at_mesh,
        K_shoaling=K_shoaling,
        boundary_Hs=boundary_Hs,
        boundary_Tp=boundary_Tp,
        boundary_direction=boundary_direction,
    )
