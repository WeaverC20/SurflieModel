"""
Surfzone Simulation Result Data Structures

Dataclass for storing forward ray tracing results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np


# Partition names for output files
PARTITION_NAMES = ['wind_sea', 'primary_swell', 'secondary_swell', 'tertiary_swell']


@dataclass
class RayPathData:
    """
    Ray path data from forward ray tracing.

    Contains sampled ray trajectories for visualization and analysis.
    Paths are stored in concatenated arrays with per-ray indexing.
    """
    # Per-ray metadata (n_sampled_rays,)
    ray_partition: np.ndarray      # Which partition (0-3), int32
    ray_start_idx: np.ndarray      # Index into path arrays where this ray starts, int64
    ray_length: np.ndarray         # Number of steps for this ray, int32
    ray_original_idx: np.ndarray   # Original ray index (for reference), int32

    # Path data (total_steps,) - all sampled rays concatenated
    path_x: np.ndarray             # UTM X position, float32
    path_y: np.ndarray             # UTM Y position, float32
    path_depth: np.ndarray         # Water depth at position, float32
    path_direction: np.ndarray     # Ray direction (radians, math convention), float32
    path_tube_width: np.ndarray    # Tube width (m), float32
    path_Hs_local: np.ndarray      # Local wave height (m), float32

    # Metadata
    n_rays_total: int              # Total rays in simulation
    sample_fraction: float         # Fraction of rays stored (e.g., 0.1 = 10%)

    @property
    def n_rays_sampled(self) -> int:
        return len(self.ray_partition)

    @property
    def total_steps(self) -> int:
        return len(self.path_x)

    def get_ray_path(self, ray_idx: int) -> dict:
        """Get path arrays for a specific sampled ray."""
        start = self.ray_start_idx[ray_idx]
        length = self.ray_length[ray_idx]
        end = start + length
        return {
            'x': self.path_x[start:end],
            'y': self.path_y[start:end],
            'depth': self.path_depth[start:end],
            'direction': self.path_direction[start:end],
            'tube_width': self.path_tube_width[start:end],
            'Hs_local': self.path_Hs_local[start:end],
            'partition': self.ray_partition[ray_idx],
            'original_idx': self.ray_original_idx[ray_idx],
        }

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"RayPathData: {self.n_rays_sampled:,} sampled rays "
            f"({self.sample_fraction*100:.0f}% of {self.n_rays_total:,} total), "
            f"{self.total_steps:,} total path steps"
        )


@dataclass
class PartitionResult:
    """
    Result for a single wave partition from forward ray tracing.

    Contains propagated wave properties (Hs, direction from ray tracing)
    and boundary period (Tp from SWAN, conserved during propagation).
    """
    partition_id: int
    partition_name: str  # "wind_sea", "primary_swell", etc.

    # Boundary values (conserved during propagation)
    boundary_Tp: np.ndarray       # Period from SWAN (conserved), shape: (n_points,)

    # Propagated values from ray tracing
    energy: np.ndarray            # Per-partition energy at mesh (J/m²)
    H_at_mesh: np.ndarray         # Hs derived from energy (m), propagated
    direction: np.ndarray         # Energy-weighted refracted direction (nautical degrees FROM)
    ray_count: np.ndarray         # Number of rays contributing to each point
    converged: np.ndarray         # Boolean mask: points hit by at least one ray

    @property
    def n_points(self) -> int:
        return len(self.H_at_mesh)

    @property
    def n_covered(self) -> int:
        return int(np.sum(self.converged))

    def summary(self) -> str:
        """Return summary string."""
        covered = self.converged
        n_covered = self.n_covered
        lines = [
            f"PartitionResult: {self.partition_name} (id={self.partition_id})",
            f"  Points: {self.n_points:,} total, {n_covered:,} covered",
        ]
        if n_covered > 0:
            H_cov = self.H_at_mesh[covered]
            dir_cov = self.direction[covered]
            lines.extend([
                f"  Hs: {H_cov.min():.2f} - {H_cov.max():.2f} m (mean: {H_cov.mean():.2f} m)",
                f"  Direction: {dir_cov.min():.0f}° - {dir_cov.max():.0f}° (mean: {dir_cov.mean():.0f}°)",
            ])
        return '\n'.join(lines)


@dataclass
class ForwardTracingResult:
    """
    Result from forward ray tracing with energy deposition.

    Forward tracing accumulates energy from all rays (all partitions,
    all directions) at each mesh point. Also contains per-partition
    data for wave statistics calculations.
    """
    # Metadata
    region_name: str
    timestamp: str
    n_partitions: int

    # Statistics
    n_points: int           # Total mesh points
    n_covered: int          # Points hit by at least one ray
    n_rays_total: int       # Total rays traced

    # Combined arrays (all partitions summed, shape: n_points)
    mesh_x: np.ndarray
    mesh_y: np.ndarray
    mesh_depth: np.ndarray
    H_at_mesh: np.ndarray      # Combined significant wave height (m)
    energy: np.ndarray         # Combined accumulated energy (J/m)
    ray_count: np.ndarray      # Total rays hitting each point

    # Per-partition data (for wave statistics)
    partitions: List[PartitionResult] = field(default_factory=list)

    # Optional ray path data (when track_paths=True)
    ray_paths: Optional[RayPathData] = None

    # Optional breaking characterization (when enable_breaking=True)
    # Keys: is_breaking, breaker_index, iribarren, breaker_type, breaking_intensity
    breaking_fields: Optional[dict] = None

    # Optional wind metadata (when wind-modified breaking is used)
    wind_metadata: Optional[dict] = None

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
                f"  Combined wave height: {H_cov.min():.2f} - {H_cov.max():.2f} m (mean: {H_cov.mean():.2f} m)",
                f"  Rays per point: {self.ray_count[covered].min()} - {self.ray_count[covered].max()} "
                f"(mean: {self.ray_count[covered].mean():.1f})",
            ])

        # Per-partition summaries
        if self.partitions:
            lines.append("  Per-partition:")
            for p in self.partitions:
                if p.n_covered > 0:
                    H_cov = p.H_at_mesh[p.converged]
                    lines.append(f"    {p.partition_name}: Hs {H_cov.min():.2f}-{H_cov.max():.2f}m, {p.n_covered:,} points")

        return '\n'.join(lines)
