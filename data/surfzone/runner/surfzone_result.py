"""
Surfzone Simulation Result Data Structures

Dataclass for storing forward ray tracing results.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np


@dataclass
class ForwardTracingResult:
    """
    Result from forward ray tracing with energy deposition.

    Forward tracing accumulates energy from all rays (all partitions,
    all directions) at each mesh point.
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
