"""
Surfzone Simulation Runner

Orchestrates the forward ray tracing surfzone simulation:
1. Load mesh and SWAN boundary data
2. Trace rays from boundary with energy deposition
3. Aggregate and save results
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
from scipy.spatial import cKDTree

from .surfzone_result import ForwardTracingResult, PartitionResult, PARTITION_NAMES

logger = logging.getLogger(__name__)


class ForwardSurfzoneRunner:
    """
    Runner for forward ray tracing with energy deposition.

    Forward tracing shoots rays from the boundary and deposits energy
    as they propagate. This naturally captures wave focusing effects.

    Example usage:
        from data.surfzone.mesh import SurfZoneMesh
        from data.surfzone.runner.swan_input_provider import SwanInputProvider

        mesh = SurfZoneMesh.load('path/to/mesh')
        swan = SwanInputProvider('path/to/swan_run')
        boundary_conditions = swan.get_boundary_from_mesh(mesh)

        runner = ForwardSurfzoneRunner(mesh, boundary_conditions)
        result = runner.run()
    """

    def __init__(
        self,
        mesh: 'SurfZoneMesh',
        boundary_conditions: 'BoundaryConditions',
        config: Optional['ForwardTracerConfig'] = None,
        track_paths: bool = False,
        sample_fraction: float = 0.1,
    ):
        """
        Initialize forward surfzone runner.

        Args:
            mesh: SurfZoneMesh with bathymetry and spatial index
            boundary_conditions: BoundaryConditions from SwanInputProvider
            config: ForwardTracerConfig (uses defaults if None)
            track_paths: If True, collect ray paths for visualization
            sample_fraction: Fraction of rays to sample for path tracking (0.0-1.0)
        """
        from .forward_ray_tracer import ForwardRayTracer, ForwardTracerConfig

        self.mesh = mesh
        self.boundary_conditions = boundary_conditions
        self.config = config or ForwardTracerConfig()
        self.track_paths = track_paths
        self.sample_fraction = sample_fraction

        # Initialize forward ray tracer with path tracking options
        self.tracer = ForwardRayTracer(
            mesh, boundary_conditions, self.config,
            track_paths=track_paths,
            sample_fraction=sample_fraction,
        )

        logger.info(
            f"ForwardSurfzoneRunner initialized: "
            f"boundary_spacing={self.config.boundary_spacing_m}m, "
            f"kernel_sigma={self.config.kernel_sigma_m}m"
            + (f", path_tracking={sample_fraction*100:.0f}%" if track_paths else "")
        )

    def run(self, region_name: str = "Southern California") -> ForwardTracingResult:
        """
        Run forward ray tracing simulation.

        Args:
            region_name: Name of the region for metadata

        Returns:
            ForwardTracingResult with wave heights, energy, per-partition data,
            and optionally ray_paths if track_paths=True
        """
        t_start = time.perf_counter()

        # Run forward tracing (now returns per-partition data and optional ray paths)
        Hs, energy, ray_counts, per_partition_data, ray_paths = self.tracer.run()

        elapsed = time.perf_counter() - t_start

        # Get mesh arrays
        arrays = self.mesh.get_numba_arrays()
        mesh_x = arrays['points_x']
        mesh_y = arrays['points_y']

        # Count rays (estimate from initializer)
        all_rays = self.tracer.initializer.create_all_rays()
        n_rays = all_rays.n_rays

        n_points = len(Hs)
        n_covered = int(np.sum(ray_counts > 0))

        logger.info(
            f"Forward simulation complete: {n_rays:,} rays in {elapsed:.1f}s, "
            f"{n_covered:,} points covered ({100*n_covered/n_points:.1f}%)"
        )

        # Build PartitionResult objects
        partitions = self._build_partition_results(
            per_partition_data, mesh_x, mesh_y, n_points
        )

        return ForwardTracingResult(
            region_name=region_name,
            timestamp=datetime.now().isoformat(),
            n_partitions=self.boundary_conditions.n_partitions,
            n_points=n_points,
            n_covered=n_covered,
            n_rays_total=n_rays,
            mesh_x=mesh_x.copy(),
            mesh_y=mesh_y.copy(),
            mesh_depth=arrays['depth'].copy(),
            H_at_mesh=Hs,
            energy=energy,
            ray_count=ray_counts,
            partitions=partitions,
            ray_paths=ray_paths,
        )

    def _build_partition_results(
        self,
        per_partition_data: dict,
        mesh_x: np.ndarray,
        mesh_y: np.ndarray,
        n_points: int,
    ) -> List[PartitionResult]:
        """
        Build PartitionResult objects from per-partition ray tracing data.

        Interpolates boundary Tp to mesh points using nearest-neighbor lookup.
        """
        # Build KD-tree for boundary point lookup
        boundary_x = self.boundary_conditions.x
        boundary_y = self.boundary_conditions.y
        boundary_tree = cKDTree(np.column_stack([boundary_x, boundary_y]))

        # Find nearest boundary point for each mesh point
        mesh_points = np.column_stack([mesh_x, mesh_y])
        _, nearest_indices = boundary_tree.query(mesh_points)

        partitions = []

        for part_idx, part_data in per_partition_data.items():
            # Get partition name
            partition_name = PARTITION_NAMES[part_idx] if part_idx < len(PARTITION_NAMES) else f"partition_{part_idx}"

            # Get SWAN partition for boundary Tp
            swan_partition = self.boundary_conditions.partitions[part_idx]

            # Interpolate boundary Tp to mesh points using nearest neighbor
            boundary_Tp = swan_partition.tp[nearest_indices]

            # Points covered by this partition
            converged = part_data['ray_counts'] > 0

            partitions.append(PartitionResult(
                partition_id=part_idx,
                partition_name=partition_name,
                boundary_Tp=boundary_Tp,
                energy=part_data['energy'],
                H_at_mesh=part_data['Hs'],
                direction=part_data['direction'],
                ray_count=part_data['ray_counts'],
                converged=converged,
            ))

        return partitions


def run_forward_surfzone_simulation(
    mesh_path: str,
    swan_run_dir: str,
    region_name: str = "Southern California",
    boundary_spacing_m: float = 50.0,
    kernel_sigma_m: float = 25.0,
    track_paths: bool = False,
    sample_fraction: float = 0.1,
) -> ForwardTracingResult:
    """
    Convenience function to run a complete forward surfzone simulation.

    Args:
        mesh_path: Path to mesh file (or directory)
        swan_run_dir: Path to SWAN run directory
        region_name: Name of region for metadata
        boundary_spacing_m: Spacing between boundary sample points
        kernel_sigma_m: Gaussian kernel width for energy deposition
        track_paths: If True, collect ray paths for visualization
        sample_fraction: Fraction of rays to sample for path tracking (0.0-1.0)

    Returns:
        ForwardTracingResult
    """
    from data.surfzone.mesh import SurfZoneMesh
    from data.surfzone.runner.swan_input_provider import SwanInputProvider
    from .forward_ray_tracer import ForwardTracerConfig

    # Load mesh
    logger.info(f"Loading mesh from: {mesh_path}")
    mesh = SurfZoneMesh.load(mesh_path)

    # Load SWAN data
    logger.info(f"Loading SWAN data from: {swan_run_dir}")
    swan = SwanInputProvider(swan_run_dir)

    # Get boundary conditions at mesh boundary points
    logger.info("Sampling SWAN data at boundary points...")
    boundary_conditions = swan.get_boundary_from_mesh(mesh)

    # Configure runner
    config = ForwardTracerConfig(
        boundary_spacing_m=boundary_spacing_m,
        kernel_sigma_m=kernel_sigma_m,
    )

    # Run simulation
    runner = ForwardSurfzoneRunner(
        mesh, boundary_conditions, config,
        track_paths=track_paths,
        sample_fraction=sample_fraction,
    )
    result = runner.run(region_name=region_name)

    return result
