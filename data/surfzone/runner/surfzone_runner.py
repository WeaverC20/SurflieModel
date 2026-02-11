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
from typing import Optional

import numpy as np

from .surfzone_result import ForwardTracingResult

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
    ):
        """
        Initialize forward surfzone runner.

        Args:
            mesh: SurfZoneMesh with bathymetry and spatial index
            boundary_conditions: BoundaryConditions from SwanInputProvider
            config: ForwardTracerConfig (uses defaults if None)
        """
        from .forward_ray_tracer import ForwardRayTracer, ForwardTracerConfig

        self.mesh = mesh
        self.boundary_conditions = boundary_conditions
        self.config = config or ForwardTracerConfig()

        # Initialize forward ray tracer
        self.tracer = ForwardRayTracer(mesh, boundary_conditions, self.config)

        logger.info(
            f"ForwardSurfzoneRunner initialized: "
            f"boundary_spacing={self.config.boundary_spacing_m}m, "
            f"kernel_sigma={self.config.kernel_sigma_m}m"
        )

    def run(self, region_name: str = "Southern California") -> ForwardTracingResult:
        """
        Run forward ray tracing simulation.

        Args:
            region_name: Name of the region for metadata

        Returns:
            ForwardTracingResult with wave heights and energy
        """
        t_start = time.perf_counter()

        # Run forward tracing
        Hs, energy, ray_counts = self.tracer.run()

        elapsed = time.perf_counter() - t_start

        # Get mesh arrays
        arrays = self.mesh.get_numba_arrays()

        # Count rays (estimate from initializer)
        all_rays = self.tracer.initializer.create_all_rays()
        n_rays = all_rays.n_rays

        n_points = len(Hs)
        n_covered = int(np.sum(ray_counts > 0))

        logger.info(
            f"Forward simulation complete: {n_rays:,} rays in {elapsed:.1f}s, "
            f"{n_covered:,} points covered ({100*n_covered/n_points:.1f}%)"
        )

        return ForwardTracingResult(
            region_name=region_name,
            timestamp=datetime.now().isoformat(),
            n_partitions=self.boundary_conditions.n_partitions,
            n_points=n_points,
            n_covered=n_covered,
            n_rays_total=n_rays,
            mesh_x=arrays['points_x'].copy(),
            mesh_y=arrays['points_y'].copy(),
            mesh_depth=arrays['depth'].copy(),
            H_at_mesh=Hs,
            energy=energy,
            ray_count=ray_counts,
        )


def run_forward_surfzone_simulation(
    mesh_path: str,
    swan_run_dir: str,
    region_name: str = "Southern California",
    boundary_spacing_m: float = 50.0,
    kernel_sigma_m: float = 25.0,
) -> ForwardTracingResult:
    """
    Convenience function to run a complete forward surfzone simulation.

    Args:
        mesh_path: Path to mesh file (or directory)
        swan_run_dir: Path to SWAN run directory
        region_name: Name of region for metadata
        boundary_spacing_m: Spacing between boundary sample points
        kernel_sigma_m: Gaussian kernel width for energy deposition

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
    runner = ForwardSurfzoneRunner(mesh, boundary_conditions, config)
    result = runner.run(region_name=region_name)

    return result
