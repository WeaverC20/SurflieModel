"""
Surfzone Simulation Runner

Orchestrates the complete surfzone simulation:
1. Load mesh and SWAN boundary data
2. Filter mesh points by depth range
3. For each point:
   - Backward ray trace to boundary with convergence
   - Forward propagate wave height along path
4. Aggregate and save results

Focus: Primary swell only (partition_id=1), 0-10m depth range.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .backward_ray_tracer import (
    BackwardRayTracer,
    BoundaryPartition,
    BoundaryDirectionLookup,
)
from .forward_propagation import forward_propagate_from_backward_path
from .surfzone_result import (
    SurfzonePointResult,
    SurfzoneSimulationResult,
    create_simulation_result,
)

logger = logging.getLogger(__name__)


@dataclass
class SurfzoneRunnerConfig:
    """Configuration for surfzone simulation."""

    # Depth filtering
    min_depth: float = 0.0   # Minimum depth (m)
    max_depth: float = 10.0  # Maximum depth (m)

    # Partition selection
    partition_id: int = 1    # 1 = primary swell (default)

    # Ray tracing parameters
    boundary_depth_threshold: float = 50.0  # Depth threshold for boundary (m)
    step_size: float = 15.0   # Ray step size (m)
    max_steps: int = 3000     # Maximum steps per ray
    max_iterations: int = 15  # Maximum convergence iterations
    convergence_tolerance: float = 0.10  # Convergence tolerance (fraction of spread)
    alpha: float = 0.6        # Gradient descent relaxation factor

    # Default directional spread (degrees) if not available from SWAN
    default_directional_spread: float = 30.0


class SurfzoneRunner:
    """
    Main runner for surfzone wave simulation.

    Coordinates backward ray tracing and forward wave propagation
    for all mesh points in the specified depth range.

    Example usage:
        from data.surfzone.mesh import SurfZoneMesh
        from data.surfzone.runner.swan_input_provider import SwanInputProvider

        mesh = SurfZoneMesh.load('path/to/mesh')
        swan = SwanInputProvider('path/to/swan_run')
        boundary_conditions = swan.get_boundary_from_mesh(mesh)

        config = SurfzoneRunnerConfig(min_depth=0.0, max_depth=10.0)
        runner = SurfzoneRunner(mesh, boundary_conditions, config)
        result = runner.run()
    """

    def __init__(
        self,
        mesh: 'SurfZoneMesh',
        boundary_conditions: 'BoundaryConditions',
        config: Optional[SurfzoneRunnerConfig] = None,
    ):
        """
        Initialize surfzone runner.

        Args:
            mesh: SurfZoneMesh with bathymetry and spatial index
            boundary_conditions: BoundaryConditions from SwanInputProvider
            config: Runner configuration (uses defaults if None)
        """
        self.mesh = mesh
        self.boundary_conditions = boundary_conditions
        self.config = config or SurfzoneRunnerConfig()

        # Initialize backward ray tracer
        self.tracer = BackwardRayTracer(
            mesh,
            boundary_depth_threshold=self.config.boundary_depth_threshold,
            step_size=self.config.step_size,
            max_steps=self.config.max_steps,
            max_iterations=self.config.max_iterations,
            convergence_tolerance=self.config.convergence_tolerance,
            alpha=self.config.alpha,
        )

        # Build boundary lookup for fast queries
        self.boundary_lookup = BoundaryDirectionLookup(boundary_conditions)

        # Get mesh arrays for forward propagation
        self.mesh_arrays = self.tracer.get_mesh_arrays()

        logger.info(
            f"SurfzoneRunner initialized: "
            f"depth_range=[{self.config.min_depth}, {self.config.max_depth}]m, "
            f"partition_id={self.config.partition_id}"
        )

    def get_filtered_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mesh points filtered to configured depth range.

        Returns:
            Tuple of (x, y, depths, indices):
            - x: UTM x coordinates
            - y: UTM y coordinates
            - depths: Water depths (m, positive)
            - indices: Original mesh indices
        """
        arrays = self.mesh.get_numba_arrays()
        depths = arrays['depth']  # Already positive for underwater

        mask = (depths >= self.config.min_depth) & (depths <= self.config.max_depth)

        x = arrays['points_x'][mask]
        y = arrays['points_y'][mask]
        d = depths[mask]
        indices = np.where(mask)[0]

        logger.info(
            f"Filtered to {len(x)} points in depth range "
            f"[{self.config.min_depth}, {self.config.max_depth}]m"
        )

        return x, y, d, indices

    def _get_primary_swell_partition(
        self,
        mesh_x: float,
        mesh_y: float,
    ) -> Optional[BoundaryPartition]:
        """
        Get primary swell partition for a mesh point.

        Uses KD-tree to find nearest boundary point and extracts
        the primary swell (partition_id=1) data.

        Args:
            mesh_x, mesh_y: Mesh point coordinates

        Returns:
            BoundaryPartition for primary swell, or None if not valid
        """
        partition_idx = self.config.partition_id

        if partition_idx >= self.boundary_lookup.n_partitions:
            return None

        # Query nearest boundary point
        _, boundary_idx = self.boundary_lookup.kdtree.query([mesh_x, mesh_y])

        # Get partition data at that boundary point
        direction = self.boundary_lookup.partition_directions[partition_idx][boundary_idx]
        hs = self.boundary_lookup.partition_hs[partition_idx][boundary_idx]
        tp = self.boundary_lookup.partition_tp[partition_idx][boundary_idx]
        is_valid = self.boundary_lookup.partition_valid[partition_idx][boundary_idx]

        if not is_valid or np.isnan(hs) or hs <= 0:
            return None

        # Get boundary location for result
        boundary_x = self.boundary_conditions.x[boundary_idx]
        boundary_y = self.boundary_conditions.y[boundary_idx]

        return BoundaryPartition(
            partition_id=partition_idx,
            Hs=hs,
            Tp=tp,
            direction=direction,
            directional_spread=self.config.default_directional_spread,
        ), boundary_x, boundary_y

    def run_single_point(
        self,
        mesh_x: float,
        mesh_y: float,
        mesh_depth: float,
    ) -> SurfzonePointResult:
        """
        Run simulation for a single mesh point.

        1. Get primary swell partition at nearest boundary point
        2. Backward trace with path storage
        3. If converged, forward propagate wave height along path
        4. Return SurfzonePointResult

        Args:
            mesh_x, mesh_y: Mesh point coordinates (UTM)
            mesh_depth: Water depth at mesh point (m)

        Returns:
            SurfzonePointResult with wave height and other properties
        """
        # Get primary swell partition
        result = self._get_primary_swell_partition(mesh_x, mesh_y)

        if result is None:
            # No valid primary swell at this location
            return SurfzonePointResult(
                mesh_x=mesh_x,
                mesh_y=mesh_y,
                mesh_depth=mesh_depth,
                boundary_x=np.nan,
                boundary_y=np.nan,
                boundary_Hs=np.nan,
                boundary_Tp=np.nan,
                boundary_direction=np.nan,
                converged=False,
                n_iterations=0,
                direction_at_mesh=np.nan,
                H_at_mesh=np.nan,
                K_shoaling=np.nan,
            )

        partition, boundary_x, boundary_y = result

        # Backward trace with path storage
        contribution = self.tracer.trace_single_partition(
            mesh_x, mesh_y, partition, store_path=True
        )

        if not contribution.converged or contribution.path_x is None:
            # Did not converge or no path stored
            return SurfzonePointResult(
                mesh_x=mesh_x,
                mesh_y=mesh_y,
                mesh_depth=mesh_depth,
                boundary_x=boundary_x,
                boundary_y=boundary_y,
                boundary_Hs=partition.Hs,
                boundary_Tp=partition.Tp,
                boundary_direction=partition.direction,
                converged=False,
                n_iterations=contribution.n_iterations,
                direction_at_mesh=contribution.direction if not np.isnan(contribution.direction) else np.nan,
                H_at_mesh=np.nan,
                K_shoaling=np.nan,
            )

        # Forward propagate wave height along path
        H_at_mesh, K_shoaling, _, _ = forward_propagate_from_backward_path(
            contribution.path_x,
            contribution.path_y,
            partition.Hs,
            partition.Tp,
            self.mesh_arrays,
        )

        return SurfzonePointResult(
            mesh_x=mesh_x,
            mesh_y=mesh_y,
            mesh_depth=mesh_depth,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
            boundary_Hs=partition.Hs,
            boundary_Tp=partition.Tp,
            boundary_direction=partition.direction,
            converged=True,
            n_iterations=contribution.n_iterations,
            direction_at_mesh=contribution.direction,
            H_at_mesh=H_at_mesh,
            K_shoaling=K_shoaling,
        )

    def run(self, region_name: str = "Southern California") -> SurfzoneSimulationResult:
        """
        Run simulation for all filtered mesh points.

        Args:
            region_name: Name of the region for metadata

        Returns:
            SurfzoneSimulationResult with all point results
        """
        # Get filtered points
        x, y, depths, indices = self.get_filtered_points()
        n_points = len(x)

        if n_points == 0:
            logger.warning("No mesh points in specified depth range")
            return create_simulation_result(
                region_name=region_name,
                depth_range=(self.config.min_depth, self.config.max_depth),
                partition_id=self.config.partition_id,
                point_results=[],
            )

        logger.info(f"Running simulation for {n_points} points...")
        t_start = time.perf_counter()

        # Process each point
        point_results = []
        n_converged = 0

        for i, (mx, my, md) in enumerate(zip(x, y, depths)):
            result = self.run_single_point(mx, my, md)
            point_results.append(result)

            if result.converged:
                n_converged += 1

            # Progress logging
            if (i + 1) % 500 == 0 or i == n_points - 1:
                elapsed = time.perf_counter() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  Progress: {i+1}/{n_points} ({100*(i+1)/n_points:.1f}%), "
                    f"converged: {n_converged}, rate: {rate:.1f} pts/sec"
                )

        # Create result
        elapsed = time.perf_counter() - t_start
        result = create_simulation_result(
            region_name=region_name,
            depth_range=(self.config.min_depth, self.config.max_depth),
            partition_id=self.config.partition_id,
            point_results=point_results,
        )

        logger.info(
            f"Simulation complete: {n_points} points in {elapsed:.1f}s "
            f"({n_points/elapsed:.1f} pts/sec), "
            f"{n_converged} converged ({100*n_converged/n_points:.1f}%)"
        )

        return result


def run_surfzone_simulation(
    mesh_path: str,
    swan_run_dir: str,
    min_depth: float = 0.0,
    max_depth: float = 10.0,
    region_name: str = "Southern California",
) -> SurfzoneSimulationResult:
    """
    Convenience function to run a complete surfzone simulation.

    Args:
        mesh_path: Path to mesh file (or directory)
        swan_run_dir: Path to SWAN run directory
        min_depth: Minimum depth filter (m)
        max_depth: Maximum depth filter (m)
        region_name: Name of region for metadata

    Returns:
        SurfzoneSimulationResult
    """
    from data.surfzone.mesh import SurfZoneMesh
    from data.surfzone.runner.swan_input_provider import SwanInputProvider

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
    config = SurfzoneRunnerConfig(
        min_depth=min_depth,
        max_depth=max_depth,
        partition_id=1,  # Primary swell
    )

    # Run simulation
    runner = SurfzoneRunner(mesh, boundary_conditions, config)
    result = runner.run(region_name=region_name)

    return result
