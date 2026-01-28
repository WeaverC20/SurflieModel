#!/usr/bin/env python3
"""
Surfzone Wave Propagation Runner

Traces wave rays from SWAN output through the surfzone mesh to predict
breaking wave locations, heights, and types.

Usage:
    python data/surfzone/run_surfzone.py --region socal
    python data/surfzone/run_surfzone.py --region socal --swan-resolution fine
    python data/surfzone/run_surfzone.py --region socal --dry-run
    python data/surfzone/run_surfzone.py --region socal --fast

The runner:
1. Loads surfzone mesh
2. Reads SWAN partition outputs (wave height, period, direction)
3. Extracts wind data from GFS
4. Traces rays from offshore boundary to shore
5. Outputs breaking field (locations, heights, types)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.surfzone.mesh import SurfZoneMesh
from data.surfzone.runner.swan_input_provider import SwanInputProvider, BoundaryConditions
from data.surfzone.runner.ray_tracer import RayTracer, RayResult
from data.surfzone.runner.output_writer import OutputWriter, BreakingField

logger = logging.getLogger(__name__)

# Directory structure
DATA_DIR = PROJECT_ROOT / "data"
SURFZONE_MESHES_DIR = DATA_DIR / "surfzone" / "meshes"
SWAN_RUNS_DIR = DATA_DIR / "swan" / "runs"
SURFZONE_RUNS_DIR = DATA_DIR / "surfzone" / "runs"
WIND_DIR = DATA_DIR / "downloaded_weather_data" / "wind"


def get_surfzone_mesh_dir(region: str) -> Path:
    """Get path to surfzone mesh directory."""
    mesh_dir = SURFZONE_MESHES_DIR / region
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Surfzone mesh directory not found: {mesh_dir}")
    return mesh_dir


def get_swan_run_dir(region: str, resolution: str) -> Path:
    """Get path to SWAN run directory."""
    run_dir = SWAN_RUNS_DIR / region / resolution / "latest"
    if not run_dir.exists():
        raise FileNotFoundError(
            f"SWAN run directory not found: {run_dir}\n"
            f"Run SWAN first: python data/swan/run_swan.py --region {region} --mesh {resolution}"
        )
    return run_dir


def get_output_dir(region: str) -> Path:
    """Get path to surfzone output directory."""
    return SURFZONE_RUNS_DIR / region / "latest"


def get_wind_data(
    mesh: SurfZoneMesh,
    wind_file: Optional[Path] = None,
) -> Tuple[float, float]:
    """
    Get wind speed and direction for the surfzone region.

    Currently returns domain-averaged wind for simplicity.
    Future: interpolate wind at each ray position.

    Args:
        mesh: SurfZoneMesh for coordinate bounds
        wind_file: Specific wind file to use (default: latest)

    Returns:
        Tuple of (wind_speed, wind_direction) in m/s and degrees nautical
    """
    import xarray as xr

    # Find latest wind file
    if wind_file is None:
        wind_files = sorted(WIND_DIR.glob("gfs_*.nc"))
        if not wind_files:
            logger.warning("No wind data found, using calm conditions")
            return 0.0, 0.0
        wind_file = wind_files[-1]

    logger.info(f"Using wind file: {wind_file.name}")

    # Get mesh bounds in lon/lat
    lon_min, lon_max = mesh.lon_range
    lat_min, lat_max = mesh.lat_range

    # Load and extract region
    try:
        with xr.open_dataset(wind_file) as ds:
            # Add buffer
            buffer = 0.5
            subset = ds.sel(
                lat=slice(lat_min - buffer, lat_max + buffer),
                lon=slice(lon_min - buffer, lon_max + buffer)
            )

            # Get first time step
            if "time" in subset.dims:
                subset = subset.isel(time=0)

            u = float(subset["u_wind"].mean())
            v = float(subset["v_wind"].mean())

        # Calculate speed and direction
        wind_speed = np.sqrt(u**2 + v**2)
        # Convert to nautical (FROM direction)
        wind_direction = (270 - np.degrees(np.arctan2(v, u))) % 360

        logger.info(f"Wind: {wind_speed:.1f} m/s from {wind_direction:.0f}°")

        return wind_speed, wind_direction

    except Exception as e:
        logger.warning(f"Failed to load wind data: {e}, using calm conditions")
        return 0.0, 0.0


class SurfzoneRunner:
    """
    Orchestrates surfzone wave propagation simulations.

    Handles the complete workflow from loading SWAN output
    to tracing rays and saving results.
    """

    def __init__(
        self,
        region: str,
        swan_resolution: str = "ultrafine",
        step_size: float = 10.0,
        max_steps: int = 1000,
        min_depth: float = 0.15,
    ):
        """
        Initialize surfzone runner.

        Args:
            region: Region name (e.g., "socal")
            swan_resolution: SWAN mesh resolution to use (e.g., "coarse")
            step_size: Ray marching step size (m)
            max_steps: Maximum steps per ray
            min_depth: Minimum depth before stopping (m)
        """
        self.region = region
        self.swan_resolution = swan_resolution
        self.step_size = step_size
        self.max_steps = max_steps
        self.min_depth = min_depth

        # Resolve paths
        self.mesh_dir = get_surfzone_mesh_dir(region)
        self.swan_run_dir = get_swan_run_dir(region, swan_resolution)
        self.output_dir = get_output_dir(region)

        # Load mesh
        logger.info(f"Loading surfzone mesh from {self.mesh_dir}")
        self.mesh = SurfZoneMesh.load(self.mesh_dir)
        logger.info(f"Loaded mesh: {len(self.mesh.points_x):,} points")

        # Initialize SWAN input provider
        logger.info(f"Loading SWAN output from {self.swan_run_dir}")
        self.swan_provider = SwanInputProvider(self.swan_run_dir)

        logger.info(f"SurfzoneRunner initialized for {region}")

    def generate_boundary_points(
        self,
        spacing_m: float = 500.0,
        target_offshore_m: float = 2000.0,
        tolerance_m: float = 500.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate boundary points at the offshore edge of the mesh.

        Finds mesh points that are approximately at the target distance from
        the coastline, then downsamples to the desired spacing. For coastal
        sections where the mesh doesn't extend to the target distance, uses
        the farthest available offshore points to ensure complete coverage.

        Args:
            spacing_m: Spacing between boundary points (m)
            target_offshore_m: Target distance from coastline (m)
            tolerance_m: Tolerance around target distance (m)

        Returns:
            Tuple of (x, y, lon, lat) arrays
        """
        from scipy.spatial import cKDTree

        # Build KDTree from all coastline points
        coastline_points = []
        for coastline in self.mesh.coastlines:
            coastline_points.extend(coastline.tolist())
        coastline_points = np.array(coastline_points)

        if len(coastline_points) == 0:
            raise ValueError("Mesh has no coastline data")

        coast_tree = cKDTree(coastline_points)

        # Get all mesh points
        mesh_points = np.column_stack([self.mesh.points_x, self.mesh.points_y])

        # Query distance to nearest coastline point for all mesh points
        distances, nearest_coast_idx = coast_tree.query(mesh_points)

        # Also filter to water only (depth > 0)
        water_mask = self.mesh.elevation < 0

        # Find points near the target offshore distance
        min_dist = target_offshore_m - tolerance_m
        max_dist = target_offshore_m + tolerance_m
        offshore_mask = (distances >= min_dist) & (distances <= max_dist)
        valid_mask = offshore_mask & water_mask

        candidate_x = self.mesh.points_x[valid_mask]
        candidate_y = self.mesh.points_y[valid_mask]

        logger.info(f"Found {len(candidate_x)} candidate points at {min_dist:.0f}-{max_dist:.0f}m offshore")

        # Check for gaps in coverage by dividing coast into segments
        # and ensuring each segment has boundary points
        y_min, y_max = self.mesh.points_y.min(), self.mesh.points_y.max()
        segment_size = spacing_m * 2  # Check coverage at 2x boundary spacing
        n_segments = int((y_max - y_min) / segment_size) + 1

        # Track which segments have coverage
        segment_has_coverage = np.zeros(n_segments, dtype=bool)
        if len(candidate_y) > 0:
            candidate_segments = ((candidate_y - y_min) / segment_size).astype(int)
            candidate_segments = np.clip(candidate_segments, 0, n_segments - 1)
            segment_has_coverage[np.unique(candidate_segments)] = True

        # For segments without coverage, find the farthest offshore points
        gap_segments = np.where(~segment_has_coverage)[0]
        if len(gap_segments) > 0:
            logger.info(f"Found {len(gap_segments)} coastal segments without boundary points at target distance")

            # Collect candidates from gap segments
            gap_x = []
            gap_y = []

            water_x = self.mesh.points_x[water_mask]
            water_y = self.mesh.points_y[water_mask]
            water_dist = distances[water_mask]

            for seg_idx in gap_segments:
                seg_y_min = y_min + seg_idx * segment_size
                seg_y_max = seg_y_min + segment_size

                # Find water points in this Y range
                in_segment = (water_y >= seg_y_min) & (water_y < seg_y_max)
                if not np.any(in_segment):
                    continue

                seg_x = water_x[in_segment]
                seg_y_pts = water_y[in_segment]
                seg_dist = water_dist[in_segment]

                # Use the farthest offshore points in this segment
                # (top 20% by distance, or at least the farthest point)
                if len(seg_dist) > 0:
                    threshold = np.percentile(seg_dist, 80)
                    far_mask = seg_dist >= threshold
                    gap_x.extend(seg_x[far_mask])
                    gap_y.extend(seg_y_pts[far_mask])

            if gap_x:
                logger.info(f"Added {len(gap_x)} points from gap segments")
                candidate_x = np.concatenate([candidate_x, np.array(gap_x)])
                candidate_y = np.concatenate([candidate_y, np.array(gap_y)])

        # Downsample to desired spacing using a greedy approach
        if len(candidate_x) > 0:
            boundary_x, boundary_y = self._downsample_points(
                candidate_x, candidate_y, spacing_m
            )
        else:
            boundary_x = np.array([])
            boundary_y = np.array([])

        # Convert to lon/lat
        if len(boundary_x) > 0:
            boundary_lon, boundary_lat = self.mesh.utm_to_lon_lat(boundary_x, boundary_y)
        else:
            boundary_lon = np.array([])
            boundary_lat = np.array([])

        logger.info(f"Generated {len(boundary_x)} boundary points")

        return boundary_x, boundary_y, boundary_lon, boundary_lat

    def _downsample_points(
        self,
        x: np.ndarray,
        y: np.ndarray,
        min_spacing: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Downsample points to approximately uniform spacing.

        Uses a greedy algorithm: iterate through points sorted by Y coordinate,
        keeping only points that are at least min_spacing from all kept points.
        """
        from scipy.spatial import cKDTree

        # Sort by Y coordinate for more consistent sampling along coast
        sort_idx = np.argsort(y)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        # Greedy selection
        kept_x = [x_sorted[0]]
        kept_y = [y_sorted[0]]

        for i in range(1, len(x_sorted)):
            # Check distance to all kept points
            px, py = x_sorted[i], y_sorted[i]
            min_dist = min(
                np.sqrt((px - kx)**2 + (py - ky)**2)
                for kx, ky in zip(kept_x, kept_y)
            )
            if min_dist >= min_spacing:
                kept_x.append(px)
                kept_y.append(py)

        return np.array(kept_x), np.array(kept_y)

    def get_boundary_conditions(
        self,
        boundary_spacing_m: float = 500.0,
    ) -> BoundaryConditions:
        """
        Get wave boundary conditions at the offshore boundary.

        Args:
            boundary_spacing_m: Spacing between boundary points (m)

        Returns:
            BoundaryConditions with wave data at each point
        """
        # Generate boundary points
        x, y, lon, lat = self.generate_boundary_points(boundary_spacing_m)

        # Get SWAN partition data at these points
        conditions = self.swan_provider.get_boundary_conditions(lon, lat, x, y)

        logger.info(conditions.summary())

        return conditions

    def run(
        self,
        dry_run: bool = False,
        boundary_spacing_m: float = 500.0,
        store_paths: bool = False,
    ) -> Optional[BreakingField]:
        """
        Execute complete surfzone simulation.

        Args:
            dry_run: If True, prepare but don't trace rays
            boundary_spacing_m: Spacing between boundary points (m)
            store_paths: Whether to store ray paths (increases memory)

        Returns:
            BreakingField if successful, None if dry run
        """
        start_time = perf_counter()
        logger.info(f"Starting surfzone run for {self.region}")

        # Clean and create output directory
        import shutil
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Get boundary conditions
        t0 = perf_counter()
        boundary = self.get_boundary_conditions(boundary_spacing_m)
        t_boundary = perf_counter() - t0
        logger.info(f"Boundary conditions loaded in {t_boundary:.1f}s")

        # Step 2: Get wind data
        t0 = perf_counter()
        wind_speed, wind_direction = get_wind_data(self.mesh)
        t_wind = perf_counter() - t0

        # Step 3: Initialize ray tracer
        t0 = perf_counter()
        tracer = RayTracer(
            self.mesh,
            step_size=self.step_size,
            max_steps=self.max_steps,
            min_depth=self.min_depth,
        )
        t_init = perf_counter() - t0
        logger.info(f"Ray tracer initialized in {t_init:.1f}s")

        if dry_run:
            logger.info("Dry run - skipping ray tracing")
            self._print_dry_run_summary(boundary, wind_speed, wind_direction)
            return None

        # Step 4: Trace rays
        t0 = perf_counter()
        results = tracer.trace_from_boundary(
            boundary,
            U_wind=wind_speed,
            wind_direction=wind_direction,
            store_paths=store_paths,
        )
        t_trace = perf_counter() - t0
        logger.info(f"Ray tracing completed in {t_trace:.1f}s ({len(results)} rays)")

        # Step 5: Create breaking field
        t0 = perf_counter()
        writer = OutputWriter(self.output_dir)
        field = writer.results_to_breaking_field(results, self.mesh)
        t_field = perf_counter() - t0

        # Step 6: Save results
        t0 = perf_counter()
        writer.save_breaking_field(field)

        # Save run metadata
        config = {
            "region": self.region,
            "swan_resolution": self.swan_resolution,
            "step_size": self.step_size,
            "max_steps": self.max_steps,
            "min_depth": self.min_depth,
            "boundary_spacing_m": boundary_spacing_m,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "n_boundary_points": boundary.n_points,
            "n_partitions": boundary.n_partitions,
            "n_rays_traced": len(results),
            "n_breaking_points": field.n_points,
        }

        timing = {
            "boundary_load_s": t_boundary,
            "wind_load_s": t_wind,
            "tracer_init_s": t_init,
            "ray_trace_s": t_trace,
            "field_create_s": t_field,
            "total_s": perf_counter() - start_time,
        }

        writer.save_run_metadata(config, timing, tracer.summary(results))
        t_save = perf_counter() - t0

        # Print summary
        total_time = perf_counter() - start_time
        logger.info(f"\nSurfzone run completed in {total_time:.1f}s")
        logger.info(f"Output directory: {self.output_dir}")
        print()
        print(field.summary())
        print()
        print(tracer.summary(results))

        return field

    def _print_dry_run_summary(
        self,
        boundary: BoundaryConditions,
        wind_speed: float,
        wind_direction: float,
    ) -> None:
        """Print summary for dry run."""
        print()
        print("=" * 60)
        print("DRY RUN SUMMARY")
        print("=" * 60)
        print()
        print(f"Region: {self.region}")
        print(f"SWAN source: {self.swan_resolution}")
        print()
        print("Mesh:")
        print(f"  Points: {len(self.mesh.points_x):,}")
        print(f"  Coastline segments: {len(self.mesh.coastlines)}")
        print()
        print("Boundary conditions:")
        print(f"  Points: {boundary.n_points}")
        print(f"  Partitions: {boundary.n_partitions}")
        for p in boundary.partitions:
            n_valid = np.sum(p.is_valid)
            if n_valid > 0:
                print(f"    {p.label}: {n_valid} valid points")
        print()
        print(f"Wind: {wind_speed:.1f} m/s from {wind_direction:.0f}°")
        print()
        print("Ray tracer config:")
        print(f"  Step size: {self.step_size} m")
        print(f"  Max steps: {self.max_steps}")
        print(f"  Min depth: {self.min_depth} m")
        print()
        print("=" * 60)


def run_surfzone(
    region: str,
    swan_resolution: str = "ultrafine",
    step_size: float = 10.0,
    max_steps: int = 1000,
    boundary_spacing_m: float = 500.0,
    dry_run: bool = False,
) -> Optional[BreakingField]:
    """
    Convenience function to run surfzone simulation.

    Args:
        region: Region name
        swan_resolution: SWAN mesh resolution
        step_size: Ray step size (m)
        max_steps: Maximum steps per ray
        boundary_spacing_m: Boundary point spacing (m)
        dry_run: If True, prepare but don't trace

    Returns:
        BreakingField if successful
    """
    runner = SurfzoneRunner(
        region=region,
        swan_resolution=swan_resolution,
        step_size=step_size,
        max_steps=max_steps,
    )
    return runner.run(dry_run=dry_run, boundary_spacing_m=boundary_spacing_m)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run surfzone wave propagation simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/surfzone/run_surfzone.py --region socal
  python data/surfzone/run_surfzone.py --region socal --swan-resolution coarse
  python data/surfzone/run_surfzone.py --region socal --dry-run
  python data/surfzone/run_surfzone.py --region socal --fast
        """
    )

    parser.add_argument(
        "--region", "-r",
        required=True,
        help="Region name (e.g., socal)"
    )
    parser.add_argument(
        "--swan-resolution", "-s",
        default="ultrafine",
        help="SWAN mesh resolution to use (default: ultrafine)"
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=10.0,
        help="Base ray marching step size in meters (default: 10, adapts to ~3m near shore)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per ray (default: 1000)"
    )
    parser.add_argument(
        "--boundary-spacing",
        type=float,
        default=500.0,
        help="Boundary point spacing in meters (default: 500)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: larger step size (50m), wider boundary spacing (1000m)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Prepare but don't trace rays (validate inputs)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Apply fast mode settings
    step_size = args.step_size
    boundary_spacing = args.boundary_spacing
    if args.fast:
        step_size = 50.0
        boundary_spacing = 1000.0
        logger.info("Fast mode enabled: step_size=50m, boundary_spacing=1000m")

    # Run
    try:
        field = run_surfzone(
            region=args.region,
            swan_resolution=args.swan_resolution,
            step_size=step_size,
            max_steps=args.max_steps,
            boundary_spacing_m=boundary_spacing,
            dry_run=args.dry_run,
        )

        sys.exit(0 if field is not None or args.dry_run else 1)

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
