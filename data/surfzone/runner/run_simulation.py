#!/usr/bin/env python3
"""
Run surfzone forward wave propagation simulation.

Usage:
    python run_simulation.py --region socal
    python run_simulation.py --region socal --swan-resolution fine
    python run_simulation.py --list-regions

Example:
    python run_simulation.py --region socal
    python run_simulation.py --region socal --min-depth 0 --max-depth 5
    python run_simulation.py --region norcal --sample-fraction 0.1
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Pin Numba thread count BEFORE importing numba (via numpy or other deps)
# This prevents thread oversubscription issues that can cause progressive slowdown
if 'NUMBA_NUM_THREADS' not in os.environ:
    # Use half of available CPUs to leave headroom for other processes
    import multiprocessing
    n_threads = max(1, multiprocessing.cpu_count() // 2)
    os.environ['NUMBA_NUM_THREADS'] = str(n_threads)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.regions.region import get_region, REGIONS
from data.surfzone.mesh import SurfZoneMesh
from data.surfzone.runner.swan_input_provider import SwanInputProvider, BoundaryConditions, WavePartition
from data.surfzone.runner.surfzone_runner import SurfzoneRunner, SurfzoneRunnerConfig
from data.surfzone.runner.output_writer import save_surfzone_result

# Partition definitions: id -> (label, filename)
PARTITIONS = {
    0: ("Wind Sea", "wind_sea"),
    1: ("Primary Swell", "primary_swell"),
    2: ("Secondary Swell", "secondary_swell"),
    3: ("Tertiary Swell", "tertiary_swell"),
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_regions():
    """List available regions with mesh and SWAN status."""
    print("\nAvailable regions:")
    print("-" * 70)

    for name in ['socal', 'central', 'norcal']:
        region = REGIONS[name]

        # Check for surfzone mesh
        mesh_dir = PROJECT_ROOT / "data" / "surfzone" / "meshes" / name
        mesh_exists = mesh_dir.exists() and any(mesh_dir.glob("*.npz"))

        # Check for SWAN runs
        swan_dir = PROJECT_ROOT / "data" / "swan" / "runs" / name
        swan_resolutions = []
        if swan_dir.exists():
            for d in sorted(swan_dir.iterdir()):
                if d.is_dir() and (d / "latest").exists():
                    swan_resolutions.append(d.name)

        # Check for existing results
        output_dir = PROJECT_ROOT / "data" / "surfzone" / "output" / name
        has_results = output_dir.exists() and any(output_dir.glob("*.npz"))

        print(f"  {name:12} - {region.display_name}")
        print(f"               Lat: [{region.lat_range[0]:.1f}, {region.lat_range[1]:.1f}]")
        print(f"               Lon: [{region.lon_range[0]:.1f}, {region.lon_range[1]:.1f}]")
        print(f"               Mesh: {'Yes' if mesh_exists else 'No (run: python scripts/generate_surfzone_mesh.py ' + name + ')'}")
        print(f"               SWAN: {', '.join(swan_resolutions) if swan_resolutions else 'None'}")
        print(f"               Results: {'Yes' if has_results else 'No'}")
        print()


def get_region_paths(region_name: str, swan_resolution: Optional[str] = None) -> dict:
    """
    Get mesh and SWAN paths for a region with auto-detection.

    Args:
        region_name: Region identifier (socal, norcal, central)
        swan_resolution: Optional SWAN resolution (coarse, fine, etc.)
                        If None, uses first available in preference order.

    Returns:
        dict with keys: 'mesh_dir', 'swan_dir', 'output_dir', 'region_display_name'

    Raises:
        FileNotFoundError: If mesh or SWAN data not found
    """
    region = get_region(region_name)

    # Mesh directory
    mesh_dir = PROJECT_ROOT / "data" / "surfzone" / "meshes" / region_name
    if not mesh_dir.exists() or not any(mesh_dir.glob("*.npz")):
        raise FileNotFoundError(
            f"No surfzone mesh found for region '{region_name}'. "
            f"Generate one with: python scripts/generate_surfzone_mesh.py {region_name}"
        )

    # SWAN directory with resolution preference
    swan_base = PROJECT_ROOT / "data" / "swan" / "runs" / region_name
    resolution_preference = ['coarse', 'medium', 'fine', 'ultrafine']

    if swan_resolution:
        swan_dir = swan_base / swan_resolution / "latest"
        if not swan_dir.exists():
            raise FileNotFoundError(
                f"SWAN run not found at: {swan_dir}"
            )
    else:
        # Auto-detect first available resolution
        swan_dir = None
        for res in resolution_preference:
            candidate = swan_base / res / "latest"
            if candidate.exists():
                swan_dir = candidate
                logger.info(f"Auto-detected SWAN resolution: {res}")
                break
        if swan_dir is None:
            raise FileNotFoundError(
                f"No SWAN runs found for region '{region_name}' in {swan_base}. "
                f"Run SWAN first: python data/swan/run_swan.py --region {region_name}"
            )

    # Output directory (region-separated)
    output_dir = PROJECT_ROOT / "data" / "surfzone" / "output" / region_name

    return {
        'mesh_dir': mesh_dir,
        'swan_dir': swan_dir,
        'output_dir': output_dir,
        'region_display_name': region.display_name,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run surfzone wave propagation simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_simulation.py --region socal
    python run_simulation.py --region socal --swan-resolution fine
    python run_simulation.py --region norcal --sample-fraction 0.1
    python run_simulation.py --list-regions
        """
    )

    # Region selection
    parser.add_argument('--region', type=str, default=None,
                        help="Region name (socal, norcal, central). Required unless using --list-regions.")
    parser.add_argument('--list-regions', action='store_true',
                        help='List available regions with mesh/SWAN status and exit')
    parser.add_argument('--swan-resolution', type=str, default=None,
                        help='SWAN resolution (coarse, fine, etc.). Default: auto-detect')

    # Depth filtering
    parser.add_argument('--min-depth', type=float, default=0.0,
                        help='Minimum depth filter (m, default: 0.0)')
    parser.add_argument('--max-depth', type=float, default=10.0,
                        help='Maximum depth filter (m, default: 10.0)')

    # Path overrides (optional, for advanced use)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory override (default: data/surfzone/output/{region}/)')
    parser.add_argument('--mesh-dir', type=str, default=None,
                        help='Mesh directory override (default: data/surfzone/meshes/{region}/)')
    parser.add_argument('--swan-dir', type=str, default=None,
                        help='SWAN run directory override (default: auto-detect)')

    # Sampling options
    parser.add_argument('--sample-fraction', type=float, default=None,
                        help='Fraction of points to sample (e.g., 0.1 for 10%%)')
    parser.add_argument('--sample-count', type=int, default=None,
                        help='Exact number of points to sample')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible sampling')

    # Performance options
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear Numba JIT cache before running (fixes slowdown from stale cache)')

    args = parser.parse_args()

    # Clear Numba cache if requested
    if args.clear_cache:
        cache_dir = Path(__file__).parent / "__pycache__"
        if cache_dir.exists():
            nbc_files = list(cache_dir.glob("*.nb*"))
            if nbc_files:
                print(f"Clearing {len(nbc_files)} Numba cache files from {cache_dir}...")
                for f in nbc_files:
                    f.unlink()
                print("Cache cleared. Functions will be recompiled on first use.")
            else:
                print("No Numba cache files found.")
        else:
            print(f"Cache directory not found: {cache_dir}")

    # Handle --list-regions
    if args.list_regions:
        list_regions()
        return

    # Require --region
    if not args.region:
        parser.print_help()
        print("\nError: --region is required. Use --list-regions to see available regions.")
        sys.exit(1)

    # Get region-based paths (auto-detect or use overrides)
    try:
        paths = get_region_paths(args.region, args.swan_resolution)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Use explicit paths if provided, otherwise use auto-detected
    mesh_dir = Path(args.mesh_dir) if args.mesh_dir else paths['mesh_dir']
    swan_dir = Path(args.swan_dir) if args.swan_dir else paths['swan_dir']
    output_dir = Path(args.output_dir) if args.output_dir else paths['output_dir']
    region_display_name = paths['region_display_name']

    logger.info("=" * 60)
    logger.info("Surfzone Wave Propagation Simulation")
    logger.info("=" * 60)
    logger.info(f"Region: {region_display_name} ({args.region})")
    logger.info(f"Mesh: {mesh_dir}")
    logger.info(f"SWAN: {swan_dir}")
    logger.info(f"Depth range: {args.min_depth} - {args.max_depth} m")
    logger.info(f"Output: {output_dir}")
    if args.sample_fraction is not None:
        logger.info(f"Sampling: {args.sample_fraction * 100:.1f}% of points")
    elif args.sample_count is not None:
        logger.info(f"Sampling: {args.sample_count} points")
    if args.seed is not None:
        logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 60)

    # Load mesh
    logger.info("Loading mesh...")
    mesh = SurfZoneMesh.load(mesh_dir)
    logger.info(f"  Loaded {len(mesh.points_x)} mesh points")

    # Load SWAN data
    logger.info("Loading SWAN data...")
    swan = SwanInputProvider(swan_dir)

    # Get boundary conditions by sampling mesh boundary points
    logger.info("Finding boundary points from mesh...")
    arrays = mesh.get_numba_arrays()
    coast_distance = arrays.get('coast_distance', None)
    offshore_distance_m = arrays.get('offshore_distance_m', 0.0)

    if coast_distance is None or offshore_distance_m <= 0:
        logger.error("Mesh does not have coast_distance data")
        sys.exit(1)

    # Find points near the offshore boundary (within 10% of threshold)
    boundary_threshold = offshore_distance_m * 0.90
    boundary_mask = coast_distance >= boundary_threshold
    boundary_indices = np.where(boundary_mask)[0]
    logger.info(f"  Found {len(boundary_indices)} mesh points near offshore boundary")

    # Subsample if too many points (for efficiency)
    max_boundary_points = 2000
    if len(boundary_indices) > max_boundary_points:
        np.random.seed(42)
        boundary_indices = np.random.choice(boundary_indices, max_boundary_points, replace=False)
        logger.info(f"  Subsampled to {max_boundary_points} points")

    # Get UTM coordinates
    boundary_x = arrays['points_x'][boundary_indices].copy()
    boundary_y = arrays['points_y'][boundary_indices].copy()

    # Convert to lon/lat for SWAN sampling
    lons, lats = mesh.utm_to_lon_lat(boundary_x, boundary_y)

    logger.info(f"  Sampling SWAN data at {len(lons)} boundary points...")
    sampled = swan.sample_at_points(lons, lats, use_nearest_fallback=True)

    # Convert to WavePartition objects
    partitions = []
    for part_id, data in sampled.items():
        hs = data['hs']
        tp = data['tp']
        direction = data['dir']
        is_valid = ~np.isnan(hs) & (hs > 0) & ~np.isnan(tp) & (tp > 0)
        partitions.append(WavePartition(
            hs=hs,
            tp=tp,
            direction=direction,
            partition_id=part_id,
            is_valid=is_valid,
        ))

    # Create boundary conditions
    boundary_conditions = BoundaryConditions(
        x=boundary_x,
        y=boundary_y,
        lon=lons,
        lat=lats,
        partitions=partitions,
    )

    logger.info(f"  {boundary_conditions.n_points} boundary points")
    logger.info(f"  {boundary_conditions.n_partitions} partitions")

    # Print partition summary
    for p in boundary_conditions.partitions:
        n_valid = p.is_valid.sum()
        if n_valid > 0:
            valid_hs = p.hs[p.is_valid]
            logger.info(f"    {p.label}: {n_valid} valid, Hs={valid_hs.min():.2f}-{valid_hs.max():.2f}m")

    # Determine which partitions to run
    # By default, run all partitions that have valid data
    partitions_to_run = []
    for p in boundary_conditions.partitions:
        n_valid = p.is_valid.sum()
        if n_valid > 0:
            partitions_to_run.append(p.partition_id)

    logger.info(f"Will run {len(partitions_to_run)} partitions: {partitions_to_run}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run simulation for each partition
    all_results = []
    for partition_id in partitions_to_run:
        partition_label, partition_filename = PARTITIONS.get(partition_id, (f"Partition {partition_id}", f"partition_{partition_id}"))

        logger.info("=" * 60)
        logger.info(f"Running simulation for partition {partition_id}: {partition_label}")
        logger.info("=" * 60)

        # Configure runner for this partition
        config = SurfzoneRunnerConfig(
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            partition_id=partition_id,
            sample_fraction=args.sample_fraction,
            sample_count=args.sample_count,
            random_seed=args.seed,
        )

        # Run simulation
        runner = SurfzoneRunner(mesh, boundary_conditions, config)
        result = runner.run(region_name=region_display_name)

        # Print summary
        logger.info("-" * 40)
        logger.info(f"Results for {partition_label}")
        logger.info("-" * 40)
        print(result.summary())

        # Save results
        logger.info("Saving results...")
        npz_path, json_path = save_surfzone_result(result, output_dir, partition_filename)
        logger.info(f"  Saved: {npz_path}")
        logger.info(f"  Saved: {json_path}")

        all_results.append((partition_id, partition_label, result))

    # Final summary
    logger.info("=" * 60)
    logger.info("All Partitions Complete")
    logger.info("=" * 60)
    for partition_id, partition_label, result in all_results:
        logger.info(f"  {partition_label}: {result.n_converged}/{result.n_sampled} converged ({result.convergence_rate:.1f}%)")

    logger.info("Done!")


if __name__ == "__main__":
    main()
