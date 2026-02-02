#!/usr/bin/env python3
"""
Run surfzone forward wave propagation simulation.

Usage:
    python run_simulation.py [--min-depth 0.0] [--max-depth 10.0] [--output-dir output/]

Example:
    python run_simulation.py
    python run_simulation.py --min-depth 0 --max-depth 5
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.surfzone.mesh import SurfZoneMesh
from data.surfzone.runner.swan_input_provider import SwanInputProvider, BoundaryConditions, WavePartition
from data.surfzone.runner.surfzone_runner import SurfzoneRunner, SurfzoneRunnerConfig
from data.surfzone.runner.output_writer import save_surfzone_result

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run surfzone wave propagation simulation')
    parser.add_argument('--min-depth', type=float, default=0.0,
                        help='Minimum depth filter (m, default: 0.0)')
    parser.add_argument('--max-depth', type=float, default=10.0,
                        help='Maximum depth filter (m, default: 10.0)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: data/surfzone/output/)')
    parser.add_argument('--mesh-dir', type=str, default=None,
                        help='Mesh directory (default: auto-detect socal mesh)')
    parser.add_argument('--swan-dir', type=str, default=None,
                        help='SWAN run directory (default: data/swan/runs/socal/coarse/latest)')
    args = parser.parse_args()

    # Default paths
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "data" / "surfzone" / "output"
    else:
        args.output_dir = Path(args.output_dir)

    if args.swan_dir is None:
        args.swan_dir = PROJECT_ROOT / "data" / "swan" / "runs" / "socal" / "coarse" / "latest"
    else:
        args.swan_dir = Path(args.swan_dir)

    # Find mesh directory
    if args.mesh_dir is None:
        # Look for socal mesh
        mesh_base = PROJECT_ROOT / "data" / "surfzone" / "meshes"
        # Try socal/socal_surfzone.npz first
        socal_mesh = mesh_base / "socal"
        if socal_mesh.exists():
            args.mesh_dir = socal_mesh
        else:
            # Fallback: look for socal_* directories
            socal_meshes = list(mesh_base.glob("socal*"))
            if socal_meshes:
                args.mesh_dir = sorted(socal_meshes)[-1]
            else:
                logger.error(f"No socal mesh found in {mesh_base}")
                logger.error("Please specify --mesh-dir")
                sys.exit(1)
    else:
        args.mesh_dir = Path(args.mesh_dir)

    logger.info("=" * 60)
    logger.info("Surfzone Wave Propagation Simulation")
    logger.info("=" * 60)
    logger.info(f"Mesh: {args.mesh_dir}")
    logger.info(f"SWAN: {args.swan_dir}")
    logger.info(f"Depth range: {args.min_depth} - {args.max_depth} m")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 60)

    # Load mesh
    logger.info("Loading mesh...")
    mesh = SurfZoneMesh.load(args.mesh_dir)
    logger.info(f"  Loaded {len(mesh.points_x)} mesh points")

    # Load SWAN data
    logger.info("Loading SWAN data...")
    swan = SwanInputProvider(args.swan_dir)

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

    # Configure runner
    config = SurfzoneRunnerConfig(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        partition_id=1,  # Primary swell
    )

    # Run simulation
    logger.info("Running simulation...")
    runner = SurfzoneRunner(mesh, boundary_conditions, config)
    result = runner.run(region_name="Southern California")

    # Print summary
    logger.info("=" * 60)
    logger.info("Results Summary")
    logger.info("=" * 60)
    print(result.summary())

    # Save results
    logger.info("Saving results...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = save_surfzone_result(result, args.output_dir, "primary_swell")
    logger.info(f"  Saved: {npz_path}")
    logger.info(f"  Saved: {json_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
