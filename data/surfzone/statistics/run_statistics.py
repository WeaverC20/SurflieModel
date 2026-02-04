#!/usr/bin/env python3
"""
Run surfzone wave statistics analysis.

Computes wave group statistics (set period, waves per set, wavelength, etc.)
from surfzone runner output files. Each partition that has been propagated
contributes to the statistics.

Usage:
    python data/surfzone/statistics/run_statistics.py --region socal
    python data/surfzone/statistics/run_statistics.py --list-regions
    python data/surfzone/statistics/run_statistics.py --list-statistics

Example:
    python data/surfzone/statistics/run_statistics.py --region socal
    python data/surfzone/statistics/run_statistics.py --region norcal --statistics set_period waves_per_set
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Add project root to path (statistics -> surfzone -> data -> project_root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.regions.region import get_region, REGIONS
from data.surfzone.runner.swan_input_provider import WavePartition
from data.surfzone.statistics import StatisticsRunner, StatisticsRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Partition file names (in order: wind_sea, primary, secondary, tertiary)
PARTITION_FILES = [
    'wind_sea',
    'primary_swell',
    'secondary_swell',
    'tertiary_swell'
]


def list_regions():
    """List available regions with surfzone output status."""
    print("\nAvailable regions for statistics:")
    print("-" * 70)

    for name in ['socal', 'central', 'norcal']:
        region = REGIONS[name]

        # Check for surfzone output
        output_dir = PROJECT_ROOT / "data" / "surfzone" / "output" / name
        partitions_found = []
        if output_dir.exists():
            for pname in PARTITION_FILES:
                if (output_dir / f"{pname}.npz").exists():
                    partitions_found.append(pname)

        # Check for existing statistics
        has_stats = output_dir.exists() and any(output_dir.glob("statistics_*.csv"))

        print(f"  {name:12} - {region.display_name}")
        print(f"               Partitions: {', '.join(partitions_found) if partitions_found else 'None'}")
        print(f"               Stats: {'Yes' if has_stats else 'No'}")
        print()


def list_statistics():
    """List all available statistics."""
    print("\nAvailable wave statistics:")
    print("-" * 70)

    for stat in StatisticsRegistry.all():
        print(f"  {stat.name:25} ({stat.units:8}) - {stat.description}")
    print()


def load_surfzone_partitions(output_dir: Path) -> Dict[str, dict]:
    """
    Load all available partition data from surfzone output.

    Returns:
        Dict mapping partition name to {'data': npz_data, 'meta': json_meta}
    """
    partitions = {}

    for pname in PARTITION_FILES:
        npz_path = output_dir / f"{pname}.npz"
        json_path = output_dir / f"{pname}.json"

        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            meta = {}
            if json_path.exists():
                with open(json_path) as f:
                    meta = json.load(f)

            partitions[pname] = {
                'data': data,
                'meta': meta
            }
            print(f"  Loaded {pname}: {meta.get('n_converged', '?')}/{meta.get('n_points', '?')} converged points")

    return partitions


def create_wave_partitions(partitions_data: Dict[str, dict]) -> List[WavePartition]:
    """
    Create WavePartition objects from surfzone output data.

    Uses the boundary_* fields which contain the SWAN partition values
    that were used for propagation.
    """
    wave_partitions = []

    for i, pname in enumerate(PARTITION_FILES):
        if pname not in partitions_data:
            continue

        data = partitions_data[pname]['data']

        # Get boundary values (these are the SWAN partition values at each point)
        hs = data['boundary_Hs']
        tp = data['boundary_Tp']
        direction = data['boundary_direction']

        # Use converged points as valid mask
        is_valid = data['converged']

        wave_partitions.append(WavePartition(
            hs=hs,
            tp=tp,
            direction=direction,
            partition_id=i,
            is_valid=is_valid
        ))

    return wave_partitions


def run_statistics(
    region_name: str,
    statistics: Optional[List[str]] = None
):
    """
    Run wave statistics for a region using surfzone output.

    Args:
        region_name: Region identifier
        statistics: List of statistics to compute (all if None)
    """
    print(f"\n{'='*70}")
    print(f"Running wave statistics for: {region_name}")
    print(f"{'='*70}\n")

    # Get paths
    region = get_region(region_name)
    output_dir = PROJECT_ROOT / "data" / "surfzone" / "output" / region_name

    if not output_dir.exists():
        raise FileNotFoundError(f"No surfzone output found for region '{region_name}'")

    print(f"Region: {region.display_name}")
    print(f"Output dir: {output_dir}")
    print()

    # Load all available partition data
    print("Loading surfzone partition data...")
    partitions_data = load_surfzone_partitions(output_dir)

    if not partitions_data:
        raise FileNotFoundError(
            f"No partition data found in {output_dir}. "
            f"Run surfzone simulation first."
        )

    # Get mesh coordinates from the first partition
    first_partition = next(iter(partitions_data.values()))
    data = first_partition['data']

    mesh_x = data['mesh_x']
    mesh_y = data['mesh_y']
    depths = data['mesh_depth']
    n_points = len(mesh_x)

    print(f"\nMesh: {n_points:,} points")

    # Convert to lon/lat (need mesh object for transformation)
    print("Converting coordinates...")
    from data.surfzone.mesh import SurfZoneMesh
    mesh = SurfZoneMesh.load(PROJECT_ROOT / "data" / "surfzone" / "meshes" / region_name)
    lons, lats = mesh.utm_to_lon_lat(mesh_x, mesh_y)

    # Create WavePartition objects
    print("\nCreating wave partitions...")
    wave_partitions = create_wave_partitions(partitions_data)
    print(f"  Created {len(wave_partitions)} partition(s)")

    if len(wave_partitions) < 2:
        print("\n  WARNING: Only 1 partition available. Set frequency statistics")
        print("           require at least 2 partitions (need beat frequency).")
        print("           Run surfzone simulation for more partitions.")

    # Create runner
    print("\nComputing statistics...")
    if statistics:
        print(f"  Selected: {', '.join(statistics)}")
        runner = StatisticsRunner(statistics=statistics)
    else:
        print(f"  All registered ({len(StatisticsRegistry.all())} statistics)")
        runner = StatisticsRunner()

    # Run statistics
    result = runner.run_from_partitions(
        partitions=wave_partitions,
        lats=lats,
        lons=lons,
        depths=depths,
        region=region_name
    )

    print(f"\nComputed {len(result.df.columns) - 4} statistic columns for {result.num_points:,} points")

    # Show summary statistics
    print("\nSummary statistics:")
    print("-" * 50)
    stat_cols = [c for c in result.df.columns if c not in ['point_id', 'lat', 'lon', 'depth']]
    for col in stat_cols[:10]:
        values = result.df[col].dropna()
        if len(values) > 0:
            print(f"  {col:25}: {values.min():.2f} - {values.max():.2f} (mean: {values.mean():.2f})")

    if len(stat_cols) > 10:
        print(f"  ... and {len(stat_cols) - 10} more columns")

    # Save results
    print(f"\nSaving to {output_dir}...")
    result.save(output_dir)

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Run wave statistics analysis on surfzone output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/surfzone/statistics/run_statistics.py --region socal
  python data/surfzone/statistics/run_statistics.py --list-regions
  python data/surfzone/statistics/run_statistics.py --list-statistics
        """
    )

    parser.add_argument(
        '--region', '-r',
        type=str,
        choices=['socal', 'norcal', 'central'],
        help='Region to analyze'
    )

    parser.add_argument(
        '--statistics', '-s',
        nargs='+',
        type=str,
        default=None,
        help='Specific statistics to compute. If not specified, computes all.'
    )

    parser.add_argument(
        '--list-regions',
        action='store_true',
        help='List available regions and their status'
    )

    parser.add_argument(
        '--list-statistics',
        action='store_true',
        help='List available statistics'
    )

    args = parser.parse_args()

    if args.list_regions:
        list_regions()
        return

    if args.list_statistics:
        list_statistics()
        return

    if not args.region:
        parser.print_help()
        print("\nError: --region is required (unless using --list-regions or --list-statistics)")
        sys.exit(1)

    try:
        run_statistics(
            region_name=args.region,
            statistics=args.statistics
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error running statistics: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
