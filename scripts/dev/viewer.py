#!/usr/bin/env python3
"""
Unified Development Viewer for SurflieModel

Launch a Panel-based interactive viewer for SWAN output, surfzone meshes,
or surfzone simulation results.

Usage:
    python scripts/dev/viewer.py
    python scripts/dev/viewer.py --region socal --view "Surfzone Results"
    python scripts/dev/viewer.py --region norcal --view "SWAN Data" --lonlat
    python scripts/dev/viewer.py --list-regions
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def list_data_availability():
    """Print available data for each region and exit."""
    from tools.viewer.data_manager import DataManager
    from tools.viewer.config import AVAILABLE_REGIONS, DATA_TYPES

    dm = DataManager(project_root)

    print("\nData availability by region:")
    print("-" * 60)

    for region in AVAILABLE_REGIONS:
        print(f"\n  {region}:")
        for dt in DATA_TYPES:
            has = dm.has_data(dt, region)
            status = "YES" if has else "no"
            print(f"    {dt:25s} {status}")

        # Show SWAN resolutions if available
        resolutions = dm.available_swan_resolutions(region)
        if resolutions:
            print(f"    SWAN resolutions: {', '.join(resolutions)}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="SurflieModel Development Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--region', '-r',
        default='socal',
        choices=['socal', 'central', 'norcal'],
        help="Region to view (default: socal)",
    )
    parser.add_argument(
        '--view', '-v',
        default='Surfzone Results',
        choices=['SWAN Data', 'Surfzone Mesh', 'Surfzone Results'],
        help="Data type to view (default: Surfzone Results)",
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5007,
        help="Port for the Panel server (default: 5007)",
    )
    parser.add_argument(
        '--lonlat',
        action='store_true',
        help="Use longitude/latitude coordinates instead of UTM",
    )
    parser.add_argument(
        '--list-regions',
        action='store_true',
        help="List available data per region and exit",
    )

    args = parser.parse_args()

    if args.list_regions:
        list_data_availability()
        return

    from tools.viewer.app import main as run_viewer

    run_viewer(
        region=args.region,
        data_type=args.view,
        port=args.port,
        use_lonlat=args.lonlat,
    )


if __name__ == "__main__":
    main()
