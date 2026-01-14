#!/usr/bin/env python3
"""
Southern California WW3 Boundary Point Extraction

Extracts WaveWatch III grid points along the western boundary of the
SoCal SWAN domain for use as spectral boundary conditions.

SoCal Region Bounds:
    Latitude:  32.0°N to 34.5°N
    Longitude: 121.0°W to 117.0°W

Western Boundary (for wave input):
    Longitude: -121.0°
    Latitude:  32.0° to 34.5°

Usage:
    python data/swan/ww3_endpoints/socal/extract.py
    python data/swan/ww3_endpoints/socal/extract.py --plot
    python data/swan/ww3_endpoints/socal/extract.py --update-region
"""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
from typing import List, Tuple

from data.swan.ww3_endpoints.extract_ww3_endpoints import (
    BoundaryLine,
    BoundaryPointSet,
    find_boundary_points,
    extract_region_boundaries,
)
from data.regions import SOCAL, get_region


# =============================================================================
# SoCal Configuration
# =============================================================================

REGION_NAME = "socal"
MESH_NAME = "socal_coarse"
OUTPUT_DIR = Path(__file__).parent  # data/swan/ww3_endpoints/socal/

# SoCal region bounds (from regions/region.py)
SOCAL_BOUNDS = {
    "lat_min": SOCAL.lat_range[0],  # 32.0
    "lat_max": SOCAL.lat_range[1],  # 34.5
    "lon_min": SOCAL.lon_range[0],  # -121.0
    "lon_max": SOCAL.lon_range[1],  # -117.0
}

# Which boundaries to extract WW3 points from
# For California, waves primarily come from the west and northwest
BOUNDARY_SIDES = ['west']  # Can add 'south', 'north' if needed


def extract_socal_ww3_points(
    sides: List[str] = None,
    save: bool = True,
    plot: bool = False,
) -> dict:
    """
    Extract WW3 boundary points for SoCal region.

    Args:
        sides: Which boundary sides to extract (default: ['west'])
        save: Whether to save to JSON file
        plot: Whether to display plots

    Returns:
        Dict mapping side name to BoundaryPointSet
    """
    if sides is None:
        sides = BOUNDARY_SIDES

    print(f"=" * 60)
    print(f"Extracting WW3 Boundary Points for SoCal")
    print(f"=" * 60)
    print(f"Region: {SOCAL.display_name}")
    print(f"Bounds: lat {SOCAL_BOUNDS['lat_min']}° to {SOCAL_BOUNDS['lat_max']}°")
    print(f"        lon {SOCAL_BOUNDS['lon_min']}° to {SOCAL_BOUNDS['lon_max']}°")
    print(f"Extracting sides: {sides}")
    print()

    boundaries = extract_region_boundaries(
        lon_min=SOCAL_BOUNDS['lon_min'],
        lon_max=SOCAL_BOUNDS['lon_max'],
        lat_min=SOCAL_BOUNDS['lat_min'],
        lat_max=SOCAL_BOUNDS['lat_max'],
        sides=sides,
        region_name=REGION_NAME,
        mesh_name=MESH_NAME,
    )

    for side, point_set in boundaries.items():
        print(f"\n{point_set.summary()}")

        if save:
            filepath = OUTPUT_DIR / f"ww3_boundary_{side}.json"
            point_set.save(filepath)

        if plot:
            plot_path = OUTPUT_DIR / f"ww3_boundary_{side}.png" if save else None
            point_set.plot(save_path=plot_path, show=True)

    return boundaries


def get_ww3_boundary_points() -> List[Tuple[float, float]]:
    """
    Get the western boundary WW3 points for SoCal.

    This is a convenience function that returns just the points
    for updating the Region class.

    Returns:
        List of (lon, lat) tuples for WW3 boundary points
    """
    # Check if points are already saved
    filepath = OUTPUT_DIR / "ww3_boundary_west.json"
    if filepath.exists():
        point_set = BoundaryPointSet.load(filepath)
        return point_set.points

    # Otherwise extract them
    boundaries = extract_socal_ww3_points(sides=['west'], save=True, plot=False)
    return boundaries['west'].points


def load_boundary_points(side: str = 'west') -> BoundaryPointSet:
    """
    Load previously saved boundary points.

    Args:
        side: Which boundary side ('west', 'east', 'north', 'south')

    Returns:
        BoundaryPointSet object
    """
    filepath = OUTPUT_DIR / f"ww3_boundary_{side}.json"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Boundary points not found: {filepath}\n"
            f"Run: python data/swan/ww3_endpoints/socal/extract.py"
        )
    return BoundaryPointSet.load(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Extract WW3 boundary points for SoCal SWAN domain"
    )
    parser.add_argument(
        "--sides", "-s",
        nargs="+",
        choices=['west', 'east', 'north', 'south'],
        default=['west'],
        help="Which boundary sides to extract (default: west)"
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Display plots of boundary points"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to JSON files"
    )
    parser.add_argument(
        "--update-region",
        action="store_true",
        help="Print code to update Region class with these points"
    )

    args = parser.parse_args()

    boundaries = extract_socal_ww3_points(
        sides=args.sides,
        save=not args.no_save,
        plot=args.plot,
    )

    if args.update_region:
        print("\n" + "=" * 60)
        print("Code to update SOCAL region ww3_boundary_points:")
        print("=" * 60)

        if 'west' in boundaries:
            points = boundaries['west'].points
            print(f"\n# In data/regions/region.py, update SOCAL definition:")
            print(f"SOCAL = Region(")
            print(f"    name=\"socal\",")
            print(f"    display_name=\"Southern California\",")
            print(f"    bounds={{")
            print(f"        \"lat\": (32.0, 34.5),")
            print(f"        \"lon\": (-121.0, -117.0),")
            print(f"    }},")
            print(f"    parent=CALIFORNIA,")
            print(f"    color=\"#E9C46A\",")
            print(f"    ww3_boundary_points={points},")
            print(f")")

    print("\nDone!")
    return boundaries


if __name__ == "__main__":
    main()