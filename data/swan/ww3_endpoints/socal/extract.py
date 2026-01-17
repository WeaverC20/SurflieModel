#!/usr/bin/env python3
"""
Southern California WW3 Boundary Point Extraction

Extracts WaveWatch III grid points along boundaries of the SoCal SWAN domain
for use as spectral boundary conditions.

SoCal Region Bounds:
    Latitude:  32.0°N to 34.5°N
    Longitude: 121.0°W to 117.0°W

Active Boundaries:
    - West: -121.0° longitude (Pacific swell from W/NW)
    - South: 32.0° latitude (Southern hemisphere swell)

Usage:
    python data/swan/ww3_endpoints/socal/extract.py
    python data/swan/ww3_endpoints/socal/extract.py --plot
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
    create_unified_boundary_config,
    UnifiedBoundaryConfig,
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

# Which boundaries to use for wave forcing
# - West: Primary Pacific swell from W/NW directions
# - South: Southern hemisphere swell (S/SW swells, especially in summer)
ACTIVE_BOUNDARIES = ['west', 'south']


def extract_boundary_config(
    active_sides: List[str] = None,
    save: bool = True,
    plot: bool = False,
) -> UnifiedBoundaryConfig:
    """
    Extract WW3 boundary points in unified multi-boundary format.

    Creates a single JSON file (ww3_boundaries.json) containing all
    boundary definitions with active boundaries marked.

    Args:
        active_sides: Which boundaries to use for forcing (default: ACTIVE_BOUNDARIES)
        save: Whether to save to JSON file
        plot: Whether to display plots of boundary points

    Returns:
        UnifiedBoundaryConfig object
    """
    if active_sides is None:
        active_sides = ACTIVE_BOUNDARIES

    print(f"=" * 60)
    print(f"Extracting WW3 Boundary Points for SoCal")
    print(f"=" * 60)
    print(f"Region: {SOCAL.display_name}")
    print(f"Bounds: lat {SOCAL_BOUNDS['lat_min']}° to {SOCAL_BOUNDS['lat_max']}°")
    print(f"        lon {SOCAL_BOUNDS['lon_min']}° to {SOCAL_BOUNDS['lon_max']}°")
    print(f"Active boundaries: {active_sides}")
    print()

    config = create_unified_boundary_config(
        lon_min=SOCAL_BOUNDS['lon_min'],
        lon_max=SOCAL_BOUNDS['lon_max'],
        lat_min=SOCAL_BOUNDS['lat_min'],
        lat_max=SOCAL_BOUNDS['lat_max'],
        active_sides=active_sides,
        region_name=REGION_NAME,
        mesh_name=MESH_NAME,
    )

    print(f"\n{config.summary()}")

    if save:
        filepath = OUTPUT_DIR / "ww3_boundaries.json"
        config.save(filepath)

    if plot:
        # Plot each boundary
        for side, point_set in config.boundaries.items():
            plot_path = OUTPUT_DIR / f"ww3_boundary_{side}.png" if save else None
            point_set.plot(save_path=plot_path, show=True)

    return config


def get_ww3_boundary_points() -> List[Tuple[float, float]]:
    """
    Get the western boundary WW3 points for SoCal.

    This is a convenience function that returns just the points
    for updating the Region class.

    Returns:
        List of (lon, lat) tuples for WW3 boundary points
    """
    config = load_boundary_config()
    if 'west' in config.boundaries:
        return config.boundaries['west'].points
    return []


def load_boundary_config() -> UnifiedBoundaryConfig:
    """
    Load the unified boundary configuration for SoCal.

    Returns:
        UnifiedBoundaryConfig object

    Raises:
        FileNotFoundError: If config not found
    """
    filepath = OUTPUT_DIR / "ww3_boundaries.json"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Boundary config not found: {filepath}\n"
            f"Run: python data/swan/ww3_endpoints/socal/extract.py"
        )
    return UnifiedBoundaryConfig.load(filepath)


def load_boundary_points(side: str = 'west') -> BoundaryPointSet:
    """
    Load boundary points for a specific side.

    Args:
        side: Which boundary side ('west', 'east', 'north', 'south')

    Returns:
        BoundaryPointSet object
    """
    config = load_boundary_config()
    if side not in config.boundaries:
        raise ValueError(
            f"Boundary '{side}' not found in config. "
            f"Available: {list(config.boundaries.keys())}"
        )
    return config.boundaries[side]


def main():
    parser = argparse.ArgumentParser(
        description="Extract WW3 boundary points for SoCal SWAN domain"
    )
    parser.add_argument(
        "--sides", "-s",
        nargs="+",
        choices=['west', 'east', 'north', 'south'],
        default=None,
        help="Which boundary sides to extract (default: west, south)"
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Display plots of boundary points"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to JSON file"
    )

    args = parser.parse_args()

    # Default sides
    sides = args.sides if args.sides else ACTIVE_BOUNDARIES

    # Extract and save boundary config
    config = extract_boundary_config(
        active_sides=sides,
        save=not args.no_save,
        plot=args.plot,
    )

    print("\nDone!")
    return config


if __name__ == "__main__":
    main()
