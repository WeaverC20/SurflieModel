#!/usr/bin/env python3
"""
Generate Surf Zone Mesh

CLI script to generate high-resolution surf zone meshes for wave ray tracing.
Uses variable resolution that's finest at the coastline and expands outward.

Usage:
    python scripts/generate_surfzone_mesh.py socal
    python scripts/generate_surfzone_mesh.py socal --min-res 5 --max-res 150
    python scripts/generate_surfzone_mesh.py --list-regions

Examples:
    # Generate SoCal mesh with default settings (10m to 100m resolution)
    python scripts/generate_surfzone_mesh.py socal

    # Generate with finer nearshore resolution
    python scripts/generate_surfzone_mesh.py socal --min-res 5

    # Generate with more aggressive stretching
    python scripts/generate_surfzone_mesh.py socal --stretch 1.12
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def list_regions():
    """List available regions."""
    from data.regions.region import REGIONS

    print("\nAvailable regions:")
    print("-" * 50)
    for name, region in REGIONS.items():
        print(f"  {name:12} - {region.display_name}")
        print(f"               Lat: [{region.lat_range[0]:.2f}, {region.lat_range[1]:.2f}]")
        print(f"               Lon: [{region.lon_range[0]:.2f}, {region.lon_range[1]:.2f}]")
    print()


def generate_mesh(
    region_name: str,
    min_res: float = 20.0,
    max_res: float = 300.0,
    offshore_dist: float = 2500.0,
    onshore_dist: float = 50.0,
    coastline_sample_res: float = 50.0,
    output_dir: Path = None,
    crm_file: Path = None,
):
    """Generate a surf zone mesh for a region."""
    from data.regions.region import get_region, REGIONS
    from data.bathymetry.usace_lidar import USACELidar
    from data.surfzone.mesh import SurfZoneMesh, SurfZoneMeshConfig

    # Validate region
    if region_name not in REGIONS:
        print(f"Error: Unknown region '{region_name}'")
        print(f"Available regions: {list(REGIONS.keys())}")
        sys.exit(1)

    region = get_region(region_name)

    print(f"\n{'='*60}")
    print(f"Generating Coastline-Following Surf Zone Mesh")
    print(f"Region: {region.display_name}")
    print(f"{'='*60}\n")

    # Configuration
    config = SurfZoneMeshConfig(
        min_resolution_m=min_res,
        max_resolution_m=max_res,
        offshore_distance_m=offshore_dist,
        onshore_distance_m=onshore_dist,
        coastline_sample_res_m=coastline_sample_res,
    )

    print("Configuration:")
    print(f"  Min resolution (at coast):  {config.min_resolution_m}m")
    print(f"  Max resolution (offshore):  {config.max_resolution_m}m")
    print(f"  Coastline sample resolution: {config.coastline_sample_res_m}m")
    print(f"  Offshore distance:          {config.offshore_distance_m}m")
    print(f"  Onshore distance:           {config.onshore_distance_m}m")
    print()

    # Load USACE Lidar data (primary source)
    print("Loading USACE Lidar data (primary source)...")
    lidar = USACELidar()

    # Check coverage overlap
    tiles = lidar.find_tiles(region.lon_range, region.lat_range)
    if not tiles:
        print(f"Error: No USACE Lidar tiles cover region '{region_name}'")
        print(f"  Region bounds: Lon {region.lon_range}, Lat {region.lat_range}")
        print(f"  Lidar bounds:  Lon [{lidar.bounds['lon_min']:.4f}, {lidar.bounds['lon_max']:.4f}]")
        print(f"                 Lat [{lidar.bounds['lat_min']:.4f}, {lidar.bounds['lat_max']:.4f}]")
        sys.exit(1)

    print(f"Found {len(tiles)} Lidar tiles covering region")
    print()

    # Load NCEI CRM data (fallback source, if provided)
    fallback_bathy = None
    if crm_file:
        print("Loading NCEI CRM data (fallback source)...")
        from data.bathymetry.ncei_crm import NCECRM
        try:
            fallback_bathy = NCECRM(crm_file)
            print()
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Continuing without fallback bathymetry...\n")

    # Generate mesh
    mesh = SurfZoneMesh.from_region(region, lidar, config, fallback_bathy=fallback_bathy)

    print()
    print(mesh.summary())

    # Save
    if output_dir is None:
        output_dir = project_root / "data" / "surfzone" / "meshes" / region_name

    print(f"\nSaving to: {output_dir}")
    mesh.save(output_dir)

    # Cleanup
    lidar.close()
    if fallback_bathy:
        fallback_bathy.close()

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")

    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="Generate variable-resolution surf zone mesh for wave ray tracing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        'region',
        nargs='?',
        help="Region name (e.g., 'socal', 'norcal', 'central')"
    )

    parser.add_argument(
        '--list-regions',
        action='store_true',
        help="List available regions and exit"
    )

    parser.add_argument(
        '--min-res',
        type=float,
        default=20.0,
        help="Finest resolution at coastline in meters (default: 20)"
    )

    parser.add_argument(
        '--max-res',
        type=float,
        default=300.0,
        help="Coarsest resolution at max offshore distance in meters (default: 300)"
    )

    parser.add_argument(
        '--offshore-dist',
        type=float,
        default=2500.0,
        help="Distance offshore from coastline in meters (default: 2500)"
    )

    parser.add_argument(
        '--onshore-dist',
        type=float,
        default=50.0,
        help="Distance onshore from coastline in meters (default: 50)"
    )

    parser.add_argument(
        '--coastline-sample-res',
        type=float,
        default=50.0,
        help="Resolution for coastline sampling in meters (default: 50)"
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help="Output directory (default: data/surfzone/meshes/{region})"
    )

    parser.add_argument(
        '--crm-file',
        type=Path,
        default=None,
        help="Path to NCEI CRM NetCDF file for fallback bathymetry (fills gaps in Lidar coverage)"
    )

    args = parser.parse_args()

    if args.list_regions:
        list_regions()
        return

    if not args.region:
        parser.print_help()
        print("\nError: region argument is required (use --list-regions to see options)")
        sys.exit(1)

    generate_mesh(
        region_name=args.region,
        min_res=args.min_res,
        max_res=args.max_res,
        offshore_dist=args.offshore_dist,
        onshore_dist=args.onshore_dist,
        coastline_sample_res=args.coastline_sample_res,
        output_dir=args.output_dir,
        crm_file=args.crm_file,
    )


if __name__ == "__main__":
    main()
