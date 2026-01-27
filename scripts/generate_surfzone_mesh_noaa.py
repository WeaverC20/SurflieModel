#!/usr/bin/env python3
"""
Generate Surf Zone Mesh using NOAA Topobathy DEM

CLI script to generate high-resolution surf zone meshes using the NOAA
2009-2011 Topobathy Elevation DEM (Coastal California) accessed via VRT.

No download required - uses GDAL's /vsicurl/ for remote access.

Usage:
    python scripts/generate_surfzone_mesh_noaa.py socal
    python scripts/generate_surfzone_mesh_noaa.py socal --min-res 10 --max-res 200
    python scripts/generate_surfzone_mesh_noaa.py --list-regions

Examples:
    # Generate SoCal mesh with default settings
    python scripts/generate_surfzone_mesh_noaa.py socal

    # Generate with finer nearshore resolution
    python scripts/generate_surfzone_mesh_noaa.py socal --min-res 10

    # Generate with larger offshore distance
    python scripts/generate_surfzone_mesh_noaa.py socal --offshore-dist 3000
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
):
    """Generate a surf zone mesh for a region using NOAA Topobathy DEM."""
    from data.regions.region import get_region, REGIONS
    from data.bathymetry.noaa_topobathy import NOAATopobathy
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
    print(f"Data Source: NOAA 2009-2011 Topobathy DEM (VRT remote access)")
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
    print(f"  Density bias:               {config.coastline_density_bias}")
    print()

    # Load NOAA Topobathy data (via VRT)
    print("Initializing NOAA Topobathy DEM (remote access)...")
    bathymetry = NOAATopobathy(verbose=True)
    print()

    # Check coverage
    tiles = bathymetry.find_tiles(region.lon_range, region.lat_range)
    if not tiles:
        print(f"Error: No NOAA Topobathy data covers region '{region_name}'")
        print(f"  Region bounds: Lon {region.lon_range}, Lat {region.lat_range}")
        b = bathymetry.bounds
        print(f"  Data bounds:   Lon [{b['lon_min']:.4f}, {b['lon_max']:.4f}]")
        print(f"                 Lat [{b['lat_min']:.4f}, {b['lat_max']:.4f}]")
        sys.exit(1)

    print("Coverage check: OK")
    print()

    # Generate mesh
    print("Starting mesh generation...")
    print("(This may take a while as data is fetched remotely)")
    print()

    mesh = SurfZoneMesh.from_region(region, bathymetry, config)

    print()
    print(mesh.summary())

    # Save
    if output_dir is None:
        output_dir = project_root / "data" / "surfzone" / "meshes" / region_name

    print(f"\nSaving to: {output_dir}")
    mesh.save(output_dir)

    # Cleanup
    bathymetry.close()

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")

    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="Generate surf zone mesh using NOAA Topobathy DEM (VRT remote access)",
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
    )


if __name__ == "__main__":
    main()
