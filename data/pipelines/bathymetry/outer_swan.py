"""
Outer SWAN grid generator from GEBCO bathymetry.

Creates the coarse outer domain for SWAN nested model runs.
This grid covers the offshore region and provides boundary conditions
for the finer inner SWAN domain.

SWAN Grid Notes:
- SWAN uses rectangular grids with uniform spacing
- Land cells are marked with EXCEPTION value (-999 depth)
- SWAN only computes waves on "wet" cells (positive depth)
- The grid should cover ocean areas; land cells are skipped in computation

Usage:
    python -m data.pipelines.bathymetry.outer_swan --generate california
    python -m data.pipelines.bathymetry.outer_swan --generate california --resolution 2000
    python -m data.pipelines.bathymetry.outer_swan --list
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.pipelines.bathymetry.config import (
    GEBCO_DIR,
    REGIONS,
)

# Output directory for outer SWAN grids
GRIDS_DIR = Path(__file__).parent.parent.parent / "grids" / "outer_swan"

# SWAN exception value for land/dry cells
SWAN_EXCEPTION = -999.0


@dataclass
class OuterSwanGrid:
    """Metadata for an outer SWAN grid."""
    name: str
    region: str
    resolution_m: float
    bounds: Tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
    n_lat: int
    n_lon: int
    n_cells: int
    n_wet_cells: int  # cells with actual bathymetry (ocean)
    n_dry_cells: int  # cells marked as land
    file_nc: str
    file_grd: str
    created_at: str


class OuterSwanGenerator:
    """
    Generate outer SWAN domain grids from GEBCO data.

    The outer SWAN grid:
    - Uses GEBCO bathymetry (~450m native resolution)
    - Covers full offshore region up to the coastline
    - Can be downsampled for faster computation
    - Provides boundary conditions for inner SWAN domains
    """

    # GEBCO native resolution
    GEBCO_NATIVE_RES_DEG = 1/240  # 15 arc-seconds
    GEBCO_NATIVE_RES_M = 463     # ~463m at mid-latitudes

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or GRIDS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        region: str,
        resolution_m: Optional[float] = None,
        name: Optional[str] = None,
    ) -> OuterSwanGrid:
        """
        Generate outer SWAN grid for a region.

        Args:
            region: Region name from config (e.g., 'california')
            resolution_m: Target resolution in meters. None = native GEBCO (~463m)
            name: Custom name for the grid. None = auto-generated

        Returns:
            OuterSwanGrid with metadata and file paths
        """
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray required. Install with: pip install xarray netCDF4")

        if region not in REGIONS:
            raise ValueError(f"Unknown region: {region}. Available: {list(REGIONS.keys())}")

        bounds = REGIONS[region]
        lat_min, lat_max, lon_min, lon_max = bounds

        # Determine resolution
        if resolution_m is None:
            resolution_m = self.GEBCO_NATIVE_RES_M
            downsample_factor = 1
        else:
            downsample_factor = max(1, int(resolution_m / self.GEBCO_NATIVE_RES_M))
            resolution_m = self.GEBCO_NATIVE_RES_M * downsample_factor

        # Generate name
        if name is None:
            res_label = f"{int(resolution_m)}m" if resolution_m < 1000 else f"{resolution_m/1000:.1f}km"
            name = f"{region}_outer_{res_label}"

        print("=" * 60)
        print(f"Generating Outer SWAN Grid: {name}")
        print("=" * 60)
        print(f"  Region: {region}")
        print(f"  Bounds: {lat_min:.2f}°N to {lat_max:.2f}°N, {lon_min:.2f}°W to {lon_max:.2f}°W")
        print(f"  Target resolution: {resolution_m:.0f}m")
        if downsample_factor > 1:
            print(f"  Downsample factor: {downsample_factor}x from native GEBCO")

        # Load GEBCO data
        gebco_path = GEBCO_DIR / f"gebco_2024_{region}.nc"
        if not gebco_path.exists():
            gebco_path = GEBCO_DIR / "gebco_2024_california.nc"
        if not gebco_path.exists():
            raise FileNotFoundError(f"GEBCO file not found. Expected: {gebco_path}")

        print(f"\nLoading GEBCO: {gebco_path}")
        ds = xr.open_dataset(gebco_path)

        # Find variable and coordinate names
        elev_var = self._find_var(ds, ["elevation", "z", "topo", "Band1"])
        lat_coord = self._find_coord(ds, ["lat", "latitude", "y"])
        lon_coord = self._find_coord(ds, ["lon", "longitude", "x"])

        # Subset to region bounds
        ds = ds.sel(**{
            lat_coord: slice(lat_min, lat_max),
            lon_coord: slice(lon_min, lon_max),
        })

        # Extract arrays
        elevation = ds[elev_var].values
        lats = ds[lat_coord].values
        lons = ds[lon_coord].values

        print(f"  GEBCO subset: {len(lats)} × {len(lons)} = {len(lats)*len(lons):,} cells")

        # Downsample if requested
        if downsample_factor > 1:
            lats = lats[::downsample_factor]
            lons = lons[::downsample_factor]
            elevation = elevation[::downsample_factor, ::downsample_factor]
            print(f"  After downsample: {len(lats)} × {len(lons)} = {len(lats)*len(lons):,} cells")

        # Calculate actual resolution
        actual_res_lat = np.abs(lats[1] - lats[0]) * 111000 if len(lats) > 1 else resolution_m
        actual_res_lon = np.abs(lons[1] - lons[0]) * 111000 * np.cos(np.radians(np.mean(lats))) if len(lons) > 1 else resolution_m

        print(f"  Actual resolution: {actual_res_lat:.0f}m (lat) × {actual_res_lon:.0f}m (lon)")

        # Count wet vs dry cells
        # GEBCO uses negative values for ocean depth, positive for land elevation
        wet_mask = elevation < 0  # Ocean cells (negative = below sea level)
        n_wet = int(np.sum(wet_mask))
        n_dry = int(np.sum(~wet_mask))
        n_total = len(lats) * len(lons)

        print(f"\n  Wet cells (ocean): {n_wet:,} ({100*n_wet/n_total:.1f}%)")
        print(f"  Dry cells (land):  {n_dry:,} ({100*n_dry/n_total:.1f}%)")

        # Create output directory for this grid
        grid_dir = self.output_dir / name
        grid_dir.mkdir(parents=True, exist_ok=True)

        # Save NetCDF (preserving GEBCO convention: negative = depth)
        nc_path = grid_dir / f"{name}.nc"
        self._save_netcdf(elevation, lats, lons, nc_path, name, region, resolution_m)

        # Save SWAN format (positive depth, exception for land)
        grd_path = grid_dir / f"{name}.grd"
        self._save_swan_grd(elevation, lats, lons, grd_path)

        ds.close()

        # Create metadata
        grid_info = OuterSwanGrid(
            name=name,
            region=region,
            resolution_m=resolution_m,
            bounds=(float(lats.min()), float(lats.max()), float(lons.min()), float(lons.max())),
            n_lat=len(lats),
            n_lon=len(lons),
            n_cells=n_total,
            n_wet_cells=n_wet,
            n_dry_cells=n_dry,
            file_nc=str(nc_path),
            file_grd=str(grd_path),
            created_at=datetime.now().isoformat(),
        )

        # Save metadata
        meta_path = grid_dir / "config.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(grid_info), f, indent=2)

        print(f"\nOutput files:")
        print(f"  NetCDF: {nc_path}")
        print(f"  SWAN:   {grd_path}")
        print(f"  Config: {meta_path}")

        return grid_info

    def _find_var(self, ds: xr.Dataset, candidates: list) -> str:
        """Find variable name from candidates."""
        for name in candidates:
            if name in ds.data_vars:
                return name
        raise ValueError(f"Variable not found. Tried: {candidates}. Available: {list(ds.data_vars)}")

    def _find_coord(self, ds: xr.Dataset, candidates: list) -> str:
        """Find coordinate name from candidates."""
        for name in candidates:
            if name in ds.coords:
                return name
        raise ValueError(f"Coordinate not found. Tried: {candidates}. Available: {list(ds.coords)}")

    def _save_netcdf(
        self,
        elevation: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        path: Path,
        name: str,
        region: str,
        resolution_m: float,
    ):
        """Save grid as NetCDF."""
        ds = xr.Dataset(
            {
                "elevation": (["lat", "lon"], elevation, {
                    "units": "m",
                    "long_name": "Elevation relative to sea level",
                    "positive": "up",
                    "description": "Negative values = ocean depth, Positive = land elevation",
                }),
            },
            coords={
                "lat": ("lat", lats, {"units": "degrees_north", "long_name": "Latitude"}),
                "lon": ("lon", lons, {"units": "degrees_east", "long_name": "Longitude"}),
            },
            attrs={
                "title": f"Outer SWAN bathymetry grid: {name}",
                "source": "GEBCO 2024",
                "region": region,
                "resolution_m": resolution_m,
                "institution": "SurflieModel",
                "created": datetime.now().isoformat(),
                "conventions": "CF-1.8",
            },
        )
        ds.to_netcdf(path)
        print(f"  Saved NetCDF: {path}")

    def _save_swan_grd(
        self,
        elevation: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        path: Path,
    ):
        """
        Save grid in SWAN format.

        SWAN expects:
        - Depth values (positive down into water)
        - Exception value for land cells
        - ASCII format, one row per line
        """
        # Convert elevation to SWAN depth
        # GEBCO: negative = below sea level (ocean), positive = above (land)
        # SWAN: positive depth = water depth, exception = land

        depth = np.where(
            elevation < 0,
            -elevation,          # Ocean: flip sign (depth positive)
            SWAN_EXCEPTION       # Land: use exception value
        )

        with open(path, "w") as f:
            f.write(f"$ Outer SWAN bathymetry grid\n")
            f.write(f"$ Source: GEBCO 2024\n")
            f.write(f"$ Created: {datetime.now().isoformat()}\n")
            f.write(f"$ Grid dimensions: {depth.shape[0]} rows (lat) x {depth.shape[1]} cols (lon)\n")
            f.write(f"$ Lat range: {lats.min():.6f} to {lats.max():.6f}\n")
            f.write(f"$ Lon range: {lons.min():.6f} to {lons.max():.6f}\n")
            f.write(f"$ Exception value for land: {SWAN_EXCEPTION}\n")
            f.write(f"$\n")

            # Write depth values row by row (from south to north)
            for row in depth:
                line = " ".join(f"{val:.2f}" for val in row)
                f.write(line + "\n")

        print(f"  Saved SWAN: {path}")

    def list_grids(self) -> list:
        """List all generated outer SWAN grids."""
        grids = []
        for config_path in self.output_dir.glob("*/config.json"):
            with open(config_path) as f:
                data = json.load(f)
                grids.append(OuterSwanGrid(**data))
        return sorted(grids, key=lambda g: g.resolution_m)

    def get_grid(self, name: str) -> Optional[OuterSwanGrid]:
        """Get a specific grid by name."""
        config_path = self.output_dir / name / "config.json"
        if not config_path.exists():
            return None
        with open(config_path) as f:
            return OuterSwanGrid(**json.load(f))

    def delete_grid(self, name: str) -> bool:
        """Delete a grid."""
        import shutil
        grid_dir = self.output_dir / name
        if grid_dir.exists():
            shutil.rmtree(grid_dir)
            print(f"Deleted: {name}")
            return True
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate outer SWAN grids from GEBCO bathymetry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate at native GEBCO resolution (~463m)
    python -m data.pipelines.bathymetry.outer_swan --generate california

    # Generate at 2km resolution (for faster computation)
    python -m data.pipelines.bathymetry.outer_swan --generate california --resolution 2000

    # Generate at 5km resolution (development/testing)
    python -m data.pipelines.bathymetry.outer_swan --generate california --resolution 5000

    # Custom name
    python -m data.pipelines.bathymetry.outer_swan --generate california --resolution 2000 --name ca_outer_dev

    # List all generated grids
    python -m data.pipelines.bathymetry.outer_swan --list

    # Delete a grid
    python -m data.pipelines.bathymetry.outer_swan --delete california_outer_5.0km

Resolution Guidelines:
    ~500m   - Native GEBCO, highest fidelity, ~5M cells for CA
    ~2km    - Good balance, ~300k cells for CA
    ~5km    - Fast iteration/testing, ~50k cells for CA
    ~10km   - Very fast, ~12k cells for CA
""",
    )

    parser.add_argument(
        "--generate", "-g",
        type=str,
        metavar="REGION",
        help="Generate outer SWAN grid for region",
    )
    parser.add_argument(
        "--resolution", "-r",
        type=float,
        help="Target resolution in meters (default: native GEBCO ~463m)",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        help="Custom name for the grid",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all generated grids",
    )
    parser.add_argument(
        "--delete", "-d",
        type=str,
        metavar="NAME",
        help="Delete a grid",
    )

    args = parser.parse_args()

    generator = OuterSwanGenerator()

    if args.list:
        grids = generator.list_grids()
        if not grids:
            print("No outer SWAN grids generated yet.")
            print("Use: --generate <region> to create one")
            return

        print("\nOuter SWAN Grids")
        print("=" * 80)
        print(f"{'Name':<30} {'Resolution':>10} {'Cells':>12} {'Wet':>10} {'Dry':>10}")
        print("-" * 80)
        for g in grids:
            res = f"{g.resolution_m/1000:.1f}km" if g.resolution_m >= 1000 else f"{g.resolution_m:.0f}m"
            print(f"{g.name:<30} {res:>10} {g.n_cells:>12,} {g.n_wet_cells:>10,} {g.n_dry_cells:>10,}")
        return

    if args.delete:
        generator.delete_grid(args.delete)
        return

    if args.generate:
        generator.generate(
            region=args.generate,
            resolution_m=args.resolution,
            name=args.name,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
