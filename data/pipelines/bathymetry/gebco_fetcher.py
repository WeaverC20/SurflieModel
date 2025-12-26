"""
GEBCO 2024 bathymetry data fetcher.

GEBCO (General Bathymetric Chart of the Oceans) provides global ocean bathymetry
at 15 arc-second (~450m) resolution. This is used for the outer SWAN domain
(25km → 3km offshore).

Data access options:
1. Web download: https://download.gebco.net/ (manual region selection)
2. OPeNDAP: https://www.gebco.net/data_and_products/gridded_bathymetry_data/gebco_2024/
3. AWS S3: s3://gebco-grid/ (public bucket)

Usage:
    python -m data.pipelines.bathymetry.gebco_fetcher --region california
    python -m data.pipelines.bathymetry.gebco_fetcher --bounds 32.0 42.0 -126.0 -117.0
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.pipelines.bathymetry.config import (
    GEBCO_DIR,
    REGIONS,
    SWAN_CONFIG,
)


class GEBCOFetcher:
    """
    Fetches GEBCO 2024 bathymetry data for the outer SWAN domain.

    GEBCO provides 15 arc-second (~450m) global bathymetry.
    We use this for the 25km → 3km offshore region where lower resolution is acceptable.
    """

    # GEBCO 2024 OPeNDAP URL (if available)
    OPENDAP_URL = "https://www.gebco.net/data_and_products/gebco_2024/gebco_2024_sub_ice_topo.nc"

    # AWS S3 bucket (public, no credentials needed)
    S3_BUCKET = "gebco-grid"
    S3_KEY = "GEBCO_2024/gebco_2024_sub_ice_topo.nc"

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize GEBCO fetcher.

        Args:
            output_dir: Directory for saving downloaded data. Defaults to GEBCO_DIR.
        """
        self.output_dir = output_dir or GEBCO_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_region(
        self,
        region_name: str,
        buffer_deg: float = 0.5,
        dry_run: bool = False,
    ) -> Optional[Path]:
        """
        Fetch GEBCO data for a predefined region.

        Args:
            region_name: Name of region (e.g., 'california', 'socal')
            buffer_deg: Buffer in degrees to add around region bounds
            dry_run: If True, only print what would be downloaded

        Returns:
            Path to downloaded NetCDF file, or None if dry_run
        """
        if region_name not in REGIONS:
            available = ", ".join(REGIONS.keys())
            raise ValueError(f"Unknown region '{region_name}'. Available: {available}")

        lat_min, lat_max, lon_min, lon_max = REGIONS[region_name]

        # Add buffer
        lat_min -= buffer_deg
        lat_max += buffer_deg
        lon_min -= buffer_deg
        lon_max += buffer_deg

        return self.fetch_bounds(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            output_name=f"gebco_2024_{region_name}.nc",
            dry_run=dry_run,
        )

    def fetch_bounds(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        output_name: Optional[str] = None,
        dry_run: bool = False,
    ) -> Optional[Path]:
        """
        Fetch GEBCO data for specified bounds.

        Args:
            lat_min: Southern latitude boundary
            lat_max: Northern latitude boundary
            lon_min: Western longitude boundary
            lon_max: Eastern longitude boundary
            output_name: Output filename (auto-generated if not provided)
            dry_run: If True, only print what would be downloaded

        Returns:
            Path to downloaded NetCDF file, or None if dry_run
        """
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray is required for GEBCO data access. Install with: pip install xarray netCDF4")

        if output_name is None:
            output_name = f"gebco_2024_{lat_min:.1f}_{lat_max:.1f}_{lon_min:.1f}_{lon_max:.1f}.nc"

        output_path = self.output_dir / output_name

        # Calculate approximate file size
        lat_cells = int((lat_max - lat_min) * 240)  # 15 arc-sec = 240 cells per degree
        lon_cells = int((lon_max - lon_min) * 240)
        approx_size_mb = (lat_cells * lon_cells * 2) / (1024 * 1024)  # int16 = 2 bytes

        print(f"GEBCO 2024 Data Fetch")
        print(f"=" * 50)
        print(f"Region bounds:")
        print(f"  Latitude:  {lat_min:.2f}°N to {lat_max:.2f}°N")
        print(f"  Longitude: {lon_min:.2f}°E to {lon_max:.2f}°E")
        print(f"Grid dimensions: {lat_cells} x {lon_cells} cells")
        print(f"Resolution: 15 arc-seconds (~450m)")
        print(f"Approximate size: {approx_size_mb:.1f} MB")
        print(f"Output: {output_path}")
        print()

        if dry_run:
            print("[DRY RUN] Would download data from GEBCO")
            self._print_download_instructions(lat_min, lat_max, lon_min, lon_max)
            return None

        # Try different data access methods
        success = False

        # Method 1: Try OPeNDAP (best for subsetting)
        if not success:
            success = self._try_opendap(lat_min, lat_max, lon_min, lon_max, output_path)

        # Method 2: Check for local full GEBCO file
        if not success:
            success = self._try_local_file(lat_min, lat_max, lon_min, lon_max, output_path)

        # Method 3: Print manual download instructions
        if not success:
            print("\nAutomatic download not available.")
            self._print_download_instructions(lat_min, lat_max, lon_min, lon_max)
            return None

        # Save metadata
        self._save_metadata(output_path, lat_min, lat_max, lon_min, lon_max)

        return output_path

    def _try_opendap(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        output_path: Path,
    ) -> bool:
        """Try to fetch data via OPeNDAP."""
        print("Attempting OPeNDAP access...")

        try:
            # GEBCO OPeNDAP endpoint
            # Note: The actual URL may need to be updated based on current GEBCO infrastructure
            opendap_url = "https://www.gebco.net/data_and_products/gridded_bathymetry_data/gebco_2024/gebco_2024_sub_ice_topo.nc"

            ds = xr.open_dataset(opendap_url, engine="netcdf4")

            # Subset to region
            subset = ds.sel(
                lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max),
            )

            # Save to local file
            subset.to_netcdf(output_path)
            print(f"Successfully downloaded via OPeNDAP: {output_path}")
            ds.close()
            return True

        except Exception as e:
            print(f"OPeNDAP access failed: {e}")
            return False

    def _try_local_file(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        output_path: Path,
    ) -> bool:
        """Try to subset from a local full GEBCO file."""
        # Check for common local file locations
        possible_paths = [
            self.output_dir / "gebco_2024.nc",
            self.output_dir / "GEBCO_2024.nc",
            self.output_dir / "gebco_2024_sub_ice_topo.nc",
            Path.home() / "data" / "gebco" / "gebco_2024.nc",
        ]

        for local_path in possible_paths:
            if local_path.exists():
                print(f"Found local GEBCO file: {local_path}")
                try:
                    ds = xr.open_dataset(local_path)
                    subset = ds.sel(
                        lat=slice(lat_min, lat_max),
                        lon=slice(lon_min, lon_max),
                    )
                    subset.to_netcdf(output_path)
                    print(f"Extracted subset to: {output_path}")
                    ds.close()
                    return True
                except Exception as e:
                    print(f"Failed to process local file: {e}")

        return False

    def _print_download_instructions(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ):
        """Print manual download instructions."""
        print()
        print("=" * 60)
        print("MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 60)
        print()
        print("1. Go to: https://download.gebco.net/")
        print()
        print("2. Select 'GEBCO_2024 Grid' as the data product")
        print()
        print("3. Enter these coordinates:")
        print(f"   North: {lat_max:.2f}")
        print(f"   South: {lat_min:.2f}")
        print(f"   West:  {lon_min:.2f}")
        print(f"   East:  {lon_max:.2f}")
        print()
        print("4. Select format: NetCDF")
        print()
        print("5. Download and save to:")
        print(f"   {self.output_dir}/")
        print()
        print("Alternative: Use GEBCO's OpenDAP service or AWS S3 bucket")
        print("  AWS: s3://gebco-grid/GEBCO_2024/")
        print()

    def _save_metadata(
        self,
        data_path: Path,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ):
        """Save metadata JSON alongside the data file."""
        metadata = {
            "source": "GEBCO 2024",
            "resolution": "15 arc-seconds (~450m)",
            "bounds": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
            "swan_domain": "outer (25km → 3km offshore)",
            "downloaded": datetime.now().isoformat(),
            "file": str(data_path),
        }

        metadata_path = data_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")

    def validate_data(self, data_path: Path) -> dict:
        """
        Validate downloaded GEBCO data.

        Returns dict with validation results.
        """
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray required for validation")

        results = {"valid": True, "errors": [], "warnings": [], "stats": {}}

        try:
            ds = xr.open_dataset(data_path)

            # Check for elevation variable
            elev_var = None
            for var in ["elevation", "z", "Band1"]:
                if var in ds.data_vars:
                    elev_var = var
                    break

            if elev_var is None:
                results["errors"].append("No elevation variable found")
                results["valid"] = False
                return results

            elev = ds[elev_var].values

            # Calculate statistics
            results["stats"] = {
                "shape": elev.shape,
                "min_depth": float(np.nanmin(elev)),
                "max_depth": float(np.nanmax(elev)),
                "mean_depth": float(np.nanmean(elev)),
                "nan_fraction": float(np.isnan(elev).sum() / elev.size),
            }

            # Check for issues
            if results["stats"]["nan_fraction"] > 0.5:
                results["warnings"].append("More than 50% of data is NaN")

            if results["stats"]["min_depth"] > 0:
                results["warnings"].append("No negative (underwater) values found")

            ds.close()

        except Exception as e:
            results["errors"].append(str(e))
            results["valid"] = False

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GEBCO 2024 bathymetry data for SWAN model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fetch California region (dry run first)
    python -m data.pipelines.bathymetry.gebco_fetcher --region california --dry-run

    # Fetch with custom bounds
    python -m data.pipelines.bathymetry.gebco_fetcher --bounds 32 42 -126 -117

    # Validate existing data
    python -m data.pipelines.bathymetry.gebco_fetcher --validate gebco_2024_california.nc
""",
    )

    parser.add_argument(
        "--region",
        type=str,
        choices=list(REGIONS.keys()),
        help="Predefined region to fetch",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=4,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Custom bounds (lat_min lat_max lon_min lon_max)",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.5,
        help="Buffer in degrees around region (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output filename (optional)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--validate",
        type=str,
        metavar="FILE",
        help="Validate existing data file",
    )

    args = parser.parse_args()

    fetcher = GEBCOFetcher()

    if args.validate:
        print(f"Validating: {args.validate}")
        results = fetcher.validate_data(Path(args.validate))
        print(json.dumps(results, indent=2))
        return

    if args.region:
        fetcher.fetch_region(
            region_name=args.region,
            buffer_deg=args.buffer,
            dry_run=args.dry_run,
        )
    elif args.bounds:
        lat_min, lat_max, lon_min, lon_max = args.bounds
        fetcher.fetch_bounds(
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            output_name=args.output,
            dry_run=args.dry_run,
        )
    else:
        parser.print_help()
        print("\nNo region or bounds specified. Use --region or --bounds.")


if __name__ == "__main__":
    main()
