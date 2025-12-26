"""
NCEI Coastal Relief Model (CRM) data fetcher.

The NCEI CRM provides high-resolution (~90m / 3 arc-second) bathymetry and topography
for the US coasts. This is used for the inner SWAN domain (3km → 500m offshore).

Data access via THREDDS/OPeNDAP:
https://www.ngdc.noaa.gov/thredds/catalog/crm/catalog.html

California is covered by CRM volumes 6, 7, and 8:
- Vol 6: Southern California
- Vol 7: Central California
- Vol 8: Northern California

Usage:
    python -m data.pipelines.bathymetry.ncei_fetcher --region california
    python -m data.pipelines.bathymetry.ncei_fetcher --bounds 32.0 42.0 -126.0 -117.0
    python -m data.pipelines.bathymetry.ncei_fetcher --region socal --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from data.pipelines.bathymetry.config import (
    NCEI_DIR,
    REGIONS,
    SWAN_CONFIG,
)


class NCEIFetcher:
    """
    Fetches NCEI Coastal Relief Model data for the inner SWAN domain.

    The CRM provides 3 arc-second (~90m) bathymetry/topography.
    We use this for the 3km → 500m offshore region where higher resolution is needed.
    """

    # THREDDS base URLs
    THREDDS_OPENDAP = "https://www.ngdc.noaa.gov/thredds/dodsC/crm/"
    THREDDS_FILESERVER = "https://www.ngdc.noaa.gov/thredds/fileServer/crm/"

    # California-specific regional files (preferred - higher quality)
    CALIFORNIA_FILES: Dict[str, Dict] = {
        "socal_3as": {
            "file": "crm_socal_3as_vers2.nc",
            "name": "Southern California 3 arc-second",
            "lat_range": (31.0, 35.0),
            "lon_range": (-122.0, -117.0),
            "resolution": "3 arc-second (~90m)",
            "size_mb": 524,
        },
        "socal_1as": {
            "file": "crm_socal_1as_vers2.nc",
            "name": "Southern California 1 arc-second",
            "lat_range": (31.0, 35.0),
            "lon_range": (-122.0, -117.0),
            "resolution": "1 arc-second (~30m)",
            "size_mb": 2177,
        },
    }

    # CRM volumes covering California (fallback - full coast coverage)
    CRM_VOLUMES: Dict[str, Dict] = {
        "vol6": {
            "file": "crm_vol6.nc",
            "name": "Southern California (Volume 6)",
            "lat_range": (31.0, 35.0),
            "lon_range": (-122.0, -117.0),
            "size_mb": 346,
        },
        "vol7": {
            "file": "crm_vol7.nc",
            "name": "Central California (Volume 7)",
            "lat_range": (35.0, 39.0),
            "lon_range": (-125.0, -120.0),
            "size_mb": 444,
        },
        "vol8": {
            "file": "crm_vol8.nc",
            "name": "Northern California (Volume 8)",
            "lat_range": (39.0, 43.0),
            "lon_range": (-126.0, -122.0),
            "size_mb": 346,
        },
    }

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize NCEI CRM fetcher.

        Args:
            output_dir: Directory for saving downloaded data. Defaults to NCEI_DIR.
        """
        self.output_dir = output_dir or NCEI_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_volumes_for_bounds(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> List[str]:
        """Determine which CRM volumes are needed for the given bounds."""
        needed = []

        for vol_id, vol_info in self.CRM_VOLUMES.items():
            vol_lat_min, vol_lat_max = vol_info["lat_range"]
            vol_lon_min, vol_lon_max = vol_info["lon_range"]

            # Check for overlap
            lat_overlap = (lat_min <= vol_lat_max) and (lat_max >= vol_lat_min)
            lon_overlap = (lon_min <= vol_lon_max) and (lon_max >= vol_lon_min)

            if lat_overlap and lon_overlap:
                needed.append(vol_id)

        return needed

    def fetch_region(
        self,
        region_name: str,
        buffer_deg: float = 0.1,
        dry_run: bool = False,
    ) -> Optional[Path]:
        """
        Fetch NCEI CRM data for a predefined region.

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
            output_name=f"crm_{region_name}.nc",
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
        Fetch NCEI CRM data for specified bounds.

        Automatically determines which CRM volumes are needed and fetches
        data from each, merging if necessary.

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
            raise ImportError("xarray is required for NCEI data access. Install with: pip install xarray netCDF4")

        if output_name is None:
            output_name = f"crm_{lat_min:.1f}_{lat_max:.1f}_{lon_min:.1f}_{lon_max:.1f}.nc"

        output_path = self.output_dir / output_name

        # Determine which volumes we need
        volumes_needed = self._get_volumes_for_bounds(lat_min, lat_max, lon_min, lon_max)

        if not volumes_needed:
            print(f"Warning: No CRM volumes cover the specified region.")
            print(f"CRM only covers US coastal waters.")
            return None

        # Calculate approximate file size
        lat_cells = int((lat_max - lat_min) * 1200)  # 3 arc-sec = 1200 cells per degree
        lon_cells = int((lon_max - lon_min) * 1200)
        approx_size_mb = (lat_cells * lon_cells * 4) / (1024 * 1024)  # float32 = 4 bytes

        print(f"NCEI Coastal Relief Model Data Fetch")
        print(f"=" * 50)
        print(f"Region bounds:")
        print(f"  Latitude:  {lat_min:.2f}°N to {lat_max:.2f}°N")
        print(f"  Longitude: {lon_min:.2f}°E to {lon_max:.2f}°E")
        print(f"Grid dimensions: ~{lat_cells} x {lon_cells} cells")
        print(f"Resolution: 3 arc-seconds (~90m)")
        print(f"Approximate size: {approx_size_mb:.1f} MB")
        print(f"CRM volumes needed: {', '.join(volumes_needed)}")
        print(f"Output: {output_path}")
        print()

        if dry_run:
            print("[DRY RUN] Would download from NCEI THREDDS:")
            for vol in volumes_needed:
                url = f"{self.THREDDS_BASE}{self.CRM_VOLUMES[vol]['file']}"
                print(f"  - {self.CRM_VOLUMES[vol]['name']}: {url}")
            return None

        # Fetch data from each volume
        datasets = []
        for vol_id in volumes_needed:
            vol_info = self.CRM_VOLUMES[vol_id]
            print(f"\nFetching {vol_info['name']} ({vol_id})...")

            ds = self._fetch_volume(
                vol_id=vol_id,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
            )

            if ds is not None:
                datasets.append(ds)

        if not datasets:
            print("No data retrieved. Check network connection and bounds.")
            return None

        # Merge datasets if multiple volumes
        if len(datasets) == 1:
            merged = datasets[0]
        else:
            print(f"\nMerging {len(datasets)} datasets...")
            merged = xr.merge(datasets, compat="override")

        # Save to file
        print(f"\nSaving to: {output_path}")
        merged.to_netcdf(output_path)

        # Clean up
        for ds in datasets:
            ds.close()

        # Save metadata
        self._save_metadata(output_path, lat_min, lat_max, lon_min, lon_max, volumes_needed)

        print(f"\nSuccess! Data saved to: {output_path}")
        return output_path

    def _download_file_direct(
        self,
        filename: str,
        output_path: Path,
    ) -> bool:
        """
        Download CRM file directly via HTTP (fileServer).

        This bypasses OPeNDAP and downloads the full file.
        """
        import subprocess

        url = f"{self.THREDDS_FILESERVER}{filename}"
        print(f"  Direct download: {url}")
        print(f"  Destination: {output_path}")

        try:
            # Use curl with progress bar
            result = subprocess.run(
                ["curl", "-L", "-o", str(output_path), "--progress-bar", url],
                capture_output=False,
                check=True,
            )
            return output_path.exists()
        except subprocess.CalledProcessError as e:
            print(f"  Download failed: {e}")
            return False
        except FileNotFoundError:
            # curl not available, try wget
            try:
                result = subprocess.run(
                    ["wget", "-O", str(output_path), "--show-progress", url],
                    capture_output=False,
                    check=True,
                )
                return output_path.exists()
            except Exception as e2:
                print(f"  Download failed (wget): {e2}")
                return False

    def _fetch_volume(
        self,
        vol_id: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        use_direct_download: bool = True,
    ) -> Optional[xr.Dataset]:
        """
        Fetch a subset from a specific CRM volume.

        Args:
            vol_id: Volume identifier (e.g., 'vol6')
            lat_min, lat_max, lon_min, lon_max: Bounds to fetch
            use_direct_download: If True, download full file then subset locally

        Returns:
            xarray Dataset or None if fetch fails
        """
        vol_info = self.CRM_VOLUMES[vol_id]

        # Clip bounds to volume extent
        vol_lat_min, vol_lat_max = vol_info["lat_range"]
        vol_lon_min, vol_lon_max = vol_info["lon_range"]

        fetch_lat_min = max(lat_min, vol_lat_min)
        fetch_lat_max = min(lat_max, vol_lat_max)
        fetch_lon_min = max(lon_min, vol_lon_min)
        fetch_lon_max = min(lon_max, vol_lon_max)

        if use_direct_download:
            # Download full file then subset locally
            cache_path = self.output_dir / f"cache_{vol_info['file']}"

            if not cache_path.exists():
                print(f"\nDownloading {vol_info['name']}...")
                print(f"  Size: ~{vol_info.get('size_mb', 'unknown')} MB")
                success = self._download_file_direct(vol_info['file'], cache_path)
                if not success:
                    return None

            try:
                print(f"  Opening local file: {cache_path}")
                ds = xr.open_dataset(cache_path)
            except Exception as e:
                print(f"  Error opening {cache_path}: {e}")
                return None

        else:
            # Try OPeNDAP (often fails)
            url = f"{self.THREDDS_OPENDAP}{vol_info['file']}"
            try:
                print(f"  Opening via OPeNDAP: {url}")
                ds = xr.open_dataset(url)
            except Exception as e:
                print(f"  OPeNDAP failed: {e}")
                print(f"  Falling back to direct download...")
                return self._fetch_volume(vol_id, lat_min, lat_max, lon_min, lon_max, use_direct_download=True)

        # Find coordinate names
        lat_coord = None
        lon_coord = None

        for name in ["lat", "latitude", "y"]:
            if name in ds.coords:
                lat_coord = name
                break

        for name in ["lon", "longitude", "x"]:
            if name in ds.coords:
                lon_coord = name
                break

        if lat_coord is None or lon_coord is None:
            print(f"  Warning: Could not identify coordinates. Available: {list(ds.coords)}")
            lat_coord = lat_coord or "y"
            lon_coord = lon_coord or "x"

        print(f"  Subsetting: lat [{fetch_lat_min:.2f}, {fetch_lat_max:.2f}], lon [{fetch_lon_min:.2f}, {fetch_lon_max:.2f}]")

        try:
            # Subset
            subset = ds.sel(
                **{
                    lat_coord: slice(fetch_lat_min, fetch_lat_max),
                    lon_coord: slice(fetch_lon_min, fetch_lon_max),
                }
            )

            # Load into memory
            subset = subset.load()
            print(f"  Retrieved shape: {dict(subset.dims)}")

            ds.close()
            return subset

        except Exception as e:
            print(f"  Error subsetting {vol_id}: {e}")
            ds.close()
            return None

    def _save_metadata(
        self,
        data_path: Path,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        volumes: List[str],
    ):
        """Save metadata JSON alongside the data file."""
        metadata = {
            "source": "NCEI Coastal Relief Model",
            "resolution": "3 arc-seconds (~90m)",
            "bounds": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
            "swan_domain": "inner (3km → 500m offshore)",
            "volumes_used": volumes,
            "downloaded": datetime.now().isoformat(),
            "file": str(data_path),
            "thredds_base": self.THREDDS_FILESERVER,
        }

        metadata_path = data_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata: {metadata_path}")

    def validate_data(self, data_path: Path) -> dict:
        """
        Validate downloaded NCEI CRM data.

        Returns dict with validation results.
        """
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray required for validation")

        results = {"valid": True, "errors": [], "warnings": [], "stats": {}}

        try:
            ds = xr.open_dataset(data_path)

            # Check for elevation variable
            elev_var = None
            for var in ["z", "elevation", "topo", "Band1"]:
                if var in ds.data_vars:
                    elev_var = var
                    break

            if elev_var is None:
                results["errors"].append(f"No elevation variable found. Available: {list(ds.data_vars)}")
                results["valid"] = False
                return results

            elev = ds[elev_var].values

            # Calculate statistics
            results["stats"] = {
                "shape": list(elev.shape),
                "min_depth": float(np.nanmin(elev)),
                "max_depth": float(np.nanmax(elev)),
                "mean_depth": float(np.nanmean(elev)),
                "nan_fraction": float(np.isnan(elev).sum() / elev.size),
                "variable_name": elev_var,
            }

            # Check for issues
            if results["stats"]["nan_fraction"] > 0.5:
                results["warnings"].append("More than 50% of data is NaN")

            if results["stats"]["min_depth"] > 0:
                results["warnings"].append("No negative (underwater) values found - may be land-only region")

            ds.close()

        except Exception as e:
            results["errors"].append(str(e))
            results["valid"] = False

        return results

    def list_available_volumes(self):
        """Print information about available CRM volumes."""
        print("NCEI Coastal Relief Model - Available Volumes")
        print("=" * 60)
        print()
        for vol_id, info in self.CRM_VOLUMES.items():
            lat_min, lat_max = info["lat_range"]
            lon_min, lon_max = info["lon_range"]
            url = f"{self.THREDDS_BASE}{info['file']}"
            print(f"{vol_id}: {info['name']}")
            print(f"  Lat: {lat_min}° to {lat_max}°")
            print(f"  Lon: {lon_min}° to {lon_max}°")
            print(f"  URL: {url}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NCEI Coastal Relief Model data for SWAN model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available CRM volumes
    python -m data.pipelines.bathymetry.ncei_fetcher --list

    # Fetch California region (dry run first)
    python -m data.pipelines.bathymetry.ncei_fetcher --region california --dry-run

    # Fetch Southern California
    python -m data.pipelines.bathymetry.ncei_fetcher --region socal

    # Fetch with custom bounds
    python -m data.pipelines.bathymetry.ncei_fetcher --bounds 32 42 -126 -117

    # Validate existing data
    python -m data.pipelines.bathymetry.ncei_fetcher --validate crm_california.nc
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
        default=0.1,
        help="Buffer in degrees around region (default: 0.1)",
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
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available CRM volumes",
    )

    args = parser.parse_args()

    fetcher = NCEIFetcher()

    if args.list:
        fetcher.list_available_volumes()
        return

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
        print("\nNo region or bounds specified. Use --region, --bounds, or --list.")


if __name__ == "__main__":
    main()
