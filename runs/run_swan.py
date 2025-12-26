#!/usr/bin/env python3
"""
Step 2: Run SWAN Computation

Uses locally stored WW3 data to:
1. Generate SWAN boundary conditions
2. Prepare SWAN input files
3. Run SWAN model (if installed)
4. Store SWAN output

Usage:
    python runs/run_swan.py
    python runs/run_swan.py --domain california_swan_2000m
    python runs/run_swan.py --prepare-only  # Just prepare files, don't run
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False


# Directories
PROJECT_ROOT = Path(__file__).parent.parent
DOMAINS_DIR = PROJECT_ROOT / "data" / "grids" / "swan_domains"
RUNS_OUTPUT_DIR = PROJECT_ROOT / "data" / "swan_runs"
WW3_STORE_PATH = PROJECT_ROOT / "data" / "zarr" / "forecasts" / "waves" / "ww3.zarr"


class SwanRunner:
    """
    Manages SWAN model runs using stored WW3 data.
    """

    # Channel Islands definitions: (name, center_lat, center_lon, radius_km)
    CHANNEL_ISLANDS = [
        ("catalina", 33.38, -118.42, 16.0),        # Santa Catalina Island
        ("san_clemente", 32.90, -118.50, 20.0),    # San Clemente Island
        ("san_nicolas", 33.25, -119.50, 12.0),     # San Nicolas Island
        ("santa_barbara", 33.47, -119.03, 6.0),    # Santa Barbara Island (small)
        ("anacapa", 34.01, -119.40, 5.0),          # Anacapa Island (small)
        ("santa_cruz", 34.02, -119.73, 18.0),      # Santa Cruz Island
        ("santa_rosa", 33.96, -120.05, 14.0),      # Santa Rosa Island
        ("san_miguel", 34.03, -120.36, 10.0),      # San Miguel Island
    ]

    # WW3 native resolution: 0.25° (~28km)
    WW3_NATIVE_SPACING_KM = 28.0

    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.domain_dir = DOMAINS_DIR / domain_name

        if not self.domain_dir.exists():
            raise FileNotFoundError(f"Domain not found: {domain_name}")

        # Load domain config
        with open(self.domain_dir / "config.json") as f:
            self.config = json.load(f)

        # Create run output directory
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = RUNS_OUTPUT_DIR / domain_name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def load_ww3_data(self) -> Optional[xr.Dataset]:
        """Load stored WW3 data."""
        if not WW3_STORE_PATH.exists():
            print(f"Error: No WW3 data found at {WW3_STORE_PATH}")
            print("Run 'python runs/fetch_ww3.py' first to download WW3 data.")
            return None

        print(f"\nLoading WW3 data from: {WW3_STORE_PATH}")
        ds = xr.open_zarr(WW3_STORE_PATH)

        # Get time range
        times = ds.time.values
        print(f"  Time range: {times[0]} to {times[-1]}")
        print(f"  Variables: {list(ds.data_vars)}")

        return ds

    def get_boundary_points(
        self,
        side: str = "W",
        spacing_km: float = None,
        include_islands: bool = True,
    ) -> List[Dict]:
        """
        Get boundary points along specified side of domain.

        Args:
            side: Boundary side (W=west/offshore, E=east, N=north, S=south)
            spacing_km: Spacing between boundary points (default: WW3 native ~28km)
            include_islands: Whether to include Channel Islands boundary points
        """
        if spacing_km is None:
            spacing_km = self.WW3_NATIVE_SPACING_KM

        nc_path = Path(self.config["file_nc"])
        ds = xr.open_dataset(nc_path)

        lats = ds["lat"].values
        lons = ds["lon"].values
        elevation = ds["elevation_masked"].values
        ds.close()

        points = []

        if side == "W":
            spacing_deg = spacing_km / 111.0
            lat_step = max(1, int(spacing_deg / np.abs(lats[1] - lats[0])))

            for i in range(0, len(lats), lat_step):
                lat = lats[i]
                row = elevation[i, :]
                wet_indices = np.where(~np.isnan(row) & (row < 0))[0]

                if len(wet_indices) > 0:
                    j = wet_indices[0]
                    lon = lons[j]
                    depth = -row[j]

                    points.append({
                        "index": len(points),
                        "lat": float(lat),
                        "lon": float(lon),
                        "depth_m": float(depth),
                    })

        # Add island boundary points if requested
        if include_islands:
            island_points = self.get_island_boundary_points(
                spacing_km, start_index=len(points)
            )
            points.extend(island_points)

        return points

    def get_island_boundary_points(
        self,
        spacing_km: float = None,
        start_index: int = 0,
    ) -> List[Dict]:
        """
        Get boundary points around Channel Islands within the domain.

        Args:
            spacing_km: Angular spacing between points around islands
            start_index: Starting index for point numbering
        """
        if spacing_km is None:
            spacing_km = self.WW3_NATIVE_SPACING_KM

        nc_path = Path(self.config["file_nc"])
        ds = xr.open_dataset(nc_path)
        domain_lats = ds["lat"].values
        domain_lons = ds["lon"].values
        elevation = ds["elevation"].values  # Full elevation, not masked
        ds.close()

        # Domain bounds
        lat_min, lat_max = domain_lats.min(), domain_lats.max()
        lon_min, lon_max = domain_lons.min(), domain_lons.max()

        points = []
        point_idx = start_index

        for island_name, center_lat, center_lon, radius_km in self.CHANNEL_ISLANDS:
            # Check if island center is within domain bounds
            if not (lat_min <= center_lat <= lat_max and lon_min <= center_lon <= lon_max):
                continue

            # Calculate number of points around the island
            circumference_km = 2 * np.pi * radius_km
            n_points = max(4, int(circumference_km / spacing_km))

            # Generate points around the island
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points

                # Convert radius to degrees
                radius_deg_lat = radius_km / 111.0
                radius_deg_lon = radius_km / (111.0 * np.cos(np.radians(center_lat)))

                pt_lat = center_lat + radius_deg_lat * np.sin(angle)
                pt_lon = center_lon + radius_deg_lon * np.cos(angle)

                # Check point is within domain
                if not (lat_min <= pt_lat <= lat_max and lon_min <= pt_lon <= lon_max):
                    continue

                # Look up depth at this location
                lat_idx = np.argmin(np.abs(domain_lats - pt_lat))
                lon_idx = np.argmin(np.abs(domain_lons - pt_lon))
                elev = elevation[lat_idx, lon_idx]

                # Skip if on land, try to find nearest wet cell
                if np.isnan(elev) or elev >= 0:
                    depth = self._find_nearest_depth(
                        elevation, domain_lats, domain_lons,
                        pt_lat, pt_lon, search_radius=5
                    )
                    if depth is None:
                        continue
                else:
                    depth = -elev

                points.append({
                    "index": point_idx,
                    "lat": float(pt_lat),
                    "lon": float(pt_lon),
                    "depth_m": float(depth),
                })
                point_idx += 1

        return points

    def _find_nearest_depth(
        self,
        elevation: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        target_lat: float,
        target_lon: float,
        search_radius: int = 5,
    ) -> Optional[float]:
        """Find depth of nearest wet cell within search radius."""
        lat_idx = np.argmin(np.abs(lats - target_lat))
        lon_idx = np.argmin(np.abs(lons - target_lon))

        for r in range(1, search_radius + 1):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    i = lat_idx + di
                    j = lon_idx + dj
                    if 0 <= i < len(lats) and 0 <= j < len(lons):
                        elev = elevation[i, j]
                        if not np.isnan(elev) and elev < 0:
                            return -elev
        return None

    def extract_boundary_conditions(
        self,
        ww3_ds: xr.Dataset,
        boundary_points: List[Dict],
    ) -> Dict[str, List[Dict]]:
        """
        Extract WW3 data at boundary points for each time step.
        """
        from scipy.interpolate import RegularGridInterpolator

        print(f"\nExtracting boundary conditions at {len(boundary_points)} points...")

        ww3_lats = ww3_ds["lat"].values
        ww3_lons = ww3_ds["lon"].values
        times = ww3_ds["time"].values

        # Ensure ascending order
        lat_ascending = ww3_lats[0] < ww3_lats[-1]
        lon_ascending = ww3_lons[0] < ww3_lons[-1]

        if not lat_ascending:
            ww3_lats = ww3_lats[::-1]
        if not lon_ascending:
            ww3_lons = ww3_lons[::-1]

        boundary_data = {}

        for t_idx, time_val in enumerate(times):
            time_str = str(time_val)[:19]

            # Get data for this time step
            hs = ww3_ds["hs"].isel(time=t_idx).values
            tp = ww3_ds["tp"].isel(time=t_idx).values
            dp = ww3_ds["dp"].isel(time=t_idx).values

            if not lat_ascending:
                hs = hs[::-1, :]
                tp = tp[::-1, :]
                dp = dp[::-1, :]
            if not lon_ascending:
                hs = hs[:, ::-1]
                tp = tp[:, ::-1]
                dp = dp[:, ::-1]

            # Fill NaN with nearest neighbor
            from scipy import ndimage

            def fill_nan(arr):
                mask = np.isnan(arr)
                if mask.all():
                    return arr
                indices = ndimage.distance_transform_edt(
                    mask, return_distances=False, return_indices=True
                )
                return arr[tuple(indices)]

            hs_filled = fill_nan(hs)
            tp_filled = fill_nan(tp)
            dp_filled = fill_nan(dp)

            # Create interpolators
            interp_hs = RegularGridInterpolator(
                (ww3_lats, ww3_lons), hs_filled,
                method='linear', bounds_error=False, fill_value=np.nan
            )
            interp_tp = RegularGridInterpolator(
                (ww3_lats, ww3_lons), tp_filled,
                method='linear', bounds_error=False, fill_value=np.nan
            )
            interp_dp = RegularGridInterpolator(
                (ww3_lats, ww3_lons), dp_filled,
                method='linear', bounds_error=False, fill_value=np.nan
            )

            # Interpolate to boundary points
            point_data = []
            for pt in boundary_points:
                coords = np.array([[pt["lat"], pt["lon"]]])
                point_data.append({
                    "index": pt["index"],
                    "lat": pt["lat"],
                    "lon": pt["lon"],
                    "hs": float(interp_hs(coords)[0]),
                    "tp": float(interp_tp(coords)[0]),
                    "dir": float(interp_dp(coords)[0]),
                    "spreading": 25.0,
                })

            boundary_data[time_str] = point_data

            # Progress
            hs_vals = [p["hs"] for p in point_data if not np.isnan(p["hs"])]
            if hs_vals:
                print(f"  {time_str}: Hs = {np.mean(hs_vals):.2f}m (range: {np.min(hs_vals):.1f} - {np.max(hs_vals):.1f}m)")

        return boundary_data

    def write_tpar_file(
        self,
        boundary_data: Dict[str, List[Dict]],
    ) -> Path:
        """Write SWAN TPAR boundary condition file."""
        tpar_path = self.run_dir / "boundary.tpar"

        with open(tpar_path, "w") as f:
            f.write("TPAR\n")
            f.write(f"$ SWAN boundary conditions from WW3\n")
            f.write(f"$ Domain: {self.domain_name}\n")
            f.write(f"$ Run ID: {self.run_id}\n")
            f.write(f"$ Generated: {datetime.now().isoformat()}\n")
            f.write("$\n")

            for time_str, points in sorted(boundary_data.items()):
                # Average along boundary
                hs_vals = [p["hs"] for p in points if not np.isnan(p["hs"])]
                tp_vals = [p["tp"] for p in points if not np.isnan(p["tp"])]
                dir_vals = [p["dir"] for p in points if not np.isnan(p["dir"])]

                if not hs_vals:
                    continue

                avg_hs = np.mean(hs_vals)
                avg_tp = np.mean(tp_vals)
                avg_dir = self._circular_mean(dir_vals)

                # Format time for SWAN: yyyymmdd.hhmmss
                dt = datetime.fromisoformat(time_str.replace("T", " ").split(".")[0])
                swan_time = dt.strftime("%Y%m%d.%H%M%S")

                f.write(f"{swan_time} {avg_hs:.2f} {avg_tp:.1f} {avg_dir:.1f} 25.0\n")

        print(f"\nWritten TPAR file: {tpar_path}")
        return tpar_path

    def prepare_swan_input(self, tpar_path: Path) -> Path:
        """
        Prepare SWAN input file for this run.
        """
        # Copy grid file to run directory
        grd_src = Path(self.config["file_grd"])
        grd_dst = self.run_dir / grd_src.name
        shutil.copy(grd_src, grd_dst)

        # Read template and customize
        template_path = Path(self.config["file_swan_input"])
        with open(template_path) as f:
            swan_input = f.read()

        # Update boundary condition line
        swan_input = swan_input.replace(
            "BOUNDSPEC SIDE W CCW CONSTANT PAR 2.0 12.0 270.0 25.0",
            f"BOUNDSPEC SIDE W CCW VARIABLE FILE '{tpar_path.name}'"
        )

        # Update grid file path
        swan_input = swan_input.replace(
            f"READINP BOTTOM 1.0 '{grd_src.name}' 1 0 FREE",
            f"READINP BOTTOM 1.0 '{grd_dst.name}' 1 0 FREE"
        )

        # Update project name
        swan_input = swan_input.replace(
            f"PROJECT '{self.domain_name}' 'run01'",
            f"PROJECT '{self.domain_name}' '{self.run_id}'"
        )

        # Write to run directory
        swan_input_path = self.run_dir / "swan_run.swn"
        with open(swan_input_path, "w") as f:
            f.write(swan_input)

        print(f"Written SWAN input: {swan_input_path}")
        return swan_input_path

    def run_swan(self, swan_input_path: Path) -> bool:
        """
        Run SWAN model if installed.
        """
        # Check if SWAN is available
        swan_cmd = shutil.which("swan") or shutil.which("swanrun")

        if swan_cmd is None:
            print("\nSWAN executable not found in PATH.")
            print("To run SWAN manually:")
            print(f"  cd {self.run_dir}")
            print(f"  swan -input {swan_input_path.name}")
            return False

        print(f"\nRunning SWAN: {swan_cmd}")
        print(f"  Working directory: {self.run_dir}")

        try:
            result = subprocess.run(
                [swan_cmd, "-input", swan_input_path.name],
                cwd=self.run_dir,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode == 0:
                print("SWAN completed successfully")
                return True
            else:
                print(f"SWAN failed with code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("SWAN timed out after 1 hour")
            return False
        except Exception as e:
            print(f"Error running SWAN: {e}")
            return False

    def save_boundary_points(self, boundary_points: List[Dict]):
        """Save boundary points to domain boundary directory for API access."""
        boundary_dir = self.domain_dir / "boundary"
        boundary_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "domain": self.domain_name,
            "n_points": len(boundary_points),
            "points": boundary_points,
            "created_at": datetime.now().isoformat(),
        }

        loc_path = boundary_dir / "boundary_points.json"
        with open(loc_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved boundary points: {loc_path}")

    def save_run_metadata(
        self,
        ww3_time_range: tuple,
        boundary_points: List[Dict],
        tpar_path: Path,
        swan_input_path: Path,
        swan_completed: bool,
    ):
        """Save run metadata for later reference."""
        metadata = {
            "run_id": self.run_id,
            "domain_name": self.domain_name,
            "created_at": datetime.now().isoformat(),
            "ww3_time_range": {
                "start": str(ww3_time_range[0]),
                "end": str(ww3_time_range[1]),
            },
            "n_boundary_points": len(boundary_points),
            "files": {
                "tpar": str(tpar_path),
                "swan_input": str(swan_input_path),
                "grid": str(self.run_dir / Path(self.config["file_grd"]).name),
            },
            "swan_completed": swan_completed,
            "domain_config": self.config,
        }

        meta_path = self.run_dir / "run_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved run metadata: {meta_path}")

    def _circular_mean(self, angles: List[float]) -> float:
        """Calculate circular mean of angles in degrees."""
        if not angles:
            return 0.0
        angles_rad = np.radians(angles)
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        return float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360)


def list_domains() -> List[str]:
    """List available SWAN domains."""
    domains = []
    for config_path in DOMAINS_DIR.glob("*/config.json"):
        domains.append(config_path.parent.name)
    return sorted(domains)


def list_runs(domain_name: str) -> List[dict]:
    """List previous runs for a domain."""
    runs = []
    runs_dir = RUNS_OUTPUT_DIR / domain_name
    if runs_dir.exists():
        for meta_path in sorted(runs_dir.glob("*/run_metadata.json"), reverse=True):
            with open(meta_path) as f:
                runs.append(json.load(f))
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Run SWAN using stored WW3 data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run SWAN for default domain
    python runs/run_swan.py

    # Run for specific domain
    python runs/run_swan.py --domain la_area_swan_500m

    # Prepare files only (don't run SWAN)
    python runs/run_swan.py --prepare-only

    # List available domains
    python runs/run_swan.py --list

    # List previous runs
    python runs/run_swan.py --runs california_swan_2000m

Workflow:
    1. First run: python runs/fetch_ww3.py
    2. Then run: python runs/run_swan.py
""",
    )

    parser.add_argument(
        "--domain", "-d",
        type=str,
        default="california_swan_2000m",
        help="SWAN domain name (default: california_swan_2000m)",
    )
    parser.add_argument(
        "--prepare-only", "-p",
        action="store_true",
        help="Only prepare input files, don't run SWAN",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available SWAN domains",
    )
    parser.add_argument(
        "--runs", "-r",
        type=str,
        metavar="DOMAIN",
        help="List previous runs for a domain",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=28.0,
        help="Boundary point spacing in km (default: 28, matching WW3 native resolution)",
    )
    parser.add_argument(
        "--no-islands",
        action="store_true",
        help="Exclude island boundary points (Channel Islands)",
    )

    args = parser.parse_args()

    if args.list:
        domains = list_domains()
        if not domains:
            print("No SWAN domains found.")
            print("Create one with: python -m data.pipelines.bathymetry.swan_domain --generate <region>")
            return

        print("\nAvailable SWAN Domains:")
        print("-" * 40)
        for d in domains:
            print(f"  {d}")
        return

    if args.runs:
        runs = list_runs(args.runs)
        if not runs:
            print(f"No runs found for {args.runs}")
            return

        print(f"\nPrevious runs for {args.runs}:")
        print("-" * 60)
        for run in runs[:10]:  # Show last 10
            status = "completed" if run.get("swan_completed") else "prepared"
            print(f"  {run['run_id']} - {status}")
        return

    # Main workflow
    print("=" * 60)
    print("SWAN Model Run")
    print("=" * 60)
    print(f"  Domain: {args.domain}")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print()

    try:
        runner = SwanRunner(args.domain)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nAvailable domains:")
        for d in list_domains():
            print(f"  {d}")
        sys.exit(1)

    # Load WW3 data
    ww3_ds = runner.load_ww3_data()
    if ww3_ds is None:
        sys.exit(1)

    # Get boundary points
    include_islands = not args.no_islands
    boundary_points = runner.get_boundary_points(
        side="W", spacing_km=args.spacing, include_islands=include_islands
    )
    print(f"\nExtracted {len(boundary_points)} boundary points")
    print(f"  Spacing: {args.spacing}km (WW3 native: 28km)")
    print(f"  Include islands: {include_islands}")

    # Save boundary points for API access
    runner.save_boundary_points(boundary_points)

    # Extract boundary conditions
    boundary_data = runner.extract_boundary_conditions(ww3_ds, boundary_points)

    # Get time range for metadata
    times = list(boundary_data.keys())
    time_range = (times[0], times[-1]) if times else (None, None)

    ww3_ds.close()

    # Write TPAR file
    tpar_path = runner.write_tpar_file(boundary_data)

    # Prepare SWAN input
    swan_input_path = runner.prepare_swan_input(tpar_path)

    # Run SWAN (unless prepare-only)
    swan_completed = False
    if not args.prepare_only:
        swan_completed = runner.run_swan(swan_input_path)

    # Save metadata
    runner.save_run_metadata(
        ww3_time_range=time_range,
        boundary_points=boundary_points,
        tpar_path=tpar_path,
        swan_input_path=swan_input_path,
        swan_completed=swan_completed,
    )

    print()
    print("=" * 60)
    print("Run Preparation Complete")
    print("=" * 60)
    print(f"  Run ID: {runner.run_id}")
    print(f"  Run directory: {runner.run_dir}")
    print()
    print("Files created:")
    print(f"  - {tpar_path.name} (boundary conditions)")
    print(f"  - {swan_input_path.name} (SWAN input)")
    print(f"  - {Path(runner.config['file_grd']).name} (bathymetry grid)")
    print()

    if not args.prepare_only and not swan_completed:
        print("To run SWAN manually:")
        print(f"  cd {runner.run_dir}")
        print(f"  swan -input swan_run.swn")


if __name__ == "__main__":
    main()
