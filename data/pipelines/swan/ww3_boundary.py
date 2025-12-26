"""
WW3 to SWAN Boundary Condition Generator.

Extracts WaveWatch III data at SWAN offshore boundary points and
generates SWAN-compatible boundary condition files.

SWAN Boundary Condition Formats:
1. TPAR (parametric): Time-series of Hs, Tp, Dir, spreading
   - Simpler, works with WW3 bulk parameters
   - Used for most operational applications

2. SPEC (spectral): Full 2D energy spectra
   - More accurate for complex sea states
   - Requires spectral WW3 output (not available from standard GRIB)

This module focuses on TPAR format since WW3 GRIB provides bulk parameters.

Usage:
    python -m data.pipelines.swan.ww3_boundary --generate california_swan_2000m
    python -m data.pipelines.swan.ww3_boundary --generate california_swan_2000m --hours 0 3 6 9 12
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False


# SWAN boundary condition output directory
BC_OUTPUT_DIR = Path(__file__).parent.parent.parent / "grids" / "swan_domains"


@dataclass
class BoundaryPoint:
    """A point along the SWAN offshore boundary."""
    index: int
    lat: float
    lon: float
    depth_m: float


@dataclass
class SwanBoundaryFile:
    """Metadata for generated SWAN boundary condition files."""
    domain_name: str
    n_boundary_points: int
    n_forecast_hours: int
    boundary_side: str  # W, E, N, S
    format_type: str    # TPAR, SPEC
    file_tpar: Optional[str]
    file_locations: str
    ww3_cycle_time: str
    forecast_hours: List[int]
    created_at: str


class WW3BoundaryConditions:
    """
    Extract WW3 data and generate SWAN boundary conditions.

    The workflow:
    1. Load SWAN domain configuration
    2. Identify offshore boundary points (western edge)
    3. Identify island boundary points (Channel Islands)
    4. Fetch WW3 data covering the boundary region
    5. Interpolate WW3 to boundary points
    6. Write SWAN TPAR boundary condition files
    """

    # Channel Islands definitions: (name, center_lat, center_lon, radius_km)
    # These islands are within the SWAN domain and need WW3 boundary conditions
    # Points will be extracted around each island at the specified spacing
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

    def __init__(self, domains_dir: Optional[Path] = None):
        self.domains_dir = domains_dir or BC_OUTPUT_DIR

    def load_domain(self, domain_name: str) -> dict:
        """Load SWAN domain configuration."""
        config_path = self.domains_dir / domain_name / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Domain not found: {domain_name}")

        with open(config_path) as f:
            return json.load(f)

    def get_boundary_points(
        self,
        domain_name: str,
        side: str = "W",
        spacing_km: float = None,
        include_islands: bool = True,
    ) -> List[BoundaryPoint]:
        """
        Get boundary points along specified side of SWAN domain.

        Args:
            domain_name: Name of SWAN domain
            side: Boundary side (W=west/offshore, E=east, N=north, S=south)
            spacing_km: Spacing between boundary points in km (default: WW3 native ~28km)
            include_islands: Whether to include boundary points around Channel Islands

        Returns:
            List of BoundaryPoint objects along the boundary
        """
        if spacing_km is None:
            spacing_km = self.WW3_NATIVE_SPACING_KM

        config = self.load_domain(domain_name)

        # Load the NetCDF to get actual boundary locations
        nc_path = Path(config["file_nc"])
        if not nc_path.exists():
            raise FileNotFoundError(f"Domain NetCDF not found: {nc_path}")

        ds = xr.open_dataset(nc_path)
        lats = ds["lat"].values
        lons = ds["lon"].values
        elevation = ds["elevation_masked"].values
        ds.close()

        points = []
        point_idx = 0

        if side == "W":
            # Western boundary - leftmost wet cells at each latitude
            spacing_deg = spacing_km / 111.0  # Approximate km to degrees
            lat_step = max(1, int(spacing_deg / np.abs(lats[1] - lats[0])))

            for i in range(0, len(lats), lat_step):
                lat = lats[i]
                row = elevation[i, :]

                # Find westernmost wet cell (first non-NaN cell)
                wet_indices = np.where(~np.isnan(row) & (row < 0))[0]
                if len(wet_indices) > 0:
                    j = wet_indices[0]  # Westernmost wet cell
                    lon = lons[j]
                    depth = -row[j]  # Convert to positive depth

                    points.append(BoundaryPoint(
                        index=point_idx,
                        lat=float(lat),
                        lon=float(lon),
                        depth_m=float(depth),
                    ))
                    point_idx += 1

        elif side == "E":
            # Eastern boundary - rightmost wet cells
            spacing_deg = spacing_km / 111.0
            lat_step = max(1, int(spacing_deg / np.abs(lats[1] - lats[0])))

            for i in range(0, len(lats), lat_step):
                lat = lats[i]
                row = elevation[i, :]
                wet_indices = np.where(~np.isnan(row) & (row < 0))[0]
                if len(wet_indices) > 0:
                    j = wet_indices[-1]  # Easternmost wet cell
                    lon = lons[j]
                    depth = -row[j]

                    points.append(BoundaryPoint(
                        index=point_idx,
                        lat=float(lat),
                        lon=float(lon),
                        depth_m=float(depth),
                    ))
                    point_idx += 1

        elif side == "N":
            # Northern boundary
            spacing_deg = spacing_km / (111.0 * np.cos(np.radians(lats[-1])))
            lon_step = max(1, int(spacing_deg / np.abs(lons[1] - lons[0])))

            for j in range(0, len(lons), lon_step):
                lon = lons[j]
                col = elevation[:, j]
                wet_indices = np.where(~np.isnan(col) & (col < 0))[0]
                if len(wet_indices) > 0:
                    i = wet_indices[-1]  # Northernmost wet cell
                    lat = lats[i]
                    depth = -col[i]

                    points.append(BoundaryPoint(
                        index=point_idx,
                        lat=float(lat),
                        lon=float(lon),
                        depth_m=float(depth),
                    ))
                    point_idx += 1

        elif side == "S":
            # Southern boundary
            spacing_deg = spacing_km / (111.0 * np.cos(np.radians(lats[0])))
            lon_step = max(1, int(spacing_deg / np.abs(lons[1] - lons[0])))

            for j in range(0, len(lons), lon_step):
                lon = lons[j]
                col = elevation[:, j]
                wet_indices = np.where(~np.isnan(col) & (col < 0))[0]
                if len(wet_indices) > 0:
                    i = wet_indices[0]  # Southernmost wet cell
                    lat = lats[i]
                    depth = -col[i]

                    points.append(BoundaryPoint(
                        index=point_idx,
                        lat=float(lat),
                        lon=float(lon),
                        depth_m=float(depth),
                    ))
                    point_idx += 1

        # Add island boundary points if requested
        if include_islands:
            island_points = self.get_island_boundary_points(
                domain_name, spacing_km, start_index=len(points)
            )
            points.extend(island_points)

        return points

    def get_island_boundary_points(
        self,
        domain_name: str,
        spacing_km: float = None,
        start_index: int = 0,
    ) -> List[BoundaryPoint]:
        """
        Get boundary points around Channel Islands within the domain.

        For each island, places points in a circle around the island center
        at the specified radius. These points receive WW3 boundary conditions
        which SWAN uses to propagate waves around and behind the islands.

        Args:
            domain_name: Name of SWAN domain
            spacing_km: Angular spacing between points around islands (default: WW3 native)
            start_index: Starting index for point numbering

        Returns:
            List of BoundaryPoint objects around islands
        """
        if spacing_km is None:
            spacing_km = self.WW3_NATIVE_SPACING_KM

        config = self.load_domain(domain_name)

        # Load domain bounds and bathymetry for depth lookup
        nc_path = Path(config["file_nc"])
        ds = xr.open_dataset(nc_path)
        domain_lats = ds["lat"].values
        domain_lons = ds["lon"].values
        elevation = ds["elevation"].values  # Use full elevation, not masked
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
            # Circumference = 2 * pi * r, divide by spacing
            circumference_km = 2 * np.pi * radius_km
            n_points = max(4, int(circumference_km / spacing_km))

            # Generate points around the island
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points

                # Convert radius to degrees (approximate)
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

                # Skip if on land (positive elevation)
                if np.isnan(elev) or elev >= 0:
                    # Try to find nearest wet cell
                    depth = self._find_nearest_depth(
                        elevation, domain_lats, domain_lons,
                        pt_lat, pt_lon, search_radius=5
                    )
                    if depth is None:
                        continue
                else:
                    depth = -elev

                points.append(BoundaryPoint(
                    index=point_idx,
                    lat=float(pt_lat),
                    lon=float(pt_lon),
                    depth_m=float(depth),
                ))
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

    async def fetch_ww3_for_boundary(
        self,
        boundary_points: List[BoundaryPoint],
        forecast_hours: List[int],
        buffer_deg: float = 0.5,
    ) -> Dict[int, List[Dict]]:
        """
        Fetch WW3 data covering the boundary region.

        Args:
            boundary_points: List of boundary points
            forecast_hours: List of forecast hours to fetch
            buffer_deg: Buffer around boundary extent in degrees

        Returns:
            Dict mapping forecast_hour -> WW3 data dict
        """
        from data.pipelines.wave.wavewatch_fetcher import WaveWatchFetcher

        # Calculate bounding box for WW3 fetch
        lats = [p.lat for p in boundary_points]
        lons = [p.lon for p in boundary_points]

        min_lat = min(lats) - buffer_deg
        max_lat = max(lats) + buffer_deg
        min_lon = min(lons) - buffer_deg
        max_lon = max(lons) + buffer_deg

        print(f"\nFetching WW3 data for boundary region:")
        print(f"  Lat: {min_lat:.2f}° to {max_lat:.2f}°")
        print(f"  Lon: {min_lon:.2f}° to {max_lon:.2f}°")
        print(f"  Forecast hours: {forecast_hours}")

        fetcher = WaveWatchFetcher()
        ww3_data = {}

        for hour in forecast_hours:
            try:
                data = await fetcher.fetch_wave_grid(
                    min_lat=min_lat,
                    max_lat=max_lat,
                    min_lon=min_lon,
                    max_lon=max_lon,
                    forecast_hour=hour,
                )
                ww3_data[hour] = data
                print(f"  Fetched hour {hour:03d}: Hs={np.nanmean(data['significant_wave_height']):.2f}m")
            except Exception as e:
                print(f"  Failed hour {hour:03d}: {e}")

        return ww3_data

    def interpolate_ww3_to_points(
        self,
        ww3_data: Dict,
        boundary_points: List[BoundaryPoint],
    ) -> List[Dict]:
        """
        Interpolate WW3 data to boundary points.

        Args:
            ww3_data: WW3 data dict from fetcher
            boundary_points: List of boundary points

        Returns:
            List of dicts with interpolated wave parameters at each point
        """
        from scipy.interpolate import RegularGridInterpolator

        ww3_lats = np.array(ww3_data["lat"])
        ww3_lons = np.array(ww3_data["lon"])

        # Ensure ascending order
        if ww3_lats[0] > ww3_lats[-1]:
            ww3_lats = ww3_lats[::-1]
            reverse_lat = True
        else:
            reverse_lat = False

        if ww3_lons[0] > ww3_lons[-1]:
            ww3_lons = ww3_lons[::-1]
            reverse_lon = True
        else:
            reverse_lon = False

        # Prepare arrays for interpolation
        hs = np.array(ww3_data["significant_wave_height"])
        tp = np.array(ww3_data["peak_wave_period"])
        dir_mean = np.array(ww3_data["mean_wave_direction"])

        if reverse_lat:
            hs = hs[::-1, :]
            tp = tp[::-1, :]
            dir_mean = dir_mean[::-1, :]
        if reverse_lon:
            hs = hs[:, ::-1]
            tp = tp[:, ::-1]
            dir_mean = dir_mean[:, ::-1]

        # Replace NaN with nearest valid value for interpolation
        from scipy import ndimage

        def fill_nan(arr):
            mask = np.isnan(arr)
            if mask.all():
                return arr
            arr_filled = arr.copy()
            indices = ndimage.distance_transform_edt(
                mask, return_distances=False, return_indices=True
            )
            arr_filled = arr[tuple(indices)]
            return arr_filled

        hs_filled = fill_nan(hs)
        tp_filled = fill_nan(tp)
        dir_filled = fill_nan(dir_mean)

        # Create interpolators
        interp_hs = RegularGridInterpolator(
            (ww3_lats, ww3_lons), hs_filled,
            method='linear', bounds_error=False, fill_value=np.nan
        )
        interp_tp = RegularGridInterpolator(
            (ww3_lats, ww3_lons), tp_filled,
            method='linear', bounds_error=False, fill_value=np.nan
        )
        interp_dir = RegularGridInterpolator(
            (ww3_lats, ww3_lons), dir_filled,
            method='linear', bounds_error=False, fill_value=np.nan
        )

        # Interpolate to boundary points
        results = []
        for pt in boundary_points:
            point = np.array([[pt.lat, pt.lon]])

            hs_val = float(interp_hs(point)[0])
            tp_val = float(interp_tp(point)[0])
            dir_val = float(interp_dir(point)[0])

            results.append({
                "index": pt.index,
                "lat": pt.lat,
                "lon": pt.lon,
                "depth_m": pt.depth_m,
                "hs": hs_val,
                "tp": tp_val,
                "dir": dir_val,
                "spreading": 25.0,  # Default directional spreading (degrees)
            })

        return results

    def write_tpar_file(
        self,
        domain_name: str,
        boundary_data: Dict[int, List[Dict]],  # hour -> list of point data
        ww3_cycle_time: datetime,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Write SWAN TPAR boundary condition file.

        TPAR format for spatially-varying boundary:
        Creates one file with time-series data, and uses SWAN's
        BOUNDSPEC SIDE command with the segment option.

        For simplicity, we average along the boundary to create
        a single representative time-series. For more accuracy,
        SWAN supports segment-by-segment specification.

        Args:
            domain_name: SWAN domain name
            boundary_data: Dict mapping forecast hour to interpolated point data
            ww3_cycle_time: WW3 model cycle time
            output_dir: Output directory (default: domain directory)

        Returns:
            Path to generated TPAR file
        """
        if output_dir is None:
            output_dir = self.domains_dir / domain_name / "boundary"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sort forecast hours
        hours = sorted(boundary_data.keys())

        # Generate timestamp for filename
        timestamp = ww3_cycle_time.strftime("%Y%m%d_%H")
        tpar_path = output_dir / f"ww3_bc_{timestamp}.tpar"

        with open(tpar_path, "w") as f:
            f.write("TPAR\n")
            f.write(f"$ WW3 boundary conditions for SWAN\n")
            f.write(f"$ Domain: {domain_name}\n")
            f.write(f"$ WW3 cycle: {ww3_cycle_time.isoformat()}\n")
            f.write(f"$ Generated: {datetime.now().isoformat()}\n")
            f.write("$\n")
            f.write("$ Format: time Hs Tp Dir spreading\n")
            f.write("$ time: yyyymmdd.hhmmss\n")
            f.write("$ Hs: significant wave height (m)\n")
            f.write("$ Tp: peak period (s)\n")
            f.write("$ Dir: mean direction (degrees, nautical convention)\n")
            f.write("$ spreading: directional spreading (degrees)\n")
            f.write("$\n")

            for hour in hours:
                points_data = boundary_data[hour]

                # Average wave parameters along boundary
                hs_vals = [p["hs"] for p in points_data if not np.isnan(p["hs"])]
                tp_vals = [p["tp"] for p in points_data if not np.isnan(p["tp"])]
                dir_vals = [p["dir"] for p in points_data if not np.isnan(p["dir"])]

                if not hs_vals:
                    continue

                avg_hs = np.mean(hs_vals)
                avg_tp = np.mean(tp_vals)
                # Direction averaging needs special handling (circular mean)
                avg_dir = self._circular_mean(dir_vals)
                spreading = 25.0  # Default spreading

                # Calculate forecast valid time
                valid_time = ww3_cycle_time + timedelta(hours=hour)
                time_str = valid_time.strftime("%Y%m%d.%H%M%S")

                f.write(f"{time_str} {avg_hs:.2f} {avg_tp:.1f} {avg_dir:.1f} {spreading:.1f}\n")

        print(f"\nWritten TPAR file: {tpar_path}")
        return tpar_path

    def write_locations_file(
        self,
        domain_name: str,
        boundary_points: List[BoundaryPoint],
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Write boundary point locations file.

        This file documents where boundary conditions were extracted
        and can be used for visualization/debugging.
        """
        if output_dir is None:
            output_dir = self.domains_dir / domain_name / "boundary"
        output_dir.mkdir(parents=True, exist_ok=True)

        loc_path = output_dir / "boundary_points.json"

        data = {
            "domain": domain_name,
            "n_points": len(boundary_points),
            "points": [asdict(p) for p in boundary_points],
            "created_at": datetime.now().isoformat(),
        }

        with open(loc_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Written locations file: {loc_path}")
        return loc_path

    def write_swan_bc_commands(
        self,
        domain_name: str,
        tpar_path: Path,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Write SWAN input commands for using the boundary conditions.

        This creates a snippet that can be included in the SWAN input file.
        """
        if output_dir is None:
            output_dir = self.domains_dir / domain_name / "boundary"

        cmd_path = output_dir / "swan_bc_commands.txt"

        tpar_filename = tpar_path.name

        commands = f"""$----------------------------------------------------------
$ BOUNDARY CONDITIONS FROM WW3
$----------------------------------------------------------
$ Generated: {datetime.now().isoformat()}
$ Using TPAR file: {tpar_filename}
$
$ Apply time-varying boundary conditions to western (offshore) boundary
$ TPAR format provides: Hs, Tp, Dir, spreading at each time step
$
BOUNDSPEC SIDE W CCW VARIABLE FILE '{tpar_filename}'
$
$ Alternative: If you want to apply to specific segments:
$ BOUNDSPEC SEGMENT XY <x1> <y1> <x2> <y2> VARIABLE FILE '{tpar_filename}'
$----------------------------------------------------------
"""

        with open(cmd_path, "w") as f:
            f.write(commands)

        print(f"Written SWAN commands: {cmd_path}")
        return cmd_path

    async def generate(
        self,
        domain_name: str,
        forecast_hours: Optional[List[int]] = None,
        boundary_side: str = "W",
        point_spacing_km: float = None,
        include_islands: bool = True,
    ) -> SwanBoundaryFile:
        """
        Generate SWAN boundary conditions from WW3 data.

        Args:
            domain_name: Name of SWAN domain
            forecast_hours: List of forecast hours (default: 0-48 every 3h)
            boundary_side: Which boundary to apply conditions (default: W = offshore)
            point_spacing_km: Spacing between boundary extraction points (default: WW3 native ~28km)
            include_islands: Whether to include Channel Islands boundary points

        Returns:
            SwanBoundaryFile metadata
        """
        # Use WW3 native spacing by default
        if point_spacing_km is None:
            point_spacing_km = self.WW3_NATIVE_SPACING_KM

        print("=" * 60)
        print(f"Generating SWAN Boundary Conditions: {domain_name}")
        print("=" * 60)

        # Default forecast hours: 3-hourly for first 2 days
        if forecast_hours is None:
            forecast_hours = list(range(0, 49, 3))

        # Load domain config
        config = self.load_domain(domain_name)
        print(f"\nDomain: {domain_name}")
        print(f"  Lat: {config['lat_min']:.2f}° to {config['lat_max']:.2f}°")
        print(f"  Lon: {config['lon_min']:.2f}° to {config['lon_max']:.2f}°")
        print(f"  Offshore boundary: {config['offshore_boundary_km']}km")

        # Get boundary points
        print(f"\nExtracting {boundary_side} boundary points (spacing: {point_spacing_km}km)...")
        print(f"  Include islands: {include_islands}")
        boundary_points = self.get_boundary_points(
            domain_name, side=boundary_side, spacing_km=point_spacing_km,
            include_islands=include_islands
        )
        print(f"  Found {len(boundary_points)} boundary points")

        if len(boundary_points) < 2:
            raise ValueError(f"Not enough boundary points found on side {boundary_side}")

        # Fetch WW3 data
        print(f"\nFetching WW3 data for {len(forecast_hours)} forecast hours...")
        ww3_data = await self.fetch_ww3_for_boundary(
            boundary_points, forecast_hours
        )

        if not ww3_data:
            raise RuntimeError("Failed to fetch any WW3 data")

        # Get cycle time from first successful fetch
        first_data = list(ww3_data.values())[0]
        cycle_time = datetime.fromisoformat(first_data["cycle_time"])

        # Interpolate WW3 to boundary points for each forecast hour
        print(f"\nInterpolating WW3 to boundary points...")
        boundary_data = {}
        for hour, data in ww3_data.items():
            interpolated = self.interpolate_ww3_to_points(data, boundary_points)
            boundary_data[hour] = interpolated

            # Summary stats
            hs_vals = [p["hs"] for p in interpolated if not np.isnan(p["hs"])]
            if hs_vals:
                print(f"  Hour {hour:03d}: Hs = {np.min(hs_vals):.1f} - {np.max(hs_vals):.1f}m (mean: {np.mean(hs_vals):.2f}m)")

        # Write output files
        output_dir = self.domains_dir / domain_name / "boundary"

        tpar_path = self.write_tpar_file(
            domain_name, boundary_data, cycle_time, output_dir
        )

        loc_path = self.write_locations_file(
            domain_name, boundary_points, output_dir
        )

        self.write_swan_bc_commands(domain_name, tpar_path, output_dir)

        # Create metadata
        result = SwanBoundaryFile(
            domain_name=domain_name,
            n_boundary_points=len(boundary_points),
            n_forecast_hours=len(boundary_data),
            boundary_side=boundary_side,
            format_type="TPAR",
            file_tpar=str(tpar_path),
            file_locations=str(loc_path),
            ww3_cycle_time=cycle_time.isoformat(),
            forecast_hours=sorted(boundary_data.keys()),
            created_at=datetime.now().isoformat(),
        )

        # Save metadata
        meta_path = output_dir / "boundary_config.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        print(f"\n" + "=" * 60)
        print("SWAN Boundary Conditions Generated Successfully")
        print("=" * 60)
        print(f"  TPAR file: {tpar_path}")
        print(f"  Locations: {loc_path}")
        print(f"  WW3 cycle: {cycle_time.isoformat()}")
        print(f"  Forecast hours: {sorted(boundary_data.keys())}")
        print(f"\nTo use in SWAN, include the commands from:")
        print(f"  {output_dir / 'swan_bc_commands.txt'}")

        return result

    def _circular_mean(self, angles: List[float]) -> float:
        """Calculate circular mean of angles in degrees."""
        if not angles:
            return 0.0
        angles_rad = np.radians(angles)
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))
        mean_rad = np.arctan2(mean_sin, mean_cos)
        return float(np.degrees(mean_rad) % 360)

    def list_domains(self) -> List[str]:
        """List available SWAN domains."""
        domains = []
        for config_path in self.domains_dir.glob("*/config.json"):
            domains.append(config_path.parent.name)
        return sorted(domains)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SWAN boundary conditions from WW3 data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate boundary conditions for a domain
    python -m data.pipelines.swan.ww3_boundary --generate california_swan_2000m

    # Specify forecast hours
    python -m data.pipelines.swan.ww3_boundary --generate california_swan_2000m --hours 0 3 6 9 12

    # List available domains
    python -m data.pipelines.swan.ww3_boundary --list

    # View boundary points for a domain
    python -m data.pipelines.swan.ww3_boundary --points california_swan_2000m

Workflow:
    1. First create a SWAN domain with swan_domain.py
    2. Then generate WW3 boundary conditions with this script
    3. Copy the SWAN commands to your .swn input file
    4. Run SWAN with the generated boundary files
""",
    )

    parser.add_argument(
        "--generate", "-g",
        type=str,
        metavar="DOMAIN",
        help="Generate boundary conditions for SWAN domain",
    )
    parser.add_argument(
        "--hours",
        type=int,
        nargs="+",
        help="Forecast hours to fetch (default: 0-48 every 3h)",
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
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available SWAN domains",
    )
    parser.add_argument(
        "--points", "-p",
        type=str,
        metavar="DOMAIN",
        help="Show boundary points for a domain",
    )

    args = parser.parse_args()

    bc_gen = WW3BoundaryConditions()

    if args.list:
        domains = bc_gen.list_domains()
        if not domains:
            print("No SWAN domains found.")
            print("Create one with: python -m data.pipelines.bathymetry.swan_domain --generate <region>")
            return

        print("\nAvailable SWAN Domains:")
        print("-" * 40)
        for d in domains:
            print(f"  {d}")
        return

    if args.points:
        include_islands = not args.no_islands
        points = bc_gen.get_boundary_points(
            args.points, side="W", spacing_km=args.spacing,
            include_islands=include_islands
        )
        print(f"\nBoundary points for {args.points}:")
        print(f"  Spacing: {args.spacing}km (WW3 native: 28km)")
        print(f"  Include islands: {include_islands}")
        print(f"\n{'Index':<6} {'Lat':>10} {'Lon':>10} {'Depth (m)':>10}")
        print("-" * 40)
        for p in points:
            print(f"{p.index:<6} {p.lat:>10.4f} {p.lon:>10.4f} {p.depth_m:>10.1f}")
        print(f"\nTotal: {len(points)} points")
        return

    if args.generate:
        asyncio.run(bc_gen.generate(
            domain_name=args.generate,
            forecast_hours=args.hours,
            point_spacing_km=args.spacing,
            include_islands=not args.no_islands,
        ))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
