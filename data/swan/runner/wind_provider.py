"""
Wind Provider for SWAN

Extracts wind data from downloaded GFS NetCDF files and formats it for SWAN input.
Uses bilinear interpolation to resample GFS (0.25°) to the mesh resolution.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

# Default location for downloaded wind data
# Path: data/downloaded_weather_data/wind/ (relative to project root)
DEFAULT_WIND_DIR = Path(__file__).parent.parent.parent / "downloaded_weather_data" / "wind"


@dataclass
class WindData:
    """
    Wind data prepared for SWAN input.

    Attributes:
        u_wind: 2D array of u-component (m/s), shape (ny+1, nx+1)
        v_wind: 2D array of v-component (m/s), shape (ny+1, nx+1)
        lons: 1D array of longitudes
        lats: 1D array of latitudes
        nx: Number of grid cells in x direction
        ny: Number of grid cells in y direction
        dx: Grid spacing in x (degrees)
        dy: Grid spacing in y (degrees)
        origin: (lon, lat) of lower-left corner
        timestamp: Valid time of wind data
        source_file: Path to source NetCDF file
    """
    u_wind: np.ndarray
    v_wind: np.ndarray
    lons: np.ndarray
    lats: np.ndarray
    nx: int
    ny: int
    dx: float
    dy: float
    origin: Tuple[float, float]
    timestamp: datetime
    source_file: Path


class WindProvider:
    """
    Extracts wind data from downloaded GFS NetCDF files for SWAN input.

    Reads from: data/downloaded_weather_data/wind/gfs_YYYYMMDD_HHz.nc
    Interpolates to mesh resolution using bilinear interpolation.
    Writes SWAN-compatible ASCII files.

    Example usage:
        provider = WindProvider()
        wind_data = provider.extract_for_mesh(mesh_metadata, forecast_hour=0)
        provider.write_swan_files(wind_data, output_dir)
    """

    def __init__(self, wind_dir: Optional[Path] = None):
        """
        Initialize WindProvider.

        Args:
            wind_dir: Directory containing downloaded GFS NetCDF files.
                      Defaults to data/downloaded_weather_data/wind/
        """
        if wind_dir is None:
            self.wind_dir = DEFAULT_WIND_DIR
        else:
            self.wind_dir = Path(wind_dir)

        if not self.wind_dir.exists():
            raise FileNotFoundError(f"Wind data directory not found: {self.wind_dir}")

    def get_latest_wind_file(self) -> Path:
        """
        Find the most recent downloaded GFS file.

        Returns:
            Path to the most recent NetCDF file

        Raises:
            FileNotFoundError: If no wind files exist
        """
        nc_files = sorted(self.wind_dir.glob("gfs_*.nc"))
        if not nc_files:
            raise FileNotFoundError(f"No wind files found in {self.wind_dir}")

        latest = nc_files[-1]
        logger.info(f"Using wind file: {latest.name}")
        return latest

    def extract_for_mesh(
        self,
        mesh_metadata: dict,
        forecast_hour: int = 0,
        wind_file: Optional[Path] = None
    ) -> WindData:
        """
        Extract and interpolate wind data for a mesh.

        Reads GFS data, extracts the region covering the mesh, and
        interpolates from GFS resolution (0.25°) to mesh resolution
        using bilinear interpolation.

        Args:
            mesh_metadata: Mesh JSON metadata containing origin, nx, ny, dx, dy
            forecast_hour: Forecast hour index to extract (0 = analysis/current)
            wind_file: Specific file to use (default: latest available)

        Returns:
            WindData with u/v components interpolated to mesh grid
        """
        if wind_file is None:
            wind_file = self.get_latest_wind_file()

        # Extract mesh bounds and grid parameters
        lon_min = mesh_metadata["origin"][0]
        lat_min = mesh_metadata["origin"][1]
        nx = mesh_metadata["nx"]
        ny = mesh_metadata["ny"]
        dx = mesh_metadata["dx"]
        dy = mesh_metadata["dy"]

        lon_max = lon_min + nx * dx
        lat_max = lat_min + ny * dy

        # Add buffer to ensure GFS coverage for interpolation edges
        buffer = 0.5  # degrees (2 GFS grid cells)

        logger.info(f"Extracting wind for region: "
                   f"lon=[{lon_min:.2f}, {lon_max:.2f}], "
                   f"lat=[{lat_min:.2f}, {lat_max:.2f}]")

        # Open NetCDF and extract region
        with xr.open_dataset(wind_file) as ds:
            # Select spatial subset with buffer
            subset = ds.sel(
                lat=slice(lat_min - buffer, lat_max + buffer),
                lon=slice(lon_min - buffer, lon_max + buffer)
            )

            # Select time (forecast hour)
            if "time" in subset.dims and len(subset.time) > 1:
                n_times = len(subset.time)
                time_idx = min(forecast_hour, n_times - 1)
                subset = subset.isel(time=time_idx)
                timestamp = subset.time.values
                # Convert numpy datetime64 to python datetime
                timestamp = timestamp.astype('datetime64[us]').astype(datetime)
            elif "time" in subset.dims:
                subset = subset.isel(time=0)
                timestamp = subset.time.values.astype('datetime64[us]').astype(datetime)
            else:
                timestamp = datetime.now()

            # Get GFS grid coordinates and data
            gfs_lons = subset["lon"].values
            gfs_lats = subset["lat"].values
            gfs_u = subset["u_wind"].values
            gfs_v = subset["v_wind"].values

        logger.info(f"GFS subset: {len(gfs_lons)} x {len(gfs_lats)} points")

        # Create mesh grid coordinates (cell centers, matching SWAN grid points)
        # SWAN grid has nx+1 points in x, ny+1 points in y
        mesh_lons = np.linspace(lon_min, lon_max, nx + 1)
        mesh_lats = np.linspace(lat_min, lat_max, ny + 1)

        # Interpolate GFS to mesh resolution using bilinear interpolation
        u_interp = self._bilinear_interpolate(gfs_lons, gfs_lats, gfs_u, mesh_lons, mesh_lats)
        v_interp = self._bilinear_interpolate(gfs_lons, gfs_lats, gfs_v, mesh_lons, mesh_lats)

        # Handle any NaN values (land in GFS) by filling with nearest valid
        u_interp = self._fill_nan_nearest(u_interp)
        v_interp = self._fill_nan_nearest(v_interp)

        logger.info(f"Interpolated to mesh: {len(mesh_lons)} x {len(mesh_lats)} points")

        # Calculate wind speed stats for logging
        wind_speed = np.sqrt(u_interp**2 + v_interp**2)
        logger.info(f"Wind speed range: {wind_speed.min():.1f} - {wind_speed.max():.1f} m/s")

        return WindData(
            u_wind=u_interp,
            v_wind=v_interp,
            lons=mesh_lons,
            lats=mesh_lats,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            origin=(lon_min, lat_min),
            timestamp=timestamp,
            source_file=wind_file
        )

    def _bilinear_interpolate(
        self,
        src_lons: np.ndarray,
        src_lats: np.ndarray,
        src_data: np.ndarray,
        dst_lons: np.ndarray,
        dst_lats: np.ndarray
    ) -> np.ndarray:
        """
        Perform bilinear interpolation from source grid to destination grid.

        Args:
            src_lons: Source longitude coordinates (1D)
            src_lats: Source latitude coordinates (1D)
            src_data: Source data array (2D: lat x lon)
            dst_lons: Destination longitude coordinates (1D)
            dst_lats: Destination latitude coordinates (1D)

        Returns:
            Interpolated data on destination grid (2D: lat x lon)
        """
        # Create interpolator (scipy expects (lat, lon) order for 2D data)
        # bounds_error=False and fill_value=nan for points outside domain
        interpolator = RegularGridInterpolator(
            (src_lats, src_lons),
            src_data,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        # Create destination mesh grid
        dst_lon_grid, dst_lat_grid = np.meshgrid(dst_lons, dst_lats)

        # Stack coordinates for interpolator
        points = np.stack([dst_lat_grid.ravel(), dst_lon_grid.ravel()], axis=-1)

        # Interpolate and reshape
        result = interpolator(points).reshape(dst_lat_grid.shape)

        return result

    def _fill_nan_nearest(self, data: np.ndarray) -> np.ndarray:
        """
        Fill NaN values with nearest valid neighbor.

        This handles land points in GFS data that fall within the ocean
        mesh domain.

        Args:
            data: 2D array potentially containing NaN values

        Returns:
            Array with NaN values filled
        """
        if not np.any(np.isnan(data)):
            return data

        from scipy.ndimage import distance_transform_edt

        # Find valid (non-NaN) mask
        valid_mask = ~np.isnan(data)

        if not np.any(valid_mask):
            logger.warning("All wind data is NaN - using zero wind")
            return np.zeros_like(data)

        # Get indices of nearest valid points
        _, nearest_indices = distance_transform_edt(
            ~valid_mask,
            return_distances=True,
            return_indices=True
        )

        # Fill NaN with nearest valid values
        filled = data[tuple(nearest_indices)]

        n_filled = np.sum(~valid_mask)
        if n_filled > 0:
            logger.info(f"Filled {n_filled} NaN wind values with nearest neighbor")

        return filled

    def write_swan_files(
        self,
        wind_data: WindData,
        output_dir: Path
    ) -> Path:
        """
        Write wind data to SWAN ASCII format.

        Creates a single file with U (x-component) followed by V (y-component).
        SWAN expects this format for vector quantities: all U values first,
        then all V values.

        Format uses IDLA=4 layout: row-by-row from bottom-left (SW corner),
        proceeding west to east, then south to north.

        Args:
            wind_data: WindData object with interpolated wind
            output_dir: Directory to write files

        Returns:
            Path to combined wind file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        wind_path = output_dir / "wind.dat"

        # SWAN format for vector (wind): all U values, then all V values
        # IDLA=4: rows from south to north, columns west to east
        with open(wind_path, 'w') as f:
            # Write U component (x-direction, eastward)
            for row in wind_data.u_wind:
                line = ' '.join(f'{val:.4f}' for val in row)
                f.write(line + '\n')
            # Write V component (y-direction, northward)
            for row in wind_data.v_wind:
                line = ' '.join(f'{val:.4f}' for val in row)
                f.write(line + '\n')

        logger.info(f"Wrote wind file: {wind_path.name}")

        return wind_path

    def generate_inpgrid_command(self, wind_data: WindData) -> str:
        """
        Generate SWAN INPGRID WIND command.

        Args:
            wind_data: WindData object with grid parameters

        Returns:
            SWAN command string for wind input grid
        """
        lon_origin, lat_origin = wind_data.origin

        # INPGRID WIND REG xp yp alp mx my dx dy
        # alp = 0 (grid not rotated)
        # mx, my = number of meshes (cells), so nx and ny
        return (
            f"INPGRID WIND REG {lon_origin:.4f} {lat_origin:.4f} 0 "
            f"{wind_data.nx} {wind_data.ny} {wind_data.dx:.6f} {wind_data.dy:.6f}"
        )

    def generate_readinp_command(self) -> str:
        """
        Generate SWAN READINP WIND command.

        Uses a single combined file with U values followed by V values.

        Returns:
            SWAN command string
        """
        # READINP WIND fac 'fname' idla nhedf nhedvec FREE
        # fac = 1.0 (no scaling)
        # fname = wind.dat (combined U and V)
        # idla = 4 (row-by-row from SW corner)
        # nhedf = 0 (no file header lines)
        # nhedvec = 0 (no header between vector components)
        # FREE = free format
        return "READINP WIND 1. 'wind.dat' 4 0 0 FREE"

    def summary(self, wind_data: WindData) -> str:
        """Return summary of extracted wind data."""
        wind_speed = np.sqrt(wind_data.u_wind**2 + wind_data.v_wind**2)
        wind_dir = (270 - np.degrees(np.arctan2(wind_data.v_wind, wind_data.u_wind))) % 360

        lines = [
            "Wind Data Summary",
            f"  Source: {wind_data.source_file.name}",
            f"  Timestamp: {wind_data.timestamp}",
            f"  Grid: {wind_data.nx + 1} x {wind_data.ny + 1} points",
            f"  Resolution: {wind_data.dx:.4f}° x {wind_data.dy:.4f}°",
            f"  Origin: ({wind_data.origin[0]:.2f}°, {wind_data.origin[1]:.2f}°)",
            f"  Wind speed: {wind_speed.min():.1f} - {wind_speed.max():.1f} m/s (mean: {wind_speed.mean():.1f})",
            f"  Mean direction: {wind_dir.mean():.0f}° (from)",
        ]
        return '\n'.join(lines)