"""
Current Provider for SWAN

Extracts ocean current data from RTOFS for SWAN input.
Uses mesh bounds directly (like WindProvider) - no region name configuration needed.

SWAN expects currents as UX (eastward) and UY (northward) velocity components
in m/s, written as all UX values followed by all UY values (like wind).

References:
- SWAN User Manual: https://swanmodel.sourceforge.io/online_doc/swanuse/node26.html
- INPGRID CURRENT and READINP CURRENT commands
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


@dataclass
class CurrentData:
    """
    Current data prepared for SWAN input.

    Attributes:
        u_current: 2D array of u-component (eastward, m/s), shape (ny+1, nx+1)
        v_current: 2D array of v-component (northward, m/s), shape (ny+1, nx+1)
        lons: 1D array of longitudes
        lats: 1D array of latitudes
        nx: Number of grid cells in x direction
        ny: Number of grid cells in y direction
        dx: Grid spacing in x (degrees)
        dy: Grid spacing in y (degrees)
        origin: (lon, lat) of lower-left corner
        timestamp: Valid time of current data
        metadata: Additional metadata from RTOFS
    """
    u_current: np.ndarray
    v_current: np.ndarray
    lons: np.ndarray
    lats: np.ndarray
    nx: int
    ny: int
    dx: float
    dy: float
    origin: Tuple[float, float]
    timestamp: datetime
    metadata: Dict


class CurrentProvider:
    """
    Extracts ocean current data from RTOFS for SWAN input.

    Uses mesh bounds directly (like WindProvider) - no region configuration needed.
    Fetches RTOFS data, interpolates to mesh resolution, and writes SWAN files.

    Example usage:
        provider = CurrentProvider()
        current_data = await provider.extract_for_mesh(mesh_metadata, forecast_hour=0)
        provider.write_swan_files(current_data, output_dir)

    SWAN commands generated:
        INPGRID CURRENT REG xp yp 0 mx my dx dy
        READINP CURRENT 1. 'current.dat' 4 0 0 FREE
    """

    def __init__(self):
        """Initialize CurrentProvider."""
        pass

    async def extract_for_mesh(
        self,
        mesh_metadata: dict,
        forecast_hour: int = 0
    ) -> CurrentData:
        """
        Extract and interpolate current data for a mesh.

        Fetches RTOFS data for the mesh bounds and interpolates to mesh resolution
        using bilinear interpolation.

        Args:
            mesh_metadata: Mesh JSON metadata containing origin, nx, ny, dx, dy
            forecast_hour: Forecast hour index to extract (0 = current)

        Returns:
            CurrentData with u/v components interpolated to mesh grid
        """
        # Import here to avoid circular imports
        from data.pipelines.ocean_tiles.rtofs_fetcher import RTOFSFetcher

        # Extract mesh bounds and grid parameters
        lon_min = mesh_metadata["origin"][0]
        lat_min = mesh_metadata["origin"][1]
        nx = mesh_metadata["nx"]
        ny = mesh_metadata["ny"]
        dx = mesh_metadata["dx"]
        dy = mesh_metadata["dy"]

        lon_max = lon_min + nx * dx
        lat_max = lat_min + ny * dy

        # Build bounds dict for RTOFS fetcher
        bounds = {
            'min_lat': lat_min,
            'max_lat': lat_max,
            'min_lon': lon_min,
            'max_lon': lon_max,
        }

        logger.info(f"Extracting currents for mesh bounds: "
                   f"lon=[{lon_min:.2f}, {lon_max:.2f}], "
                   f"lat=[{lat_min:.2f}, {lat_max:.2f}]")

        # Fetch RTOFS data using bounds directly (no region name needed)
        fetcher = RTOFSFetcher()
        rtofs_data = await fetcher.fetch_for_bounds(
            bounds=bounds,
            forecast_hour=forecast_hour
        )

        if rtofs_data is None:
            raise RuntimeError(f"Failed to fetch RTOFS data for bounds: {bounds}")

        # Extract data arrays
        rtofs_u = rtofs_data['u_velocity']
        rtofs_v = rtofs_data['v_velocity']
        rtofs_lons = rtofs_data['lons']
        rtofs_lats = rtofs_data['lats']
        metadata = rtofs_data['metadata']

        logger.info(f"RTOFS data shape: {rtofs_u.shape}")

        # Handle RTOFS curvilinear grid - need to regrid to regular grid first
        # RTOFS uses 2D lat/lon arrays for curvilinear coordinates
        if rtofs_lons.ndim == 2:
            rtofs_u, rtofs_v, rtofs_lons_1d, rtofs_lats_1d = self._regrid_curvilinear(
                rtofs_u, rtofs_v, rtofs_lons, rtofs_lats
            )
        else:
            rtofs_lons_1d = rtofs_lons
            rtofs_lats_1d = rtofs_lats

        # Create mesh grid coordinates (cell centers, matching SWAN grid points)
        # SWAN grid has nx+1 points in x, ny+1 points in y
        mesh_lons = np.linspace(lon_min, lon_max, nx + 1)
        mesh_lats = np.linspace(lat_min, lat_max, ny + 1)

        # Interpolate RTOFS to mesh resolution using bilinear interpolation
        u_interp = self._bilinear_interpolate(rtofs_lons_1d, rtofs_lats_1d, rtofs_u, mesh_lons, mesh_lats)
        v_interp = self._bilinear_interpolate(rtofs_lons_1d, rtofs_lats_1d, rtofs_v, mesh_lons, mesh_lats)

        # Handle any NaN values (land) by filling with zero (no current)
        u_interp = self._fill_nan_zero(u_interp)
        v_interp = self._fill_nan_zero(v_interp)

        logger.info(f"Interpolated to mesh: {len(mesh_lons)} x {len(mesh_lats)} points")

        # Calculate current speed stats for logging
        current_speed = np.sqrt(u_interp**2 + v_interp**2)
        logger.info(f"Current speed range: {current_speed.min():.3f} - {current_speed.max():.3f} m/s")

        # Parse timestamp from metadata
        valid_time_str = metadata.get('valid_time', '')
        if valid_time_str:
            try:
                timestamp = datetime.fromisoformat(valid_time_str)
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        return CurrentData(
            u_current=u_interp,
            v_current=v_interp,
            lons=mesh_lons,
            lats=mesh_lats,
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            origin=(lon_min, lat_min),
            timestamp=timestamp,
            metadata=metadata
        )

    def _regrid_curvilinear(
        self,
        u: np.ndarray,
        v: np.ndarray,
        lons_2d: np.ndarray,
        lats_2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Regrid curvilinear data to a regular lat/lon grid.

        RTOFS uses a curvilinear grid with 2D lat/lon arrays.
        We regrid to a regular grid for easier interpolation to the mesh.

        Args:
            u: U velocity on curvilinear grid
            v: V velocity on curvilinear grid
            lons_2d: 2D longitude array
            lats_2d: 2D latitude array

        Returns:
            Tuple of (u_regular, v_regular, lons_1d, lats_1d)
        """
        from scipy.interpolate import griddata

        # Get bounds
        lon_min, lon_max = np.nanmin(lons_2d), np.nanmax(lons_2d)
        lat_min, lat_max = np.nanmin(lats_2d), np.nanmax(lats_2d)

        # Create regular grid at similar resolution
        # RTOFS is ~1/12° (~0.083°), use slightly coarser for efficiency
        resolution = 0.1  # degrees
        lons_1d = np.arange(lon_min, lon_max + resolution, resolution)
        lats_1d = np.arange(lat_min, lat_max + resolution, resolution)

        # Create mesh for interpolation target
        lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)

        # Flatten source coordinates and data
        points = np.column_stack([lons_2d.ravel(), lats_2d.ravel()])

        # Handle masked arrays
        u_flat = np.ma.filled(u.ravel(), np.nan) if hasattr(u, 'mask') else u.ravel()
        v_flat = np.ma.filled(v.ravel(), np.nan) if hasattr(v, 'mask') else v.ravel()

        # Remove NaN points for interpolation
        valid = ~(np.isnan(u_flat) | np.isnan(v_flat) | np.isnan(points[:, 0]) | np.isnan(points[:, 1]))
        points_valid = points[valid]
        u_valid = u_flat[valid]
        v_valid = v_flat[valid]

        if len(points_valid) < 10:
            logger.warning("Too few valid points for regridding, using zeros")
            return (
                np.zeros((len(lats_1d), len(lons_1d))),
                np.zeros((len(lats_1d), len(lons_1d))),
                lons_1d,
                lats_1d
            )

        # Interpolate to regular grid
        u_regular = griddata(points_valid, u_valid, (lon_grid, lat_grid), method='linear')
        v_regular = griddata(points_valid, v_valid, (lon_grid, lat_grid), method='linear')

        logger.info(f"Regridded curvilinear {u.shape} to regular {u_regular.shape}")

        return u_regular, v_regular, lons_1d, lats_1d

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
        # Handle NaN in source data for interpolation
        src_data_filled = np.nan_to_num(src_data, nan=0.0)

        # Create interpolator (scipy expects (lat, lon) order for 2D data)
        try:
            interpolator = RegularGridInterpolator(
                (src_lats, src_lons),
                src_data_filled,
                method='linear',
                bounds_error=False,
                fill_value=0.0  # Use zero for out-of-bounds (land/edge)
            )
        except ValueError as e:
            logger.warning(f"Interpolation setup failed: {e}, using zeros")
            return np.zeros((len(dst_lats), len(dst_lons)))

        # Create destination mesh grid
        dst_lon_grid, dst_lat_grid = np.meshgrid(dst_lons, dst_lats)

        # Stack coordinates for interpolator
        points = np.stack([dst_lat_grid.ravel(), dst_lon_grid.ravel()], axis=-1)

        # Interpolate and reshape
        result = interpolator(points).reshape(dst_lat_grid.shape)

        return result

    def _fill_nan_zero(self, data: np.ndarray) -> np.ndarray:
        """
        Fill NaN values with zero (no current).

        For ocean currents, NaN typically represents land, so zero is appropriate.

        Args:
            data: 2D array potentially containing NaN values

        Returns:
            Array with NaN values replaced by zero
        """
        if not np.any(np.isnan(data)):
            return data

        n_nan = np.sum(np.isnan(data))
        data = np.nan_to_num(data, nan=0.0)

        if n_nan > 0:
            logger.debug(f"Filled {n_nan} NaN current values with zero")

        return data

    def write_swan_files(
        self,
        current_data: CurrentData,
        output_dir: Path
    ) -> Path:
        """
        Write current data to SWAN ASCII format.

        Creates a single file with U (x-component) followed by V (y-component).
        SWAN expects this format for vector quantities: all U values first,
        then all V values.

        Format uses IDLA=4 layout: row-by-row from bottom-left (SW corner),
        proceeding west to east, then south to north.

        Args:
            current_data: CurrentData object with interpolated currents
            output_dir: Directory to write files

        Returns:
            Path to current file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        current_path = output_dir / "current.dat"

        # SWAN format for vector (current): all U values, then all V values
        # IDLA=4: rows from south to north, columns west to east
        with open(current_path, 'w') as f:
            # Write U component (x-direction, eastward)
            for row in current_data.u_current:
                line = ' '.join(f'{val:.6f}' for val in row)
                f.write(line + '\n')
            # Write V component (y-direction, northward)
            for row in current_data.v_current:
                line = ' '.join(f'{val:.6f}' for val in row)
                f.write(line + '\n')

        logger.info(f"Wrote current file: {current_path.name}")

        return current_path

    def generate_inpgrid_command(self, current_data: CurrentData) -> str:
        """
        Generate SWAN INPGRID CURRENT command.

        Args:
            current_data: CurrentData object with grid parameters

        Returns:
            SWAN command string for current input grid
        """
        lon_origin, lat_origin = current_data.origin

        # INPGRID CURRENT REG xp yp alp mx my dx dy
        # alp = 0 (grid not rotated)
        # mx, my = number of meshes (cells), so nx and ny
        return (
            f"INPGRID CURRENT REG {lon_origin:.4f} {lat_origin:.4f} 0 "
            f"{current_data.nx} {current_data.ny} {current_data.dx:.6f} {current_data.dy:.6f}"
        )

    def generate_readinp_command(self) -> str:
        """
        Generate SWAN READINP CURRENT command.

        Uses a single combined file with U values followed by V values.

        Returns:
            SWAN command string
        """
        # READINP CURRENT fac 'fname' idla nhedf nhedvec FREE
        # fac = 1.0 (no scaling)
        # fname = current.dat (combined U and V)
        # idla = 4 (row-by-row from SW corner)
        # nhedf = 0 (no file header lines)
        # nhedvec = 0 (no header between vector components)
        # FREE = free format
        return "READINP CURRENT 1. 'current.dat' 4 0 0 FREE"

    def summary(self, current_data: CurrentData) -> str:
        """Return summary of extracted current data."""
        current_speed = np.sqrt(current_data.u_current**2 + current_data.v_current**2)
        current_dir = (90 - np.degrees(np.arctan2(current_data.v_current, current_data.u_current))) % 360

        lines = [
            "Current Data Summary",
            f"  Valid time: {current_data.timestamp}",
            f"  Grid: {current_data.nx + 1} x {current_data.ny + 1} points",
            f"  Resolution: {current_data.dx:.4f}° x {current_data.dy:.4f}°",
            f"  Origin: ({current_data.origin[0]:.2f}°, {current_data.origin[1]:.2f}°)",
            f"  Current speed: {current_speed.min():.3f} - {current_speed.max():.3f} m/s (mean: {current_speed.mean():.3f})",
            f"  Mean direction: {current_dir.mean():.0f}° (toward)",
        ]
        return '\n'.join(lines)
