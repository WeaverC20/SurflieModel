"""
WCOFS (West Coast Operational Forecast System) Data Fetcher

Fetches ocean current data from NOAA's WCOFS model via THREDDS OPeNDAP.

Model Info:
- ROMS-based regional ocean model for the US West Coast
- ~4km resolution (348x1016 grid, 40 vertical levels)
- Updated 4x daily (00Z, 06Z, 12Z, 18Z cycles)
- 72-hour forecast + 24-hour nowcast
- Assimilates HF radar surface currents, satellite SST, satellite altimetry (4DVAR)
- ~7-8 cm/s RMSE vs HF radar (5x better than RTOFS)

THREDDS: https://opendap.co-ops.nos.noaa.gov/thredds/catalog/NOAA/WCOFS/MODELS/
File pattern: wcofs.t{CC}z.{YYYYMMDD}.2ds.{n|f}{HHH}.nc
Variables: u_sur_eastward, v_sur_northward (geographic coords on rho grid)
Coordinates: lon_rho, lat_rho (curvilinear ROMS grid)
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import httpx
import numpy as np

from .config import WCOFS_CONFIG, REGIONS, GRIB_CACHE_DIR

logger = logging.getLogger(__name__)

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False
    logger.warning("xarray not available. Install with: pip install xarray netCDF4")


class WCOFSFetcher:
    """Fetches ocean current data from NOAA WCOFS model via THREDDS."""

    THREDDS_BASE = "https://opendap.co-ops.nos.noaa.gov/thredds"

    # WCOFS cycles run at 03Z (primary), may also have 09Z, 15Z, 21Z
    CYCLE_HOURS = [3, 9, 15, 21]

    # Variable names in WCOFS 2ds files (geographic coordinates, on rho grid)
    U_VAR_NAMES = ['u_sur_eastward', 'u_eastward', 'u_sur', 'ubar_eastward']
    V_VAR_NAMES = ['v_sur_northward', 'v_northward', 'v_sur', 'vbar_northward']
    LAT_NAMES = ['lat_rho', 'latitude', 'lat']
    LON_NAMES = ['lon_rho', 'longitude', 'lon']

    def __init__(self):
        """Initialize WCOFS fetcher."""
        self.cache_dir = GRIB_CACHE_DIR

    def _build_file_url(
        self,
        model_date: datetime,
        cycle_hour: int,
        forecast_hour: int,
        service: str = "fileServer"
    ) -> str:
        """
        Build THREDDS URL for a WCOFS 2ds file.

        Args:
            model_date: Model run date
            cycle_hour: Cycle hour (3, 9, 15, 21)
            forecast_hour: Forecast hour (0-72) or nowcast hour
            service: THREDDS service ('fileServer' for HTTP, 'dodsC' for OPeNDAP)

        Returns:
            Full URL to the file
        """
        date_str = model_date.strftime('%Y%m%d')
        year = model_date.strftime('%Y')
        month = model_date.strftime('%m')
        day = model_date.strftime('%d')

        # Forecast files use 'f' prefix, nowcast uses 'n'
        prefix = 'f' if forecast_hour >= 0 else 'n'
        hour_num = abs(forecast_hour)

        filename = f"wcofs.t{cycle_hour:02d}z.{date_str}.2ds.{prefix}{hour_num:03d}.nc"
        path = f"NOAA/WCOFS/MODELS/{year}/{month}/{day}/{filename}"

        return f"{self.THREDDS_BASE}/{service}/{path}"

    def _get_cache_path(self, model_date: datetime, cycle_hour: int, forecast_hour: int) -> Path:
        """Get local cache file path."""
        date_str = model_date.strftime('%Y%m%d')
        return self.cache_dir / f"wcofs_{date_str}_t{cycle_hour:02d}z_f{forecast_hour:03d}.nc"

    async def fetch_current_grid(
        self,
        region_name: str = "california",
        forecast_hour: int = 0,
        model_date: Optional[datetime] = None
    ) -> Optional[Dict]:
        """
        Fetch WCOFS current data for a region.

        Args:
            region_name: Name of region from config (e.g., 'california')
            forecast_hour: Forecast hour (0-72)
            model_date: Model run date (defaults to today)

        Returns:
            Dictionary with u_velocity, v_velocity, current_speed, current_direction,
            lons, lats, metadata — same structure as RTOFSFetcher for compatibility.
        """
        if not XARRAY_AVAILABLE:
            logger.error("xarray not available")
            return None

        if region_name not in REGIONS:
            logger.error(f"Unknown region: {region_name}")
            return None

        region = REGIONS[region_name]
        bounds = region.bounds

        return await self.fetch_for_bounds(
            bounds=bounds,
            forecast_hour=forecast_hour,
            model_date=model_date
        )

    async def fetch_for_bounds(
        self,
        bounds: Dict[str, float],
        forecast_hour: int = 0,
        model_date: Optional[datetime] = None
    ) -> Optional[Dict]:
        """
        Fetch WCOFS current data for a geographic bounding box.

        Args:
            bounds: Geographic bounds dict with min_lat, max_lat, min_lon, max_lon
            forecast_hour: Forecast hour (0-72)
            model_date: Model run date (defaults to today)

        Returns:
            Dictionary with u_velocity, v_velocity, current_speed, current_direction,
            lons, lats, metadata.
        """
        if not XARRAY_AVAILABLE:
            logger.error("xarray not available")
            return None

        if model_date is None:
            model_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        # Clamp forecast hour to WCOFS range
        forecast_hour = min(max(forecast_hour, 0), 72)

        logger.info(f"Fetching WCOFS data for bounds at forecast hour {forecast_hour}")
        logger.info(f"Model date: {model_date.strftime('%Y-%m-%d')}")
        logger.info(f"Bounds: lat=[{bounds['min_lat']:.2f}, {bounds['max_lat']:.2f}], "
                   f"lon=[{bounds['min_lon']:.2f}, {bounds['max_lon']:.2f}]")

        # Try cycles from most recent to oldest, with day fallback
        for day_offset in range(3):
            try_date = model_date - timedelta(days=day_offset)

            if day_offset > 0:
                logger.info(f"Trying fallback to {try_date.strftime('%Y-%m-%d')}")

            for cycle_hour in reversed(self.CYCLE_HOURS):
                try:
                    data = await self._fetch_wcofs_file(
                        model_date=try_date,
                        cycle_hour=cycle_hour,
                        forecast_hour=forecast_hour,
                        bounds=bounds
                    )
                    if data is not None:
                        if day_offset > 0:
                            data['metadata']['fallback_days'] = day_offset
                        data['metadata']['cycle_hour'] = cycle_hour
                        return data
                except Exception as e:
                    logger.debug(f"Failed WCOFS {try_date.strftime('%Y-%m-%d')} t{cycle_hour:02d}z: {e}")
                    continue

        logger.error("Failed to fetch WCOFS data after trying multiple cycles and days")
        return None

    async def _fetch_wcofs_file(
        self,
        model_date: datetime,
        cycle_hour: int,
        forecast_hour: int,
        bounds: Dict[str, float]
    ) -> Optional[Dict]:
        """
        Fetch and parse a single WCOFS 2ds file.

        Downloads via HTTP fileServer (more reliable than OPeNDAP for WCOFS).
        """
        cache_file = self._get_cache_path(model_date, cycle_hour, forecast_hour)

        # Check cache
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=6):
                logger.info(f"Using cached WCOFS data: {cache_file.name}")
                try:
                    return self._parse_wcofs_netcdf(cache_file, model_date, cycle_hour, forecast_hour, bounds)
                except Exception as e:
                    logger.warning(f"Failed to read cache: {e}")

        # Download file
        file_url = self._build_file_url(model_date, cycle_hour, forecast_hour, service="fileServer")
        logger.info(f"Downloading WCOFS from: {file_url}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(file_url)

                if response.status_code == 404:
                    logger.debug(f"WCOFS file not found (404): {file_url}")
                    return None

                response.raise_for_status()
                nc_bytes = response.content

                if len(nc_bytes) < 1000:
                    logger.warning(f"WCOFS response too small ({len(nc_bytes)} bytes)")
                    return None

                logger.info(f"Downloaded {len(nc_bytes) / (1024*1024):.1f} MB of WCOFS data")

                # Cache the file
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    f.write(nc_bytes)

                return self._parse_wcofs_netcdf(cache_file, model_date, cycle_hour, forecast_hour, bounds)

        except httpx.HTTPStatusError as e:
            logger.debug(f"HTTP error fetching WCOFS: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error fetching WCOFS: {e}")
            return None

    def _parse_wcofs_netcdf(
        self,
        nc_file: Path,
        model_date: datetime,
        cycle_hour: int,
        forecast_hour: int,
        bounds: Dict[str, float]
    ) -> Optional[Dict]:
        """
        Parse WCOFS NetCDF data into usable arrays.

        Returns same dict structure as RTOFSFetcher for compatibility.
        """
        try:
            ds = xr.open_dataset(nc_file)

            logger.info(f"WCOFS variables: {list(ds.data_vars.keys())}")
            logger.info(f"WCOFS dimensions: {dict(ds.dims)}")

            # Find U/V variables
            u_velocity = None
            v_velocity = None
            u_var_name = None
            v_var_name = None

            for name in self.U_VAR_NAMES:
                if name in ds.data_vars:
                    u_var_name = name
                    u_var = ds[name]
                    # Squeeze out time dimension if present
                    if 'ocean_time' in u_var.dims:
                        u_var = u_var.isel(ocean_time=0)
                    elif 'time' in u_var.dims:
                        u_var = u_var.isel(time=0)
                    u_velocity = u_var.values
                    logger.info(f"Found U velocity as '{name}', shape: {u_velocity.shape}")
                    break

            for name in self.V_VAR_NAMES:
                if name in ds.data_vars:
                    v_var_name = name
                    v_var = ds[name]
                    if 'ocean_time' in v_var.dims:
                        v_var = v_var.isel(ocean_time=0)
                    elif 'time' in v_var.dims:
                        v_var = v_var.isel(time=0)
                    v_velocity = v_var.values
                    logger.info(f"Found V velocity as '{name}', shape: {v_velocity.shape}")
                    break

            if u_velocity is None or v_velocity is None:
                logger.error(f"Could not find U/V velocity in WCOFS. Available: {list(ds.data_vars.keys())}")
                ds.close()
                return None

            # Find coordinate arrays
            lats = None
            lons = None

            for name in self.LAT_NAMES:
                if name in ds.coords or name in ds.data_vars or name in ds.dims:
                    lats = ds[name].values
                    logger.info(f"Found latitudes as '{name}', shape: {lats.shape}")
                    break

            for name in self.LON_NAMES:
                if name in ds.coords or name in ds.data_vars or name in ds.dims:
                    lons = ds[name].values
                    logger.info(f"Found longitudes as '{name}', shape: {lons.shape}")
                    break

            if lats is None or lons is None:
                logger.error("Could not find lat/lon coordinates in WCOFS data")
                ds.close()
                return None

            ds.close()

            # Extract region of interest
            # WCOFS coordinates should already be in -180/180 range
            if lons.ndim == 2:
                # Curvilinear ROMS grid (2D lat/lon)
                region_mask = (
                    (lats >= bounds['min_lat']) &
                    (lats <= bounds['max_lat']) &
                    (lons >= bounds['min_lon']) &
                    (lons <= bounds['max_lon'])
                )

                if not region_mask.any():
                    logger.warning("No WCOFS data in requested region")
                    return None

                rows, cols = np.where(region_mask)
                y_min, y_max = rows.min(), rows.max() + 1
                x_min, x_max = cols.min(), cols.max() + 1

                u_velocity = u_velocity[y_min:y_max, x_min:x_max]
                v_velocity = v_velocity[y_min:y_max, x_min:x_max]
                lats_region = lats[y_min:y_max, x_min:x_max]
                lons_region = lons[y_min:y_max, x_min:x_max]
            else:
                # Regular grid (1D lat/lon)
                lat_mask = (lats >= bounds['min_lat']) & (lats <= bounds['max_lat'])
                lon_mask = (lons >= bounds['min_lon']) & (lons <= bounds['max_lon'])

                if not lat_mask.any() or not lon_mask.any():
                    logger.warning("No WCOFS data in requested region")
                    return None

                lat_idx = np.where(lat_mask)[0]
                lon_idx = np.where(lon_mask)[0]

                u_velocity = u_velocity[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
                v_velocity = v_velocity[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
                lats_region = lats[lat_idx[0]:lat_idx[-1]+1]
                lons_region = lons[lon_idx[0]:lon_idx[-1]+1]
                # Make 2D for consistent output
                lons_region, lats_region = np.meshgrid(lons_region, lats_region)

            # Calculate derived fields
            current_speed = np.sqrt(u_velocity**2 + v_velocity**2)
            current_direction = (90 - np.arctan2(v_velocity, u_velocity) * 180 / np.pi) % 360

            valid_time = model_date.replace(hour=cycle_hour) + timedelta(hours=forecast_hour)

            result = {
                'u_velocity': u_velocity,
                'v_velocity': v_velocity,
                'current_speed': current_speed,
                'current_direction': current_direction,
                'lons': lons_region,
                'lats': lats_region,
                'metadata': {
                    'model': 'wcofs',
                    'model_time': model_date.isoformat(),
                    'cycle_hour': cycle_hour,
                    'forecast_hour': forecast_hour,
                    'valid_time': valid_time.isoformat(),
                    'shape': current_speed.shape,
                    'bounds': {
                        'min_lat': float(np.nanmin(lats_region)),
                        'max_lat': float(np.nanmax(lats_region)),
                        'min_lon': float(np.nanmin(lons_region)),
                        'max_lon': float(np.nanmax(lons_region)),
                    },
                    'units': {
                        'velocity': 'm/s',
                        'direction': 'degrees (oceanographic convention)'
                    }
                }
            }

            logger.info(f"Parsed WCOFS data: shape {current_speed.shape}, "
                       f"speed range {np.nanmin(current_speed):.3f}-{np.nanmax(current_speed):.3f} m/s")

            return result

        except Exception as e:
            logger.error(f"Failed to parse WCOFS NetCDF: {e}", exc_info=True)
            return None

    @staticmethod
    def ms_to_knots(speed_ms: float) -> float:
        """Convert speed from m/s to knots."""
        return speed_ms * 1.94384

    @staticmethod
    def knots_to_ms(speed_knots: float) -> float:
        """Convert speed from knots to m/s."""
        return speed_knots / 1.94384
