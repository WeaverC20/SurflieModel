"""
RTOFS (Real-Time Ocean Forecast System) Data Fetcher

Fetches ocean current data from NOAA's Global RTOFS model for tile generation.

Model Info:
- Global HYCOM-based ocean forecast
- 1/12° resolution (~9km)
- Updated daily at 00Z
- 8-day forecast (192 hours)
- Provides U/V velocity, temperature, salinity

Documentation: https://polar.ncep.noaa.gov/global/
"""

import logging
import math
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path

import httpx
import numpy as np

from .config import RTOFS_CONFIG, REGIONS, GRIB_CACHE_DIR

logger = logging.getLogger(__name__)

# Try to import GRIB parsing libraries
try:
    import xarray as xr
    import cfgrib
    GRIB_PARSING_AVAILABLE = True
except ImportError:
    GRIB_PARSING_AVAILABLE = False
    logger.warning("cfgrib/xarray not available. Install with: pip install cfgrib xarray")


class RTOFSFetcher:
    """Fetches ocean current data from NOAA RTOFS model"""

    def __init__(self):
        """Initialize RTOFS fetcher"""
        # Direct NetCDF file access (filter service is not operational)
        self.prod_url = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtofs/prod/"

    async def fetch_current_grid(
        self,
        region_name: str = "california",
        forecast_hour: int = 0,
        model_date: Optional[datetime] = None
    ) -> Optional[Dict]:
        """
        Fetch RTOFS current data for an entire region grid

        Args:
            region_name: Name of region from config (e.g., 'california')
            forecast_hour: Forecast hour (0-192)
            model_date: Model run date (defaults to today at 00Z)

        Returns:
            Dictionary containing:
            - u_velocity: 2D array of eastward velocity (m/s)
            - v_velocity: 2D array of northward velocity (m/s)
            - lons: 1D array of longitudes
            - lats: 1D array of latitudes
            - current_speed: 2D array of current speed (m/s)
            - current_direction: 2D array of current direction (degrees)
            - metadata: Model run info
        """
        if not GRIB_PARSING_AVAILABLE:
            logger.error("GRIB parsing libraries not available")
            return None

        if region_name not in REGIONS:
            logger.error(f"Unknown region: {region_name}")
            return None

        region = REGIONS[region_name]
        bounds = region.bounds

        # Default to today's 00Z run
        if model_date is None:
            model_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        logger.info(f"Fetching RTOFS data for {region_name} at forecast hour {forecast_hour}")
        logger.info(f"Model date: {model_date.strftime('%Y-%m-%d %HZ')}")
        logger.info(f"Region bounds: {bounds}")

        # Try current model run, then fallback to previous days
        for day_offset in [0, 1, 2]:
            try_date = model_date - timedelta(days=day_offset)

            if day_offset > 0:
                logger.info(f"Trying fallback to {try_date.strftime('%Y-%m-%d')}")

            try:
                data = await self._fetch_rtofs_grib(
                    model_date=try_date,
                    forecast_hour=forecast_hour,
                    bounds=bounds
                )

                if data is not None:
                    if day_offset > 0:
                        data['metadata']['fallback_days'] = day_offset
                    return data

            except Exception as e:
                logger.warning(f"Failed to fetch RTOFS for {try_date.strftime('%Y-%m-%d')}: {e}")
                continue

        logger.error(f"Failed to fetch RTOFS data after trying 3 days")
        return None

    async def _fetch_rtofs_grib(
        self,
        model_date: datetime,
        forecast_hour: int,
        bounds: Dict[str, float]
    ) -> Optional[Dict]:
        """
        Fetch and parse RTOFS NetCDF data

        Args:
            model_date: Model run date
            forecast_hour: Forecast hour (0-192)
            bounds: Geographic bounds dict with min_lat, max_lat, min_lon, max_lon

        Returns:
            Parsed data dictionary or None
        """
        date_str = model_date.strftime('%Y%m%d')

        # RTOFS file naming: rtofs_glo_2ds_f{NNN}_diag.nc
        file_name = f"rtofs_glo_2ds_f{forecast_hour:03d}_diag.nc"

        # Full URL to NetCDF file
        file_url = f"{self.prod_url}rtofs.{date_str}/{file_name}"

        # Check cache first
        cache_file = GRIB_CACHE_DIR / f"rtofs_{date_str}_f{forecast_hour:03d}.nc"

        if cache_file.exists():
            # Check if cache is recent (less than 1 day old)
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(days=1):
                logger.info(f"Using cached RTOFS data: {cache_file}")
                try:
                    return self._parse_rtofs_netcdf(cache_file, model_date, forecast_hour, bounds)
                except Exception as e:
                    logger.warning(f"Failed to read cache: {e}")

        # Download full NetCDF file
        logger.info(f"Downloading RTOFS NetCDF from: {file_url}")

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:  # Longer timeout for large files
                response = await client.get(file_url)

                if response.status_code == 404:
                    logger.warning(f"RTOFS data not found (404) for {date_str} hour {forecast_hour}")
                    return None

                response.raise_for_status()
                nc_bytes = response.content

                if len(nc_bytes) < 1000:
                    logger.warning(f"RTOFS response too small ({len(nc_bytes)} bytes), likely empty")
                    return None

                logger.info(f"Downloaded {len(nc_bytes) / (1024*1024):.1f} MB of RTOFS data")

                # Cache the NetCDF file
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    f.write(nc_bytes)
                logger.info(f"Cached RTOFS data to {cache_file}")

                # Parse the data
                return self._parse_rtofs_netcdf(cache_file, model_date, forecast_hour, bounds)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching RTOFS: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching RTOFS: {e}")
            return None

    def _parse_rtofs_netcdf(
        self,
        nc_file: Path,
        model_date: datetime,
        forecast_hour: int,
        bounds: Dict[str, float]
    ) -> Optional[Dict]:
        """
        Parse RTOFS NetCDF data into usable arrays

        Args:
            nc_file: Path to NetCDF file
            model_date: Model run date
            forecast_hour: Forecast hour
            bounds: Geographic bounds for region extraction

        Returns:
            Parsed data dictionary
        """
        if not GRIB_PARSING_AVAILABLE:
            return None

        try:
            # Open NetCDF file with xarray
            ds = xr.open_dataset(nc_file)

            logger.info(f"RTOFS dataset variables: {list(ds.data_vars.keys())}")
            logger.info(f"RTOFS dataset dimensions: {dict(ds.dims)}")
            logger.info(f"RTOFS dataset coords: {list(ds.coords.keys())}")

            # Extract coordinate arrays
            # RTOFS uses 'Latitude' and 'Longitude' (capital L)
            lat_names = ['Latitude', 'latitude', 'lat', 'Y']
            lon_names = ['Longitude', 'longitude', 'lon', 'X']

            lats = None
            lons = None

            for name in lat_names:
                if name in ds.coords or name in ds.dims:
                    lats = ds[name].values
                    logger.info(f"Found latitudes as '{name}'")
                    break

            for name in lon_names:
                if name in ds.coords or name in ds.dims:
                    lons = ds[name].values
                    logger.info(f"Found longitudes as '{name}'")
                    break

            if lats is None or lons is None:
                logger.error(f"Could not find lat/lon coordinates")
                return None

            # Extract region of interest
            # RTOFS uses curvilinear grid (2D lat/lon arrays)
            # Convert bounds to 0-360 if needed (RTOFS uses 0-360 longitude)
            min_lon = bounds['min_lon'] if bounds['min_lon'] >= 0 else bounds['min_lon'] + 360
            max_lon = bounds['max_lon'] if bounds['max_lon'] >= 0 else bounds['max_lon'] + 360

            logger.info(f"Searching for region: lat [{bounds['min_lat']}, {bounds['max_lat']}], lon [{min_lon}, {max_lon}]")
            logger.info(f"Grid lat range: [{np.min(lats):.2f}, {np.max(lats):.2f}]")
            logger.info(f"Grid lon range: [{np.min(lons):.2f}, {np.max(lons):.2f}]")

            # Find indices for region in 2D grid
            region_mask = (
                (lats >= bounds['min_lat']) &
                (lats <= bounds['max_lat']) &
                (lons >= min_lon) &
                (lons <= max_lon)
            )

            if not region_mask.any():
                logger.warning("No data in requested region")
                return None

            # Find bounding box of region in grid coordinates
            rows, cols = np.where(region_mask)
            y_min, y_max = rows.min(), rows.max() + 1
            x_min, x_max = cols.min(), cols.max() + 1

            logger.info(f"Region found at grid indices: Y[{y_min}:{y_max}], X[{x_min}:{x_max}]")

            y_slice = slice(y_min, y_max)
            x_slice = slice(x_min, x_max)

            # Extract U and V velocity components
            # RTOFS variable names: 'u_barotropic_velocity' and 'v_barotropic_velocity'
            u_var_names = ['u_barotropic_velocity', 'u_velocity', 'u', 'uvel', 'water_u']
            v_var_names = ['v_barotropic_velocity', 'v_velocity', 'v', 'vvel', 'water_v']

            u_velocity = None
            v_velocity = None

            for var in u_var_names:
                if var in ds.data_vars:
                    u_var = ds[var]
                    # Extract time dimension if present
                    if 'MT' in u_var.dims:
                        u_var = u_var.isel(MT=0)
                    # Extract surface layer if 3D
                    if 'Depth' in u_var.dims:
                        u_var = u_var.isel(Depth=0)
                    elif 'depth' in u_var.dims:
                        u_var = u_var.isel(depth=0)
                    # Extract region
                    u_velocity = u_var[y_slice, x_slice].values
                    logger.info(f"Found U velocity as '{var}', shape: {u_velocity.shape}")
                    break

            for var in v_var_names:
                if var in ds.data_vars:
                    v_var = ds[var]
                    # Extract time dimension if present
                    if 'MT' in v_var.dims:
                        v_var = v_var.isel(MT=0)
                    # Extract surface layer if 3D
                    if 'Depth' in v_var.dims:
                        v_var = v_var.isel(Depth=0)
                    elif 'depth' in v_var.dims:
                        v_var = v_var.isel(depth=0)
                    # Extract region
                    v_velocity = v_var[y_slice, x_slice].values
                    logger.info(f"Found V velocity as '{var}', shape: {v_velocity.shape}")
                    break

            if u_velocity is None or v_velocity is None:
                logger.error(f"Could not find U/V velocity in RTOFS data. Available: {list(ds.data_vars.keys())}")
                return None

            # Extract coordinates for region (2D lat/lon arrays)
            lats_region = lats[y_slice, x_slice]
            lons_region = lons[y_slice, x_slice]

            # Convert longitude from 0-360 to -180 to 180 for tile generation
            lons_region = np.where(lons_region > 180, lons_region - 360, lons_region)

            # Calculate current speed and direction
            current_speed = np.sqrt(u_velocity**2 + v_velocity**2)  # m/s

            # Current direction (oceanographic convention: direction TOWARD which current flows)
            current_direction = (90 - np.arctan2(v_velocity, u_velocity) * 180 / np.pi) % 360

            # Calculate valid time
            valid_time = model_date + timedelta(hours=forecast_hour)

            result = {
                'u_velocity': u_velocity,
                'v_velocity': v_velocity,
                'current_speed': current_speed,
                'current_direction': current_direction,
                'lons': lons_region,
                'lats': lats_region,
                'metadata': {
                    'model': 'rtofs',
                    'model_time': model_date.isoformat(),
                    'forecast_hour': forecast_hour,
                    'valid_time': valid_time.isoformat(),
                    'shape': current_speed.shape,
                    'bounds': {
                        'min_lat': float(np.min(lats_region)),
                        'max_lat': float(np.max(lats_region)),
                        'min_lon': float(np.min(lons_region)),
                        'max_lon': float(np.max(lons_region)),
                    },
                    'units': {
                        'velocity': 'm/s',
                        'direction': 'degrees (oceanographic convention)'
                    }
                }
            }

            ds.close()

            logger.info(f"Parsed RTOFS data: shape {current_speed.shape}, "
                       f"speed range {np.nanmin(current_speed):.3f}-{np.nanmax(current_speed):.3f} m/s")

            return result

        except Exception as e:
            logger.error(f"Failed to parse RTOFS NetCDF: {e}", exc_info=True)
            return None

    @staticmethod
    def ms_to_knots(speed_ms: float) -> float:
        """Convert speed from m/s to knots"""
        return speed_ms * 1.94384

    @staticmethod
    def knots_to_ms(speed_knots: float) -> float:
        """Convert speed from knots to m/s"""
        return speed_knots / 1.94384
