"""
GFS Wind Forecast Fetcher

Fetches wind forecast data from NOAA's Global Forecast System (GFS).
"""

import logging
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import httpx

logger = logging.getLogger(__name__)

# Try to import GRIB parsing libraries
try:
    import xarray as xr
    import cfgrib
    GRIB_PARSING_AVAILABLE = True
except ImportError:
    GRIB_PARSING_AVAILABLE = False
    logger.warning("cfgrib/xarray not available. Install with: pip install cfgrib xarray")


class GFSWindFetcher:
    """Fetches wind forecast data from NOAA GFS model

    GFS provides global weather forecasts including wind speed and direction.
    Data available via NOMADS server in GRIB2 format.

    Documentation: https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
    """

    def __init__(self):
        """Initialize GFS wind fetcher"""
        self.nomads_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

    def _get_latest_cycle(self) -> Tuple[datetime, int]:
        """Get the latest available GFS cycle

        GFS runs 4 times daily at 00, 06, 12, 18 UTC
        Data typically available 3-4 hours after cycle time

        Returns:
            Tuple of (cycle_time, cycle_hour)
        """
        now = datetime.utcnow()

        # Find most recent cycle (00, 06, 12, 18)
        cycle_hour = (now.hour // 6) * 6
        cycle_time = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

        # If less than 4 hours since cycle, use previous cycle
        if (now - cycle_time) < timedelta(hours=4):
            cycle_time -= timedelta(hours=6)
            cycle_hour = cycle_time.hour

        return cycle_time, cycle_hour

    async def fetch_wind_grid(
        self,
        min_lat: float = 32.0,
        max_lat: float = 42.0,
        min_lon: float = -125.0,
        max_lon: float = -117.0,
        forecast_hour: int = 0
    ) -> Dict:
        """Fetch wind forecast grid for specified region

        Args:
            min_lat: Minimum latitude (degrees North)
            max_lat: Maximum latitude (degrees North)
            min_lon: Minimum longitude (degrees East, use negative for West)
            max_lon: Maximum longitude (degrees East, use negative for West)
            forecast_hour: Forecast hour (0 = current, 3, 6, 9, etc.)

        Returns:
            Dict containing:
                - lat: Array of latitudes
                - lon: Array of longitudes
                - u_wind: U-component of wind (m/s) at 10m height
                - v_wind: V-component of wind (m/s) at 10m height
                - wind_speed: Wind speed magnitude (m/s)
                - wind_direction: Wind direction (degrees, meteorological convention)
                - forecast_time: ISO timestamp of forecast
                - cycle_time: ISO timestamp of model cycle
        """
        logger.info(f"Fetching GFS wind forecast for {min_lat}°N to {max_lat}°N, {min_lon}°E to {max_lon}°E")

        if not GRIB_PARSING_AVAILABLE:
            logger.error("GRIB parsing libraries not available. Install with: pip install cfgrib xarray")
            return self._create_synthetic_data(min_lat, max_lat, min_lon, max_lon, forecast_hour)

        cycle_time, cycle_hour = self._get_latest_cycle()

        # Normalize longitude to 0-360 for NOAA data
        left_lon = min_lon if min_lon >= 0 else min_lon + 360
        right_lon = max_lon if max_lon >= 0 else max_lon + 360

        # Try up to 3 previous model runs
        for run_offset in [0, 6, 12]:
            try_time = cycle_time - timedelta(hours=run_offset)
            try_hour = try_time.hour

            params = {
                "file": f"gfs.t{try_hour:02d}z.pgrb2.0p25.f{forecast_hour:03d}",
                "lev_10_m_above_ground": "on",
                "var_UGRD": "on",  # U-component of wind
                "var_VGRD": "on",  # V-component of wind
                "var_GUST": "on",  # Wind gust
                "subregion": "",
                "leftlon": left_lon,
                "rightlon": right_lon,
                "toplat": max_lat,
                "bottomlat": min_lat,
                "dir": f"/gfs.{try_time.strftime('%Y%m%d')}/{try_hour:02d}/atmos"
            }

            try:
                logger.info(f"Fetching GFS data from {try_time.isoformat()} run, forecast hour {forecast_hour}")

                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.get(self.nomads_url, params=params)
                    response.raise_for_status()

                    logger.info(f"Downloaded {len(response.content) / 1024:.1f} KB of GFS wind data")

                    # Parse GRIB2 data
                    result = self._parse_grib2_wind(response.content, try_time, forecast_hour, min_lon, max_lon)
                    if result:
                        return result
                    else:
                        logger.warning("Failed to parse GRIB2 data, trying next model run...")
                        continue

            except httpx.HTTPError as e:
                logger.warning(f"Model run {try_time.isoformat()} not available: {e}")
                if run_offset < 12:
                    continue
                else:
                    logger.error("All model runs failed, falling back to synthetic data")
                    return self._create_synthetic_data(min_lat, max_lat, min_lon, max_lon, forecast_hour)

        # If we get here, all attempts failed
        logger.error("Failed to fetch GFS data, using synthetic fallback")
        return self._create_synthetic_data(min_lat, max_lat, min_lon, max_lon, forecast_hour)

    def _parse_grib2_wind(
        self,
        grib_bytes: bytes,
        cycle_time: datetime,
        forecast_hour: int,
        original_min_lon: float,
        original_max_lon: float
    ) -> Optional[Dict]:
        """Parse GRIB2 wind data into grid arrays

        Args:
            grib_bytes: Raw GRIB2 data
            cycle_time: Model cycle time
            forecast_hour: Forecast hour
            original_min_lon: Original minimum longitude (may be negative)
            original_max_lon: Original maximum longitude (may be negative)

        Returns:
            Dictionary with wind grid data or None if parsing fails
        """
        try:
            # Write to temporary file (cfgrib requires file path)
            with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False, mode='wb') as tmp:
                tmp.write(grib_bytes)
                tmp_path = tmp.name

            try:
                # Open with xarray/cfgrib
                ds = xr.open_dataset(tmp_path, engine='cfgrib')

                # Extract U and V wind components at 10m
                u_wind = ds['u10'].values  # U-component (eastward)
                v_wind = ds['v10'].values  # V-component (northward)
                lats = ds['latitude'].values
                lons = ds['longitude'].values

                # Convert longitudes back to -180/180 if needed
                if original_min_lon < 0:
                    lons = np.where(lons > 180, lons - 360, lons)

                # Calculate wind speed and direction
                wind_speed = np.sqrt(u_wind**2 + v_wind**2)
                # Meteorological convention: direction FROM which wind blows
                wind_direction = (270 - np.arctan2(v_wind, u_wind) * 180 / np.pi) % 360

                forecast_time = cycle_time + timedelta(hours=forecast_hour)

                ds.close()

                logger.info(f"Successfully parsed GFS grid: {len(lats)} x {len(lons)} points")

                return {
                    "lat": lats.tolist(),
                    "lon": lons.tolist(),
                    "u_wind": u_wind.tolist(),
                    "v_wind": v_wind.tolist(),
                    "wind_speed": wind_speed.tolist(),
                    "wind_direction": wind_direction.tolist(),
                    "forecast_time": forecast_time.isoformat(),
                    "cycle_time": cycle_time.isoformat(),
                    "forecast_hour": forecast_hour,
                    "resolution_deg": 0.25,
                    "model": "GFS",
                    "units": {
                        "wind_speed": "m/s",
                        "wind_direction": "degrees (meteorological)",
                        "lat": "degrees_north",
                        "lon": "degrees_east"
                    }
                }

            finally:
                # Clean up temp file
                import os
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Failed to parse GRIB2 wind data: {e}")
            return None

    def _create_synthetic_data(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        forecast_hour: int
    ) -> Dict:
        """Create synthetic wind data as fallback

        This is used when real data fetching fails.
        """
        logger.warning("Using synthetic wind data - real GFS data not available")

        cycle_time, _ = self._get_latest_cycle()

        # Create coordinate arrays
        lat_resolution = 0.25
        lon_resolution = 0.25

        lats = np.arange(min_lat, max_lat + lat_resolution, lat_resolution)
        lons = np.arange(min_lon, max_lon + lon_resolution, lon_resolution)

        height, width = len(lats), len(lons)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

        # Simple land mask for California coast
        is_land = np.zeros((height, width), dtype=bool)
        for i in range(height):
            for j in range(width):
                lat = lats[i]
                lon = lons[j]
                if lat < 34:
                    is_land[i, j] = lon > -118
                elif lat < 38:
                    is_land[i, j] = lon > -120
                else:
                    is_land[i, j] = lon > -124

        # Synthetic prevailing NW winds
        u_wind = np.full((height, width), 8.0)
        v_wind = np.full((height, width), -3.0)

        distance_from_coast = np.abs(lon_grid - (-120))
        latitude_factor = (lat_grid - min_lat) / (max_lat - min_lat)

        u_wind = u_wind + 2 * distance_from_coast + 3 * latitude_factor
        v_wind = v_wind - 2 * latitude_factor

        u_wind += np.random.randn(height, width) * 0.5
        v_wind += np.random.randn(height, width) * 0.5

        u_wind[is_land] = np.nan
        v_wind[is_land] = np.nan

        wind_speed = np.sqrt(u_wind**2 + v_wind**2)
        wind_direction = (270 - np.arctan2(v_wind, u_wind) * 180 / np.pi) % 360

        forecast_time = cycle_time + timedelta(hours=forecast_hour)

        return {
            "lat": lats.tolist(),
            "lon": lons.tolist(),
            "u_wind": u_wind.tolist(),
            "v_wind": v_wind.tolist(),
            "wind_speed": wind_speed.tolist(),
            "wind_direction": wind_direction.tolist(),
            "forecast_time": forecast_time.isoformat(),
            "cycle_time": cycle_time.isoformat(),
            "forecast_hour": forecast_hour,
            "resolution_deg": lat_resolution,
            "model": "GFS (synthetic fallback)",
            "units": {
                "wind_speed": "m/s",
                "wind_direction": "degrees (meteorological)",
                "lat": "degrees_north",
                "lon": "degrees_east"
            }
        }
