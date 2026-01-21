"""
WaveWatch III Wave Forecast Fetcher

Fetches wave forecast data from NOAA's WaveWatch III model.
Supports optional storage to Zarr via WaveDataStore.
"""

import logging
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import httpx

logger = logging.getLogger(__name__)

# Try to import storage module
try:
    from data.storage import WaveDataStore, MetadataDB, get_storage_config
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    logger.debug("Storage module not available - fetch-only mode")

# Try to import GRIB parsing libraries
try:
    import xarray as xr
    import cfgrib
    GRIB_PARSING_AVAILABLE = True
except ImportError:
    GRIB_PARSING_AVAILABLE = False
    logger.warning("cfgrib/xarray not available. Install with: pip install cfgrib xarray")


class WaveWatchFetcher:
    """Fetches wave forecast data from NOAA WaveWatch III model

    WaveWatch III is NOAA's operational wave prediction model providing
    significant wave height, period, and direction forecasts.

    Documentation: https://polar.ncep.noaa.gov/waves/
    """

    def __init__(self):
        """Initialize WaveWatch III fetcher"""
        self.nomads_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfswave.pl"

    def _get_latest_cycle(self) -> Tuple[datetime, int]:
        """Get the latest available WaveWatch III cycle

        WaveWatch III runs 4 times daily at 00, 06, 12, 18 UTC

        Returns:
            Tuple of (cycle_time, cycle_hour)
        """
        now = datetime.utcnow()

        # Find most recent cycle (00, 06, 12, 18)
        cycle_hour = (now.hour // 6) * 6
        cycle_time = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

        # If less than 5 hours since cycle, use previous cycle
        if (now - cycle_time) < timedelta(hours=5):
            cycle_time -= timedelta(hours=6)
            cycle_hour = cycle_time.hour

        return cycle_time, cycle_hour

    async def fetch_wave_grid(
        self,
        min_lat: float = 32.0,
        max_lat: float = 42.0,
        min_lon: float = -125.0,
        max_lon: float = -117.0,
        forecast_hour: int = 0
    ) -> Dict:
        """Fetch wave forecast grid for specified region

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
                - significant_wave_height: Significant wave height (m)
                - peak_wave_period: Peak wave period (s)
                - mean_wave_direction: Mean wave direction (degrees)
                - wind_sea_height: Wind sea height (m)
                - swell_height: Swell height (m)
                - forecast_time: ISO timestamp of forecast
                - cycle_time: ISO timestamp of model cycle
        """
        logger.info(f"Fetching WaveWatch III wave forecast for {min_lat}°N to {max_lat}°N, {min_lon}°E to {max_lon}°E")

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
                "file": f"gfswave.t{try_hour:02d}z.global.0p25.f{forecast_hour:03d}.grib2",
                "lev_surface": "on",
                # Combined sea state
                "var_HTSGW": "on",  # Significant wave height (combined)
                "var_PERPW": "on",  # Peak wave period (combined)
                "var_DIRPW": "on",  # Mean wave direction (combined)
                # Wind waves
                "var_WVHGT": "on",  # Wind wave height
                "var_WVPER": "on",  # Wind wave period
                "var_WVDIR": "on",  # Wind wave direction
                # Primary swell (partition 1)
                "var_SWELL": "on",  # Primary swell height
                "var_SWPER": "on",  # Primary swell period
                "var_SWDIR": "on",  # Primary swell direction
                # Secondary swell (partition 2)
                "var_SWELL_2": "on",  # Secondary swell height
                "var_SWPER_2": "on",  # Secondary swell period
                "var_SWDIR_2": "on",  # Secondary swell direction
                # Tertiary swell (partition 3)
                "var_SWELL_3": "on",  # Tertiary swell height
                "var_SWPER_3": "on",  # Tertiary swell period
                "var_SWDIR_3": "on",  # Tertiary swell direction
                "subregion": "",
                "leftlon": left_lon,
                "rightlon": right_lon,
                "toplat": max_lat,
                "bottomlat": min_lat,
                "dir": f"/gfs.{try_time.strftime('%Y%m%d')}/{try_hour:02d}/wave/gridded"
            }

            try:
                logger.info(f"Fetching WaveWatch III data from {try_time.isoformat()} run, forecast hour {forecast_hour}")

                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.get(self.nomads_url, params=params)
                    response.raise_for_status()

                    logger.info(f"Downloaded {len(response.content) / 1024:.1f} KB of WaveWatch III data")

                    # Parse GRIB2 data
                    result = self._parse_grib2_wave(response.content, try_time, forecast_hour, min_lon, max_lon)
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
        logger.error("Failed to fetch WaveWatch III data, using synthetic fallback")
        return self._create_synthetic_data(min_lat, max_lat, min_lon, max_lon, forecast_hour)

    def _parse_grib2_wave(
        self,
        grib_bytes: bytes,
        cycle_time: datetime,
        forecast_hour: int,
        original_min_lon: float,
        original_max_lon: float
    ) -> Optional[Dict]:
        """Parse GRIB2 wave data into grid arrays

        Args:
            grib_bytes: Raw GRIB2 data
            cycle_time: Model cycle time
            forecast_hour: Forecast hour
            original_min_lon: Original minimum longitude (may be negative)
            original_max_lon: Original maximum longitude (may be negative)

        Returns:
            Dictionary with wave grid data or None if parsing fails
        """
        try:
            # Write to temporary file (cfgrib requires file path)
            with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False, mode='wb') as tmp:
                tmp.write(grib_bytes)
                tmp_path = tmp.name

            try:
                # Open with xarray/cfgrib
                ds = xr.open_dataset(tmp_path, engine='cfgrib')

                # Extract wave variables
                # Note: Variable names may vary - log what's available
                logger.info(f"Available variables: {list(ds.data_vars.keys())}")

                lats = ds['latitude'].values
                lons = ds['longitude'].values

                # Convert longitudes back to -180/180 if needed
                if original_min_lon < 0:
                    lons = np.where(lons > 180, lons - 360, lons)

                # Extract wave height - try different possible variable names
                if 'swh' in ds:
                    significant_wave_height = ds['swh'].values
                elif 'htsgw' in ds:
                    significant_wave_height = ds['htsgw'].values
                else:
                    logger.warning("Could not find wave height variable")
                    significant_wave_height = np.zeros((len(lats), len(lons)))

                # Extract peak period
                if 'perpw' in ds:
                    peak_wave_period = ds['perpw'].values
                elif 'pp1d' in ds:
                    peak_wave_period = ds['pp1d'].values
                else:
                    logger.warning("Could not find peak period variable")
                    peak_wave_period = np.full((len(lats), len(lons)), 10.0)

                # Extract mean direction
                if 'dirpw' in ds:
                    mean_wave_direction = ds['dirpw'].values
                elif 'mwd' in ds:
                    mean_wave_direction = ds['mwd'].values
                else:
                    logger.warning("Could not find wave direction variable")
                    mean_wave_direction = np.full((len(lats), len(lons)), 315.0)

                # Extract wind wave components
                if 'wvhgt' in ds:
                    wind_wave_height = ds['wvhgt'].values
                else:
                    wind_wave_height = significant_wave_height * 0.3

                if 'wvper' in ds:
                    wind_wave_period = ds['wvper'].values
                else:
                    wind_wave_period = np.full_like(significant_wave_height, 5.0)

                if 'wvdir' in ds:
                    wind_wave_direction = ds['wvdir'].values
                else:
                    wind_wave_direction = mean_wave_direction.copy()

                # Extract primary swell (partition 1)
                if 'swell' in ds:
                    primary_swell_height = ds['swell'].values
                elif 'shts' in ds:  # Alternative variable name
                    primary_swell_height = ds['shts'].values
                else:
                    primary_swell_height = significant_wave_height * 0.7

                if 'swper' in ds:
                    primary_swell_period = ds['swper'].values
                elif 'mpts' in ds:
                    primary_swell_period = ds['mpts'].values
                else:
                    primary_swell_period = peak_wave_period.copy()

                if 'swdir' in ds:
                    primary_swell_direction = ds['swdir'].values
                elif 'mdts' in ds:
                    primary_swell_direction = ds['mdts'].values
                else:
                    primary_swell_direction = mean_wave_direction.copy()

                # Extract secondary swell (partition 2) - may not always be present
                secondary_swell_height = None
                secondary_swell_period = None
                secondary_swell_direction = None

                # Try different variable name patterns for secondary swell
                for height_var in ['swell_2', 'shts_2', 'swh2']:
                    if height_var in ds:
                        secondary_swell_height = ds[height_var].values
                        break

                for period_var in ['swper_2', 'mpts_2', 'mwp2']:
                    if period_var in ds:
                        secondary_swell_period = ds[period_var].values
                        break

                for dir_var in ['swdir_2', 'mdts_2', 'mwd2']:
                    if dir_var in ds:
                        secondary_swell_direction = ds[dir_var].values
                        break

                # Extract tertiary swell (partition 3) - may not always be present
                tertiary_swell_height = None
                tertiary_swell_period = None
                tertiary_swell_direction = None

                for height_var in ['swell_3', 'shts_3', 'swh3']:
                    if height_var in ds:
                        tertiary_swell_height = ds[height_var].values
                        break

                for period_var in ['swper_3', 'mpts_3', 'mwp3']:
                    if period_var in ds:
                        tertiary_swell_period = ds[period_var].values
                        break

                for dir_var in ['swdir_3', 'mdts_3', 'mwd3']:
                    if dir_var in ds:
                        tertiary_swell_direction = ds[dir_var].values
                        break

                forecast_time = cycle_time + timedelta(hours=forecast_hour)

                ds.close()

                logger.info(f"Successfully parsed WaveWatch III grid: {len(lats)} x {len(lons)} points")
                logger.info(f"Secondary swell available: {secondary_swell_height is not None}")
                logger.info(f"Tertiary swell available: {tertiary_swell_height is not None}")

                result = {
                    "lat": lats.tolist(),
                    "lon": lons.tolist(),
                    # Combined sea state
                    "significant_wave_height": significant_wave_height.tolist(),
                    "peak_wave_period": peak_wave_period.tolist(),
                    "mean_wave_direction": mean_wave_direction.tolist(),
                    # Wind waves
                    "wind_wave_height": wind_wave_height.tolist(),
                    "wind_wave_period": wind_wave_period.tolist(),
                    "wind_wave_direction": wind_wave_direction.tolist(),
                    # Primary swell
                    "primary_swell_height": primary_swell_height.tolist(),
                    "primary_swell_period": primary_swell_period.tolist(),
                    "primary_swell_direction": primary_swell_direction.tolist(),
                    # Metadata
                    "forecast_time": forecast_time.isoformat(),
                    "cycle_time": cycle_time.isoformat(),
                    "forecast_hour": forecast_hour,
                    "resolution_deg": 0.25,
                    "model": "WaveWatch III",
                    "units": {
                        "wave_height": "m",
                        "wave_period": "s",
                        "wave_direction": "degrees (direction from)",
                        "lat": "degrees_north",
                        "lon": "degrees_east"
                    }
                }

                # Add secondary swell if available
                if secondary_swell_height is not None:
                    result["secondary_swell_height"] = secondary_swell_height.tolist()
                if secondary_swell_period is not None:
                    result["secondary_swell_period"] = secondary_swell_period.tolist()
                if secondary_swell_direction is not None:
                    result["secondary_swell_direction"] = secondary_swell_direction.tolist()

                # Add tertiary swell if available
                if tertiary_swell_height is not None:
                    result["tertiary_swell_height"] = tertiary_swell_height.tolist()
                if tertiary_swell_period is not None:
                    result["tertiary_swell_period"] = tertiary_swell_period.tolist()
                if tertiary_swell_direction is not None:
                    result["tertiary_swell_direction"] = tertiary_swell_direction.tolist()

                # Keep legacy fields for backwards compatibility
                result["wind_sea_height"] = result["wind_wave_height"]
                result["swell_height"] = result["primary_swell_height"]

                return result

            finally:
                # Clean up temp file
                import os
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Failed to parse GRIB2 wave data: {e}")
            return None

    def _create_synthetic_data(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        forecast_hour: int
    ) -> Dict:
        """Create synthetic wave data as fallback

        This is used when real data fetching fails.
        """
        logger.warning("Using synthetic wave data - real WaveWatch III data not available")

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

        # Distance from coast (roughly)
        distance_from_coast = np.abs(lon_grid - (-120))

        # Synthetic wave heights - larger offshore, smaller nearshore
        significant_wave_height = 1.5 + 1.5 * distance_from_coast / np.abs(min_lon - (-120)) + np.random.randn(height, width) * 0.3
        significant_wave_height = np.maximum(0.5, significant_wave_height)

        # Peak period (typically 8-14 seconds)
        peak_wave_period = 10 + 2 * distance_from_coast / np.abs(min_lon - (-120)) + np.random.randn(height, width) * 1

        # Mean direction (mostly from NW in California)
        mean_wave_direction = 315 + np.random.randn(height, width) * 20

        # Wind waves (short period, locally generated)
        wind_wave_height = significant_wave_height * 0.3
        wind_wave_period = np.full((height, width), 5.0) + np.random.randn(height, width) * 0.5
        wind_wave_direction = mean_wave_direction + np.random.randn(height, width) * 30

        # Primary swell (long period, from distant storms)
        primary_swell_height = significant_wave_height * 0.6
        primary_swell_period = peak_wave_period + np.random.randn(height, width) * 1
        primary_swell_direction = 290 + np.random.randn(height, width) * 15  # WNW

        # Secondary swell (weaker, different direction)
        secondary_swell_height = significant_wave_height * 0.2
        secondary_swell_period = np.full((height, width), 8.0) + np.random.randn(height, width) * 1
        secondary_swell_direction = 200 + np.random.randn(height, width) * 15  # SSW

        # Mask land areas
        for arr in [significant_wave_height, peak_wave_period, mean_wave_direction,
                    wind_wave_height, wind_wave_period, wind_wave_direction,
                    primary_swell_height, primary_swell_period, primary_swell_direction,
                    secondary_swell_height, secondary_swell_period, secondary_swell_direction]:
            arr[is_land] = np.nan

        forecast_time = cycle_time + timedelta(hours=forecast_hour)

        return {
            "lat": lats.tolist(),
            "lon": lons.tolist(),
            # Combined sea state
            "significant_wave_height": significant_wave_height.tolist(),
            "peak_wave_period": peak_wave_period.tolist(),
            "mean_wave_direction": mean_wave_direction.tolist(),
            # Wind waves
            "wind_wave_height": wind_wave_height.tolist(),
            "wind_wave_period": wind_wave_period.tolist(),
            "wind_wave_direction": wind_wave_direction.tolist(),
            # Primary swell
            "primary_swell_height": primary_swell_height.tolist(),
            "primary_swell_period": primary_swell_period.tolist(),
            "primary_swell_direction": primary_swell_direction.tolist(),
            # Secondary swell
            "secondary_swell_height": secondary_swell_height.tolist(),
            "secondary_swell_period": secondary_swell_period.tolist(),
            "secondary_swell_direction": secondary_swell_direction.tolist(),
            # Legacy compatibility
            "wind_sea_height": wind_wave_height.tolist(),
            "swell_height": primary_swell_height.tolist(),
            # Metadata
            "forecast_time": forecast_time.isoformat(),
            "cycle_time": cycle_time.isoformat(),
            "forecast_hour": forecast_hour,
            "resolution_deg": lat_resolution,
            "model": "WaveWatch III (synthetic fallback)",
            "units": {
                "wave_height": "m",
                "wave_period": "s",
                "wave_direction": "degrees (direction from)",
                "lat": "degrees_north",
                "lon": "degrees_east"
            }
        }

    async def fetch_and_store(
        self,
        min_lat: float = 32.0,
        max_lat: float = 42.0,
        min_lon: float = -125.0,
        max_lon: float = -117.0,
        forecast_hours: Optional[List[int]] = None
    ) -> Optional[str]:
        """
        Fetch wave forecast and store to Zarr.

        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
            forecast_hours: List of forecast hours to fetch (default: 0-72 every 3h)

        Returns:
            Path to Zarr store or None if storage unavailable
        """
        if not STORAGE_AVAILABLE:
            logger.error("Storage module not available")
            return None

        if forecast_hours is None:
            # 3-hourly for first 48 hours, then 24-hourly out to 16 days (384 hours)
            forecast_hours = list(range(0, 49, 3)) + list(range(72, 385, 24))

        cycle_time, _ = self._get_latest_cycle()
        data_list = []

        logger.info(f"Fetching WaveWatch III for {len(forecast_hours)} forecast hours")

        for hour in forecast_hours:
            try:
                data = await self.fetch_wave_grid(
                    min_lat=min_lat,
                    max_lat=max_lat,
                    min_lon=min_lon,
                    max_lon=max_lon,
                    forecast_hour=hour
                )
                if data and 'synthetic' not in data.get('model', '').lower():
                    data_list.append(data)
                    logger.info(f"Fetched forecast hour {hour}")
            except Exception as e:
                logger.warning(f"Failed to fetch hour {hour}: {e}")

        if not data_list:
            logger.error("No data fetched successfully")
            return None

        # Store to Zarr
        store = WaveDataStore()
        store.store_forecast_range(data_list, cycle_time)

        # Record in metadata database
        config = get_storage_config()
        db = MetadataDB()
        db.record_forecast_cycle(
            model='ww3',
            cycle_time=cycle_time,
            store_path=store.store_path,
            min_forecast_hour=min(forecast_hours),
            max_forecast_hour=max(forecast_hours),
            bounds={
                'min_lat': min_lat,
                'max_lat': max_lat,
                'min_lon': min_lon,
                'max_lon': max_lon
            }
        )

        logger.info(f"Stored WaveWatch III forecast to {store.store_path}")
        return store.store_path
