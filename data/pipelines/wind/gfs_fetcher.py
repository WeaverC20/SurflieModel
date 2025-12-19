"""
GFS Wind Forecast Fetcher

Fetches wind forecast data from NOAA's Global Forecast System (GFS).
Supports optional storage to Zarr via WindDataStore.
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
    from data.storage import WindDataStore, MetadataDB, get_storage_config
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


class GFSWindFetcher:
    """Fetches wind forecast data from NOAA GFS model

    GFS provides global weather forecasts including wind speed and direction.
    Data available via NOMADS server in GRIB2 format.

    For historical data, uses AWS Open Data Registry (noaa-gfs-bdp-pds bucket).

    Documentation: https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
    """

    def __init__(self):
        """Initialize GFS wind fetcher"""
        self.nomads_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        self.aws_base_url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

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

    async def fetch_and_store(
        self,
        min_lat: float = 32.0,
        max_lat: float = 42.0,
        min_lon: float = -125.0,
        max_lon: float = -117.0,
        forecast_hours: Optional[List[int]] = None
    ) -> Optional[str]:
        """
        Fetch wind forecast and store to Zarr.

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

        logger.info(f"Fetching GFS wind for {len(forecast_hours)} forecast hours")

        for hour in forecast_hours:
            try:
                data = await self.fetch_wind_grid(
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
        store = WindDataStore()
        store.store_forecast_range(data_list, cycle_time)

        # Record in metadata database
        config = get_storage_config()
        db = MetadataDB()
        db.record_forecast_cycle(
            model='gfs',
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

        logger.info(f"Stored GFS wind forecast to {store.store_path}")
        return store.store_path

    # -------------------------------------------------------------------------
    # Historical Data Fetching (from AWS Open Data)
    # -------------------------------------------------------------------------

    async def fetch_historical_wind_grid(
        self,
        date: datetime,
        cycle_hour: int = 0,
        min_lat: float = 32.0,
        max_lat: float = 42.0,
        min_lon: float = -125.0,
        max_lon: float = -117.0,
    ) -> Optional[Dict]:
        """Fetch historical GFS analysis for a specific date/cycle from AWS

        Uses the AWS Open Data Registry (noaa-gfs-bdp-pds bucket) which has
        GFS data from approximately 2021 onward.

        Args:
            date: Date to fetch (time component ignored, uses cycle_hour)
            cycle_hour: Model cycle hour (0, 6, 12, or 18 UTC)
            min_lat: Minimum latitude (degrees North)
            max_lat: Maximum latitude (degrees North)
            min_lon: Minimum longitude (degrees East, use negative for West)
            max_lon: Maximum longitude (degrees East, use negative for West)

        Returns:
            Dict containing wind grid data or None if fetch failed
        """
        if not GRIB_PARSING_AVAILABLE:
            logger.error("GRIB parsing libraries not available for historical fetch")
            return None

        # Validate cycle hour
        if cycle_hour not in [0, 6, 12, 18]:
            logger.warning(f"Invalid cycle_hour {cycle_hour}, rounding to nearest valid")
            cycle_hour = (cycle_hour // 6) * 6

        # Construct AWS S3 URL for the analysis file (f000)
        date_str = date.strftime('%Y%m%d')
        file_name = f"gfs.t{cycle_hour:02d}z.pgrb2.0p25.f000"
        url = f"{self.aws_base_url}/gfs.{date_str}/{cycle_hour:02d}/atmos/{file_name}"

        logger.info(f"Fetching historical GFS from AWS: {date_str} {cycle_hour:02d}Z")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # First, check if the file exists
                head_response = await client.head(url)
                if head_response.status_code == 404:
                    logger.warning(f"Historical data not available for {date_str} {cycle_hour:02d}Z")
                    return None

                # Download the full GRIB file
                response = await client.get(url)
                response.raise_for_status()

                logger.info(f"Downloaded {len(response.content) / 1024 / 1024:.1f} MB of historical GFS data")

                # Parse GRIB2 and extract region
                cycle_time = date.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
                result = self._parse_grib2_wind_region(
                    response.content,
                    cycle_time,
                    min_lat, max_lat, min_lon, max_lon
                )

                if result:
                    result['source'] = 'aws_historical'
                    return result

                return None

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch historical GFS data: {e}")
            return None

    def _parse_grib2_wind_region(
        self,
        grib_bytes: bytes,
        cycle_time: datetime,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float
    ) -> Optional[Dict]:
        """Parse GRIB2 wind data and extract a regional subset

        Args:
            grib_bytes: Raw GRIB2 data
            cycle_time: Model cycle time
            min_lat, max_lat, min_lon, max_lon: Region bounds

        Returns:
            Dictionary with wind grid data or None if parsing fails
        """
        try:
            import os

            # Write to temporary file (cfgrib requires file path)
            with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False, mode='wb') as tmp:
                tmp.write(grib_bytes)
                tmp_path = tmp.name

            try:
                # Open with xarray/cfgrib - filter for 10m winds
                ds = xr.open_dataset(
                    tmp_path,
                    engine='cfgrib',
                    filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 10}
                )

                # Normalize longitude for subsetting (GFS uses 0-360)
                lon_min_360 = min_lon if min_lon >= 0 else min_lon + 360
                lon_max_360 = max_lon if max_lon >= 0 else max_lon + 360

                # Subset to region
                ds_region = ds.sel(
                    latitude=slice(max_lat, min_lat),  # Note: lat is descending in GFS
                    longitude=slice(lon_min_360, lon_max_360)
                )

                # Extract U and V wind components at 10m
                u_wind = ds_region['u10'].values
                v_wind = ds_region['v10'].values
                lats = ds_region['latitude'].values
                lons = ds_region['longitude'].values

                # Convert longitudes back to -180/180 if needed
                if min_lon < 0:
                    lons = np.where(lons > 180, lons - 360, lons)

                # Calculate wind speed and direction
                wind_speed = np.sqrt(u_wind**2 + v_wind**2)
                wind_direction = (270 - np.arctan2(v_wind, u_wind) * 180 / np.pi) % 360

                ds.close()

                logger.info(f"Parsed historical GFS grid: {len(lats)} x {len(lons)} points")

                return {
                    "lat": lats.tolist(),
                    "lon": lons.tolist(),
                    "u_wind": u_wind.tolist(),
                    "v_wind": v_wind.tolist(),
                    "wind_speed": wind_speed.tolist(),
                    "wind_direction": wind_direction.tolist(),
                    "time": cycle_time.isoformat(),
                    "cycle_time": cycle_time.isoformat(),
                    "forecast_hour": 0,
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
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Failed to parse historical GRIB2 wind data: {e}")
            return None

    async def fetch_historical_range(
        self,
        start_date: datetime,
        end_date: datetime,
        resolution_hours: int = 3,
        min_lat: float = 32.0,
        max_lat: float = 42.0,
        min_lon: float = -125.0,
        max_lon: float = -117.0,
        store_path: Optional[str] = None,
        append: bool = False,
    ) -> Optional[str]:
        """Fetch and store a range of historical GFS data for validation

        Downloads historical GFS analysis data from AWS and stores it in a
        Zarr archive for later use in model validation.

        Args:
            start_date: Start of date range
            end_date: End of date range
            resolution_hours: Time resolution (3 or 6 hours)
            min_lat, max_lat, min_lon, max_lon: Region bounds
            store_path: Optional custom path for Zarr store
            append: If True, append to existing dataset instead of overwriting

        Returns:
            Path to Zarr store or None if failed
        """
        if not STORAGE_AVAILABLE:
            logger.error("Storage module not available")
            return None

        if resolution_hours not in [3, 6]:
            logger.warning(f"Invalid resolution_hours {resolution_hours}, using 6")
            resolution_hours = 6

        # Generate list of cycle times to fetch
        cycle_hours = [0, 6, 12, 18] if resolution_hours == 6 else [0, 3, 6, 9, 12, 15, 18, 21]
        # For 3-hourly, we still use analysis files (f000) from each GFS cycle
        # GFS runs at 00, 06, 12, 18 - for intermediate hours we'd need forecast files
        # For simplicity, use 6-hourly analysis for now
        if resolution_hours == 3:
            logger.info("Note: 3-hourly resolution uses GFS cycles + 3-hour forecasts")

        # Determine store path
        config = get_storage_config()
        config.ensure_directories()
        if store_path is None:
            store_path = config.get_historical_wind_store_path()

        logger.info(f"Fetching historical GFS from {start_date.date()} to {end_date.date()}")
        logger.info(f"Resolution: {resolution_hours}-hourly, storing to {store_path}")

        # Collect all data
        all_data = []
        current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

        while current_date <= end_date:
            for hour in [0, 6, 12, 18]:
                cycle_time = current_date.replace(hour=hour)
                if cycle_time < start_date or cycle_time > end_date:
                    continue

                # Fetch this cycle
                data = await self.fetch_historical_wind_grid(
                    date=cycle_time,
                    cycle_hour=hour,
                    min_lat=min_lat,
                    max_lat=max_lat,
                    min_lon=min_lon,
                    max_lon=max_lon
                )

                if data:
                    all_data.append(data)
                    logger.info(f"Fetched {cycle_time.isoformat()}")
                else:
                    logger.warning(f"Missing data for {cycle_time.isoformat()}")

                # If 3-hourly, also fetch +3 hour forecast
                if resolution_hours == 3:
                    forecast_data = await self._fetch_historical_forecast(
                        date=cycle_time,
                        cycle_hour=hour,
                        forecast_hour=3,
                        min_lat=min_lat,
                        max_lat=max_lat,
                        min_lon=min_lon,
                        max_lon=max_lon
                    )
                    if forecast_data:
                        all_data.append(forecast_data)

            current_date += timedelta(days=1)

        if not all_data:
            logger.error("No historical data fetched successfully")
            return None

        # Store to Zarr
        logger.info(f"Storing {len(all_data)} historical wind snapshots")
        self._store_historical_to_zarr(all_data, store_path, append=append)

        # Record in metadata database
        db = MetadataDB()
        db.record_forecast_cycle(
            model='gfs_historical',
            cycle_time=start_date,
            store_path=store_path,
            min_forecast_hour=0,
            max_forecast_hour=0,
            bounds={
                'min_lat': min_lat,
                'max_lat': max_lat,
                'min_lon': min_lon,
                'max_lon': max_lon
            },
            status='ready'
        )

        logger.info(f"Stored historical GFS data to {store_path}")
        return store_path

    async def _fetch_historical_forecast(
        self,
        date: datetime,
        cycle_hour: int,
        forecast_hour: int,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float
    ) -> Optional[Dict]:
        """Fetch a specific forecast hour from historical GFS data

        Used for 3-hourly resolution to get intermediate times.
        """
        if not GRIB_PARSING_AVAILABLE:
            return None

        date_str = date.strftime('%Y%m%d')
        file_name = f"gfs.t{cycle_hour:02d}z.pgrb2.0p25.f{forecast_hour:03d}"
        url = f"{self.aws_base_url}/gfs.{date_str}/{cycle_hour:02d}/atmos/{file_name}"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                head_response = await client.head(url)
                if head_response.status_code == 404:
                    return None

                response = await client.get(url)
                response.raise_for_status()

                cycle_time = date.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
                valid_time = cycle_time + timedelta(hours=forecast_hour)

                result = self._parse_grib2_wind_region(
                    response.content,
                    valid_time,
                    min_lat, max_lat, min_lon, max_lon
                )

                if result:
                    result['source'] = 'aws_historical'
                    result['forecast_hour'] = forecast_hour
                    result['cycle_time'] = cycle_time.isoformat()
                    return result

                return None

        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch f{forecast_hour:03d}: {e}")
            return None

    def _store_historical_to_zarr(self, data_list: List[Dict], store_path: str, append: bool = False):
        """Store historical wind data to Zarr format

        Args:
            data_list: List of wind data dictionaries
            store_path: Path to Zarr store
            append: If True, append to existing dataset
        """
        import pandas as pd
        from pathlib import Path

        # Sort by time
        data_list.sort(key=lambda x: x['time'])

        # Extract coordinates from first entry
        lats = np.array(data_list[0]['lat'])
        lons = np.array(data_list[0]['lon'])
        times = pd.to_datetime([d['time'] for d in data_list])

        # Stack data arrays
        u_wind = np.stack([np.array(d['u_wind']) for d in data_list], axis=0)
        v_wind = np.stack([np.array(d['v_wind']) for d in data_list], axis=0)
        wind_speed = np.stack([np.array(d['wind_speed']) for d in data_list], axis=0)
        wind_direction = np.stack([np.array(d['wind_direction']) for d in data_list], axis=0)

        # Create xarray Dataset
        ds = xr.Dataset(
            {
                'u10': (['time', 'latitude', 'longitude'], u_wind),
                'v10': (['time', 'latitude', 'longitude'], v_wind),
                'wind_speed': (['time', 'latitude', 'longitude'], wind_speed),
                'wind_direction': (['time', 'latitude', 'longitude'], wind_direction),
            },
            coords={
                'time': times,
                'latitude': lats,
                'longitude': lons,
            },
            attrs={
                'model': 'GFS',
                'source': 'AWS Open Data (noaa-gfs-bdp-pds)',
                'description': 'Historical GFS wind data for validation',
                'created': datetime.utcnow().isoformat(),
            }
        )

        # Add variable attributes
        ds['u10'].attrs = {'units': 'm/s', 'long_name': 'U-component of wind at 10m'}
        ds['v10'].attrs = {'units': 'm/s', 'long_name': 'V-component of wind at 10m'}
        ds['wind_speed'].attrs = {'units': 'm/s', 'long_name': 'Wind speed at 10m'}
        ds['wind_direction'].attrs = {'units': 'degrees', 'long_name': 'Wind direction (meteorological)'}

        # Chunking strategy via encoding (works without dask)
        time_chunk = min(24, len(times))
        encoding = {
            'u10': {'chunks': (time_chunk, len(lats), len(lons))},
            'v10': {'chunks': (time_chunk, len(lats), len(lons))},
            'wind_speed': {'chunks': (time_chunk, len(lats), len(lons))},
            'wind_direction': {'chunks': (time_chunk, len(lats), len(lons))},
        }

        # Write to Zarr
        store_exists = Path(store_path).exists()

        if append and store_exists:
            # Load existing dataset and merge
            existing_ds = xr.open_zarr(store_path)
            existing_times = set(existing_ds.time.values)
            new_times = set(ds.time.values)

            # Filter out times that already exist
            times_to_add = new_times - existing_times
            if not times_to_add:
                logger.info("All requested times already exist in dataset, nothing to append")
                existing_ds.close()
                return

            # Filter new data to only include new times
            ds = ds.sel(time=sorted(list(times_to_add)))

            # Append along time dimension
            ds.to_zarr(store_path, mode='a', append_dim='time', consolidated=True)
            logger.info(f"Appended {len(times_to_add)} new time steps to {store_path}")
            existing_ds.close()
        else:
            ds.to_zarr(store_path, mode='w', consolidated=True, encoding=encoding)
            logger.info(f"Wrote historical data to {store_path}: {len(times)} time steps")
