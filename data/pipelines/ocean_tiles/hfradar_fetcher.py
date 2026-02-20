"""
HF Radar Surface Currents Fetcher

Fetches real-time surface current observations from the NOAA/UCSD HF Radar network
via CoastWatch ERDDAP with geographic subsetting.

Data Info:
- 62 HF radars along California coast (SCCOOS + CeNCOOS)
- 6km resolution (ucsdHfrW6 dataset), also available at 2km and 1km
- Hourly observations
- No forecast — real-time/near-real-time observations only
- Ground truth for ocean surface currents (~2km accuracy reference)

ERDDAP: https://coastwatch.pfeg.noaa.gov/erddap/griddap/ucsdHfrW6
Variables: water_u (eastward, m/s), water_v (northward, m/s)
Coordinates: latitude (30.25-49.99°N), longitude (-130.36 to -115.81°W)
Resolution: ~0.054° lat × ~0.062° lon (~6km)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path

import httpx
import numpy as np

from .config import GRIB_CACHE_DIR

logger = logging.getLogger(__name__)

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False
    logger.warning("xarray not available. Install with: pip install xarray netCDF4")


# ERDDAP dataset configurations
HFRADAR_DATASETS = {
    '6km': {
        'dataset_id': 'ucsdHfrW6',
        'description': 'US West Coast, 6km resolution, hourly',
        'resolution_km': 6,
    },
    '2km': {
        'dataset_id': 'ucsdHfrW2',
        'description': 'US West Coast, 2km resolution, hourly',
        'resolution_km': 2,
    },
    '1km': {
        'dataset_id': 'ucsdHfrW1',
        'description': 'US West Coast, 1km resolution, hourly',
        'resolution_km': 1,
    },
}


class HFRadarFetcher:
    """Fetches surface current observations from HF Radar via ERDDAP."""

    ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"

    def __init__(self, resolution: str = '6km'):
        """
        Initialize HF Radar fetcher.

        Args:
            resolution: Data resolution - '6km' (default, broader coverage),
                       '2km' (higher res, less coverage), or '1km'
        """
        if resolution not in HFRADAR_DATASETS:
            raise ValueError(f"Invalid resolution: {resolution}. Choose from {list(HFRADAR_DATASETS.keys())}")

        self.resolution = resolution
        self.dataset = HFRADAR_DATASETS[resolution]
        self.dataset_id = self.dataset['dataset_id']
        self.cache_dir = GRIB_CACHE_DIR

    def _build_erddap_url(
        self,
        time_str: str,
        bounds: Dict[str, float],
        file_format: str = 'nc'
    ) -> str:
        """
        Build ERDDAP griddap URL for geographic/temporal subset.

        Args:
            time_str: ISO time string for the observation (e.g., '2026-02-19T12:00:00Z')
            bounds: Geographic bounds dict with min_lat, max_lat, min_lon, max_lon
            file_format: Output format ('nc', 'csv', 'json')

        Returns:
            Full ERDDAP griddap URL with subsetting constraints
        """
        lat_min = bounds['min_lat']
        lat_max = bounds['max_lat']
        lon_min = bounds['min_lon']
        lon_max = bounds['max_lon']

        # ERDDAP griddap URL format:
        # datasetID.fileType?var[(time)][(lat_min):(lat_max)][(lon_min):(lon_max)]
        query = (
            f"water_u[({time_str})][({lat_min}):({lat_max})][({lon_min}):({lon_max})],"
            f"water_v[({time_str})][({lat_min}):({lat_max})][({lon_min}):({lon_max})]"
        )

        return f"{self.ERDDAP_BASE}/{self.dataset_id}.{file_format}?{query}"

    def _build_latest_url(self, bounds: Dict[str, float], file_format: str = 'nc') -> str:
        """Build ERDDAP URL requesting the most recent observation."""
        lat_min = bounds['min_lat']
        lat_max = bounds['max_lat']
        lon_min = bounds['min_lon']
        lon_max = bounds['max_lon']

        # Use 'last' keyword for most recent time
        query = (
            f"water_u[(last)][({lat_min}):({lat_max})][({lon_min}):({lon_max})],"
            f"water_v[(last)][({lat_min}):({lat_max})][({lon_min}):({lon_max})]"
        )

        return f"{self.ERDDAP_BASE}/{self.dataset_id}.{file_format}?{query}"

    def _get_cache_path(self, time_str: str) -> Path:
        """Get local cache file path."""
        safe_time = time_str.replace(':', '').replace('-', '').replace('T', '_').replace('Z', '')
        return self.cache_dir / f"hfradar_{self.resolution}_{safe_time}.nc"

    async def fetch_current_grid(
        self,
        region_name: str = "california",
        observation_time: Optional[datetime] = None
    ) -> Optional[Dict]:
        """
        Fetch HF Radar current observations for a region.

        Args:
            region_name: Region name (uses same config as WCOFS)
            observation_time: Specific observation time (default: most recent)

        Returns:
            Dictionary with u_velocity, v_velocity, current_speed, current_direction,
            lons, lats, metadata — same structure as WCOFSFetcher/RTOFSFetcher.
        """
        from .config import REGIONS

        if region_name not in REGIONS:
            logger.error(f"Unknown region: {region_name}")
            return None

        region = REGIONS[region_name]
        return await self.fetch_for_bounds(
            bounds=region.bounds,
            observation_time=observation_time
        )

    async def fetch_for_bounds(
        self,
        bounds: Dict[str, float],
        observation_time: Optional[datetime] = None
    ) -> Optional[Dict]:
        """
        Fetch HF Radar current observations for a bounding box.

        Args:
            bounds: Geographic bounds dict with min_lat, max_lat, min_lon, max_lon
            observation_time: Specific observation time (default: most recent available)

        Returns:
            Dictionary with u_velocity, v_velocity, current_speed, current_direction,
            lons, lats, metadata.
        """
        if not XARRAY_AVAILABLE:
            logger.error("xarray not available")
            return None

        logger.info(f"Fetching HF Radar ({self.resolution}) data for bounds")
        logger.info(f"Bounds: lat=[{bounds['min_lat']:.2f}, {bounds['max_lat']:.2f}], "
                   f"lon=[{bounds['min_lon']:.2f}, {bounds['max_lon']:.2f}]")

        # Build URL — use specific time or most recent
        if observation_time:
            time_str = observation_time.strftime('%Y-%m-%dT%H:00:00Z')
            url = self._build_erddap_url(time_str, bounds)
        else:
            time_str = 'latest'
            url = self._build_latest_url(bounds)

        # Check cache (only for specific times, not 'latest')
        if observation_time:
            cache_file = self._get_cache_path(time_str)
            if cache_file.exists():
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age < timedelta(hours=1):
                    logger.info(f"Using cached HF Radar data: {cache_file.name}")
                    try:
                        return self._parse_erddap_netcdf(cache_file)
                    except Exception as e:
                        logger.warning(f"Failed to read cache: {e}")
        else:
            cache_file = None

        # Download subset from ERDDAP
        logger.info(f"Downloading HF Radar subset from ERDDAP")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url)

                if response.status_code == 404:
                    logger.warning("HF Radar data not available for requested time/region")
                    return None

                if response.status_code != 200:
                    logger.warning(f"ERDDAP returned status {response.status_code}: {response.text[:200]}")
                    # Try falling back to recent hours if specific time failed
                    if observation_time:
                        return await self._try_recent_hours(bounds, observation_time)
                    return None

                nc_bytes = response.content

                if len(nc_bytes) < 100:
                    logger.warning(f"HF Radar response too small ({len(nc_bytes)} bytes)")
                    return None

                logger.info(f"Downloaded {len(nc_bytes) / 1024:.1f} KB of HF Radar data")

                # Save to temp file for xarray parsing
                if cache_file:
                    save_path = cache_file
                else:
                    save_path = self.cache_dir / f"hfradar_{self.resolution}_latest.nc"

                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(nc_bytes)

                return self._parse_erddap_netcdf(save_path)

        except Exception as e:
            logger.error(f"Error fetching HF Radar data: {e}")
            return None

    async def _try_recent_hours(
        self,
        bounds: Dict[str, float],
        base_time: datetime,
        max_hours_back: int = 6
    ) -> Optional[Dict]:
        """Try fetching data from recent hours if the requested time isn't available."""
        for hours_back in range(1, max_hours_back + 1):
            try_time = base_time - timedelta(hours=hours_back)
            logger.info(f"Trying {hours_back}h earlier: {try_time.strftime('%Y-%m-%dT%H:00:00Z')}")

            time_str = try_time.strftime('%Y-%m-%dT%H:00:00Z')
            url = self._build_erddap_url(time_str, bounds)

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(url)
                    if response.status_code == 200 and len(response.content) > 100:
                        save_path = self.cache_dir / f"hfradar_{self.resolution}_fallback.nc"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(save_path, 'wb') as f:
                            f.write(response.content)
                        result = self._parse_erddap_netcdf(save_path)
                        if result:
                            result['metadata']['fallback_hours'] = hours_back
                            return result
            except Exception:
                continue

        return None

    def _parse_erddap_netcdf(self, nc_file: Path) -> Optional[Dict]:
        """
        Parse ERDDAP NetCDF response into standard dict format.

        ERDDAP returns data on a regular grid with standard coordinate names.
        """
        try:
            ds = xr.open_dataset(nc_file)

            # ERDDAP uses standard names: latitude, longitude, time
            lats = ds['latitude'].values
            lons = ds['longitude'].values

            # Get velocity components
            u_velocity = ds['water_u'].values
            v_velocity = ds['water_v'].values

            # Squeeze time dimension if present
            if u_velocity.ndim == 3:
                u_velocity = u_velocity[0]  # First (only) time step
                v_velocity = v_velocity[0]

            # Get the observation time
            if 'time' in ds.coords:
                obs_time = ds['time'].values
                if hasattr(obs_time, '__len__') and len(obs_time) > 0:
                    obs_time = obs_time[0]
                obs_time_str = str(np.datetime_as_string(obs_time, unit='s'))
            else:
                obs_time_str = datetime.utcnow().isoformat()

            ds.close()

            # Create 2D lat/lon grids for consistent output
            lons_2d, lats_2d = np.meshgrid(lons, lats)

            # Calculate derived fields (handling NaN from land/gaps)
            current_speed = np.sqrt(
                np.where(np.isnan(u_velocity), 0, u_velocity)**2 +
                np.where(np.isnan(v_velocity), 0, v_velocity)**2
            )
            # Set speed to NaN where either component is NaN
            current_speed = np.where(
                np.isnan(u_velocity) | np.isnan(v_velocity),
                np.nan,
                current_speed
            )

            current_direction = np.where(
                np.isnan(u_velocity) | np.isnan(v_velocity),
                np.nan,
                (90 - np.arctan2(v_velocity, u_velocity) * 180 / np.pi) % 360
            )

            result = {
                'u_velocity': u_velocity,
                'v_velocity': v_velocity,
                'current_speed': current_speed,
                'current_direction': current_direction,
                'lons': lons_2d,
                'lats': lats_2d,
                'metadata': {
                    'model': 'hfradar',
                    'dataset': self.dataset_id,
                    'resolution': self.resolution,
                    'observation_time': obs_time_str,
                    'valid_time': obs_time_str,
                    'shape': current_speed.shape,
                    'bounds': {
                        'min_lat': float(np.nanmin(lats)),
                        'max_lat': float(np.nanmax(lats)),
                        'min_lon': float(np.nanmin(lons)),
                        'max_lon': float(np.nanmax(lons)),
                    },
                    'units': {
                        'velocity': 'm/s',
                        'direction': 'degrees (oceanographic convention)'
                    },
                    'type': 'observation',
                }
            }

            valid_count = np.sum(~np.isnan(current_speed))
            total_count = current_speed.size
            coverage = valid_count / total_count * 100

            logger.info(f"Parsed HF Radar data: shape {current_speed.shape}, "
                       f"speed range {np.nanmin(current_speed):.3f}-{np.nanmax(current_speed):.3f} m/s, "
                       f"coverage: {coverage:.0f}%")

            return result

        except Exception as e:
            logger.error(f"Failed to parse HF Radar NetCDF: {e}", exc_info=True)
            return None

    @staticmethod
    def ms_to_knots(speed_ms: float) -> float:
        """Convert speed from m/s to knots."""
        return speed_ms * 1.94384
