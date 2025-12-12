"""
Current Data Store

Zarr-based storage for RTOFS ocean current forecast data.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from .base import BaseDataStore
from .config import StorageConfig, get_storage_config

logger = logging.getLogger(__name__)


class CurrentDataStore(BaseDataStore):
    """Store for RTOFS ocean current forecast data"""

    @property
    def default_store_path(self) -> str:
        return self.config.get_current_store_path()

    @property
    def model_name(self) -> str:
        return 'rtofs'

    @property
    def variables(self) -> List[str]:
        return ['u_velocity', 'v_velocity', 'current_speed', 'current_direction', 'sst']

    def store_forecast(
        self,
        data: Dict,
        cycle_time: Optional[datetime] = None
    ):
        """
        Store current forecast data from RTOFS fetcher output.

        Args:
            data: Dictionary from RTOFSFetcher.fetch_current_grid()
            cycle_time: Model cycle time (parsed from metadata if not provided)
        """
        metadata = data.get('metadata', {})

        if cycle_time is None:
            cycle_time = datetime.fromisoformat(metadata.get('model_time', datetime.utcnow().isoformat()))

        # Get arrays - RTOFS returns numpy arrays directly
        u_velocity = data['u_velocity']
        v_velocity = data['v_velocity']
        current_speed = data['current_speed']
        current_direction = data['current_direction']
        lats = data['lats']
        lons = data['lons']

        # Get forecast time
        valid_time = datetime.fromisoformat(metadata.get('valid_time', cycle_time.isoformat()))

        # Handle 2D lat/lon grids (curvilinear) vs 1D (regular)
        if lats.ndim == 2:
            # RTOFS uses curvilinear grid - need to regrid or store as-is
            # For now, extract 1D coordinates by averaging
            lat_1d = lats.mean(axis=1)
            lon_1d = lons.mean(axis=0)
            logger.info(f"Converted 2D grid ({lats.shape}) to 1D coords")
        else:
            lat_1d = lats
            lon_1d = lons

        # Extract SST if available
        if 'sst' in data:
            sst = data['sst']
        else:
            sst = np.full_like(current_speed, np.nan)

        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                'u_velocity': (['time', 'lat', 'lon'], u_velocity[np.newaxis, :, :].astype(np.float32)),
                'v_velocity': (['time', 'lat', 'lon'], v_velocity[np.newaxis, :, :].astype(np.float32)),
                'current_speed': (['time', 'lat', 'lon'], current_speed[np.newaxis, :, :].astype(np.float32)),
                'current_direction': (['time', 'lat', 'lon'], current_direction[np.newaxis, :, :].astype(np.float32)),
                'sst': (['time', 'lat', 'lon'], sst[np.newaxis, :, :].astype(np.float32)),
            },
            coords={
                'time': [np.datetime64(valid_time)],
                'lat': lat_1d.astype(np.float64),
                'lon': lon_1d.astype(np.float64),
            },
            attrs={
                'model': 'RTOFS',
                'source': 'NOAA NOMADS',
                'resolution_deg': 1/12,  # ~9km
                'forecast_hour': metadata.get('forecast_hour', 0),
            }
        )

        # Add variable attributes
        ds['u_velocity'].attrs = {
            'long_name': 'Eastward sea water velocity',
            'units': 'm/s',
            'standard_name': 'eastward_sea_water_velocity'
        }
        ds['v_velocity'].attrs = {
            'long_name': 'Northward sea water velocity',
            'units': 'm/s',
            'standard_name': 'northward_sea_water_velocity'
        }
        ds['current_speed'].attrs = {
            'long_name': 'Sea water speed',
            'units': 'm/s',
            'standard_name': 'sea_water_speed'
        }
        ds['current_direction'].attrs = {
            'long_name': 'Sea water direction',
            'units': 'degrees',
            'comment': 'Direction TOWARD which current flows (oceanographic convention)'
        }
        ds['sst'].attrs = {
            'long_name': 'Sea surface temperature',
            'units': 'degC',
            'standard_name': 'sea_surface_temperature'
        }

        self.write_forecast(ds, cycle_time, mode='w')
        logger.info(f"Stored RTOFS current forecast for cycle {cycle_time.isoformat()}")

    def store_forecast_range(
        self,
        data_list: List[Dict],
        cycle_time: datetime
    ):
        """
        Store multiple forecast hours as a single dataset.

        Args:
            data_list: List of dictionaries from RTOFSFetcher for different forecast hours
            cycle_time: Model cycle time
        """
        if not data_list:
            logger.warning("No data to store")
            return

        # Sort by forecast hour
        data_list = sorted(data_list, key=lambda x: x.get('metadata', {}).get('forecast_hour', 0))

        # Get coordinate arrays from first entry
        first_data = data_list[0]
        lats = first_data['lats']
        lons = first_data['lons']

        # Handle 2D grids
        if lats.ndim == 2:
            lat_1d = lats.mean(axis=1)
            lon_1d = lons.mean(axis=0)
        else:
            lat_1d = lats
            lon_1d = lons

        # Initialize arrays
        n_times = len(data_list)
        n_lat = len(lat_1d)
        n_lon = len(lon_1d)

        times = []
        u_velocity = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        v_velocity = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        current_speed = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        current_direction = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        sst = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)

        for i, data in enumerate(data_list):
            metadata = data.get('metadata', {})
            valid_time = metadata.get('valid_time', cycle_time.isoformat())
            times.append(np.datetime64(datetime.fromisoformat(valid_time)))

            u_velocity[i] = data['u_velocity']
            v_velocity[i] = data['v_velocity']
            current_speed[i] = data['current_speed']
            current_direction[i] = data['current_direction']

            if 'sst' in data:
                sst[i] = data['sst']
            else:
                sst[i] = np.nan

        ds = xr.Dataset(
            data_vars={
                'u_velocity': (['time', 'lat', 'lon'], u_velocity),
                'v_velocity': (['time', 'lat', 'lon'], v_velocity),
                'current_speed': (['time', 'lat', 'lon'], current_speed),
                'current_direction': (['time', 'lat', 'lon'], current_direction),
                'sst': (['time', 'lat', 'lon'], sst),
            },
            coords={
                'time': times,
                'lat': lat_1d.astype(np.float64),
                'lon': lon_1d.astype(np.float64),
            },
            attrs={
                'model': 'RTOFS',
                'source': 'NOAA NOMADS',
                'resolution_deg': 1/12,
                'min_forecast_hour': data_list[0].get('metadata', {}).get('forecast_hour', 0),
                'max_forecast_hour': data_list[-1].get('metadata', {}).get('forecast_hour', 0),
            }
        )

        # Add variable attributes
        ds['u_velocity'].attrs = {'long_name': 'Eastward sea water velocity', 'units': 'm/s'}
        ds['v_velocity'].attrs = {'long_name': 'Northward sea water velocity', 'units': 'm/s'}
        ds['current_speed'].attrs = {'long_name': 'Sea water speed', 'units': 'm/s'}
        ds['current_direction'].attrs = {'long_name': 'Sea water direction', 'units': 'degrees'}
        ds['sst'].attrs = {'long_name': 'Sea surface temperature', 'units': 'degC'}

        self.write_forecast(ds, cycle_time, mode='w')
        logger.info(f"Stored RTOFS current forecast range ({n_times} hours) for cycle {cycle_time.isoformat()}")

    def to_dict(
        self,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        forecast_hour: int = 0
    ) -> Dict:
        """
        Export data as dictionary (compatible with existing API format).

        Args:
            bounds: Optional spatial bounds
            forecast_hour: Forecast hour to extract

        Returns:
            Dictionary in same format as RTOFSFetcher output
        """
        ds = self.get_forecast(bounds=bounds)

        if 'time' in ds.dims and len(ds.time) > forecast_hour:
            ds = ds.isel(time=forecast_hour)
        elif 'time' in ds.dims:
            ds = ds.isel(time=0)

        return {
            'u_velocity': ds.u_velocity.values if 'u_velocity' in ds else None,
            'v_velocity': ds.v_velocity.values if 'v_velocity' in ds else None,
            'current_speed': ds.current_speed.values if 'current_speed' in ds else None,
            'current_direction': ds.current_direction.values if 'current_direction' in ds else None,
            'lats': ds.lat.values,
            'lons': ds.lon.values,
            'metadata': {
                'model': 'rtofs',
                'model_time': ds.attrs.get('cycle_time'),
                'forecast_hour': forecast_hour,
                'valid_time': str(ds.time.values) if 'time' in ds.coords else None,
                'shape': ds.current_speed.shape if 'current_speed' in ds else None,
                'bounds': {
                    'min_lat': float(ds.lat.min()),
                    'max_lat': float(ds.lat.max()),
                    'min_lon': float(ds.lon.min()),
                    'max_lon': float(ds.lon.max()),
                },
                'units': {
                    'velocity': 'm/s',
                    'direction': 'degrees (oceanographic convention)'
                }
            }
        }
