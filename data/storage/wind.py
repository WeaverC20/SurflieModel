"""
Wind Data Store

Zarr-based storage for GFS wind forecast data.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from .base import BaseDataStore
from .config import StorageConfig, get_storage_config

logger = logging.getLogger(__name__)


class WindDataStore(BaseDataStore):
    """Store for GFS wind forecast data"""

    @property
    def default_store_path(self) -> str:
        return self.config.get_wind_store_path()

    @property
    def model_name(self) -> str:
        return 'gfs'

    @property
    def variables(self) -> List[str]:
        return ['u10', 'v10', 'gust', 'wind_speed', 'wind_direction']

    def store_forecast(
        self,
        data: Dict,
        cycle_time: Optional[datetime] = None
    ):
        """
        Store wind forecast data from GFS fetcher output.

        Args:
            data: Dictionary from GFSWindFetcher.fetch_wind_grid()
            cycle_time: Model cycle time (parsed from data if not provided)
        """
        if cycle_time is None:
            cycle_time = datetime.fromisoformat(data['cycle_time'])

        # Convert lists to numpy arrays
        lat = np.array(data['lat'])
        lon = np.array(data['lon'])
        u_wind = np.array(data['u_wind'])
        v_wind = np.array(data['v_wind'])
        wind_speed = np.array(data['wind_speed'])
        wind_direction = np.array(data['wind_direction'])

        # Get forecast time
        forecast_time = datetime.fromisoformat(data['forecast_time'])

        # Handle gust if present (may not be in all fetcher outputs)
        if 'gust' in data and data['gust'] is not None:
            gust = np.array(data['gust'])
        else:
            # Estimate gust as 1.3x wind speed if not available
            gust = wind_speed * 1.3

        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                'u10': (['time', 'lat', 'lon'], u_wind[np.newaxis, :, :].astype(np.float32)),
                'v10': (['time', 'lat', 'lon'], v_wind[np.newaxis, :, :].astype(np.float32)),
                'gust': (['time', 'lat', 'lon'], gust[np.newaxis, :, :].astype(np.float32)),
                'wind_speed': (['time', 'lat', 'lon'], wind_speed[np.newaxis, :, :].astype(np.float32)),
                'wind_direction': (['time', 'lat', 'lon'], wind_direction[np.newaxis, :, :].astype(np.float32)),
            },
            coords={
                'time': [np.datetime64(forecast_time)],
                'lat': lat.astype(np.float64),
                'lon': lon.astype(np.float64),
            },
            attrs={
                'model': 'GFS',
                'source': 'NOAA NOMADS',
                'resolution_deg': data.get('resolution_deg', 0.25),
                'forecast_hour': data.get('forecast_hour', 0),
            }
        )

        # Add variable attributes
        ds['u10'].attrs = {
            'long_name': 'Eastward wind component at 10m',
            'units': 'm/s',
            'standard_name': 'eastward_wind'
        }
        ds['v10'].attrs = {
            'long_name': 'Northward wind component at 10m',
            'units': 'm/s',
            'standard_name': 'northward_wind'
        }
        ds['gust'].attrs = {
            'long_name': 'Wind gust at 10m',
            'units': 'm/s'
        }
        ds['wind_speed'].attrs = {
            'long_name': 'Wind speed magnitude at 10m',
            'units': 'm/s'
        }
        ds['wind_direction'].attrs = {
            'long_name': 'Wind direction (meteorological)',
            'units': 'degrees',
            'comment': 'Direction FROM which wind blows'
        }

        self.write_forecast(ds, cycle_time, mode='w')
        logger.info(f"Stored GFS wind forecast for cycle {cycle_time.isoformat()}")

    def store_forecast_range(
        self,
        data_list: List[Dict],
        cycle_time: datetime
    ):
        """
        Store multiple forecast hours as a single dataset.

        Args:
            data_list: List of dictionaries from GFSWindFetcher for different forecast hours
            cycle_time: Model cycle time
        """
        if not data_list:
            logger.warning("No data to store")
            return

        # Sort by forecast hour
        data_list = sorted(data_list, key=lambda x: x.get('forecast_hour', 0))

        # Get coordinate arrays from first entry
        lat = np.array(data_list[0]['lat'])
        lon = np.array(data_list[0]['lon'])

        # Initialize arrays
        n_times = len(data_list)
        n_lat = len(lat)
        n_lon = len(lon)

        times = []
        u10 = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        v10 = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        gust = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        wind_speed = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        wind_direction = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)

        for i, data in enumerate(data_list):
            times.append(np.datetime64(datetime.fromisoformat(data['forecast_time'])))
            u10[i] = np.array(data['u_wind'])
            v10[i] = np.array(data['v_wind'])
            wind_speed[i] = np.array(data['wind_speed'])
            wind_direction[i] = np.array(data['wind_direction'])

            if 'gust' in data and data['gust'] is not None:
                gust[i] = np.array(data['gust'])
            else:
                gust[i] = wind_speed[i] * 1.3

        ds = xr.Dataset(
            data_vars={
                'u10': (['time', 'lat', 'lon'], u10),
                'v10': (['time', 'lat', 'lon'], v10),
                'gust': (['time', 'lat', 'lon'], gust),
                'wind_speed': (['time', 'lat', 'lon'], wind_speed),
                'wind_direction': (['time', 'lat', 'lon'], wind_direction),
            },
            coords={
                'time': times,
                'lat': lat.astype(np.float64),
                'lon': lon.astype(np.float64),
            },
            attrs={
                'model': 'GFS',
                'source': 'NOAA NOMADS',
                'resolution_deg': data_list[0].get('resolution_deg', 0.25),
                'min_forecast_hour': data_list[0].get('forecast_hour', 0),
                'max_forecast_hour': data_list[-1].get('forecast_hour', 0),
            }
        )

        # Add variable attributes
        ds['u10'].attrs = {'long_name': 'Eastward wind component at 10m', 'units': 'm/s'}
        ds['v10'].attrs = {'long_name': 'Northward wind component at 10m', 'units': 'm/s'}
        ds['gust'].attrs = {'long_name': 'Wind gust at 10m', 'units': 'm/s'}
        ds['wind_speed'].attrs = {'long_name': 'Wind speed magnitude at 10m', 'units': 'm/s'}
        ds['wind_direction'].attrs = {'long_name': 'Wind direction (meteorological)', 'units': 'degrees'}

        self.write_forecast(ds, cycle_time, mode='w')
        logger.info(f"Stored GFS wind forecast range ({n_times} hours) for cycle {cycle_time.isoformat()}")

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
            Dictionary in same format as GFSWindFetcher output
        """
        ds = self.get_forecast(bounds=bounds)

        if 'time' in ds.dims and len(ds.time) > forecast_hour:
            ds = ds.isel(time=forecast_hour)
        elif 'time' in ds.dims:
            ds = ds.isel(time=0)

        # Handle the case where ds might be a DataArray after isel
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()

        return {
            'lat': ds.lat.values.tolist(),
            'lon': ds.lon.values.tolist(),
            'u_wind': ds.u10.values.tolist() if 'u10' in ds else None,
            'v_wind': ds.v10.values.tolist() if 'v10' in ds else None,
            'wind_speed': ds.wind_speed.values.tolist() if 'wind_speed' in ds else None,
            'wind_direction': ds.wind_direction.values.tolist() if 'wind_direction' in ds else None,
            'gust': ds.gust.values.tolist() if 'gust' in ds else None,
            'forecast_time': str(ds.time.values) if 'time' in ds.coords else None,
            'cycle_time': ds.attrs.get('cycle_time'),
            'forecast_hour': forecast_hour,
            'resolution_deg': ds.attrs.get('resolution_deg', 0.25),
            'model': 'GFS',
            'units': {
                'wind_speed': 'm/s',
                'wind_direction': 'degrees (meteorological)',
                'lat': 'degrees_north',
                'lon': 'degrees_east'
            }
        }
