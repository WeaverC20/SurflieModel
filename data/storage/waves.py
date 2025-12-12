"""
Wave Data Store

Zarr-based storage for WaveWatch III wave forecast data.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from .base import BaseDataStore
from .config import StorageConfig, get_storage_config

logger = logging.getLogger(__name__)


class WaveDataStore(BaseDataStore):
    """Store for WaveWatch III wave forecast data"""

    @property
    def default_store_path(self) -> str:
        return self.config.get_wave_store_path()

    @property
    def model_name(self) -> str:
        return 'ww3'

    @property
    def variables(self) -> List[str]:
        return ['hs', 'tp', 'dp', 'hs_wind', 'hs_swell', 'tp_swell', 'dp_swell']

    def store_forecast(
        self,
        data: Dict,
        cycle_time: Optional[datetime] = None
    ):
        """
        Store wave forecast data from WaveWatch fetcher output.

        Args:
            data: Dictionary from WaveWatchFetcher.fetch_wave_grid()
            cycle_time: Model cycle time (parsed from data if not provided)
        """
        if cycle_time is None:
            cycle_time = datetime.fromisoformat(data['cycle_time'])

        # Convert lists to numpy arrays
        lat = np.array(data['lat'])
        lon = np.array(data['lon'])
        hs = np.array(data['significant_wave_height'])
        tp = np.array(data['peak_wave_period'])
        dp = np.array(data['mean_wave_direction'])

        # Get forecast time
        forecast_time = datetime.fromisoformat(data['forecast_time'])

        # Optional swell components
        hs_wind = np.array(data.get('wind_sea_height', hs * 0.3))
        hs_swell = np.array(data.get('swell_height', hs * 0.7))
        tp_swell = np.array(data.get('swell_period', tp))
        dp_swell = np.array(data.get('swell_direction', dp))

        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                'hs': (['time', 'lat', 'lon'], hs[np.newaxis, :, :].astype(np.float32)),
                'tp': (['time', 'lat', 'lon'], tp[np.newaxis, :, :].astype(np.float32)),
                'dp': (['time', 'lat', 'lon'], dp[np.newaxis, :, :].astype(np.float32)),
                'hs_wind': (['time', 'lat', 'lon'], hs_wind[np.newaxis, :, :].astype(np.float32)),
                'hs_swell': (['time', 'lat', 'lon'], hs_swell[np.newaxis, :, :].astype(np.float32)),
                'tp_swell': (['time', 'lat', 'lon'], tp_swell[np.newaxis, :, :].astype(np.float32)),
                'dp_swell': (['time', 'lat', 'lon'], dp_swell[np.newaxis, :, :].astype(np.float32)),
            },
            coords={
                'time': [np.datetime64(forecast_time)],
                'lat': lat.astype(np.float64),
                'lon': lon.astype(np.float64),
            },
            attrs={
                'model': 'WaveWatch III',
                'source': 'NOAA NOMADS',
                'resolution_deg': data.get('resolution_deg', 0.25),
                'forecast_hour': data.get('forecast_hour', 0),
            }
        )

        # Add variable attributes
        ds['hs'].attrs = {
            'long_name': 'Significant wave height',
            'units': 'm',
            'standard_name': 'sea_surface_wave_significant_height'
        }
        ds['tp'].attrs = {
            'long_name': 'Peak wave period',
            'units': 's',
            'standard_name': 'sea_surface_wave_period_at_variance_spectral_density_maximum'
        }
        ds['dp'].attrs = {
            'long_name': 'Peak wave direction',
            'units': 'degrees',
            'standard_name': 'sea_surface_wave_from_direction',
            'comment': 'Direction FROM which waves propagate'
        }
        ds['hs_wind'].attrs = {
            'long_name': 'Wind sea significant height',
            'units': 'm'
        }
        ds['hs_swell'].attrs = {
            'long_name': 'Primary swell significant height',
            'units': 'm'
        }
        ds['tp_swell'].attrs = {
            'long_name': 'Primary swell peak period',
            'units': 's'
        }
        ds['dp_swell'].attrs = {
            'long_name': 'Primary swell direction',
            'units': 'degrees'
        }

        self.write_forecast(ds, cycle_time, mode='w')
        logger.info(f"Stored WaveWatch III forecast for cycle {cycle_time.isoformat()}")

    def store_forecast_range(
        self,
        data_list: List[Dict],
        cycle_time: datetime
    ):
        """
        Store multiple forecast hours as a single dataset.

        Args:
            data_list: List of dictionaries from WaveWatchFetcher for different forecast hours
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
        hs = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        tp = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        dp = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        hs_wind = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        hs_swell = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        tp_swell = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)
        dp_swell = np.zeros((n_times, n_lat, n_lon), dtype=np.float32)

        for i, data in enumerate(data_list):
            times.append(np.datetime64(datetime.fromisoformat(data['forecast_time'])))
            hs[i] = np.array(data['significant_wave_height'])
            tp[i] = np.array(data['peak_wave_period'])
            dp[i] = np.array(data['mean_wave_direction'])

            hs_wind[i] = np.array(data.get('wind_sea_height', hs[i] * 0.3))
            hs_swell[i] = np.array(data.get('swell_height', hs[i] * 0.7))
            tp_swell[i] = np.array(data.get('swell_period', tp[i]))
            dp_swell[i] = np.array(data.get('swell_direction', dp[i]))

        ds = xr.Dataset(
            data_vars={
                'hs': (['time', 'lat', 'lon'], hs),
                'tp': (['time', 'lat', 'lon'], tp),
                'dp': (['time', 'lat', 'lon'], dp),
                'hs_wind': (['time', 'lat', 'lon'], hs_wind),
                'hs_swell': (['time', 'lat', 'lon'], hs_swell),
                'tp_swell': (['time', 'lat', 'lon'], tp_swell),
                'dp_swell': (['time', 'lat', 'lon'], dp_swell),
            },
            coords={
                'time': times,
                'lat': lat.astype(np.float64),
                'lon': lon.astype(np.float64),
            },
            attrs={
                'model': 'WaveWatch III',
                'source': 'NOAA NOMADS',
                'resolution_deg': data_list[0].get('resolution_deg', 0.25),
                'min_forecast_hour': data_list[0].get('forecast_hour', 0),
                'max_forecast_hour': data_list[-1].get('forecast_hour', 0),
            }
        )

        # Add variable attributes
        ds['hs'].attrs = {'long_name': 'Significant wave height', 'units': 'm'}
        ds['tp'].attrs = {'long_name': 'Peak wave period', 'units': 's'}
        ds['dp'].attrs = {'long_name': 'Peak wave direction', 'units': 'degrees'}
        ds['hs_wind'].attrs = {'long_name': 'Wind sea significant height', 'units': 'm'}
        ds['hs_swell'].attrs = {'long_name': 'Primary swell significant height', 'units': 'm'}
        ds['tp_swell'].attrs = {'long_name': 'Primary swell peak period', 'units': 's'}
        ds['dp_swell'].attrs = {'long_name': 'Primary swell direction', 'units': 'degrees'}

        self.write_forecast(ds, cycle_time, mode='w')
        logger.info(f"Stored WaveWatch III forecast range ({n_times} hours) for cycle {cycle_time.isoformat()}")

    def get_boundary_conditions(
        self,
        locations: List[Tuple[float, float]],
        time_range: Optional[Tuple[str, str]] = None,
        output_format: str = 'parametric'
    ) -> xr.Dataset:
        """
        Extract wave boundary conditions for model input.

        Args:
            locations: List of (lat, lon) tuples for boundary points
            time_range: Optional (start, end) time range
            output_format: 'parametric' for Hs/Tp/Dp or 'spectral' (future)

        Returns:
            xarray Dataset with boundary condition data
        """
        ds = self.open()

        if time_range:
            ds = ds.sel(time=slice(time_range[0], time_range[1]))

        # Extract data at each location
        datasets = []
        for i, (lat, lon) in enumerate(locations):
            point_ds = ds.sel(lat=lat, lon=lon, method='nearest')
            point_ds = point_ds.expand_dims({'location': [i]})
            point_ds['location_lat'] = ('location', [lat])
            point_ds['location_lon'] = ('location', [lon])
            datasets.append(point_ds)

        # Combine all locations
        result = xr.concat(datasets, dim='location')

        return result

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
            Dictionary in same format as WaveWatchFetcher output
        """
        ds = self.get_forecast(bounds=bounds)

        if 'time' in ds.dims and len(ds.time) > forecast_hour:
            ds = ds.isel(time=forecast_hour)
        elif 'time' in ds.dims:
            ds = ds.isel(time=0)

        return {
            'lat': ds.lat.values.tolist(),
            'lon': ds.lon.values.tolist(),
            'significant_wave_height': ds.hs.values.tolist() if 'hs' in ds else None,
            'peak_wave_period': ds.tp.values.tolist() if 'tp' in ds else None,
            'mean_wave_direction': ds.dp.values.tolist() if 'dp' in ds else None,
            'wind_sea_height': ds.hs_wind.values.tolist() if 'hs_wind' in ds else None,
            'swell_height': ds.hs_swell.values.tolist() if 'hs_swell' in ds else None,
            'swell_period': ds.tp_swell.values.tolist() if 'tp_swell' in ds else None,
            'swell_direction': ds.dp_swell.values.tolist() if 'dp_swell' in ds else None,
            'forecast_time': str(ds.time.values) if 'time' in ds.coords else None,
            'cycle_time': ds.attrs.get('cycle_time'),
            'forecast_hour': forecast_hour,
            'resolution_deg': ds.attrs.get('resolution_deg', 0.25),
            'model': 'WaveWatch III',
            'units': {
                'significant_wave_height': 'm',
                'peak_wave_period': 's',
                'mean_wave_direction': 'degrees (direction from)',
                'lat': 'degrees_north',
                'lon': 'degrees_east'
            }
        }
