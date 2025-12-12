"""
Base Data Store

Abstract base class for all data stores (wind, waves, currents, bathymetry).
Provides common functionality for Zarr I/O and spatial queries.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from .config import StorageConfig, RegionConfig, get_storage_config

logger = logging.getLogger(__name__)


class BaseDataStore(ABC):
    """Abstract base class for data stores"""

    def __init__(
        self,
        store_path: Optional[str] = None,
        config: Optional[StorageConfig] = None
    ):
        """
        Initialize data store.

        Args:
            store_path: Path to Zarr store (overrides config)
            config: Storage configuration (uses global if not provided)
        """
        self.config = config or get_storage_config()
        self._store_path = store_path
        self._ds: Optional[xr.Dataset] = None

    @property
    @abstractmethod
    def default_store_path(self) -> str:
        """Return the default store path for this data type"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name (e.g., 'gfs', 'ww3', 'rtofs')"""
        pass

    @property
    @abstractmethod
    def variables(self) -> List[str]:
        """Return list of variable names stored in this store"""
        pass

    @property
    def store_path(self) -> str:
        """Get the store path"""
        return self._store_path or self.default_store_path

    def exists(self) -> bool:
        """Check if the store exists"""
        path = Path(self.store_path)
        # Check for both zarr v2 (.zattrs) and v3 (zarr.json) formats
        return path.exists() and (
            (path / '.zattrs').exists() or (path / 'zarr.json').exists()
        )

    def open(self) -> xr.Dataset:
        """
        Open the Zarr store as an xarray Dataset.

        Returns:
            xarray Dataset with lazy loading
        """
        if self._ds is not None:
            return self._ds

        if not self.exists():
            raise FileNotFoundError(f"Store not found: {self.store_path}")

        self._ds = xr.open_zarr(self.store_path)
        return self._ds

    def close(self):
        """Close the dataset if open"""
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def get_forecast(
        self,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        time_range: Optional[Tuple[str, str]] = None,
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        """
        Get forecast data for a region and time range.

        Args:
            bounds: (min_lat, max_lat, min_lon, max_lon) or None for full extent
            time_range: (start_time, end_time) as ISO strings or None for all times
            variables: List of variable names or None for all

        Returns:
            xarray Dataset with requested data
        """
        ds = self.open()

        # Select variables
        if variables:
            ds = ds[variables]

        # Spatial subsetting
        if bounds:
            min_lat, max_lat, min_lon, max_lon = bounds
            ds = ds.sel(
                lat=slice(min_lat, max_lat),
                lon=slice(min_lon, max_lon)
            )

        # Time subsetting
        if time_range:
            start, end = time_range
            ds = ds.sel(time=slice(start, end))

        return ds

    def get_timeseries(
        self,
        lat: float,
        lon: float,
        hours_ahead: int = 72,
        variables: Optional[List[str]] = None
    ) -> xr.Dataset:
        """
        Get time series at a specific point.

        Args:
            lat: Latitude
            lon: Longitude
            hours_ahead: Number of forecast hours to include
            variables: List of variable names or None for all

        Returns:
            xarray Dataset with time dimension only
        """
        ds = self.open()

        if variables:
            ds = ds[variables]

        # Select nearest point
        ds = ds.sel(lat=lat, lon=lon, method='nearest')

        # Limit forecast hours
        if 'time' in ds.dims and len(ds.time) > hours_ahead:
            ds = ds.isel(time=slice(0, hours_ahead))

        return ds

    def get_latest_cycle_time(self) -> Optional[datetime]:
        """Get the cycle time of the latest stored forecast"""
        try:
            ds = self.open()
            if 'cycle_time' in ds.attrs:
                return datetime.fromisoformat(ds.attrs['cycle_time'])
            return None
        except FileNotFoundError:
            return None

    def write_forecast(
        self,
        ds: xr.Dataset,
        cycle_time: datetime,
        mode: str = 'w'
    ):
        """
        Write forecast data to the Zarr store.

        Args:
            ds: xarray Dataset with forecast data
            cycle_time: Model cycle/initialization time
            mode: 'w' to overwrite, 'a' to append
        """
        # Close existing connection
        self.close()

        # Add metadata
        ds.attrs['cycle_time'] = cycle_time.isoformat()
        ds.attrs['model'] = self.model_name
        ds.attrs['updated_at'] = datetime.utcnow().isoformat()

        # Ensure parent directory exists
        store_path = Path(self.store_path)
        store_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure chunking based on dimensions
        chunks = self._get_chunks(ds)

        # Configure encoding with chunking (zarr handles compression automatically)
        encoding = {}
        for var in ds.data_vars:
            encoding[var] = {'chunks': chunks.get(var, None)}

        if mode == 'w' or not self.exists():
            ds.to_zarr(self.store_path, mode='w', encoding=encoding)
            logger.info(f"Created new Zarr store at {self.store_path}")
        else:
            # Append along time dimension
            ds.to_zarr(self.store_path, mode='a', append_dim='time')
            logger.info(f"Appended data to {self.store_path}")

    def _get_chunks(self, ds: xr.Dataset) -> Dict:
        """
        Determine optimal chunking for the dataset.

        Args:
            ds: Dataset to chunk

        Returns:
            Dictionary mapping variable names to chunk tuples
        """
        chunks = {}
        time_chunk = self.config.time_chunk_size

        for var in ds.data_vars:
            dims = ds[var].dims
            var_chunks = []
            for dim in dims:
                if dim == 'time':
                    var_chunks.append(time_chunk)
                elif dim in ('lat', 'latitude'):
                    var_chunks.append(len(ds[dim]))  # Full spatial extent
                elif dim in ('lon', 'longitude'):
                    var_chunks.append(len(ds[dim]))  # Full spatial extent
                else:
                    var_chunks.append(len(ds[dim]))
            chunks[var] = tuple(var_chunks)

        return chunks

    def delete(self):
        """Delete the Zarr store"""
        self.close()
        import shutil
        path = Path(self.store_path)
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"Deleted store at {self.store_path}")

    def get_info(self) -> Dict:
        """Get information about the store"""
        if not self.exists():
            return {'exists': False, 'path': self.store_path}

        ds = self.open()
        return {
            'exists': True,
            'path': self.store_path,
            'model': self.model_name,
            'variables': list(ds.data_vars.keys()),
            'dimensions': dict(ds.dims),
            'cycle_time': ds.attrs.get('cycle_time'),
            'updated_at': ds.attrs.get('updated_at'),
            'size_mb': self._get_store_size_mb()
        }

    def _get_store_size_mb(self) -> float:
        """Calculate store size in MB"""
        path = Path(self.store_path)
        if not path.exists():
            return 0.0
        total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return total / (1024 * 1024)
