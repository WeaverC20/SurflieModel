"""
Bathymetry Data Store

Zarr-based storage for bathymetry/elevation data.
Supports multiple resolutions for different modeling needs.

Data Sources (planned):
- GEBCO 2023: Global ~500m resolution
- NOAA NCEI Coastal DEMs: ~30-90m resolution
- High-resolution lidar: ~1-10m resolution for specific sites
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from .base import BaseDataStore
from .config import StorageConfig, get_storage_config
from .metadata import MetadataDB

logger = logging.getLogger(__name__)


class BathymetryStore(BaseDataStore):
    """Store for bathymetry/elevation data

    Note: This is a placeholder implementation. The actual data fetching
    from GEBCO/NOAA will be implemented in a future phase.
    """

    def __init__(
        self,
        name: str = "global",
        store_path: Optional[str] = None,
        config: Optional[StorageConfig] = None
    ):
        """
        Initialize bathymetry store.

        Args:
            name: Name of this bathymetry dataset (e.g., 'global', 'california', 'huntington')
            store_path: Path to Zarr store (overrides default)
            config: Storage configuration
        """
        super().__init__(store_path=store_path, config=config)
        self._name = name

    @property
    def default_store_path(self) -> str:
        base = Path(self.config.base_path) / self.config.bathymetry_dir
        return str(base / f"{self._name}.zarr")

    @property
    def model_name(self) -> str:
        return 'bathymetry'

    @property
    def variables(self) -> List[str]:
        return ['elevation', 'slope', 'aspect']

    def store_bathymetry(
        self,
        elevation: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        source: str = "unknown",
        resolution_m: float = 100.0
    ):
        """
        Store bathymetry data.

        Args:
            elevation: 2D array of elevation/depth values (negative = underwater)
            lat: 1D array of latitudes
            lon: 1D array of longitudes
            source: Data source name (e.g., 'GEBCO_2023', 'NOAA_NCEI')
            resolution_m: Approximate resolution in meters
        """
        # Calculate slope and aspect if we have enough resolution
        slope = np.zeros_like(elevation)
        aspect = np.zeros_like(elevation)

        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                'elevation': (['lat', 'lon'], elevation.astype(np.float32)),
                'slope': (['lat', 'lon'], slope.astype(np.float32)),
                'aspect': (['lat', 'lon'], aspect.astype(np.float32)),
            },
            coords={
                'lat': lat.astype(np.float64),
                'lon': lon.astype(np.float64),
            },
            attrs={
                'name': self._name,
                'source': source,
                'resolution_m': resolution_m,
                'vertical_datum': 'MSL',
            }
        )

        # Add variable attributes
        ds['elevation'].attrs = {
            'long_name': 'Elevation/Bathymetry',
            'units': 'm',
            'positive': 'up',
            'comment': 'Negative values indicate depth below sea level'
        }
        ds['slope'].attrs = {
            'long_name': 'Seafloor slope',
            'units': 'degrees'
        }
        ds['aspect'].attrs = {
            'long_name': 'Slope aspect/direction',
            'units': 'degrees'
        }

        # Ensure parent directory exists
        store_path = Path(self.store_path)
        store_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to Zarr
        ds.to_zarr(self.store_path, mode='w')

        # Record in metadata database
        db = MetadataDB(config=self.config)
        db.record_bathymetry(
            name=self._name,
            store_path=self.store_path,
            resolution_m=resolution_m,
            bounds={
                'min_lat': float(lat.min()),
                'max_lat': float(lat.max()),
                'min_lon': float(lon.min()),
                'max_lon': float(lon.max())
            },
            source=source
        )

        logger.info(f"Stored bathymetry '{self._name}' to {self.store_path}")

    def get_bathymetry(
        self,
        bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> xr.Dataset:
        """
        Get bathymetry data for a region.

        Args:
            bounds: (min_lat, max_lat, min_lon, max_lon) or None for full extent

        Returns:
            xarray Dataset with elevation data
        """
        ds = self.open()

        if bounds:
            min_lat, max_lat, min_lon, max_lon = bounds
            ds = ds.sel(
                lat=slice(min_lat, max_lat),
                lon=slice(min_lon, max_lon)
            )

        return ds

    def interpolate_to_grid(
        self,
        target_lat: np.ndarray,
        target_lon: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate bathymetry to a target grid.

        Args:
            target_lat: 1D array of target latitudes
            target_lon: 1D array of target longitudes

        Returns:
            2D array of interpolated elevation values
        """
        ds = self.open()

        # Use xarray's built-in interpolation
        interp = ds['elevation'].interp(lat=target_lat, lon=target_lon)

        return interp.values


def list_available_bathymetry(config: Optional[StorageConfig] = None) -> List[Dict]:
    """
    List all available bathymetry datasets.

    Returns:
        List of bathymetry info dictionaries
    """
    db = MetadataDB(config=config)

    # Get all bathymetry records
    with db._get_connection() as conn:
        rows = conn.execute('''
            SELECT * FROM bathymetry_tiles ORDER BY resolution_m ASC
        ''').fetchall()

        return [dict(row) for row in rows]


def get_best_bathymetry_for_region(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    config: Optional[StorageConfig] = None
) -> Optional[BathymetryStore]:
    """
    Find the best (highest resolution) bathymetry for a region.

    Args:
        min_lat, max_lat, min_lon, max_lon: Region bounds

    Returns:
        BathymetryStore for the best available data, or None
    """
    db = MetadataDB(config=config)
    tiles = db.find_bathymetry_for_region(min_lat, max_lat, min_lon, max_lon)

    if not tiles:
        return None

    # First tile is highest resolution (sorted by resolution_m ASC)
    best = tiles[0]
    return BathymetryStore(name=best['name'], store_path=best['store_path'], config=config)
