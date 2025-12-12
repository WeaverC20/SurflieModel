"""
Data Storage Module

Provides unified access to Zarr-based storage for oceanographic data.
Supports local filesystem and cloud storage (S3/GCS) via fsspec.
"""

from .config import StorageConfig, REGIONS, RegionConfig, get_storage_config, set_storage_config
from .metadata import MetadataDB
from .base import BaseDataStore
from .wind import WindDataStore
from .waves import WaveDataStore
from .currents import CurrentDataStore
from .bathymetry import BathymetryStore, list_available_bathymetry, get_best_bathymetry_for_region
from .saved_runs import SavedRunManager

__all__ = [
    # Config
    'StorageConfig',
    'REGIONS',
    'RegionConfig',
    'get_storage_config',
    'set_storage_config',
    # Database
    'MetadataDB',
    # Data Stores
    'BaseDataStore',
    'WindDataStore',
    'WaveDataStore',
    'CurrentDataStore',
    'BathymetryStore',
    # Utilities
    'list_available_bathymetry',
    'get_best_bathymetry_for_region',
    'SavedRunManager',
]
