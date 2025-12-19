"""
Storage Configuration

Defines storage paths, region configurations, and fsspec settings for
local and cloud storage backends.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class RegionConfig:
    """Configuration for a geographic region"""
    name: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    resolution_deg: float = 0.25

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return bounds as (min_lat, max_lat, min_lon, max_lon)"""
        return (self.min_lat, self.max_lat, self.min_lon, self.max_lon)

    @property
    def bounds_dict(self) -> Dict[str, float]:
        """Return bounds as dictionary"""
        return {
            'min_lat': self.min_lat,
            'max_lat': self.max_lat,
            'min_lon': self.min_lon,
            'max_lon': self.max_lon
        }


# Predefined regions (expandable)
REGIONS: Dict[str, RegionConfig] = {
    'california': RegionConfig(
        name='california',
        min_lat=32.0,
        max_lat=42.0,
        min_lon=-125.0,
        max_lon=-117.0,
        resolution_deg=0.25
    ),
    'southern_california': RegionConfig(
        name='southern_california',
        min_lat=32.0,
        max_lat=35.0,
        min_lon=-121.0,
        max_lon=-117.0,
        resolution_deg=0.25
    ),
    'huntington_beach': RegionConfig(
        name='huntington_beach',
        min_lat=33.5,
        max_lat=33.8,
        min_lon=-118.1,
        max_lon=-117.9,
        resolution_deg=0.01  # Higher resolution for local modeling
    ),
}


@dataclass
class StorageConfig:
    """Configuration for data storage backend"""

    # Base paths
    base_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / 'zarr')
    metadata_db_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / 'metadata.db')

    # Zarr store paths (relative to base_path)
    wind_store: str = 'forecasts/wind/gfs.zarr'
    wave_store: str = 'forecasts/waves/ww3.zarr'
    current_store: str = 'forecasts/currents/rtofs.zarr'
    saved_runs_dir: str = 'saved_runs'
    bathymetry_dir: str = 'static/bathymetry'

    # Historical data for validation
    historical_wind_store: str = 'historical/wind/gfs_historical.zarr'

    # Cloud storage settings (for future S3/GCS support)
    storage_backend: str = 'local'  # 'local', 's3', 'gcs'
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None
    gcs_bucket: Optional[str] = None
    gcs_prefix: Optional[str] = None

    # Default region
    default_region: str = 'california'

    # Chunking configuration for Zarr stores
    time_chunk_size: int = 24  # Hours per chunk

    def __post_init__(self):
        """Ensure paths are Path objects and create directories"""
        if isinstance(self.base_path, str):
            self.base_path = Path(self.base_path)
        if isinstance(self.metadata_db_path, str):
            self.metadata_db_path = Path(self.metadata_db_path)

    def get_wind_store_path(self) -> str:
        """Get full path to wind Zarr store"""
        if self.storage_backend == 'local':
            return str(self.base_path / self.wind_store)
        elif self.storage_backend == 's3':
            return f"s3://{self.s3_bucket}/{self.s3_prefix}/{self.wind_store}"
        elif self.storage_backend == 'gcs':
            return f"gcs://{self.gcs_bucket}/{self.gcs_prefix}/{self.wind_store}"
        return str(self.base_path / self.wind_store)

    def get_wave_store_path(self) -> str:
        """Get full path to wave Zarr store"""
        if self.storage_backend == 'local':
            return str(self.base_path / self.wave_store)
        elif self.storage_backend == 's3':
            return f"s3://{self.s3_bucket}/{self.s3_prefix}/{self.wave_store}"
        elif self.storage_backend == 'gcs':
            return f"gcs://{self.gcs_bucket}/{self.gcs_prefix}/{self.wave_store}"
        return str(self.base_path / self.wave_store)

    def get_current_store_path(self) -> str:
        """Get full path to current Zarr store"""
        if self.storage_backend == 'local':
            return str(self.base_path / self.current_store)
        elif self.storage_backend == 's3':
            return f"s3://{self.s3_bucket}/{self.s3_prefix}/{self.current_store}"
        elif self.storage_backend == 'gcs':
            return f"gcs://{self.gcs_bucket}/{self.gcs_prefix}/{self.current_store}"
        return str(self.base_path / self.current_store)

    def get_saved_runs_path(self) -> str:
        """Get path to saved runs directory"""
        if self.storage_backend == 'local':
            return str(self.base_path / self.saved_runs_dir)
        elif self.storage_backend == 's3':
            return f"s3://{self.s3_bucket}/{self.s3_prefix}/{self.saved_runs_dir}"
        elif self.storage_backend == 'gcs':
            return f"gcs://{self.gcs_bucket}/{self.gcs_prefix}/{self.saved_runs_dir}"
        return str(self.base_path / self.saved_runs_dir)

    def get_historical_wind_store_path(self) -> str:
        """Get full path to historical wind Zarr store"""
        if self.storage_backend == 'local':
            return str(self.base_path / self.historical_wind_store)
        elif self.storage_backend == 's3':
            return f"s3://{self.s3_bucket}/{self.s3_prefix}/{self.historical_wind_store}"
        elif self.storage_backend == 'gcs':
            return f"gcs://{self.gcs_bucket}/{self.gcs_prefix}/{self.historical_wind_store}"
        return str(self.base_path / self.historical_wind_store)

    def ensure_directories(self):
        """Create necessary directories for local storage"""
        if self.storage_backend == 'local':
            (self.base_path / 'forecasts' / 'wind').mkdir(parents=True, exist_ok=True)
            (self.base_path / 'forecasts' / 'waves').mkdir(parents=True, exist_ok=True)
            (self.base_path / 'forecasts' / 'currents').mkdir(parents=True, exist_ok=True)
            (self.base_path / 'historical' / 'wind').mkdir(parents=True, exist_ok=True)
            (self.base_path / self.saved_runs_dir).mkdir(parents=True, exist_ok=True)
            (self.base_path / self.bathymetry_dir).mkdir(parents=True, exist_ok=True)
            self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)


# Global config instance (can be overridden)
_config: Optional[StorageConfig] = None


def get_storage_config() -> StorageConfig:
    """Get the current storage configuration"""
    global _config
    if _config is None:
        _config = StorageConfig()
        # Check for environment variable overrides
        if os.environ.get('SURFLIE_STORAGE_BACKEND'):
            _config.storage_backend = os.environ['SURFLIE_STORAGE_BACKEND']
        if os.environ.get('SURFLIE_S3_BUCKET'):
            _config.s3_bucket = os.environ['SURFLIE_S3_BUCKET']
        if os.environ.get('SURFLIE_S3_PREFIX'):
            _config.s3_prefix = os.environ['SURFLIE_S3_PREFIX']
        if os.environ.get('SURFLIE_DATA_PATH'):
            _config.base_path = Path(os.environ['SURFLIE_DATA_PATH'])
    return _config


def set_storage_config(config: StorageConfig):
    """Set a custom storage configuration"""
    global _config
    _config = config
