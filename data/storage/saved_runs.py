"""
Saved Run Manager

Manages snapshots of forecast data for analysis and comparison.
Allows saving current forecast state and loading historical runs.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import xarray as xr

from .config import StorageConfig, get_storage_config
from .metadata import MetadataDB
from .wind import WindDataStore
from .waves import WaveDataStore
from .currents import CurrentDataStore

logger = logging.getLogger(__name__)


class SavedRunManager:
    """Manages saved forecast snapshots for analysis"""

    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize saved run manager.

        Args:
            config: Storage configuration
        """
        self.config = config or get_storage_config()
        self.db = MetadataDB(config=self.config)
        self._saved_runs_path = Path(self.config.get_saved_runs_path())
        self._saved_runs_path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        description: Optional[str] = None,
        models: Optional[List[str]] = None
    ) -> str:
        """
        Save current forecast state as a named snapshot.

        Args:
            name: Unique name for this saved run
            description: Optional description
            models: List of models to save ('wind', 'waves', 'currents')
                   Defaults to all available

        Returns:
            Path to saved run directory
        """
        if models is None:
            models = ['wind', 'waves', 'currents']

        # Create directory for this run
        run_path = self._saved_runs_path / name
        if run_path.exists():
            raise ValueError(f"Saved run '{name}' already exists. Use a different name or delete it first.")

        run_path.mkdir(parents=True)
        logger.info(f"Creating saved run '{name}' at {run_path}")

        # Get cycle time from current data
        cycle_time = None
        saved_models = []

        for model in models:
            try:
                if model == 'wind':
                    store = WindDataStore(config=self.config)
                    if store.exists():
                        src_path = Path(store.store_path)
                        dst_path = run_path / 'wind' / 'gfs.zarr'
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(src_path, dst_path)
                        saved_models.append('wind')
                        if cycle_time is None:
                            cycle_time = store.get_latest_cycle_time()
                        logger.info(f"Saved wind data to {dst_path}")

                elif model == 'waves':
                    store = WaveDataStore(config=self.config)
                    if store.exists():
                        src_path = Path(store.store_path)
                        dst_path = run_path / 'waves' / 'ww3.zarr'
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(src_path, dst_path)
                        saved_models.append('waves')
                        if cycle_time is None:
                            cycle_time = store.get_latest_cycle_time()
                        logger.info(f"Saved wave data to {dst_path}")

                elif model == 'currents':
                    store = CurrentDataStore(config=self.config)
                    if store.exists():
                        src_path = Path(store.store_path)
                        dst_path = run_path / 'currents' / 'rtofs.zarr'
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copytree(src_path, dst_path)
                        saved_models.append('currents')
                        if cycle_time is None:
                            cycle_time = store.get_latest_cycle_time()
                        logger.info(f"Saved current data to {dst_path}")

            except Exception as e:
                logger.warning(f"Failed to save {model}: {e}")

        if not saved_models:
            shutil.rmtree(run_path)
            raise ValueError("No data available to save")

        # Record in database
        self.db.record_saved_run(
            name=name,
            store_path=str(run_path),
            cycle_time=cycle_time,
            description=description,
            models=saved_models
        )

        logger.info(f"Saved run '{name}' created with models: {saved_models}")
        return str(run_path)

    def load(
        self,
        name: str,
        data_type: str = 'wind'
    ) -> xr.Dataset:
        """
        Load data from a saved run.

        Args:
            name: Name of the saved run
            data_type: Type of data to load ('wind', 'waves', 'currents')

        Returns:
            xarray Dataset
        """
        run_info = self.db.get_saved_run(name)
        if not run_info:
            raise ValueError(f"Saved run '{name}' not found")

        run_path = Path(run_info['store_path'])

        if data_type == 'wind':
            store_path = run_path / 'wind' / 'gfs.zarr'
        elif data_type == 'waves':
            store_path = run_path / 'waves' / 'ww3.zarr'
        elif data_type == 'currents':
            store_path = run_path / 'currents' / 'rtofs.zarr'
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        if not store_path.exists():
            raise ValueError(f"Data type '{data_type}' not available in saved run '{name}'")

        return xr.open_zarr(store_path)

    def load_wind(self, name: str) -> WindDataStore:
        """Load wind data store from a saved run"""
        run_info = self.db.get_saved_run(name)
        if not run_info:
            raise ValueError(f"Saved run '{name}' not found")

        run_path = Path(run_info['store_path'])
        store_path = str(run_path / 'wind' / 'gfs.zarr')

        return WindDataStore(store_path=store_path, config=self.config)

    def load_waves(self, name: str) -> WaveDataStore:
        """Load wave data store from a saved run"""
        run_info = self.db.get_saved_run(name)
        if not run_info:
            raise ValueError(f"Saved run '{name}' not found")

        run_path = Path(run_info['store_path'])
        store_path = str(run_path / 'waves' / 'ww3.zarr')

        return WaveDataStore(store_path=store_path, config=self.config)

    def load_currents(self, name: str) -> CurrentDataStore:
        """Load current data store from a saved run"""
        run_info = self.db.get_saved_run(name)
        if not run_info:
            raise ValueError(f"Saved run '{name}' not found")

        run_path = Path(run_info['store_path'])
        store_path = str(run_path / 'currents' / 'rtofs.zarr')

        return CurrentDataStore(store_path=store_path, config=self.config)

    def list_runs(self) -> List[Dict]:
        """List all saved runs"""
        return self.db.list_saved_runs()

    def get_run_info(self, name: str) -> Optional[Dict]:
        """Get information about a saved run"""
        run_info = self.db.get_saved_run(name)
        if not run_info:
            return None

        # Add size information
        run_path = Path(run_info['store_path'])
        if run_path.exists():
            total_size = sum(
                f.stat().st_size
                for f in run_path.rglob('*')
                if f.is_file()
            )
            run_info['size_mb'] = total_size / (1024 * 1024)

        return run_info

    def delete(self, name: str):
        """Delete a saved run"""
        run_info = self.db.get_saved_run(name)
        if not run_info:
            raise ValueError(f"Saved run '{name}' not found")

        run_path = Path(run_info['store_path'])
        if run_path.exists():
            shutil.rmtree(run_path)
            logger.info(f"Deleted saved run directory: {run_path}")

        self.db.delete_saved_run(name)
        logger.info(f"Deleted saved run '{name}'")

    def compare_runs(
        self,
        name1: str,
        name2: str,
        data_type: str = 'waves',
        variable: str = 'hs'
    ) -> xr.Dataset:
        """
        Compare two saved runs.

        Args:
            name1: First saved run name
            name2: Second saved run name
            data_type: Type of data to compare
            variable: Variable to compare

        Returns:
            xarray Dataset with difference (run2 - run1)
        """
        ds1 = self.load(name1, data_type)
        ds2 = self.load(name2, data_type)

        # Compute difference
        diff = ds2[variable] - ds1[variable]

        return xr.Dataset({
            f'{variable}_diff': diff,
            f'{variable}_run1': ds1[variable],
            f'{variable}_run2': ds2[variable],
        })
