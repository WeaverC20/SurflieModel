"""
Metadata Database

SQLite-based index for tracking available forecast cycles and data status.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .config import StorageConfig, get_storage_config

logger = logging.getLogger(__name__)


class MetadataDB:
    """SQLite database for tracking forecast metadata"""

    def __init__(self, db_path: Optional[str] = None, config: Optional[StorageConfig] = None):
        """
        Initialize metadata database.

        Args:
            db_path: Path to SQLite database (overrides config)
            config: Storage configuration
        """
        self.config = config or get_storage_config()
        self._db_path = db_path or str(self.config.metadata_db_path)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS forecast_cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    cycle_time DATETIME NOT NULL,
                    min_forecast_hour INTEGER DEFAULT 0,
                    max_forecast_hour INTEGER DEFAULT 0,
                    min_lat REAL,
                    max_lat REAL,
                    min_lon REAL,
                    max_lon REAL,
                    store_path TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'ready',
                    UNIQUE(model, cycle_time)
                );

                CREATE TABLE IF NOT EXISTS saved_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cycle_time DATETIME,
                    store_path TEXT NOT NULL,
                    models TEXT
                );

                CREATE TABLE IF NOT EXISTS bathymetry_tiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    resolution_m REAL,
                    min_lat REAL,
                    max_lat REAL,
                    min_lon REAL,
                    max_lon REAL,
                    store_path TEXT NOT NULL,
                    source TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_cycles_model_time
                    ON forecast_cycles(model, cycle_time DESC);

                CREATE INDEX IF NOT EXISTS idx_cycles_status
                    ON forecast_cycles(status);

                CREATE INDEX IF NOT EXISTS idx_bathy_bounds
                    ON bathymetry_tiles(min_lat, max_lat, min_lon, max_lon);
            ''')
            logger.info(f"Initialized metadata database at {self._db_path}")

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Forecast Cycles
    # -------------------------------------------------------------------------

    def record_forecast_cycle(
        self,
        model: str,
        cycle_time: datetime,
        store_path: str,
        min_forecast_hour: int = 0,
        max_forecast_hour: int = 384,
        bounds: Optional[Dict[str, float]] = None,
        status: str = 'ready'
    ) -> int:
        """
        Record a new forecast cycle in the database.

        Args:
            model: Model name ('gfs', 'ww3', 'rtofs')
            cycle_time: Model initialization time
            store_path: Path to Zarr store
            min_forecast_hour: First forecast hour
            max_forecast_hour: Last forecast hour
            bounds: Geographic bounds dict
            status: Status ('downloading', 'ready', 'archived')

        Returns:
            Row ID of inserted record
        """
        with self._get_connection() as conn:
            cursor = conn.execute('''
                INSERT OR REPLACE INTO forecast_cycles
                (model, cycle_time, store_path, min_forecast_hour, max_forecast_hour,
                 min_lat, max_lat, min_lon, max_lon, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                model,
                cycle_time.isoformat(),
                store_path,
                min_forecast_hour,
                max_forecast_hour,
                bounds.get('min_lat') if bounds else None,
                bounds.get('max_lat') if bounds else None,
                bounds.get('min_lon') if bounds else None,
                bounds.get('max_lon') if bounds else None,
                status
            ))
            return cursor.lastrowid

    def get_latest_cycle(self, model: str) -> Optional[Dict]:
        """
        Get the most recent forecast cycle for a model.

        Args:
            model: Model name

        Returns:
            Cycle info dict or None
        """
        with self._get_connection() as conn:
            row = conn.execute('''
                SELECT * FROM forecast_cycles
                WHERE model = ? AND status = 'ready'
                ORDER BY cycle_time DESC
                LIMIT 1
            ''', (model,)).fetchone()

            if row:
                return dict(row)
            return None

    def get_cycle_by_time(self, model: str, cycle_time: datetime) -> Optional[Dict]:
        """Get a specific forecast cycle"""
        with self._get_connection() as conn:
            row = conn.execute('''
                SELECT * FROM forecast_cycles
                WHERE model = ? AND cycle_time = ?
            ''', (model, cycle_time.isoformat())).fetchone()

            if row:
                return dict(row)
            return None

    def list_cycles(
        self,
        model: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """List forecast cycles with optional filtering"""
        query = 'SELECT * FROM forecast_cycles WHERE 1=1'
        params = []

        if model:
            query += ' AND model = ?'
            params.append(model)
        if status:
            query += ' AND status = ?'
            params.append(status)

        query += ' ORDER BY cycle_time DESC LIMIT ?'
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def update_cycle_status(self, model: str, cycle_time: datetime, status: str):
        """Update the status of a forecast cycle"""
        with self._get_connection() as conn:
            conn.execute('''
                UPDATE forecast_cycles
                SET status = ?
                WHERE model = ? AND cycle_time = ?
            ''', (status, model, cycle_time.isoformat()))

    def delete_old_cycles(self, model: str, keep_count: int = 1):
        """Delete old cycles, keeping only the most recent N"""
        with self._get_connection() as conn:
            # Get IDs to keep
            keep_ids = conn.execute('''
                SELECT id FROM forecast_cycles
                WHERE model = ?
                ORDER BY cycle_time DESC
                LIMIT ?
            ''', (model, keep_count)).fetchall()

            keep_ids = [row['id'] for row in keep_ids]

            if keep_ids:
                placeholders = ','.join(['?' for _ in keep_ids])
                conn.execute(f'''
                    DELETE FROM forecast_cycles
                    WHERE model = ? AND id NOT IN ({placeholders})
                ''', [model] + keep_ids)

    # -------------------------------------------------------------------------
    # Saved Runs
    # -------------------------------------------------------------------------

    def record_saved_run(
        self,
        name: str,
        store_path: str,
        cycle_time: Optional[datetime] = None,
        description: Optional[str] = None,
        models: Optional[List[str]] = None
    ) -> int:
        """Record a saved run"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO saved_runs (name, store_path, cycle_time, description, models)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                name,
                store_path,
                cycle_time.isoformat() if cycle_time else None,
                description,
                ','.join(models) if models else None
            ))
            return cursor.lastrowid

    def get_saved_run(self, name: str) -> Optional[Dict]:
        """Get a saved run by name"""
        with self._get_connection() as conn:
            row = conn.execute('''
                SELECT * FROM saved_runs WHERE name = ?
            ''', (name,)).fetchone()

            if row:
                result = dict(row)
                if result.get('models'):
                    result['models'] = result['models'].split(',')
                return result
            return None

    def list_saved_runs(self) -> List[Dict]:
        """List all saved runs"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM saved_runs ORDER BY created_at DESC
            ''').fetchall()

            results = []
            for row in rows:
                result = dict(row)
                if result.get('models'):
                    result['models'] = result['models'].split(',')
                results.append(result)
            return results

    def delete_saved_run(self, name: str):
        """Delete a saved run record"""
        with self._get_connection() as conn:
            conn.execute('DELETE FROM saved_runs WHERE name = ?', (name,))

    # -------------------------------------------------------------------------
    # Bathymetry
    # -------------------------------------------------------------------------

    def record_bathymetry(
        self,
        name: str,
        store_path: str,
        resolution_m: float,
        bounds: Dict[str, float],
        source: Optional[str] = None
    ) -> int:
        """Record a bathymetry tile"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                INSERT OR REPLACE INTO bathymetry_tiles
                (name, store_path, resolution_m, min_lat, max_lat, min_lon, max_lon, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                name,
                store_path,
                resolution_m,
                bounds['min_lat'],
                bounds['max_lat'],
                bounds['min_lon'],
                bounds['max_lon'],
                source
            ))
            return cursor.lastrowid

    def get_bathymetry(self, name: str) -> Optional[Dict]:
        """Get bathymetry tile by name"""
        with self._get_connection() as conn:
            row = conn.execute('''
                SELECT * FROM bathymetry_tiles WHERE name = ?
            ''', (name,)).fetchone()

            if row:
                return dict(row)
            return None

    def find_bathymetry_for_region(
        self,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float
    ) -> List[Dict]:
        """Find bathymetry tiles that cover a region"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM bathymetry_tiles
                WHERE min_lat <= ? AND max_lat >= ?
                  AND min_lon <= ? AND max_lon >= ?
                ORDER BY resolution_m ASC
            ''', (min_lat, max_lat, min_lon, max_lon)).fetchall()

            return [dict(row) for row in rows]
