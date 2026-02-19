"""
Data Manager for the dev viewer.

Provides lazy-loaded, cached access to SWAN outputs, surfzone meshes,
simulation results, partition data, ray paths, spot configs, and buoy data.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DataManager:
    """
    Centralized data loader with in-memory caching.

    All heavy data modules (data.swan, data.surfzone, data.spots) are
    imported lazily inside methods to avoid import-order issues when
    sys.path hasn't been configured yet.
    """

    def __init__(self, project_root: Path):
        self._project_root = Path(project_root)
        self._cache: Dict[tuple, Any] = {}

    # -----------------------------------------------------------------
    # SWAN
    # -----------------------------------------------------------------

    def get_swan_output(self, region: str, resolution: str = 'coarse'):
        """Load SWAN output for a region/resolution (cached)."""
        key = ('swan', region, resolution)
        if key not in self._cache:
            from data.swan.analysis.output_reader import read_swan_output

            run_dir = (
                self._project_root / "data" / "swan" / "runs"
                / region / resolution / "latest"
            )
            self._cache[key] = read_swan_output(run_dir)
        return self._cache[key]

    def available_swan_resolutions(self, region: str) -> List[str]:
        """Return sorted list of SWAN resolutions that have a 'latest' symlink."""
        swan_dir = self._project_root / "data" / "swan" / "runs" / region
        if not swan_dir.exists():
            return []
        return sorted([
            d.name for d in swan_dir.iterdir()
            if d.is_dir() and (d / "latest").exists()
        ])

    # -----------------------------------------------------------------
    # Surfzone Mesh
    # -----------------------------------------------------------------

    def get_mesh(self, region: str):
        """Load the SurfZoneMesh for a region (cached)."""
        key = ('mesh', region)
        if key not in self._cache:
            from data.surfzone.mesh import SurfZoneMesh

            mesh_dir = self._project_root / "data" / "surfzone" / "meshes" / region
            self._cache[key] = SurfZoneMesh.load(mesh_dir)
        return self._cache[key]

    # -----------------------------------------------------------------
    # Forward Tracing Result
    # -----------------------------------------------------------------

    def get_result(self, region: str):
        """Load the ForwardTracingResult for a region (cached)."""
        key = ('result', region)
        if key not in self._cache:
            from data.surfzone.runner.output_writer import load_forward_result

            result_path = (
                self._project_root / "data" / "surfzone" / "output"
                / region / "forward_result.npz"
            )
            self._cache[key] = load_forward_result(result_path)
        return self._cache[key]

    # -----------------------------------------------------------------
    # Per-partition Data
    # -----------------------------------------------------------------

    def get_partition_data(self, region: str) -> Dict[str, Dict[str, Any]]:
        """Load per-partition .npz files for a region (cached)."""
        key = ('partitions', region)
        if key not in self._cache:
            import numpy as np

            output_dir = (
                self._project_root / "data" / "surfzone" / "output" / region
            )
            partitions: Dict[str, Dict[str, Any]] = {}
            for name in ['wind_sea', 'primary_swell', 'secondary_swell', 'tertiary_swell']:
                path = output_dir / f"{name}.npz"
                if path.exists():
                    data = np.load(path)
                    partitions[name] = {k: data[k] for k in data.files}
            self._cache[key] = partitions
        return self._cache[key]

    # -----------------------------------------------------------------
    # Ray Paths
    # -----------------------------------------------------------------

    def get_ray_paths(self, region: str) -> Optional[Dict[str, Any]]:
        """Load ray_paths.npz for a region (cached). Returns None if missing."""
        key = ('rays', region)
        if key not in self._cache:
            import numpy as np

            path = (
                self._project_root / "data" / "surfzone" / "output"
                / region / "ray_paths.npz"
            )
            if path.exists():
                data = np.load(path)
                self._cache[key] = {k: data[k] for k in data.files}
            else:
                self._cache[key] = None
        return self._cache[key]

    # -----------------------------------------------------------------
    # Surf Spots
    # -----------------------------------------------------------------

    def get_spots(self, region: str) -> list:
        """Load surf spot configs for a region (cached)."""
        key = ('spots', region)
        if key not in self._cache:
            from data.spots.spot import load_spots_config

            try:
                self._cache[key] = load_spots_config(region)
            except Exception:
                self._cache[key] = []
        return self._cache[key]

    # -----------------------------------------------------------------
    # Buoy Data (NDBC + CDIP with partitioned spectral data)
    # -----------------------------------------------------------------

    def get_buoy_data(self, region: str) -> List[Dict[str, Any]]:
        """Fetch NDBC + CDIP buoy data with partitioned swell components (cached).

        Filters buoys to the region's geographic extent, fetches partitioned
        spectral data concurrently, and returns a list of buoy dicts.
        Each buoy has: station_id, name, network, lat, lon, timestamp,
        partitions (with r1 confidence), combined params, and error info.
        """
        key = ('buoys', region)
        if key not in self._cache:
            self._cache[key] = self._fetch_buoy_data(region)
        return self._cache[key]

    def _fetch_buoy_data(self, region: str) -> List[Dict[str, Any]]:
        """Internal: fetch buoy data from NDBC and CDIP APIs."""
        from data.regions.region import get_region

        reg = get_region(region)
        # Pad bounds by 0.5° to include offshore buoys near the edges
        lat_min = reg.lat_range[0] - 0.5
        lat_max = reg.lat_range[1] + 0.5
        lon_min = reg.lon_range[0] - 0.5
        lon_max = reg.lon_range[1] + 0.5

        try:
            return asyncio.run(
                self._fetch_buoys_async(lat_min, lat_max, lon_min, lon_max)
            )
        except RuntimeError:
            # Event loop already running (e.g. inside Jupyter/Panel)
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(
                    self._fetch_buoys_async(lat_min, lat_max, lon_min, lon_max)
                )
            except ImportError:
                logger.warning(
                    "nest_asyncio not installed; cannot fetch buoys in running event loop. "
                    "Install with: pip install nest_asyncio"
                )
                return []

    async def _fetch_buoys_async(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    ) -> List[Dict[str, Any]]:
        """Concurrently fetch NDBC and CDIP buoy data within bounds."""
        from data.pipelines.buoy.fetcher import NDBCBuoyFetcher, CDIPBuoyFetcher

        ndbc_fetcher = NDBCBuoyFetcher()
        cdip_fetcher = CDIPBuoyFetcher()

        # Filter buoys to region bounds using the known station dicts
        from backend.api.app.routers.buoys import NDBC_CALIFORNIA_BUOYS
        ndbc_in_region = {
            sid: info for sid, info in NDBC_CALIFORNIA_BUOYS.items()
            if lat_min <= info['lat'] <= lat_max and lon_min <= info['lon'] <= lon_max
        }

        cdip_in_region = {
            sid: info for sid, info in cdip_fetcher.CALIFORNIA_BUOYS.items()
            if lat_min <= info['lat'] <= lat_max and lon_min <= info['lon'] <= lon_max
        }

        logger.info(
            f"Fetching buoys: {len(ndbc_in_region)} NDBC, {len(cdip_in_region)} CDIP "
            f"(bounds: lat [{lat_min:.1f}, {lat_max:.1f}], lon [{lon_min:.1f}, {lon_max:.1f}])"
        )

        tasks = []

        # NDBC tasks — fetch_partitioned_spectral_data for r1 confidence
        for sid, info in ndbc_in_region.items():
            tasks.append(self._fetch_single_ndbc(ndbc_fetcher, sid, info))

        # CDIP tasks — fetch_partitioned_wave_data (ERDDAP + 9-band, skips THREDDS)
        for sid, info in cdip_in_region.items():
            tasks.append(self._fetch_single_cdip(cdip_fetcher, sid, info))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        buoys = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Buoy fetch error: {r}")
                continue
            if r is not None:
                buoys.append(r)

        # Filter out buoys with stale data (>48h old) — likely decommissioned
        from datetime import datetime, timezone, timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
        fresh_buoys = []
        for b in buoys:
            ts = b.get('timestamp')
            if ts is None:
                fresh_buoys.append(b)  # Keep if no timestamp (partitions-only)
                continue
            try:
                if isinstance(ts, str):
                    ts_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                else:
                    ts_dt = ts
                if ts_dt >= cutoff:
                    fresh_buoys.append(b)
                else:
                    logger.info(f"Skipping stale buoy {b['station_id']}: {ts}")
            except (ValueError, TypeError):
                fresh_buoys.append(b)  # Keep if unparsable

        logger.info(f"Fetched {len(fresh_buoys)} buoys ({len(buoys) - len(fresh_buoys)} stale filtered)")
        return fresh_buoys

    async def _fetch_single_ndbc(
        self, fetcher, station_id: str, info: Dict,
    ) -> Optional[Dict[str, Any]]:
        """Fetch partitioned spectral data for one NDBC buoy."""
        try:
            data = await asyncio.wait_for(
                fetcher.fetch_partitioned_spectral_data(station_id),
                timeout=15.0,
            )
            if data.get('status') == 'error':
                logger.warning(f"NDBC {station_id}: {data.get('error', 'unknown error')}")
                return None
            return {
                'station_id': station_id,
                'name': info['name'],
                'network': 'NDBC',
                'lat': info['lat'],
                'lon': info['lon'],
                'timestamp': data.get('timestamp'),
                'partitions': data.get('partitions', []),
                'combined': data.get('combined'),
                'error': None,
            }
        except asyncio.TimeoutError:
            logger.warning(f"NDBC {station_id}: timeout after 15s")
            return None
        except Exception as e:
            logger.warning(f"NDBC {station_id}: {e}")
            return None

    async def _fetch_single_cdip(
        self, fetcher, station_id: str, info: Dict,
    ) -> Optional[Dict[str, Any]]:
        """Fetch partitioned wave data for one CDIP buoy."""
        try:
            data = await asyncio.wait_for(
                fetcher.fetch_partitioned_wave_data(station_id),
                timeout=15.0,
            )
            if data.get('status') == 'error':
                logger.warning(f"CDIP {station_id}: {data.get('error', 'unknown error')}")
                return None
            return {
                'station_id': f'CDIP-{station_id}',
                'name': info['name'],
                'network': 'CDIP',
                'lat': info['lat'],
                'lon': info['lon'],
                'depth_m': info.get('depth_m'),
                'timestamp': data.get('timestamp'),
                'partitions': data.get('partitions', []),
                'combined': data.get('combined'),
                'error': None,
            }
        except asyncio.TimeoutError:
            logger.warning(f"CDIP {station_id}: timeout after 15s")
            return None
        except Exception as e:
            logger.warning(f"CDIP {station_id}: {e}")
            return None

    # -----------------------------------------------------------------
    # Data Availability
    # -----------------------------------------------------------------

    def has_data(self, data_type: str, region: str) -> bool:
        """Check whether data of a given type exists for a region."""
        if data_type == 'SWAN Data':
            return len(self.available_swan_resolutions(region)) > 0
        elif data_type == 'Surfzone Mesh':
            mesh_dir = self._project_root / "data" / "surfzone" / "meshes" / region
            return mesh_dir.exists() and any(mesh_dir.glob("*.npz"))
        elif data_type == 'Surfzone Results':
            result_path = (
                self._project_root / "data" / "surfzone" / "output"
                / region / "forward_result.npz"
            )
            return result_path.exists()
        return False

    # -----------------------------------------------------------------
    # Cache Management
    # -----------------------------------------------------------------

    def clear_cache(self, region: Optional[str] = None):
        """Clear all cached data, or only entries for a specific region."""
        if region:
            self._cache = {k: v for k, v in self._cache.items() if k[1] != region}
        else:
            self._cache.clear()
