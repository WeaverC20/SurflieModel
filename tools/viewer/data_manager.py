"""
Data Manager for the dev viewer.

Provides lazy-loaded, cached access to SWAN outputs, surfzone meshes,
simulation results, partition data, ray paths, and spot configs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional


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
