"""
SWAN Output Reader

Loads and processes SWAN model output files.
SWAN outputs MATLAB v4 format files which require scipy.io.loadmat().

Supports:
- Integrated outputs (Hsig, Tps, Dir)
- Partition outputs (wind sea + up to 3 swells with Hs, Tp, Dir each)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat

logger = logging.getLogger(__name__)


# Maximum number of partitions SWAN can output (0=wind sea, 1-9=swells)
MAX_PARTITIONS = 4  # We use wind sea + 3 swells

# Human-readable labels for partitions
PARTITION_LABELS = {
    0: "Wind Sea",
    1: "Primary Swell",
    2: "Secondary Swell",
    3: "Tertiary Swell",
}


@dataclass
class WavePartitionGrid:
    """
    A single wave partition (wind sea or swell) across the entire grid.

    Attributes:
        hs: Significant wave height (m), 2D array
        tp: Peak period (s), 2D array
        dir: Wave direction (degrees, nautical - FROM), 2D array
        partition_id: Partition index (0=wind sea, 1+=swells)
        label: Human-readable label
    """
    hs: np.ndarray
    tp: np.ndarray
    dir: np.ndarray
    partition_id: int
    label: str

    def mask_invalid(self, exception_value: float = -99.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return masked arrays with invalid points as NaN."""
        hs_masked = np.where(
            (self.hs == exception_value) | (self.hs <= 0),
            np.nan, self.hs
        )
        tp_masked = np.where(
            (self.tp == exception_value) | (self.tp <= 0),
            np.nan, self.tp
        )
        dir_masked = np.where(
            (self.dir == exception_value) | (self.hs <= 0),  # Use hs to mask dir too
            np.nan, self.dir
        )
        return hs_masked, tp_masked, dir_masked


@dataclass
class SwanOutput:
    """
    Container for SWAN model output data.

    Attributes:
        hsig: Significant wave height (m), 2D array
        tps: Peak wave period (s), 2D array
        dir: Mean wave direction (degrees), 2D array
        lons: Longitude coordinates, 1D array
        lats: Latitude coordinates, 1D array
        mesh_name: Name of the mesh used
        region_name: Name of the region
        exception_value: Value used for land/invalid points
        partitions: List of WavePartitionGrid objects (if partition outputs available)
    """
    hsig: np.ndarray
    tps: np.ndarray
    dir: np.ndarray
    lons: np.ndarray
    lats: np.ndarray
    mesh_name: str
    region_name: str
    exception_value: float = -99.0
    partitions: List[WavePartitionGrid] = field(default_factory=list)

    def mask_land(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return masked arrays with land points as NaN.

        Returns:
            Tuple of (hsig_masked, tps_masked, dir_masked)
        """
        hsig_masked = np.where(self.hsig == self.exception_value, np.nan, self.hsig)
        tps_masked = np.where(self.tps == self.exception_value, np.nan, self.tps)
        dir_masked = np.where(self.dir == self.exception_value, np.nan, self.dir)
        return hsig_masked, tps_masked, dir_masked

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """Get map extent as (lon_min, lon_max, lat_min, lat_max)."""
        return (self.lons[0], self.lons[-1], self.lats[0], self.lats[-1])

    @property
    def has_partitions(self) -> bool:
        """Check if partition data is available."""
        return len(self.partitions) > 0

    def get_partition(self, partition_id: int) -> Optional[WavePartitionGrid]:
        """Get a specific partition by ID."""
        for p in self.partitions:
            if p.partition_id == partition_id:
                return p
        return None

    @property
    def wind_sea(self) -> Optional[WavePartitionGrid]:
        """Get wind sea partition (id=0) if available."""
        return self.get_partition(0)

    @property
    def swells(self) -> List[WavePartitionGrid]:
        """Get all swell partitions (id>0)."""
        return [p for p in self.partitions if p.partition_id > 0]

    def summary(self) -> str:
        """Return summary of output data."""
        hsig_masked, tps_masked, dir_masked = self.mask_land()
        lines = [
            f"SWAN Output: {self.region_name}/{self.mesh_name}",
            f"  Grid: {self.hsig.shape[1]} x {self.hsig.shape[0]}",
            f"  Extent: lon [{self.lons[0]:.2f}, {self.lons[-1]:.2f}], lat [{self.lats[0]:.2f}, {self.lats[-1]:.2f}]",
            f"  Hsig: {np.nanmin(hsig_masked):.2f} - {np.nanmax(hsig_masked):.2f} m",
            f"  Tps: {np.nanmin(tps_masked):.1f} - {np.nanmax(tps_masked):.1f} s",
            f"  Dir: {np.nanmin(dir_masked):.0f} - {np.nanmax(dir_masked):.0f} deg",
        ]

        if self.has_partitions:
            lines.append(f"  Partitions: {len(self.partitions)}")
            for p in self.partitions:
                hs_m, tp_m, _ = p.mask_invalid(self.exception_value)
                if np.any(~np.isnan(hs_m)):
                    lines.append(
                        f"    {p.label}: Hs={np.nanmax(hs_m):.2f}m, "
                        f"Tp={np.nanmean(tp_m[~np.isnan(tp_m)]):.1f}s"
                    )

        return '\n'.join(lines)


class SwanOutputReader:
    """
    Reads SWAN output files from a run directory.

    Example usage:
        reader = SwanOutputReader("data/swan/runs/socal/coarse/latest")
        output = reader.read()
        print(output.summary())
    """

    def __init__(self, run_dir: str | Path):
        """
        Initialize reader for a SWAN run directory.

        Args:
            run_dir: Path to run directory containing output files
        """
        self.run_dir = Path(run_dir)

        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        # Load mesh metadata from INPUT file comments or find mesh JSON
        self.mesh_metadata = self._load_mesh_metadata()

    def _load_mesh_metadata(self) -> Dict:
        """Load mesh metadata from run directory or mesh directory."""
        # Try to parse mesh info from INPUT file
        input_file = self.run_dir / "INPUT"
        if input_file.exists():
            with open(input_file) as f:
                content = f.read()

            # Extract mesh name and region from comments
            mesh_name = None
            region_name = None
            for line in content.split('\n'):
                if line.startswith('$ Mesh:'):
                    mesh_name = line.split(':')[1].strip()
                elif line.startswith('$ Region:'):
                    region_name = line.split(':')[1].strip()

            # Try to find mesh JSON in meshes directory
            if mesh_name and region_name:
                # Construct path to mesh directory
                # run_dir is: .../data/swan/runs/{region}/{mesh}/latest/
                # We need to go up 6 levels to get to project root
                project_root = self.run_dir.parent.parent.parent.parent.parent.parent
                mesh_parts = mesh_name.split('_')
                if len(mesh_parts) >= 2:
                    mesh_type = mesh_parts[-1]  # e.g., "coarse"
                    mesh_dir = project_root / "data" / "meshes" / region_name / mesh_type
                    json_path = mesh_dir / f"{mesh_name}.json"
                    if json_path.exists():
                        with open(json_path) as f:
                            return json.load(f)

        # Fallback: return minimal metadata
        return {
            "name": "unknown",
            "region_name": "unknown",
            "origin": [-121.0, 32.0],
            "nx": 75,
            "ny": 56,
            "dx": 0.054,
            "dy": 0.045,
            "exception_value": -99.0
        }

    def _build_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build lon/lat coordinate arrays from mesh metadata."""
        origin = self.mesh_metadata["origin"]
        nx = self.mesh_metadata["nx"]
        ny = self.mesh_metadata["ny"]
        dx = self.mesh_metadata["dx"]
        dy = self.mesh_metadata["dy"]

        lons = np.linspace(origin[0], origin[0] + nx * dx, nx + 1)
        lats = np.linspace(origin[1], origin[1] + ny * dy, ny + 1)

        return lons, lats

    def _load_mat_file(self, path: Path) -> np.ndarray:
        """
        Load a SWAN MATLAB output file.

        SWAN outputs MATLAB v4 format. The variable name in the file
        matches the output type (e.g., 'Hsig', 'Tps', 'Dir').

        Args:
            path: Path to .mat file

        Returns:
            2D numpy array
        """
        mat_data = loadmat(str(path))
        # Find the data variable (not metadata keys starting with __)
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if not data_keys:
            raise ValueError(f"No data found in {path}")
        return mat_data[data_keys[0]]

    def _read_partitions(self) -> List[WavePartitionGrid]:
        """
        Read wave partition output files if available.

        SWAN outputs partition data with variables named HsPT01-HsPT06 (up to 6 peaks),
        TpPT01-TpPT06, and DirPT01-DirPT06 within each .mat file. Each peak represents
        a different spectral partition (wind sea, primary swell, secondary swell, etc.).

        We read from phs0.mat, ptp0.mat, pdir0.mat and extract HsPT01-04, etc. for
        the first 4 partitions (wind sea + 3 swells).

        Returns:
            List of WavePartitionGrid objects for available partitions
        """
        partitions = []

        # SWAN stores all partitions in a single file with variables HsPT01-HsPT06
        hs_path = self.run_dir / "phs0.mat"
        tp_path = self.run_dir / "ptp0.mat"
        dir_path = self.run_dir / "pdir0.mat"

        if not all(p.exists() for p in [hs_path, tp_path, dir_path]):
            return partitions

        try:
            hs_data = loadmat(str(hs_path))
            tp_data = loadmat(str(tp_path))
            dir_data = loadmat(str(dir_path))

            # SWAN uses HsPT01, HsPT02, ... TpPT01, TpPT02, ... DrPT01, ...
            for part_id in range(MAX_PARTITIONS):
                # Variable names use 1-based indexing with 2-digit padding
                var_idx = f"{part_id + 1:02d}"
                hs_var = f"HsPT{var_idx}"
                tp_var = f"TpPT{var_idx}"
                dir_var = f"DrPT{var_idx}"  # Note: "Dr" not "Dir"

                # Check if this partition exists
                if hs_var not in hs_data or tp_var not in tp_data or dir_var not in dir_data:
                    continue

                hs = hs_data[hs_var]
                tp = tp_data[tp_var]
                wave_dir = dir_data[dir_var]

                # Check if partition has any valid data
                valid_hs = hs[(~np.isnan(hs)) & (hs > 0)]
                if len(valid_hs) == 0:
                    logger.debug(f"Partition {part_id} has no valid data, skipping")
                    continue

                label = PARTITION_LABELS.get(part_id, f"Partition {part_id}")

                partition = WavePartitionGrid(
                    hs=hs,
                    tp=tp,
                    dir=wave_dir,
                    partition_id=part_id,
                    label=label
                )
                partitions.append(partition)
                logger.debug(f"Loaded partition {part_id}: {label} (max Hs={np.nanmax(valid_hs):.2f}m)")

        except Exception as e:
            logger.warning(f"Failed to load partition data: {e}")

        if partitions:
            logger.info(f"Loaded {len(partitions)} wave partitions")

        return partitions

    def read(self) -> SwanOutput:
        """
        Read all SWAN output files.

        Returns:
            SwanOutput object containing all data
        """
        hsig_path = self.run_dir / "hsig.mat"
        tps_path = self.run_dir / "tps.mat"
        dir_path = self.run_dir / "dir.mat"

        # Check files exist
        for path in [hsig_path, tps_path, dir_path]:
            if not path.exists():
                raise FileNotFoundError(f"Output file not found: {path}")

        # Load MATLAB format data
        hsig = self._load_mat_file(hsig_path)
        tps = self._load_mat_file(tps_path)
        dir_data = self._load_mat_file(dir_path)

        # Build coordinates
        lons, lats = self._build_coordinates()

        # Read partition data if available
        partitions = self._read_partitions()

        logger.info(f"Loaded SWAN output: {hsig.shape}")

        return SwanOutput(
            hsig=hsig,
            tps=tps,
            dir=dir_data,
            lons=lons,
            lats=lats,
            mesh_name=self.mesh_metadata.get("name", "unknown"),
            region_name=self.mesh_metadata.get("region_name", "unknown"),
            exception_value=self.mesh_metadata.get("exception_value", -99.0),
            partitions=partitions
        )

    def read_single(self, variable: str) -> np.ndarray:
        """
        Read a single output variable.

        Args:
            variable: Variable name ("hsig", "tps", or "dir")

        Returns:
            2D numpy array
        """
        path = self.run_dir / f"{variable}.mat"
        if not path.exists():
            raise FileNotFoundError(f"Output file not found: {path}")
        return self._load_mat_file(path)


def read_swan_output(run_dir: str | Path) -> SwanOutput:
    """
    Convenience function to read SWAN output.

    Args:
        run_dir: Path to run directory

    Returns:
        SwanOutput object
    """
    reader = SwanOutputReader(run_dir)
    return reader.read()