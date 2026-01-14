"""
SWAN Output Reader

Loads and processes SWAN model output files.
SWAN outputs MATLAB v4 format files which require scipy.io.loadmat().
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.io import loadmat

logger = logging.getLogger(__name__)


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
    """
    hsig: np.ndarray
    tps: np.ndarray
    dir: np.ndarray
    lons: np.ndarray
    lats: np.ndarray
    mesh_name: str
    region_name: str
    exception_value: float = -99.0

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
                project_root = self.run_dir.parent.parent.parent.parent.parent
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

        logger.info(f"Loaded SWAN output: {hsig.shape}")

        return SwanOutput(
            hsig=hsig,
            tps=tps,
            dir=dir_data,
            lons=lons,
            lats=lats,
            mesh_name=self.mesh_metadata.get("name", "unknown"),
            region_name=self.mesh_metadata.get("region_name", "unknown"),
            exception_value=self.mesh_metadata.get("exception_value", -99.0)
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