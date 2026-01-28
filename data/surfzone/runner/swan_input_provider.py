"""
SWAN Input Provider for Surfzone Model

Reads SWAN partition outputs and interpolates to surfzone boundary points.
The SWAN grid is coarser than the surfzone mesh, so we use bilinear interpolation
to sample wave conditions at the 2.5km offshore boundary.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

logger = logging.getLogger(__name__)

# Maximum partitions (0=wind sea, 1-3=swells)
MAX_PARTITIONS = 4

# SWAN exception value for land/invalid points
EXCEPTION_VALUE = -99.0


@dataclass
class WavePartition:
    """
    A single wave partition at a point or array of points.

    Attributes:
        hs: Significant wave height (m)
        tp: Peak period (s)
        direction: Wave direction (degrees, nautical - FROM)
        partition_id: 0=wind sea, 1+=swells
        is_valid: Boolean mask for valid data
    """
    hs: np.ndarray
    tp: np.ndarray
    direction: np.ndarray
    partition_id: int
    is_valid: np.ndarray

    @property
    def label(self) -> str:
        labels = {0: "Wind Sea", 1: "Primary Swell", 2: "Secondary Swell", 3: "Tertiary Swell"}
        return labels.get(self.partition_id, f"Partition {self.partition_id}")


@dataclass
class BoundaryConditions:
    """
    Wave boundary conditions at surfzone boundary points.

    Attributes:
        x: UTM x coordinates of boundary points (m)
        y: UTM y coordinates of boundary points (m)
        lon: Longitude of boundary points
        lat: Latitude of boundary points
        partitions: List of WavePartition objects (1-4 partitions)
        n_points: Number of boundary points
    """
    x: np.ndarray
    y: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    partitions: List[WavePartition]

    @property
    def n_points(self) -> int:
        return len(self.x)

    @property
    def n_partitions(self) -> int:
        return len(self.partitions)

    def get_valid_partitions_at(self, idx: int) -> List[WavePartition]:
        """Get partitions with valid data at a specific point index."""
        return [p for p in self.partitions if p.is_valid[idx]]

    def summary(self) -> str:
        lines = [
            f"Boundary Conditions: {self.n_points} points, {self.n_partitions} partitions",
        ]
        for p in self.partitions:
            n_valid = np.sum(p.is_valid)
            if n_valid > 0:
                valid_hs = p.hs[p.is_valid]
                lines.append(
                    f"  {p.label}: {n_valid} valid points, "
                    f"Hs={valid_hs.min():.2f}-{valid_hs.max():.2f}m"
                )
        return '\n'.join(lines)


class SwanInputProvider:
    """
    Extracts wave partition data from SWAN output and interpolates to boundary points.

    Example usage:
        provider = SwanInputProvider("data/swan/runs/socal/coarse/latest")
        boundary_lons = np.array([...])  # Points at 2.5km offshore
        boundary_lats = np.array([...])
        conditions = provider.get_boundary_conditions(boundary_lons, boundary_lats)
    """

    def __init__(self, run_dir: str | Path):
        """
        Initialize provider for a SWAN run directory.

        Args:
            run_dir: Path to SWAN run directory containing output .mat files
        """
        self.run_dir = Path(run_dir)

        if not self.run_dir.exists():
            raise FileNotFoundError(f"SWAN run directory not found: {self.run_dir}")

        # Load mesh metadata to get grid coordinates
        self.mesh_metadata = self._load_mesh_metadata()

        # Build coordinate arrays
        self.lons, self.lats = self._build_coordinates()

        # Load partition data and build interpolators (lazy)
        self._interpolators: Optional[Dict[str, RegularGridInterpolator]] = None
        self._partition_data: Optional[Dict[int, Dict[str, np.ndarray]]] = None

    def _load_mesh_metadata(self) -> Dict:
        """Load mesh metadata from INPUT file or mesh directory."""
        input_file = self.run_dir / "INPUT"
        if input_file.exists():
            with open(input_file) as f:
                content = f.read()

            mesh_name = None
            region_name = None
            for line in content.split('\n'):
                if line.startswith('$ Mesh:'):
                    mesh_name = line.split(':')[1].strip()
                elif line.startswith('$ Region:'):
                    region_name = line.split(':')[1].strip()

            if mesh_name and region_name:
                # Find mesh JSON
                project_root = self.run_dir.parent.parent.parent.parent.parent.parent
                mesh_parts = mesh_name.split('_')
                if len(mesh_parts) >= 2:
                    mesh_type = mesh_parts[-1]
                    mesh_dir = project_root / "data" / "meshes" / region_name / mesh_type
                    json_path = mesh_dir / f"{mesh_name}.json"
                    if json_path.exists():
                        with open(json_path) as f:
                            return json.load(f)

        # Fallback: try to infer from directory structure
        # run_dir format: .../runs/{region}/{resolution}/latest
        try:
            resolution = self.run_dir.parent.name
            region = self.run_dir.parent.parent.name
            project_root = self.run_dir.parent.parent.parent.parent.parent
            mesh_name = f"{region}_{resolution}"
            json_path = project_root / "data" / "meshes" / region / resolution / f"{mesh_name}.json"
            if json_path.exists():
                with open(json_path) as f:
                    return json.load(f)
        except Exception:
            pass

        raise FileNotFoundError(
            f"Could not find mesh metadata for {self.run_dir}. "
            "Ensure SWAN INPUT file has mesh/region comments or mesh JSON exists."
        )

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

    def _load_partition_data(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Load all partition data from SWAN output files.

        Returns:
            Dict mapping partition_id to dict of {'hs': array, 'tp': array, 'dir': array}
        """
        hs_path = self.run_dir / "phs0.mat"
        tp_path = self.run_dir / "ptp0.mat"
        dir_path = self.run_dir / "pdir0.mat"

        if not all(p.exists() for p in [hs_path, tp_path, dir_path]):
            raise FileNotFoundError(
                f"Partition files not found in {self.run_dir}. "
                "Expected phs0.mat, ptp0.mat, pdir0.mat"
            )

        hs_data = loadmat(str(hs_path))
        tp_data = loadmat(str(tp_path))
        dir_data = loadmat(str(dir_path))

        partitions = {}

        for part_id in range(MAX_PARTITIONS):
            var_idx = f"{part_id + 1:02d}"
            hs_var = f"HsPT{var_idx}"
            tp_var = f"TpPT{var_idx}"
            dir_var = f"DrPT{var_idx}"

            if hs_var not in hs_data:
                continue

            hs = hs_data[hs_var].astype(np.float64)
            tp = tp_data[tp_var].astype(np.float64)
            wave_dir = dir_data[dir_var].astype(np.float64)

            # Mask invalid values
            invalid = (hs == EXCEPTION_VALUE) | (hs <= 0) | np.isnan(hs)
            if np.all(invalid):
                logger.debug(f"Partition {part_id} has no valid data, skipping")
                continue

            partitions[part_id] = {
                'hs': hs,
                'tp': tp,
                'dir': wave_dir,
            }

            valid_hs = hs[~invalid]
            logger.info(
                f"Loaded partition {part_id}: Hs={valid_hs.min():.2f}-{valid_hs.max():.2f}m, "
                f"{np.sum(~invalid)} valid points"
            )

        if not partitions:
            raise ValueError(f"No valid partition data found in {self.run_dir}")

        return partitions

    def _build_interpolators(self) -> Dict[str, RegularGridInterpolator]:
        """
        Build interpolators for all partition variables.

        Returns:
            Dict mapping "{part_id}_{var}" to RegularGridInterpolator
        """
        if self._partition_data is None:
            self._partition_data = self._load_partition_data()

        interpolators = {}

        for part_id, data in self._partition_data.items():
            for var_name, values in data.items():
                key = f"{part_id}_{var_name}"
                # SWAN output is (ny+1, nx+1), indexed as (lat, lon)
                interpolators[key] = RegularGridInterpolator(
                    (self.lats, self.lons),
                    values,
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )

        return interpolators

    def _ensure_interpolators(self) -> None:
        """Ensure interpolators are built."""
        if self._interpolators is None:
            self._interpolators = self._build_interpolators()

    def sample_at_points(
        self,
        lons: np.ndarray,
        lats: np.ndarray
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Sample all partition data at arbitrary lon/lat points.

        Args:
            lons: Longitude coordinates
            lats: Latitude coordinates

        Returns:
            Dict mapping partition_id to dict of {'hs': array, 'tp': array, 'dir': array}
        """
        self._ensure_interpolators()

        lons = np.atleast_1d(lons)
        lats = np.atleast_1d(lats)

        # Stack coordinates for interpolator (expects (lat, lon) order)
        points = np.stack([lats, lons], axis=-1)

        results = {}

        for part_id in self._partition_data.keys():
            hs = self._interpolators[f"{part_id}_hs"](points)
            tp = self._interpolators[f"{part_id}_tp"](points)
            direction = self._interpolators[f"{part_id}_dir"](points)

            results[part_id] = {
                'hs': hs,
                'tp': tp,
                'dir': direction,
            }

        return results

    def get_boundary_conditions(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        x_utm: Optional[np.ndarray] = None,
        y_utm: Optional[np.ndarray] = None,
    ) -> BoundaryConditions:
        """
        Get wave boundary conditions at specified points.

        Args:
            lons: Longitude coordinates of boundary points
            lats: Latitude coordinates of boundary points
            x_utm: Optional UTM x coordinates (if already computed)
            y_utm: Optional UTM y coordinates (if already computed)

        Returns:
            BoundaryConditions object with wave data at each point
        """
        lons = np.atleast_1d(lons).astype(np.float64)
        lats = np.atleast_1d(lats).astype(np.float64)

        if x_utm is None or y_utm is None:
            x_utm = np.zeros_like(lons)
            y_utm = np.zeros_like(lats)

        # Sample SWAN output at these points
        sampled = self.sample_at_points(lons, lats)

        # Convert to WavePartition objects
        partitions = []
        for part_id, data in sampled.items():
            hs = data['hs']
            tp = data['tp']
            direction = data['dir']

            # Determine valid points (valid in SWAN output and interpolated successfully)
            is_valid = ~np.isnan(hs) & (hs > 0) & ~np.isnan(tp) & (tp > 0)

            partitions.append(WavePartition(
                hs=hs,
                tp=tp,
                direction=direction,
                partition_id=part_id,
                is_valid=is_valid,
            ))

        return BoundaryConditions(
            x=x_utm,
            y=y_utm,
            lon=lons,
            lat=lats,
            partitions=partitions,
        )

    def get_boundary_from_mesh(
        self,
        mesh: 'SurfZoneMesh',
        offshore_distance_m: float = 2500.0,
    ) -> BoundaryConditions:
        """
        Get boundary conditions at the offshore edge of a surfzone mesh.

        Samples points at approximately the specified offshore distance from
        the coastline, using the mesh's iso-distance contours.

        Args:
            mesh: SurfZoneMesh object
            offshore_distance_m: Distance from shore for boundary (default 2500m)

        Returns:
            BoundaryConditions at the offshore boundary
        """
        # Find points near the offshore boundary
        # Use the mesh's depth field - points at ~offshore_distance should have
        # depths around 20-50m typically

        # For now, use a simple approach: sample points at regular intervals
        # along the offshore edge of the mesh bounds
        # TODO: Use actual iso-distance contour from mesh

        if mesh.coastlines is None or len(mesh.coastlines) == 0:
            raise ValueError("Mesh has no coastline data")

        # Get mesh bounds in UTM
        x_min, x_max = mesh.points_x.min(), mesh.points_x.max()
        y_min, y_max = mesh.points_y.min(), mesh.points_y.max()

        # Sample boundary points along the offshore edge
        # This is a simplified approach - ideally we'd use the actual
        # iso-distance contour at 2.5km from shore
        n_points_x = int((x_max - x_min) / 500)  # ~500m spacing
        n_points_y = int((y_max - y_min) / 500)

        boundary_x = []
        boundary_y = []

        # Sample along edges of mesh domain
        # West edge
        boundary_x.extend([x_min] * n_points_y)
        boundary_y.extend(np.linspace(y_min, y_max, n_points_y))

        # South edge
        boundary_x.extend(np.linspace(x_min, x_max, n_points_x))
        boundary_y.extend([y_min] * n_points_x)

        boundary_x = np.array(boundary_x)
        boundary_y = np.array(boundary_y)

        # Filter to points that are actually in water (depth > 0)
        depths = np.array([mesh.get_depth_at_point(x, y) for x, y in zip(boundary_x, boundary_y)])
        valid_water = ~np.isnan(depths) & (depths > 0)

        boundary_x = boundary_x[valid_water]
        boundary_y = boundary_y[valid_water]

        # Convert to lon/lat
        lons, lats = mesh.utm_to_lon_lat(boundary_x, boundary_y)

        return self.get_boundary_conditions(lons, lats, boundary_x, boundary_y)

    @property
    def grid_extent(self) -> Tuple[float, float, float, float]:
        """Get SWAN grid extent as (lon_min, lon_max, lat_min, lat_max)."""
        return (self.lons[0], self.lons[-1], self.lats[0], self.lats[-1])

    def summary(self) -> str:
        """Return summary of available SWAN data."""
        self._ensure_interpolators()

        lines = [
            f"SWAN Input Provider: {self.run_dir}",
            f"  Grid: {len(self.lons)} x {len(self.lats)} points",
            f"  Extent: lon [{self.lons[0]:.2f}, {self.lons[-1]:.2f}], "
            f"lat [{self.lats[0]:.2f}, {self.lats[-1]:.2f}]",
            f"  Partitions: {len(self._partition_data)}",
        ]

        labels = {0: "Wind Sea", 1: "Primary Swell", 2: "Secondary Swell", 3: "Tertiary Swell"}
        for part_id, data in self._partition_data.items():
            hs = data['hs']
            valid = ~np.isnan(hs) & (hs > 0) & (hs != EXCEPTION_VALUE)
            if np.any(valid):
                lines.append(
                    f"    {labels.get(part_id, f'P{part_id}')}: "
                    f"Hs={hs[valid].min():.2f}-{hs[valid].max():.2f}m"
                )

        return '\n'.join(lines)
