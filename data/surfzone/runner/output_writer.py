"""
Output Writer for Surfzone Model

Aggregates ray tracing results into a continuous breaking field
and saves to disk.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ray_tracer import RayResult
from .wave_physics import BREAKER_TYPE_LABELS

logger = logging.getLogger(__name__)


@dataclass
class BreakingField:
    """
    Aggregated breaking wave field.

    Contains all breaking locations, heights, and types from ray tracing.

    Attributes:
        break_x: UTM x coordinates of breaking locations (m)
        break_y: UTM y coordinates of breaking locations (m)
        break_lon: Longitude of breaking locations
        break_lat: Latitude of breaking locations
        break_height: Wave height at breaking (m)
        break_depth: Water depth at breaking (m)
        break_period: Wave period at breaking (s)
        breaker_type: Breaker type code (0=spilling, 1=plunging, 2=collapsing, 3=surging)
        iribarren: Iribarren number at breaking
        beach_slope: Local beach slope at breaking
        partition_id: Source wave partition ID
        n_points: Number of breaking points
    """
    break_x: np.ndarray
    break_y: np.ndarray
    break_lon: np.ndarray
    break_lat: np.ndarray
    break_height: np.ndarray
    break_depth: np.ndarray
    break_period: np.ndarray
    breaker_type: np.ndarray
    iribarren: np.ndarray
    beach_slope: np.ndarray
    partition_id: np.ndarray

    @property
    def n_points(self) -> int:
        return len(self.break_x)

    def filter_by_type(self, breaker_type: int) -> 'BreakingField':
        """Return subset filtered by breaker type."""
        mask = self.breaker_type == breaker_type
        return BreakingField(
            break_x=self.break_x[mask],
            break_y=self.break_y[mask],
            break_lon=self.break_lon[mask],
            break_lat=self.break_lat[mask],
            break_height=self.break_height[mask],
            break_depth=self.break_depth[mask],
            break_period=self.break_period[mask],
            breaker_type=self.breaker_type[mask],
            iribarren=self.iribarren[mask],
            beach_slope=self.beach_slope[mask],
            partition_id=self.partition_id[mask],
        )

    def filter_by_partition(self, partition_id: int) -> 'BreakingField':
        """Return subset filtered by partition."""
        mask = self.partition_id == partition_id
        return BreakingField(
            break_x=self.break_x[mask],
            break_y=self.break_y[mask],
            break_lon=self.break_lon[mask],
            break_lat=self.break_lat[mask],
            break_height=self.break_height[mask],
            break_depth=self.break_depth[mask],
            break_period=self.break_period[mask],
            breaker_type=self.breaker_type[mask],
            iribarren=self.iribarren[mask],
            beach_slope=self.beach_slope[mask],
            partition_id=self.partition_id[mask],
        )

    def summary(self) -> str:
        """Return summary of breaking field."""
        if self.n_points == 0:
            return "BreakingField: 0 points"

        # Count by breaker type
        type_counts = {}
        for code, label in BREAKER_TYPE_LABELS.items():
            count = np.sum(self.breaker_type == code)
            if count > 0:
                type_counts[label] = count

        lines = [
            f"BreakingField: {self.n_points} breaking points",
            f"  Height: {self.break_height.min():.2f} - {self.break_height.max():.2f} m "
            f"(mean: {self.break_height.mean():.2f} m)",
            f"  Depth: {self.break_depth.min():.2f} - {self.break_depth.max():.2f} m",
            f"  Breaker types:",
        ]
        for label, count in type_counts.items():
            pct = 100 * count / self.n_points
            lines.append(f"    {label}: {count} ({pct:.1f}%)")

        return '\n'.join(lines)


class OutputWriter:
    """
    Writes surfzone model results to disk.

    Saves breaking field data as .npz for fast loading,
    with metadata in accompanying .json file.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize output writer.

        Args:
            output_dir: Directory to write output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def results_to_breaking_field(
        self,
        results: List[RayResult],
        mesh: 'SurfZoneMesh',
    ) -> BreakingField:
        """
        Convert ray tracing results to a BreakingField.

        Args:
            results: List of RayResult objects
            mesh: SurfZoneMesh for coordinate conversion

        Returns:
            BreakingField with all breaking locations
        """
        # Filter to only rays that broke
        breaking_results = [r for r in results if r.did_break]

        if not breaking_results:
            logger.warning("No waves broke in any rays")
            return BreakingField(
                break_x=np.array([]),
                break_y=np.array([]),
                break_lon=np.array([]),
                break_lat=np.array([]),
                break_height=np.array([]),
                break_depth=np.array([]),
                break_period=np.array([]),
                breaker_type=np.array([], dtype=np.int32),
                iribarren=np.array([]),
                beach_slope=np.array([]),
                partition_id=np.array([], dtype=np.int32),
            )

        n = len(breaking_results)

        break_x = np.array([r.break_x for r in breaking_results])
        break_y = np.array([r.break_y for r in breaking_results])
        break_height = np.array([r.break_height for r in breaking_results])
        break_depth = np.array([r.break_depth for r in breaking_results])
        break_period = np.array([r.break_period for r in breaking_results])
        breaker_type = np.array([r.breaker_type for r in breaking_results], dtype=np.int32)
        iribarren = np.array([r.iribarren for r in breaking_results])
        beach_slope = np.array([r.beach_slope for r in breaking_results])
        partition_id = np.array([r.partition_id for r in breaking_results], dtype=np.int32)

        # Convert UTM to lon/lat
        break_lon, break_lat = mesh.utm_to_lon_lat(break_x, break_y)

        logger.info(f"Created breaking field with {n} points")

        return BreakingField(
            break_x=break_x,
            break_y=break_y,
            break_lon=break_lon,
            break_lat=break_lat,
            break_height=break_height,
            break_depth=break_depth,
            break_period=break_period,
            breaker_type=breaker_type,
            iribarren=iribarren,
            beach_slope=beach_slope,
            partition_id=partition_id,
        )

    def save_breaking_field(
        self,
        field: BreakingField,
        filename: str = "breaking_field",
    ) -> Tuple[Path, Path]:
        """
        Save breaking field to disk.

        Args:
            field: BreakingField to save
            filename: Base filename (without extension)

        Returns:
            Tuple of (npz_path, json_path)
        """
        npz_path = self.output_dir / f"{filename}.npz"
        json_path = self.output_dir / f"{filename}.json"

        # Save data as compressed NPZ
        np.savez_compressed(
            npz_path,
            break_x=field.break_x,
            break_y=field.break_y,
            break_lon=field.break_lon,
            break_lat=field.break_lat,
            break_height=field.break_height,
            break_depth=field.break_depth,
            break_period=field.break_period,
            breaker_type=field.breaker_type,
            iribarren=field.iribarren,
            beach_slope=field.beach_slope,
            partition_id=field.partition_id,
        )

        logger.info(f"Saved: {npz_path}")

        # Save metadata as JSON
        metadata = {
            "n_points": field.n_points,
            "created": datetime.now().isoformat(),
            "statistics": self._compute_statistics(field),
        }

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved: {json_path}")

        return npz_path, json_path

    def _compute_statistics(self, field: BreakingField) -> Dict:
        """Compute statistics for metadata."""
        if field.n_points == 0:
            return {}

        # Count by breaker type
        type_counts = {}
        for code, label in BREAKER_TYPE_LABELS.items():
            count = int(np.sum(field.breaker_type == code))
            if count > 0:
                type_counts[label] = count

        # Count by partition
        partition_counts = {}
        for pid in np.unique(field.partition_id):
            count = int(np.sum(field.partition_id == pid))
            partition_counts[int(pid)] = count

        return {
            "height": {
                "min": float(field.break_height.min()),
                "max": float(field.break_height.max()),
                "mean": float(field.break_height.mean()),
                "std": float(field.break_height.std()),
            },
            "depth": {
                "min": float(field.break_depth.min()),
                "max": float(field.break_depth.max()),
                "mean": float(field.break_depth.mean()),
            },
            "iribarren": {
                "min": float(field.iribarren.min()),
                "max": float(field.iribarren.max()),
                "mean": float(field.iribarren.mean()),
            },
            "breaker_types": type_counts,
            "partitions": partition_counts,
            "bounds": {
                "x_min": float(field.break_x.min()),
                "x_max": float(field.break_x.max()),
                "y_min": float(field.break_y.min()),
                "y_max": float(field.break_y.max()),
                "lon_min": float(field.break_lon.min()),
                "lon_max": float(field.break_lon.max()),
                "lat_min": float(field.break_lat.min()),
                "lat_max": float(field.break_lat.max()),
            },
        }

    def save_run_metadata(
        self,
        config: Dict,
        timing: Dict,
        results_summary: str,
    ) -> Path:
        """
        Save run metadata.

        Args:
            config: Run configuration
            timing: Timing information
            results_summary: Summary string from ray tracer

        Returns:
            Path to metadata file
        """
        json_path = self.output_dir / "run_metadata.json"

        metadata = {
            "config": config,
            "timing": timing,
            "summary": results_summary,
            "created": datetime.now().isoformat(),
        }

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved: {json_path}")

        return json_path


def load_breaking_field(npz_path: Path) -> BreakingField:
    """
    Load a breaking field from disk.

    Args:
        npz_path: Path to .npz file

    Returns:
        BreakingField object
    """
    data = np.load(npz_path)

    return BreakingField(
        break_x=data['break_x'],
        break_y=data['break_y'],
        break_lon=data['break_lon'],
        break_lat=data['break_lat'],
        break_height=data['break_height'],
        break_depth=data['break_depth'],
        break_period=data['break_period'],
        breaker_type=data['breaker_type'],
        iribarren=data['iribarren'],
        beach_slope=data['beach_slope'],
        partition_id=data['partition_id'],
    )


# =============================================================================
# Surfzone Simulation Result Save/Load
# =============================================================================

def save_surfzone_result(
    result: 'SurfzoneSimulationResult',
    output_path: Path,
    filename: str = "surfzone_result",
) -> Tuple[Path, Path]:
    """
    Save surfzone simulation result to disk.

    Saves two files:
    - {filename}.npz: Compressed arrays (mesh coords, wave heights, etc.)
    - {filename}.json: Metadata and summary statistics

    Args:
        result: SurfzoneSimulationResult to save
        output_path: Directory to save to
        filename: Base filename (without extension)

    Returns:
        Tuple of (npz_path, json_path)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    npz_path = output_path / f"{filename}.npz"
    json_path = output_path / f"{filename}.json"

    # Save arrays as compressed NPZ
    np.savez_compressed(
        npz_path,
        mesh_x=result.mesh_x,
        mesh_y=result.mesh_y,
        mesh_depth=result.mesh_depth,
        sampled=result.sampled,
        H_at_mesh=result.H_at_mesh,
        converged=result.converged,
        direction_at_mesh=result.direction_at_mesh,
        K_shoaling=result.K_shoaling,
        boundary_Hs=result.boundary_Hs,
        boundary_Tp=result.boundary_Tp,
        boundary_direction=result.boundary_direction,
    )

    logger.info(f"Saved arrays: {npz_path}")

    # Compute statistics for converged points
    converged_mask = result.converged
    stats = {}

    if result.n_converged > 0:
        H_conv = result.H_at_mesh[converged_mask]
        K_conv = result.K_shoaling[converged_mask]
        depth_conv = result.mesh_depth[converged_mask]

        stats = {
            "wave_height": {
                "min": float(np.nanmin(H_conv)),
                "max": float(np.nanmax(H_conv)),
                "mean": float(np.nanmean(H_conv)),
                "std": float(np.nanstd(H_conv)),
            },
            "shoaling_coefficient": {
                "min": float(np.nanmin(K_conv)),
                "max": float(np.nanmax(K_conv)),
                "mean": float(np.nanmean(K_conv)),
            },
            "depth": {
                "min": float(np.nanmin(depth_conv)),
                "max": float(np.nanmax(depth_conv)),
                "mean": float(np.nanmean(depth_conv)),
            },
        }

    # Save metadata as JSON
    metadata = {
        "region_name": result.region_name,
        "timestamp": result.timestamp,
        "depth_range": list(result.depth_range),
        "partition_id": result.partition_id,
        "n_points": result.n_points,
        "n_sampled": result.n_sampled,
        "n_converged": result.n_converged,
        "sample_rate": result.sample_rate,
        "convergence_rate": result.convergence_rate,
        "statistics": stats,
        "bounds": {
            "x_min": float(result.mesh_x.min()) if len(result.mesh_x) > 0 else None,
            "x_max": float(result.mesh_x.max()) if len(result.mesh_x) > 0 else None,
            "y_min": float(result.mesh_y.min()) if len(result.mesh_y) > 0 else None,
            "y_max": float(result.mesh_y.max()) if len(result.mesh_y) > 0 else None,
        },
    }

    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata: {json_path}")

    return npz_path, json_path


def load_surfzone_result(npz_path: Path) -> 'SurfzoneSimulationResult':
    """
    Load surfzone simulation result from disk.

    Args:
        npz_path: Path to .npz file

    Returns:
        SurfzoneSimulationResult object

    Note:
        The point_results list will be empty when loading from disk,
        as only the aggregate arrays are stored.
    """
    from .surfzone_result import SurfzoneSimulationResult

    npz_path = Path(npz_path)
    json_path = npz_path.with_suffix('.json')

    # Load arrays
    data = np.load(npz_path)

    mesh_x = data['mesh_x']
    mesh_y = data['mesh_y']
    mesh_depth = data['mesh_depth']
    H_at_mesh = data['H_at_mesh']
    converged = data['converged']
    direction_at_mesh = data['direction_at_mesh']
    K_shoaling = data['K_shoaling']
    boundary_Hs = data['boundary_Hs']
    boundary_Tp = data['boundary_Tp']
    boundary_direction = data['boundary_direction']

    # Load sampled array (backwards compatible - default to all True for old files)
    if 'sampled' in data:
        sampled = data['sampled']
    else:
        sampled = np.ones(len(mesh_x), dtype=bool)

    # Load metadata
    if json_path.exists():
        with open(json_path) as f:
            metadata = json.load(f)
        region_name = metadata.get('region_name', 'Unknown')
        timestamp = metadata.get('timestamp', datetime.now().isoformat())
        depth_range = tuple(metadata.get('depth_range', [0.0, 10.0]))
        partition_id = metadata.get('partition_id', 1)
    else:
        region_name = 'Unknown'
        timestamp = datetime.now().isoformat()
        depth_range = (0.0, 10.0)
        partition_id = 1

    n_points = len(mesh_x)
    n_sampled = int(np.sum(sampled))
    n_converged = int(np.sum(converged))

    return SurfzoneSimulationResult(
        region_name=region_name,
        timestamp=timestamp,
        depth_range=depth_range,
        partition_id=partition_id,
        n_points=n_points,
        n_sampled=n_sampled,
        n_converged=n_converged,
        point_results=[],  # Not stored in file
        mesh_x=mesh_x,
        mesh_y=mesh_y,
        mesh_depth=mesh_depth,
        sampled=sampled,
        H_at_mesh=H_at_mesh,
        converged=converged,
        direction_at_mesh=direction_at_mesh,
        K_shoaling=K_shoaling,
        boundary_Hs=boundary_Hs,
        boundary_Tp=boundary_Tp,
        boundary_direction=boundary_direction,
    )