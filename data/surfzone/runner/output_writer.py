"""
Output Writer for Surfzone Model

Saves forward ray tracing results to disk.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .wave_physics import BREAKER_TYPE_LABELS

logger = logging.getLogger(__name__)


# =============================================================================
# Forward Tracing Result Save/Load
# =============================================================================

def save_forward_result(
    result: 'ForwardTracingResult',
    output_path: Path,
    filename: str = "forward_result",
) -> Tuple[Path, Path]:
    """
    Save forward ray tracing result to disk.

    Saves:
    - {filename}.npz: Combined arrays (mesh coords, energy, wave heights)
    - {filename}.json: Metadata and summary statistics
    - Per-partition files: wind_sea.npz, primary_swell.npz, etc. (if partitions available)

    Args:
        result: ForwardTracingResult to save
        output_path: Directory to save to
        filename: Base filename (without extension)

    Returns:
        Tuple of (npz_path, json_path)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    npz_path = output_path / f"{filename}.npz"
    json_path = output_path / f"{filename}.json"

    # Save combined arrays as compressed NPZ
    save_dict = dict(
        mesh_x=result.mesh_x,
        mesh_y=result.mesh_y,
        mesh_depth=result.mesh_depth,
        H_at_mesh=result.H_at_mesh,
        energy=result.energy,
        ray_count=result.ray_count,
    )

    # Include breaking characterization if available
    if result.breaking_fields is not None:
        for key, arr in result.breaking_fields.items():
            save_dict[f'breaking_{key}'] = arr

    np.savez_compressed(npz_path, **save_dict)

    logger.info(f"Saved combined arrays: {npz_path}")

    # Save per-partition files (compatible with statistics runner)
    if result.partitions:
        for partition in result.partitions:
            partition_path = output_path / f"{partition.partition_name}.npz"
            np.savez_compressed(
                partition_path,
                mesh_x=result.mesh_x,
                mesh_y=result.mesh_y,
                mesh_depth=result.mesh_depth,
                # Field names match what statistics/run_statistics.py expects
                boundary_Hs=partition.H_at_mesh,         # Propagated Hs (shoaled + focused)
                boundary_Tp=partition.boundary_Tp,       # Period from SWAN (conserved)
                boundary_direction=partition.direction,  # Refracted direction (from ray tracing)
                converged=partition.converged,
            )
            logger.info(f"Saved partition: {partition_path}")

    # Compute statistics for covered points
    covered_mask = result.ray_count > 0
    stats = {}

    if result.n_covered > 0:
        H_cov = result.H_at_mesh[covered_mask]
        energy_cov = result.energy[covered_mask]
        ray_cov = result.ray_count[covered_mask]
        depth_cov = result.mesh_depth[covered_mask]

        stats = {
            "wave_height": {
                "min": float(np.nanmin(H_cov)),
                "max": float(np.nanmax(H_cov)),
                "mean": float(np.nanmean(H_cov)),
                "std": float(np.nanstd(H_cov)),
            },
            "energy": {
                "min": float(np.nanmin(energy_cov)),
                "max": float(np.nanmax(energy_cov)),
                "mean": float(np.nanmean(energy_cov)),
                "total": float(np.nansum(energy_cov)),
            },
            "rays_per_point": {
                "min": int(ray_cov.min()),
                "max": int(ray_cov.max()),
                "mean": float(ray_cov.mean()),
            },
            "depth": {
                "min": float(np.nanmin(depth_cov)),
                "max": float(np.nanmax(depth_cov)),
                "mean": float(np.nanmean(depth_cov)),
            },
        }

    # Per-partition statistics
    partition_stats = {}
    if result.partitions:
        for partition in result.partitions:
            if partition.n_covered > 0:
                H_cov = partition.H_at_mesh[partition.converged]
                partition_stats[partition.partition_name] = {
                    "n_covered": partition.n_covered,
                    "Hs_min": float(np.nanmin(H_cov)),
                    "Hs_max": float(np.nanmax(H_cov)),
                    "Hs_mean": float(np.nanmean(H_cov)),
                }

    # Save metadata as JSON
    metadata = {
        "region_name": result.region_name,
        "timestamp": result.timestamp,
        "n_partitions": result.n_partitions,
        "n_points": result.n_points,
        "n_covered": result.n_covered,
        "n_rays_total": result.n_rays_total,
        "coverage_rate": result.coverage_rate,
        "statistics": stats,
        "partition_statistics": partition_stats,
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

    # Save ray paths if available
    if result.ray_paths is not None:
        ray_paths_path = save_ray_paths(result.ray_paths, output_path)
        logger.info(f"Saved ray paths: {ray_paths_path}")

    return npz_path, json_path


def save_ray_paths(
    ray_paths: 'RayPathData',
    output_path: Path,
    filename: str = "ray_paths",
) -> Path:
    """
    Save ray path data to disk.

    Args:
        ray_paths: RayPathData to save
        output_path: Directory to save to
        filename: Base filename (without extension)

    Returns:
        Path to saved .npz file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    npz_path = output_path / f"{filename}.npz"

    np.savez_compressed(
        npz_path,
        # Per-ray metadata
        ray_partition=ray_paths.ray_partition.astype(np.int32),
        ray_start_idx=ray_paths.ray_start_idx.astype(np.int64),
        ray_length=ray_paths.ray_length.astype(np.int32),
        ray_original_idx=ray_paths.ray_original_idx.astype(np.int32),
        # Path data
        path_x=ray_paths.path_x.astype(np.float32),
        path_y=ray_paths.path_y.astype(np.float32),
        path_depth=ray_paths.path_depth.astype(np.float32),
        path_direction=ray_paths.path_direction.astype(np.float32),
        path_tube_width=ray_paths.path_tube_width.astype(np.float32),
        path_Hs_local=ray_paths.path_Hs_local.astype(np.float32),
        # Metadata
        n_rays_total=np.array(ray_paths.n_rays_total),
        n_rays_sampled=np.array(ray_paths.n_rays_sampled),
        sample_fraction=np.array(ray_paths.sample_fraction),
        total_steps=np.array(ray_paths.total_steps),
    )

    return npz_path


def load_ray_paths(npz_path: Path) -> 'RayPathData':
    """
    Load ray path data from disk.

    Args:
        npz_path: Path to ray_paths.npz file

    Returns:
        RayPathData object
    """
    from .surfzone_result import RayPathData

    data = np.load(npz_path)

    return RayPathData(
        ray_partition=data['ray_partition'],
        ray_start_idx=data['ray_start_idx'],
        ray_length=data['ray_length'],
        ray_original_idx=data['ray_original_idx'],
        path_x=data['path_x'],
        path_y=data['path_y'],
        path_depth=data['path_depth'],
        path_direction=data['path_direction'],
        path_tube_width=data['path_tube_width'],
        path_Hs_local=data['path_Hs_local'],
        n_rays_total=int(data['n_rays_total']),
        sample_fraction=float(data['sample_fraction']),
    )


def load_forward_result(npz_path: Path) -> 'ForwardTracingResult':
    """
    Load forward tracing result from disk.

    Args:
        npz_path: Path to .npz file

    Returns:
        ForwardTracingResult object
    """
    from .surfzone_result import ForwardTracingResult

    npz_path = Path(npz_path)
    json_path = npz_path.with_suffix('.json')

    # Load arrays
    data = np.load(npz_path)

    mesh_x = data['mesh_x']
    mesh_y = data['mesh_y']
    mesh_depth = data['mesh_depth']
    H_at_mesh = data['H_at_mesh']
    energy = data['energy']
    ray_count = data['ray_count']

    # Load metadata
    if json_path.exists():
        with open(json_path) as f:
            metadata = json.load(f)
        region_name = metadata.get('region_name', 'Unknown')
        timestamp = metadata.get('timestamp', datetime.now().isoformat())
        n_partitions = metadata.get('n_partitions', 4)
        n_rays_total = metadata.get('n_rays_total', 0)
    else:
        region_name = 'Unknown'
        timestamp = datetime.now().isoformat()
        n_partitions = 4
        n_rays_total = 0

    n_points = len(mesh_x)
    n_covered = int(np.sum(ray_count > 0))

    # Load breaking characterization if available
    breaking_fields = None
    breaking_keys = ['is_breaking', 'breaker_index', 'iribarren', 'breaker_type', 'breaking_intensity']
    if f'breaking_{breaking_keys[0]}' in data:
        breaking_fields = {k: data[f'breaking_{k}'] for k in breaking_keys if f'breaking_{k}' in data}

    return ForwardTracingResult(
        region_name=region_name,
        timestamp=timestamp,
        n_partitions=n_partitions,
        n_points=n_points,
        n_covered=n_covered,
        n_rays_total=n_rays_total,
        mesh_x=mesh_x,
        mesh_y=mesh_y,
        mesh_depth=mesh_depth,
        H_at_mesh=H_at_mesh,
        energy=energy,
        ray_count=ray_count,
        breaking_fields=breaking_fields,
    )


# =============================================================================
# Breaking Field (for future breaking wave analysis)
# =============================================================================

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
