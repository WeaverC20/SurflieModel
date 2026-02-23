"""
Statistics Runner

Applies registered statistics to all mesh points using vectorized operations.
Provides StatisticsResult for saving/loading results.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from data.surfzone.runner.swan_input_provider import BoundaryConditions, WavePartition
from .base import StatisticFunction
from .registry import StatisticsRegistry

logger = logging.getLogger(__name__)


@dataclass
class StatisticsResult:
    """
    Full result of wave statistics computation for all mesh points.

    Attributes:
        region: Region identifier (e.g., 'socal')
        timestamp: When the computation was performed
        num_points: Number of mesh points
        df: DataFrame with all statistics
        metadata: Description of each statistic (units, etc.)
    """
    region: str
    timestamp: str
    num_points: int
    df: pd.DataFrame
    metadata: Dict

    def save(self, output_dir: Union[str, Path]):
        """
        Save to CSV + metadata JSON.

        Creates:
        - statistics_{timestamp}.csv
        - statistics_latest.csv (copy for easy access)
        - statistics_{timestamp}_meta.json
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Data file with timestamp - use CSV for compatibility
        data_path = output_dir / f"statistics_{self.timestamp}.csv"
        self.df.to_csv(data_path, index=False)

        # Also save as "latest" for easy access
        latest_path = output_dir / "statistics_latest.csv"
        self.df.to_csv(latest_path, index=False)

        # Metadata
        meta_path = output_dir / f"statistics_{self.timestamp}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Also update latest metadata
        latest_meta_path = output_dir / "statistics_latest_meta.json"
        with open(latest_meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved {self.num_points} points to {data_path}")
        print(f"Saved statistics for {self.num_points} points to {output_dir}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'StatisticsResult':
        """
        Load from CSV file.

        Args:
            path: Path to CSV file or directory containing statistics_latest.csv
        """
        path = Path(path)

        # Handle directory input
        if path.is_dir():
            path = path / "statistics_latest.csv"

        df = pd.read_csv(path)

        # Try to load metadata
        meta_path = path.with_name(path.stem + "_meta.json")
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        return cls(
            region=metadata.get('region', 'unknown'),
            timestamp=metadata.get('timestamp', ''),
            num_points=len(df),
            df=df,
            metadata=metadata
        )

    def describe(self) -> str:
        """Get a summary of the statistics."""
        lines = [
            f"Statistics Result: {self.region}",
            f"  Timestamp: {self.timestamp}",
            f"  Points: {self.num_points}",
            f"  Statistics computed:"
        ]
        for stat_info in self.metadata.get('statistics', []):
            name = stat_info.get('name', '?')
            units = stat_info.get('units', '')
            desc = stat_info.get('description', '')
            lines.append(f"    - {name} ({units}): {desc}")
        return '\n'.join(lines)


class StatisticsRunner:
    """
    Applies wave statistics to all mesh points using vectorized operations.

    Usage:
        runner = StatisticsRunner()  # All registered statistics
        result = runner.run_from_boundary(boundary, depths, region="socal")
        result.save("data/surfzone/output/socal/")

        # Or select specific statistics
        runner = StatisticsRunner(statistics=["set_period", "waves_per_set"])
    """

    def __init__(self, statistics: Optional[List[str]] = None):
        """
        Initialize the runner.

        Args:
            statistics: List of statistic names to compute.
                       If None, compute all registered statistics.
        """
        if statistics is None:
            self.statistics = StatisticsRegistry.all()
        else:
            self.statistics = [StatisticsRegistry.get(s) for s in statistics]

        if not self.statistics:
            raise ValueError("No statistics registered or specified")

    def run_from_boundary(
        self,
        boundary: BoundaryConditions,
        depths: np.ndarray,
        region: str = "unknown",
        context: Optional[Dict] = None,
    ) -> StatisticsResult:
        """
        Compute all statistics from SWAN boundary conditions.

        Uses vectorized operations - no per-point loops for efficiency.

        Args:
            boundary: BoundaryConditions from SwanInputProvider
            depths: Water depth at each point, shape (n_points,)
            region: Region identifier for metadata

        Returns:
            StatisticsResult with all computed statistics
        """
        n_points = boundary.n_points
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"Computing {len(self.statistics)} statistics for {n_points} points")

        # Base columns
        data = {
            'point_id': np.arange(n_points),
            'lat': boundary.lat,
            'lon': boundary.lon,
            'depth': depths
        }

        # Compute each statistic
        extra_kwargs = context or {}
        metadata_stats = []
        for stat in self.statistics:
            logger.debug(f"Computing {stat.name}...")

            output = stat.compute_vectorized(
                partitions=boundary.partitions,
                depths=depths,
                lats=boundary.lat,
                lons=boundary.lon,
                **extra_kwargs,
            )

            # Handle multi-column output
            if output.values.ndim == 1:
                data[stat.name] = output.values
            else:
                # Multi-column: use extra['columns'] if provided
                cols = (output.extra.get('columns', stat.output_columns)
                        if output.extra else stat.output_columns)
                for i, col in enumerate(cols):
                    if i < output.values.shape[1]:
                        data[col] = output.values[:, i]

            metadata_stats.append({
                'name': stat.name,
                'units': stat.units,
                'description': stat.description,
                'columns': stat.output_columns
            })

        df = pd.DataFrame(data)

        return StatisticsResult(
            region=region,
            timestamp=timestamp,
            num_points=n_points,
            df=df,
            metadata={
                'region': region,
                'timestamp': timestamp,
                'num_points': n_points,
                'statistics': metadata_stats
            }
        )

    def run_from_partitions(
        self,
        partitions: List[WavePartition],
        lats: np.ndarray,
        lons: np.ndarray,
        depths: np.ndarray,
        region: str = "unknown",
        context: Optional[Dict] = None,
    ) -> StatisticsResult:
        """
        Compute statistics from raw WavePartition arrays.

        Alternative to run_from_boundary when you have partitions directly.

        Args:
            partitions: List of WavePartition objects
            lats: Latitude at each point
            lons: Longitude at each point
            depths: Water depth at each point
            region: Region identifier

        Returns:
            StatisticsResult
        """
        n_points = len(depths)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        data = {
            'point_id': np.arange(n_points),
            'lat': lats,
            'lon': lons,
            'depth': depths
        }

        extra_kwargs = context or {}
        metadata_stats = []
        for stat in self.statistics:
            output = stat.compute_vectorized(
                partitions=partitions,
                depths=depths,
                lats=lats,
                lons=lons,
                **extra_kwargs,
            )

            if output.values.ndim == 1:
                data[stat.name] = output.values
            else:
                cols = (output.extra.get('columns', stat.output_columns)
                        if output.extra else stat.output_columns)
                for i, col in enumerate(cols):
                    if i < output.values.shape[1]:
                        data[col] = output.values[:, i]

            metadata_stats.append({
                'name': stat.name,
                'units': stat.units,
                'description': stat.description,
                'columns': stat.output_columns
            })

        df = pd.DataFrame(data)

        return StatisticsResult(
            region=region,
            timestamp=timestamp,
            num_points=n_points,
            df=df,
            metadata={
                'region': region,
                'timestamp': timestamp,
                'num_points': n_points,
                'statistics': metadata_stats
            }
        )
