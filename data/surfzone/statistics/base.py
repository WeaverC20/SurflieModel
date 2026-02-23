"""
Base classes for pluggable wave statistics.

Statistics follow a protocol that supports both vectorized computation
(for efficiency) and single-point computation (for debugging/interactive use).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Import WavePartition from the existing SWAN provider
from data.surfzone.runner.swan_input_provider import WavePartition


@dataclass
class StatisticOutput:
    """
    Output of a statistic computation.

    Attributes:
        name: Statistic identifier
        values: Computed values, shape (n_points,) or (n_points, n_columns)
        units: Physical units (e.g., 's', 'm', 'ratio')
        description: Human-readable description
        extra: Additional metadata (e.g., column names for multi-column output)
    """
    name: str
    values: np.ndarray
    units: str
    description: str
    extra: Optional[Dict[str, Any]] = None


class StatisticFunction(ABC):
    """
    Base class for all pluggable wave statistics.

    Subclasses must implement:
        - name: Unique identifier
        - units: Physical units
        - compute_vectorized: Vectorized computation across all mesh points

    Optional overrides:
        - description: Human-readable description
        - output_columns: Column names for multi-column output
        - compute_single: Single-point computation (default uses vectorized)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this statistic."""
        pass

    @property
    @abstractmethod
    def units(self) -> str:
        """Physical units for the output (e.g., 's', 'm', 'ratio')."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of what this statistic measures."""
        return ""

    @property
    def output_columns(self) -> List[str]:
        """
        Column names for the output.

        Override this for multi-column statistics (e.g., per-partition values).
        Default returns [self.name] for single-column output.
        """
        return [self.name]

    @abstractmethod
    def compute_vectorized(
        self,
        partitions: List[WavePartition],
        depths: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        **kwargs,
    ) -> StatisticOutput:
        """
        Compute statistic for ALL mesh points at once (vectorized).

        This is the primary interface for efficient batch processing.
        Implementations should use numpy vectorization.

        Args:
            partitions: List of WavePartition objects (1-4 partitions).
                       Each partition has arrays of shape (n_points,) for
                       hs, tp, direction, and is_valid.
            depths: Water depth at each point, shape (n_points,)
            lats: Latitude of each point, shape (n_points,)
            lons: Longitude of each point, shape (n_points,)
            **kwargs: Additional context (e.g., slopes for breaking statistics)

        Returns:
            StatisticOutput with values of shape (n_points,) or (n_points, k)
        """
        pass

    def compute_single(
        self,
        partitions: List[Dict[str, float]],
        depth: float,
        lat: float,
        lon: float,
        **kwargs,
    ) -> Union[float, Dict[str, Any]]:
        """
        Compute statistic for a SINGLE point.

        Useful for debugging, interactive viewing, and testing.
        Default implementation wraps the vectorized method.

        Args:
            partitions: List of dicts with keys 'hs', 'tp', 'dir'
                       Example: [{'hs': 2.0, 'tp': 15, 'dir': 310}, ...]
            depth: Water depth at the point (m)
            lat: Latitude
            lon: Longitude
            **kwargs: Additional context (forwarded to compute_vectorized)

        Returns:
            Single value or dict of values
        """
        # Convert to WavePartition format with single-element arrays
        wave_parts = []
        for i, p in enumerate(partitions):
            wave_parts.append(WavePartition(
                hs=np.array([p['hs']]),
                tp=np.array([p['tp']]),
                direction=np.array([p.get('dir', p.get('direction', 0))]),
                partition_id=i,
                is_valid=np.array([True])
            ))

        output = self.compute_vectorized(
            wave_parts,
            depths=np.array([depth]),
            lats=np.array([lat]),
            lons=np.array([lon]),
            **kwargs,
        )

        # Extract single value
        if output.values.ndim == 1:
            return float(output.values[0])
        else:
            # Multi-column: return dict
            cols = output.extra.get('columns', self.output_columns) if output.extra else self.output_columns
            return {col: float(output.values[0, i]) for i, col in enumerate(cols)}
