"""
WW3 Boundary Fetcher for SWAN

Fetches WaveWatch III data at specific boundary points for SWAN
boundary conditions in stationary mode.
"""

import asyncio
import json
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import the existing WaveWatch fetcher
from data.pipelines.wave.wavewatch_fetcher import WaveWatchFetcher

logger = logging.getLogger(__name__)


@dataclass
class BoundaryPoint:
    """A single boundary point with its wave data."""
    lon: float
    lat: float
    index: int  # Index along the boundary (0 = first point)

    # Wave data (populated after fetch)
    times: Optional[List[datetime]] = None
    hs: Optional[List[float]] = None  # Significant wave height (m)
    tp: Optional[List[float]] = None  # Peak period (s)
    dir: Optional[List[float]] = None  # Direction (degrees)
    spread: Optional[List[float]] = None  # Directional spread (degrees)


class WW3BoundaryFetcher:
    """
    Fetches WW3 data at boundary points for SWAN stationary runs.

    Uses the existing WaveWatchFetcher to get grid data, then extracts
    values at the specific boundary points defined in the boundary JSON files.

    Example usage:
        fetcher = WW3BoundaryFetcher()
        metadata, points = await fetcher.fetch_boundary(
            boundary_file="data/swan/ww3_endpoints/socal/ww3_boundary_west.json",
            forecast_hours=[0]  # Single hour for stationary mode
        )
        # Use points[i].hs[0], points[i].tp[0], etc. for wave parameters
    """

    # Default directional spreading when not available from WW3
    DEFAULT_SPREAD = 25.0  # degrees

    def __init__(self):
        self.ww3_fetcher = WaveWatchFetcher()

    def load_boundary_points(self, boundary_file: str | Path) -> Tuple[Dict, List[BoundaryPoint]]:
        """
        Load boundary point definitions from JSON file.

        Args:
            boundary_file: Path to boundary JSON file

        Returns:
            Tuple of (metadata dict, list of BoundaryPoint objects)
        """
        boundary_file = Path(boundary_file)

        with open(boundary_file) as f:
            data = json.load(f)

        points = []
        for i, (lon, lat) in enumerate(data["points"]):
            points.append(BoundaryPoint(lon=lon, lat=lat, index=i))

        logger.info(f"Loaded {len(points)} boundary points from {boundary_file.name}")
        return data, points

    def _get_bounding_box(self, points: List[BoundaryPoint], padding: float = 0.5) -> Tuple[float, float, float, float]:
        """
        Get bounding box around boundary points with padding.

        Args:
            points: List of boundary points
            padding: Degrees of padding around points

        Returns:
            (min_lat, max_lat, min_lon, max_lon)
        """
        lons = [p.lon for p in points]
        lats = [p.lat for p in points]

        return (
            min(lats) - padding,
            max(lats) + padding,
            min(lons) - padding,
            max(lons) + padding
        )

    def _extract_point_value(
        self,
        grid_lats: np.ndarray,
        grid_lons: np.ndarray,
        grid_data: np.ndarray,
        point_lat: float,
        point_lon: float
    ) -> float:
        """
        Extract value at a specific point from grid data.

        Uses nearest neighbor interpolation (points should align with WW3 grid).

        Args:
            grid_lats: 1D array of latitudes
            grid_lons: 1D array of longitudes
            grid_data: 2D array of values [lat, lon]
            point_lat: Target latitude
            point_lon: Target longitude

        Returns:
            Value at the point (or NaN if not found)
        """
        # Find nearest grid indices
        lat_idx = np.argmin(np.abs(grid_lats - point_lat))
        lon_idx = np.argmin(np.abs(grid_lons - point_lon))

        # Check if we're close enough (within 0.15 degrees)
        if abs(grid_lats[lat_idx] - point_lat) > 0.15 or abs(grid_lons[lon_idx] - point_lon) > 0.15:
            logger.warning(f"Point ({point_lat}, {point_lon}) not well aligned with grid")

        # Handle 2D grid data
        if isinstance(grid_data, list):
            grid_data = np.array(grid_data)

        value = grid_data[lat_idx, lon_idx]

        # Convert NaN-like values
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan

        return float(value)

    async def fetch_boundary(
        self,
        boundary_file: str | Path,
        forecast_hours: Optional[List[int]] = None
    ) -> Tuple[Dict, List[BoundaryPoint]]:
        """
        Fetch WW3 data at all boundary points for specified forecast hours.

        Args:
            boundary_file: Path to boundary JSON file
            forecast_hours: List of forecast hours to fetch (default: [0] for stationary)

        Returns:
            Tuple of (metadata dict, list of BoundaryPoint objects with data)
        """
        if forecast_hours is None:
            forecast_hours = [0]  # Default to current hour for stationary mode

        # Load boundary definition
        metadata, points = self.load_boundary_points(boundary_file)

        # Get bounding box for WW3 fetch
        min_lat, max_lat, min_lon, max_lon = self._get_bounding_box(points)

        logger.info(f"Fetching WW3 data for bounding box: "
                   f"lat [{min_lat:.2f}, {max_lat:.2f}], lon [{min_lon:.2f}, {max_lon:.2f}]")

        # Initialize time series storage for each point
        for point in points:
            point.times = []
            point.hs = []
            point.tp = []
            point.dir = []
            point.spread = []

        # Fetch data for each forecast hour
        for hour in forecast_hours:
            logger.info(f"Fetching forecast hour {hour}...")

            try:
                grid_data = await self.ww3_fetcher.fetch_wave_grid(
                    min_lat=min_lat,
                    max_lat=max_lat,
                    min_lon=min_lon,
                    max_lon=max_lon,
                    forecast_hour=hour
                )

                # Parse the forecast time
                forecast_time = datetime.fromisoformat(grid_data["forecast_time"])

                # Extract grid arrays
                grid_lats = np.array(grid_data["lat"])
                grid_lons = np.array(grid_data["lon"])
                hs_grid = np.array(grid_data["significant_wave_height"])
                tp_grid = np.array(grid_data["peak_wave_period"])
                dir_grid = np.array(grid_data["mean_wave_direction"])

                # Extract values at each boundary point
                for point in points:
                    hs = self._extract_point_value(grid_lats, grid_lons, hs_grid, point.lat, point.lon)
                    tp = self._extract_point_value(grid_lats, grid_lons, tp_grid, point.lat, point.lon)
                    wave_dir = self._extract_point_value(grid_lats, grid_lons, dir_grid, point.lat, point.lon)

                    # Use default spread (WW3 bulk params don't include spreading)
                    spread = self.DEFAULT_SPREAD

                    point.times.append(forecast_time)
                    point.hs.append(hs)
                    point.tp.append(tp)
                    point.dir.append(wave_dir)
                    point.spread.append(spread)

                logger.info(f"  Hour {hour}: Hs={point.hs[-1]:.2f}m, Tp={point.tp[-1]:.1f}s, Dir={point.dir[-1]:.0f}°")

            except Exception as e:
                logger.error(f"Failed to fetch hour {hour}: {e}")
                # Fill with NaN for this timestep
                for point in points:
                    point.times.append(None)
                    point.hs.append(np.nan)
                    point.tp.append(np.nan)
                    point.dir.append(np.nan)
                    point.spread.append(np.nan)

        return metadata, points