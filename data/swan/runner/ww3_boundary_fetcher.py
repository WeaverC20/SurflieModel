"""
WW3 Boundary Fetcher for SWAN

Fetches WaveWatch III data at specific boundary points for SWAN
boundary conditions in stationary mode.

Uses unified boundary config format (ww3_boundaries.json) with
multiple active boundaries.
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
class WavePartition:
    """A single wave partition (wind sea or swell component)."""
    hs: float   # Significant wave height (m)
    tp: float   # Peak period (s)
    dir: float  # Direction (degrees, nautical convention - direction FROM)
    spread: float = 25.0  # Directional spreading (degrees)


@dataclass
class BoundaryPoint:
    """A single boundary point with its wave data including spectral partitions."""
    lon: float
    lat: float
    index: int  # Index along the boundary (0 = first point)

    # Combined wave data (populated after fetch)
    times: Optional[List[datetime]] = None
    hs: Optional[List[float]] = None  # Significant wave height (m)
    tp: Optional[List[float]] = None  # Peak period (s)
    dir: Optional[List[float]] = None  # Direction (degrees)
    spread: Optional[List[float]] = None  # Directional spread (degrees)

    # Spectral partitions (for spectral boundary conditions)
    wind_waves: Optional[List[WavePartition]] = None      # Wind sea component
    primary_swell: Optional[List[WavePartition]] = None   # Primary swell
    secondary_swell: Optional[List[WavePartition]] = None # Secondary swell (may be None)
    tertiary_swell: Optional[List[WavePartition]] = None  # Tertiary swell (may be None)


class WW3BoundaryFetcher:
    """
    Fetches WW3 data at boundary points for SWAN stationary runs.

    Uses the existing WaveWatchFetcher to get grid data, then extracts
    values at the specific boundary points defined in the unified
    boundary config file (ww3_boundaries.json).

    Example usage:
        fetcher = WW3BoundaryFetcher()
        config, boundary_points = await fetcher.fetch_boundary(
            boundary_file="data/swan/ww3_endpoints/socal/ww3_boundaries.json",
            forecast_hours=[0]  # Single hour for stationary mode
        )
        # boundary_points["west"][i].hs[0], etc. for wave parameters
    """

    # Default directional spreading when not available from WW3
    DEFAULT_SPREAD = 25.0  # degrees

    def __init__(self):
        self.ww3_fetcher = WaveWatchFetcher()

    def load_boundary_config(self, boundary_file: str | Path) -> Dict:
        """
        Load unified boundary configuration from JSON file.

        Args:
            boundary_file: Path to unified boundary JSON file (ww3_boundaries.json)

        Returns:
            Dict with config data including 'boundaries' and 'active_boundaries'
        """
        boundary_file = Path(boundary_file)

        with open(boundary_file) as f:
            data = json.load(f)

        # Validate unified format
        if "boundaries" not in data:
            raise ValueError(
                f"File {boundary_file.name} is not in unified format. "
                f"Expected 'boundaries' key with boundary definitions."
            )

        if "active_boundaries" not in data:
            raise ValueError(
                f"File {boundary_file.name} missing 'active_boundaries' key."
            )

        logger.info(f"Loaded boundary config from {boundary_file.name}")
        logger.info(f"  Active boundaries: {data['active_boundaries']}")

        return data

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
        forecast_hours: Optional[List[int]] = None,
        boundaries_to_fetch: Optional[List[str]] = None,
    ) -> Tuple[Dict, Dict[str, List[BoundaryPoint]]]:
        """
        Fetch WW3 data for all active boundaries from unified config file.

        Fetches WW3 data once for a bounding box that covers all active
        boundaries, then extracts values for each boundary's points.

        Args:
            boundary_file: Path to unified boundary JSON file (ww3_boundaries.json)
            forecast_hours: List of forecast hours to fetch (default: [0] for stationary)
            boundaries_to_fetch: Which boundaries to fetch (default: all active)

        Returns:
            Tuple of:
                - config dict (from the unified config)
                - dict mapping boundary side to list of BoundaryPoint objects
        """
        if forecast_hours is None:
            forecast_hours = [0]

        # Load unified config
        config = self.load_boundary_config(boundary_file)

        # Determine which boundaries to fetch
        if boundaries_to_fetch is None:
            boundaries_to_fetch = config["active_boundaries"]

        # Validate requested boundaries exist
        for side in boundaries_to_fetch:
            if side not in config["boundaries"]:
                raise ValueError(f"Boundary '{side}' not found in config")

        logger.info(f"Fetching WW3 data for boundaries: {boundaries_to_fetch}")

        # Collect all points from all boundaries to determine overall bounding box
        all_points = []
        for side in boundaries_to_fetch:
            boundary_data = config["boundaries"][side]
            for lon, lat in boundary_data["points"]:
                all_points.append(BoundaryPoint(lon=lon, lat=lat, index=0))

        # Get combined bounding box
        min_lat, max_lat, min_lon, max_lon = self._get_bounding_box(all_points)

        logger.info(f"Combined bounding box: "
                   f"lat [{min_lat:.2f}, {max_lat:.2f}], lon [{min_lon:.2f}, {max_lon:.2f}]")

        # Create BoundaryPoint objects for each boundary
        boundary_points: Dict[str, List[BoundaryPoint]] = {}
        for side in boundaries_to_fetch:
            boundary_data = config["boundaries"][side]
            points = []
            for i, (lon, lat) in enumerate(boundary_data["points"]):
                point = BoundaryPoint(lon=lon, lat=lat, index=i)
                # Initialize time series storage
                point.times = []
                point.hs = []
                point.tp = []
                point.dir = []
                point.spread = []
                point.wind_waves = []
                point.primary_swell = []
                point.secondary_swell = []
                point.tertiary_swell = []
                points.append(point)
            boundary_points[side] = points

        # Fetch data for each forecast hour (single fetch covers all boundaries)
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

                # Extract partition grids
                wind_hs_grid = np.array(grid_data.get("wind_wave_height", hs_grid * 0.3))
                wind_tp_grid = np.array(grid_data.get("wind_wave_period", np.full_like(hs_grid, 5.0)))
                wind_dir_grid = np.array(grid_data.get("wind_wave_direction", dir_grid))

                swell1_hs_grid = np.array(grid_data.get("primary_swell_height", hs_grid * 0.7))
                swell1_tp_grid = np.array(grid_data.get("primary_swell_period", tp_grid))
                swell1_dir_grid = np.array(grid_data.get("primary_swell_direction", dir_grid))

                has_secondary = "secondary_swell_height" in grid_data
                if has_secondary:
                    swell2_hs_grid = np.array(grid_data["secondary_swell_height"])
                    swell2_tp_grid = np.array(grid_data["secondary_swell_period"])
                    swell2_dir_grid = np.array(grid_data["secondary_swell_direction"])

                has_tertiary = "tertiary_swell_height" in grid_data
                if has_tertiary:
                    swell3_hs_grid = np.array(grid_data["tertiary_swell_height"])
                    swell3_tp_grid = np.array(grid_data["tertiary_swell_period"])
                    swell3_dir_grid = np.array(grid_data["tertiary_swell_direction"])

                # Extract values for each boundary's points
                for side, points in boundary_points.items():
                    for point in points:
                        # Combined sea state
                        hs = self._extract_point_value(grid_lats, grid_lons, hs_grid, point.lat, point.lon)
                        tp = self._extract_point_value(grid_lats, grid_lons, tp_grid, point.lat, point.lon)
                        wave_dir = self._extract_point_value(grid_lats, grid_lons, dir_grid, point.lat, point.lon)
                        spread = self.DEFAULT_SPREAD

                        point.times.append(forecast_time)
                        point.hs.append(hs)
                        point.tp.append(tp)
                        point.dir.append(wave_dir)
                        point.spread.append(spread)

                        # Wind wave partition
                        wind_hs = self._extract_point_value(grid_lats, grid_lons, wind_hs_grid, point.lat, point.lon)
                        wind_tp = self._extract_point_value(grid_lats, grid_lons, wind_tp_grid, point.lat, point.lon)
                        wind_dir = self._extract_point_value(grid_lats, grid_lons, wind_dir_grid, point.lat, point.lon)
                        point.wind_waves.append(WavePartition(
                            hs=wind_hs, tp=wind_tp, dir=wind_dir, spread=30.0
                        ))

                        # Primary swell partition
                        swell1_hs = self._extract_point_value(grid_lats, grid_lons, swell1_hs_grid, point.lat, point.lon)
                        swell1_tp = self._extract_point_value(grid_lats, grid_lons, swell1_tp_grid, point.lat, point.lon)
                        swell1_dir = self._extract_point_value(grid_lats, grid_lons, swell1_dir_grid, point.lat, point.lon)
                        point.primary_swell.append(WavePartition(
                            hs=swell1_hs, tp=swell1_tp, dir=swell1_dir, spread=20.0
                        ))

                        # Secondary swell partition
                        if has_secondary:
                            swell2_hs = self._extract_point_value(grid_lats, grid_lons, swell2_hs_grid, point.lat, point.lon)
                            swell2_tp = self._extract_point_value(grid_lats, grid_lons, swell2_tp_grid, point.lat, point.lon)
                            swell2_dir = self._extract_point_value(grid_lats, grid_lons, swell2_dir_grid, point.lat, point.lon)
                            point.secondary_swell.append(WavePartition(
                                hs=swell2_hs, tp=swell2_tp, dir=swell2_dir, spread=20.0
                            ))
                        else:
                            point.secondary_swell.append(None)

                        # Tertiary swell partition
                        if has_tertiary:
                            swell3_hs = self._extract_point_value(grid_lats, grid_lons, swell3_hs_grid, point.lat, point.lon)
                            swell3_tp = self._extract_point_value(grid_lats, grid_lons, swell3_tp_grid, point.lat, point.lon)
                            swell3_dir = self._extract_point_value(grid_lats, grid_lons, swell3_dir_grid, point.lat, point.lon)
                            point.tertiary_swell.append(WavePartition(
                                hs=swell3_hs, tp=swell3_tp, dir=swell3_dir, spread=20.0
                            ))
                        else:
                            point.tertiary_swell.append(None)

                    # Log summary for this boundary
                    avg_hs = sum(p.hs[-1] for p in points) / len(points)
                    logger.info(f"  {side} boundary: avg Hs={avg_hs:.2f}m ({len(points)} points)")

            except Exception as e:
                logger.error(f"Failed to fetch hour {hour}: {e}")
                # Fill with NaN for this timestep
                for points in boundary_points.values():
                    for point in points:
                        point.times.append(None)
                        point.hs.append(np.nan)
                        point.tp.append(np.nan)
                        point.dir.append(np.nan)
                        point.spread.append(np.nan)
                        point.wind_waves.append(None)
                        point.primary_swell.append(None)
                        point.secondary_swell.append(None)
                        point.tertiary_swell.append(None)

        return config, boundary_points
