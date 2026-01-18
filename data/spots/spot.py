"""
Surf Spot Class

Defines individual surf spots within regions. Each spot has a precise location
and can extract wave predictions from SWAN model output.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from data.regions.region import Region
    from data.swan.analysis.output_reader import SwanOutput, WavePartitionGrid


@dataclass
class SpotForecast:
    """
    Wave forecast data extracted at a specific spot.

    Contains the spectral partition data (wind sea + swells) at the spot location.
    """
    spot_name: str
    lat: float
    lon: float

    # Grid indices where data was extracted
    grid_i: int  # longitude index
    grid_j: int  # latitude index

    # Distance from spot to nearest grid point (km)
    distance_km: float

    # Partition data: list of (hs, tp, dir, label) tuples
    # Each partition is (significant_height_m, peak_period_s, direction_deg, label)
    partitions: List[Tuple[float, float, float, str]]

    def __repr__(self) -> str:
        lines = [f"SpotForecast: {self.spot_name}"]
        lines.append(f"  Location: ({self.lat:.4f}, {self.lon:.4f})")
        lines.append(f"  Grid point: ({self.grid_i}, {self.grid_j}), {self.distance_km:.2f} km away")
        lines.append(f"  Partitions:")
        for hs, tp, direction, label in self.partitions:
            if not np.isnan(hs) and hs > 0:
                lines.append(f"    {label}: Hs={hs:.2f}m, Tp={tp:.1f}s, Dir={direction:.0f}°")
        return "\n".join(lines)


@dataclass
class SurfSpot:
    """
    A specific surf spot location.

    Attributes:
        name: Unique identifier (e.g., "huntington_pier")
        display_name: Human-readable name (e.g., "Huntington Beach Pier")
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees (negative for west)
        region: Parent Region this spot belongs to
    """
    name: str
    display_name: str
    lat: float
    lon: float
    region: Region

    def get_forecast(self, swan_output: SwanOutput, max_search_radius: int = 5) -> SpotForecast:
        """
        Extract wave forecast at this spot from SWAN output.

        Finds the nearest valid (non-NaN) grid point and extracts all spectral
        partition data. If the nearest point is on land, searches nearby points.

        Args:
            swan_output: SwanOutput object containing model results
            max_search_radius: Maximum grid cells to search from nearest point

        Returns:
            SpotForecast with partition data at this location
        """
        # Find nearest valid grid point (searches nearby if nearest is on land)
        i, j, distance_km = self._find_nearest_valid_grid_point(
            swan_output.lons,
            swan_output.lats,
            swan_output.hsig,
            max_search_radius
        )

        # Extract partition data at this point
        partitions = []
        for partition in swan_output.partitions:
            hs = partition.hs[j, i]
            tp = partition.tp[j, i]
            direction = partition.dir[j, i]

            # Check for invalid/land values
            if hs == swan_output.exception_value:
                hs = np.nan
                tp = np.nan
                direction = np.nan

            partitions.append((hs, tp, direction, partition.label))

        return SpotForecast(
            spot_name=self.name,
            lat=self.lat,
            lon=self.lon,
            grid_i=i,
            grid_j=j,
            distance_km=distance_km,
            partitions=partitions,
        )

    def _find_nearest_valid_grid_point(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        data: np.ndarray,
        max_search_radius: int = 5
    ) -> Tuple[int, int, float]:
        """
        Find the nearest valid (non-NaN) grid point to this spot.

        If the nearest point has NaN data (land), searches in expanding
        squares around the point until valid data is found.

        Args:
            lons: 1D array of longitude values
            lats: 1D array of latitude values
            data: 2D array to check for valid values (e.g., hsig)
            max_search_radius: Maximum grid cells to search

        Returns:
            Tuple of (i_index, j_index, distance_km)
        """
        # Find nearest indices
        i_nearest = np.argmin(np.abs(lons - self.lon))
        j_nearest = np.argmin(np.abs(lats - self.lat))

        ny, nx = data.shape

        # Check if nearest point is valid
        if not np.isnan(data[j_nearest, i_nearest]):
            distance_km = self._haversine_km(
                self.lat, self.lon,
                lats[j_nearest], lons[i_nearest]
            )
            return i_nearest, j_nearest, distance_km

        # Search in expanding radius for valid point
        best_i, best_j = i_nearest, j_nearest
        best_distance = float('inf')

        for radius in range(1, max_search_radius + 1):
            # Check all points at this radius (square ring)
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    # Only check points on the edge of the square
                    if abs(di) != radius and abs(dj) != radius:
                        continue

                    i = i_nearest + di
                    j = j_nearest + dj

                    # Bounds check
                    if i < 0 or i >= nx or j < 0 or j >= ny:
                        continue

                    # Check if valid
                    if np.isnan(data[j, i]):
                        continue

                    # Calculate distance
                    distance = self._haversine_km(
                        self.lat, self.lon,
                        lats[j], lons[i]
                    )

                    if distance < best_distance:
                        best_distance = distance
                        best_i, best_j = i, j

            # If we found a valid point at this radius, use it
            if best_distance < float('inf'):
                return best_i, best_j, best_distance

        # No valid point found, return nearest anyway
        distance_km = self._haversine_km(
            self.lat, self.lon,
            lats[j_nearest], lons[i_nearest]
        )
        return i_nearest, j_nearest, distance_km

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two points in km."""
        R = 6371  # Earth's radius in km

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return R * c

    def __repr__(self) -> str:
        return (
            f"SurfSpot(name='{self.name}', display_name='{self.display_name}', "
            f"lat={self.lat}, lon={self.lon}, region='{self.region.name}')"
        )


# =============================================================================
# Southern California Surf Spots
# =============================================================================

# Import region for spot definitions
from data.regions.region import SOCAL

# Example spots - add more as needed
HUNTINGTON_PIER = SurfSpot(
    name="huntington_pier",
    display_name="Huntington Beach Pier",
    lat=33.6556,
    lon=-117.9999,
    region=SOCAL,
)

TRESTLES = SurfSpot(
    name="trestles",
    display_name="Trestles",
    lat=33.3825,
    lon=-117.5883,
    region=SOCAL,
)

BLACKS_BEACH = SurfSpot(
    name="blacks_beach",
    display_name="Black's Beach",
    lat=32.8894,
    lon=-117.2528,
    region=SOCAL,
)

# Registry of all spots
SPOTS = {
    "huntington_pier": HUNTINGTON_PIER,
    "trestles": TRESTLES,
    "blacks_beach": BLACKS_BEACH,
}

# Group spots by region
SOCAL_SPOTS = [HUNTINGTON_PIER, TRESTLES, BLACKS_BEACH]


def get_spot(name: str) -> SurfSpot:
    """Get a spot by name."""
    if name not in SPOTS:
        raise ValueError(f"Unknown spot: {name}. Available: {list(SPOTS.keys())}")
    return SPOTS[name]


def get_spots_for_region(region_name: str) -> List[SurfSpot]:
    """Get all spots in a region."""
    return [spot for spot in SPOTS.values() if spot.region.name == region_name]
