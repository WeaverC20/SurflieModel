#!/usr/bin/env python3
"""
WW3 Boundary Endpoint Extractor

Finds the nearest WaveWatch III grid points along a boundary line
for use as SWAN spectral boundary conditions.

WW3 Global Model Grid:
    - Resolution: 0.25° (approximately 25km at mid-latitudes)
    - Grid points at: ..., -121.25, -121.0, -120.75, ... (longitude)
    - Grid points at: ..., 32.0, 32.25, 32.5, ... (latitude)

Usage:
    from data.swan.ww3_endpoints.extract_ww3_endpoints import (
        WW3Grid,
        find_boundary_points,
        BoundaryLine,
    )

    # Define a vertical boundary line (constant longitude)
    line = BoundaryLine.vertical(lon=-121.0, lat_min=32.0, lat_max=34.5)

    # Find nearest WW3 points
    points = find_boundary_points(line)
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from enum import Enum
import json
import numpy as np

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# =============================================================================
# WW3 Grid Definition
# =============================================================================

@dataclass
class WW3Grid:
    """
    WaveWatch III global grid specification.

    The WW3 global model uses a regular lat/lon grid at 0.25° resolution.
    Grid points are located at exact multiples of 0.25°.
    """

    resolution_deg: float = 0.25  # Grid spacing in degrees

    # Global grid extent (full globe)
    lon_min: float = -180.0
    lon_max: float = 180.0
    lat_min: float = -90.0
    lat_max: float = 90.0

    def snap_to_grid(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        Snap a coordinate to the nearest WW3 grid point.

        Args:
            lon: Longitude in degrees
            lat: Latitude in degrees

        Returns:
            (lon, lat) of nearest grid point
        """
        snapped_lon = round(lon / self.resolution_deg) * self.resolution_deg
        snapped_lat = round(lat / self.resolution_deg) * self.resolution_deg
        return (snapped_lon, snapped_lat)

    def get_grid_points_in_range(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
    ) -> List[Tuple[float, float]]:
        """
        Get all WW3 grid points within a bounding box.

        Args:
            lon_min, lon_max: Longitude range
            lat_min, lat_max: Latitude range

        Returns:
            List of (lon, lat) grid points
        """
        # Snap bounds to grid
        lon_start = np.ceil(lon_min / self.resolution_deg) * self.resolution_deg
        lon_end = np.floor(lon_max / self.resolution_deg) * self.resolution_deg
        lat_start = np.ceil(lat_min / self.resolution_deg) * self.resolution_deg
        lat_end = np.floor(lat_max / self.resolution_deg) * self.resolution_deg

        lons = np.arange(lon_start, lon_end + self.resolution_deg / 2, self.resolution_deg)
        lats = np.arange(lat_start, lat_end + self.resolution_deg / 2, self.resolution_deg)

        points = []
        for lat in lats:
            for lon in lons:
                points.append((round(lon, 4), round(lat, 4)))

        return points

    def get_grid_points_along_longitude(
        self,
        lon: float,
        lat_min: float,
        lat_max: float,
    ) -> List[Tuple[float, float]]:
        """
        Get WW3 grid points along a line of constant longitude.

        Args:
            lon: Longitude of the line
            lat_min, lat_max: Latitude range

        Returns:
            List of (lon, lat) grid points, sorted south to north
        """
        # Snap longitude to nearest grid point
        snapped_lon = round(lon / self.resolution_deg) * self.resolution_deg

        # Get latitude points
        lat_start = np.ceil(lat_min / self.resolution_deg) * self.resolution_deg
        lat_end = np.floor(lat_max / self.resolution_deg) * self.resolution_deg

        lats = np.arange(lat_start, lat_end + self.resolution_deg / 2, self.resolution_deg)

        points = [(round(snapped_lon, 4), round(lat, 4)) for lat in lats]
        return sorted(points, key=lambda p: p[1])  # Sort by latitude

    def get_grid_points_along_latitude(
        self,
        lat: float,
        lon_min: float,
        lon_max: float,
    ) -> List[Tuple[float, float]]:
        """
        Get WW3 grid points along a line of constant latitude.

        Args:
            lat: Latitude of the line
            lon_min, lon_max: Longitude range

        Returns:
            List of (lon, lat) grid points, sorted west to east
        """
        # Snap latitude to nearest grid point
        snapped_lat = round(lat / self.resolution_deg) * self.resolution_deg

        # Get longitude points
        lon_start = np.ceil(lon_min / self.resolution_deg) * self.resolution_deg
        lon_end = np.floor(lon_max / self.resolution_deg) * self.resolution_deg

        lons = np.arange(lon_start, lon_end + self.resolution_deg / 2, self.resolution_deg)

        points = [(round(lon, 4), round(snapped_lat, 4)) for lon in lons]
        return sorted(points, key=lambda p: p[0])  # Sort by longitude


# Default WW3 grid instance
WW3_GRID = WW3Grid()


# =============================================================================
# Boundary Line Definition
# =============================================================================

class BoundaryType(Enum):
    """Type of boundary line."""
    VERTICAL = "vertical"      # Constant longitude (west or east boundary)
    HORIZONTAL = "horizontal"  # Constant latitude (north or south boundary)
    ARBITRARY = "arbitrary"    # General line between two points


@dataclass
class BoundaryLine:
    """
    Defines a boundary line for WW3 point extraction.

    Attributes:
        boundary_type: Type of boundary (vertical, horizontal, arbitrary)
        points: List of (lon, lat) points defining the line
        name: Optional name for this boundary
        side: Which side of the domain (west, east, north, south)
    """

    boundary_type: BoundaryType
    points: List[Tuple[float, float]]
    name: Optional[str] = None
    side: Optional[str] = None  # 'west', 'east', 'north', 'south'

    @classmethod
    def vertical(
        cls,
        lon: float,
        lat_min: float,
        lat_max: float,
        name: Optional[str] = None,
        side: str = "west",
    ) -> "BoundaryLine":
        """
        Create a vertical boundary line (constant longitude).

        Args:
            lon: Longitude of the line
            lat_min: Southern extent
            lat_max: Northern extent
            name: Optional name
            side: 'west' or 'east'

        Returns:
            BoundaryLine object
        """
        return cls(
            boundary_type=BoundaryType.VERTICAL,
            points=[(lon, lat_min), (lon, lat_max)],
            name=name or f"vertical_lon_{lon}",
            side=side,
        )

    @classmethod
    def horizontal(
        cls,
        lat: float,
        lon_min: float,
        lon_max: float,
        name: Optional[str] = None,
        side: str = "south",
    ) -> "BoundaryLine":
        """
        Create a horizontal boundary line (constant latitude).

        Args:
            lat: Latitude of the line
            lon_min: Western extent
            lon_max: Eastern extent
            name: Optional name
            side: 'north' or 'south'

        Returns:
            BoundaryLine object
        """
        return cls(
            boundary_type=BoundaryType.HORIZONTAL,
            points=[(lon_min, lat), (lon_max, lat)],
            name=name or f"horizontal_lat_{lat}",
            side=side,
        )

    @property
    def lon_range(self) -> Tuple[float, float]:
        """Get longitude range of the line."""
        lons = [p[0] for p in self.points]
        return (min(lons), max(lons))

    @property
    def lat_range(self) -> Tuple[float, float]:
        """Get latitude range of the line."""
        lats = [p[1] for p in self.points]
        return (min(lats), max(lats))


# =============================================================================
# Boundary Point Extraction
# =============================================================================

@dataclass
class BoundaryPointSet:
    """
    A set of WW3 boundary points for SWAN.

    Attributes:
        points: List of (lon, lat) WW3 grid points
        boundary_line: The boundary line these points are from
        grid: WW3 grid specification used
        region_name: Name of the region
        mesh_name: Name of the associated mesh
    """

    points: List[Tuple[float, float]]
    boundary_line: BoundaryLine
    grid: WW3Grid
    region_name: Optional[str] = None
    mesh_name: Optional[str] = None

    @property
    def n_points(self) -> int:
        """Number of boundary points."""
        return len(self.points)

    def save(self, filepath: Union[str, Path]) -> Path:
        """
        Save boundary points to JSON file.

        Args:
            filepath: Path to save to

        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "region_name": self.region_name,
            "mesh_name": self.mesh_name,
            "boundary": {
                "type": self.boundary_line.boundary_type.value,
                "side": self.boundary_line.side,
                "name": self.boundary_line.name,
                "definition": self.boundary_line.points,
            },
            "ww3_grid": {
                "resolution_deg": self.grid.resolution_deg,
            },
            "points": self.points,
            "n_points": self.n_points,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {self.n_points} boundary points to: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BoundaryPointSet":
        """
        Load boundary points from JSON file.

        Args:
            filepath: Path to load from

        Returns:
            BoundaryPointSet object
        """
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            data = json.load(f)

        boundary_line = BoundaryLine(
            boundary_type=BoundaryType(data["boundary"]["type"]),
            points=[tuple(p) for p in data["boundary"]["definition"]],
            name=data["boundary"]["name"],
            side=data["boundary"]["side"],
        )

        return cls(
            points=[tuple(p) for p in data["points"]],
            boundary_line=boundary_line,
            grid=WW3Grid(resolution_deg=data["ww3_grid"]["resolution_deg"]),
            region_name=data.get("region_name"),
            mesh_name=data.get("mesh_name"),
        )

    def summary(self) -> str:
        """Return a summary string."""
        lines = [
            f"WW3 Boundary Points: {self.boundary_line.name}",
            f"  Region: {self.region_name}",
            f"  Mesh: {self.mesh_name}",
            f"  Boundary type: {self.boundary_line.boundary_type.value}",
            f"  Boundary side: {self.boundary_line.side}",
            f"  Number of points: {self.n_points}",
            f"  WW3 resolution: {self.grid.resolution_deg}°",
            f"  Points:",
        ]

        for i, (lon, lat) in enumerate(self.points):
            lines.append(f"    [{i}] ({lon:.2f}°, {lat:.2f}°)")

        return '\n'.join(lines)

    def plot(
        self,
        ax=None,
        figsize: Tuple[int, int] = (10, 8),
        show_region: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ):
        """
        Plot the boundary points on a map.

        Args:
            ax: Matplotlib axes (creates new if None)
            figsize: Figure size
            show_region: Whether to show region bounds
            save_path: Path to save figure
            show: Whether to display
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        if ax is None:
            fig, ax = plt.subplots(
                figsize=figsize,
                subplot_kw={'projection': ccrs.PlateCarree()}
            )

        # Plot boundary points
        lons = [p[0] for p in self.points]
        lats = [p[1] for p in self.points]

        ax.scatter(
            lons, lats,
            c='red', s=100, marker='o',
            transform=ccrs.PlateCarree(),
            zorder=10,
            label=f'WW3 Points ({self.n_points})'
        )

        # Add point labels
        for i, (lon, lat) in enumerate(self.points):
            ax.text(
                lon + 0.1, lat, str(i),
                transform=ccrs.PlateCarree(),
                fontsize=8,
            )

        # Plot boundary line
        line_lons = [p[0] for p in self.boundary_line.points]
        line_lats = [p[1] for p in self.boundary_line.points]
        ax.plot(
            line_lons, line_lats,
            'b-', linewidth=2,
            transform=ccrs.PlateCarree(),
            label='Boundary Line'
        )

        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.STATES, linewidth=0.5)

        # Set extent with some padding
        lon_min, lon_max = self.boundary_line.lon_range
        lat_min, lat_max = self.boundary_line.lat_range
        padding = 1.0
        ax.set_extent([
            lon_min - padding, lon_max + padding,
            lat_min - padding, lat_max + padding
        ], crs=ccrs.PlateCarree())

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        ax.legend(loc='upper right')
        ax.set_title(f"WW3 Boundary Points: {self.boundary_line.name}")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return ax


def find_boundary_points(
    boundary_line: BoundaryLine,
    grid: WW3Grid = None,
    region_name: Optional[str] = None,
    mesh_name: Optional[str] = None,
) -> BoundaryPointSet:
    """
    Find WW3 grid points along a boundary line.

    Args:
        boundary_line: The boundary line to extract points from
        grid: WW3 grid specification (uses default if None)
        region_name: Name of the region
        mesh_name: Name of the associated mesh

    Returns:
        BoundaryPointSet with the extracted points
    """
    if grid is None:
        grid = WW3_GRID

    if boundary_line.boundary_type == BoundaryType.VERTICAL:
        # Constant longitude line
        lon = boundary_line.points[0][0]  # First point's longitude
        lat_min, lat_max = boundary_line.lat_range
        points = grid.get_grid_points_along_longitude(lon, lat_min, lat_max)

    elif boundary_line.boundary_type == BoundaryType.HORIZONTAL:
        # Constant latitude line
        lat = boundary_line.points[0][1]  # First point's latitude
        lon_min, lon_max = boundary_line.lon_range
        points = grid.get_grid_points_along_latitude(lat, lon_min, lon_max)

    else:
        raise NotImplementedError("Arbitrary boundary lines not yet supported")

    return BoundaryPointSet(
        points=points,
        boundary_line=boundary_line,
        grid=grid,
        region_name=region_name,
        mesh_name=mesh_name,
    )


def extract_region_boundaries(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    sides: List[str] = None,
    region_name: Optional[str] = None,
    mesh_name: Optional[str] = None,
) -> dict:
    """
    Extract WW3 boundary points for all sides of a rectangular region.

    Args:
        lon_min, lon_max: Longitude bounds
        lat_min, lat_max: Latitude bounds
        sides: Which sides to extract ('west', 'east', 'north', 'south')
               Default is ['west'] for typical wave propagation
        region_name: Name of the region
        mesh_name: Name of the mesh

    Returns:
        Dict mapping side name to BoundaryPointSet
    """
    if sides is None:
        sides = ['west']  # Default: waves typically come from the west

    boundaries = {}

    if 'west' in sides:
        line = BoundaryLine.vertical(lon_min, lat_min, lat_max, side='west')
        boundaries['west'] = find_boundary_points(
            line, region_name=region_name, mesh_name=mesh_name
        )

    if 'east' in sides:
        line = BoundaryLine.vertical(lon_max, lat_min, lat_max, side='east')
        boundaries['east'] = find_boundary_points(
            line, region_name=region_name, mesh_name=mesh_name
        )

    if 'south' in sides:
        line = BoundaryLine.horizontal(lat_min, lon_min, lon_max, side='south')
        boundaries['south'] = find_boundary_points(
            line, region_name=region_name, mesh_name=mesh_name
        )

    if 'north' in sides:
        line = BoundaryLine.horizontal(lat_max, lon_min, lon_max, side='north')
        boundaries['north'] = find_boundary_points(
            line, region_name=region_name, mesh_name=mesh_name
        )

    return boundaries