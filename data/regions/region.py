#!/usr/bin/env python3
"""
Region Class for SWAN Modeling

Defines geographic subregions for wave modeling. Each region has bounds,
optional WW3 boundary points, and associated meshes.

Regions can have parent regions (e.g., NorCal is a subregion of California).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from data.bathymetry.gebco import GEBCOBathymetry
    from data.regions.mesh import Mesh


@dataclass
class Region:
    """
    A geographic region for SWAN wave modeling.

    Attributes:
        name: Short identifier (e.g., "norcal")
        display_name: Human-readable name (e.g., "Northern California")
        bounds: Geographic bounds as {"lat": (min, max), "lon": (min, max)}
        parent: Optional parent region (e.g., California for NorCal)
        color: Color for plotting this region's bounds
        ww3_boundary_points: List of (lon, lat) points for WW3 boundary extraction
        meshes: List of Mesh objects associated with this region
    """

    name: str
    display_name: str
    bounds: dict  # {"lat": (min, max), "lon": (min, max)}
    parent: Optional[Region] = None
    color: str = "red"

    # Placeholders for future implementation
    ww3_boundary_points: List[Tuple[float, float]] = field(default_factory=list)
    meshes: List[Mesh] = field(default_factory=list)

    @property
    def lat_range(self) -> Tuple[float, float]:
        """Get latitude range as (min, max)."""
        return self.bounds["lat"]

    @property
    def lon_range(self) -> Tuple[float, float]:
        """Get longitude range as (min, max)."""
        return self.bounds["lon"]

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point as (lon, lat)."""
        lat_center = (self.bounds["lat"][0] + self.bounds["lat"][1]) / 2
        lon_center = (self.bounds["lon"][0] + self.bounds["lon"][1]) / 2
        return (lon_center, lat_center)

    @property
    def width_deg(self) -> float:
        """Width in degrees longitude."""
        return self.bounds["lon"][1] - self.bounds["lon"][0]

    @property
    def height_deg(self) -> float:
        """Height in degrees latitude."""
        return self.bounds["lat"][1] - self.bounds["lat"][0]

    def get_gebco_subset(self, gebco: GEBCOBathymetry) -> dict:
        """
        Extract GEBCO bathymetry data for this region.

        Args:
            gebco: GEBCOBathymetry instance

        Returns:
            dict with keys: 'depth', 'lats', 'lons', 'elevation'
        """
        lat_min, lat_max = self.lat_range
        lon_min, lon_max = self.lon_range

        lat_mask = (gebco.lats >= lat_min) & (gebco.lats <= lat_max)
        lon_mask = (gebco.lons >= lon_min) & (gebco.lons <= lon_max)

        return {
            'depth': gebco.depth[np.ix_(lat_mask, lon_mask)],
            'elevation': gebco.elevation[np.ix_(lat_mask, lon_mask)],
            'lats': gebco.lats[lat_mask],
            'lons': gebco.lons[lon_mask],
        }

    def plot_bounds(
        self,
        ax: plt.Axes,
        linewidth: float = 2.0,
        linestyle: str = '-',
        fill: bool = False,
        alpha: float = 0.2,
        label: bool = True,
        transform=None,
    ) -> Rectangle:
        """
        Draw this region's bounds as a rectangle on a matplotlib axes.

        Args:
            ax: Matplotlib axes to draw on
            linewidth: Width of rectangle border
            linestyle: Style of border line
            fill: Whether to fill the rectangle
            alpha: Transparency for fill
            label: Whether to add region name label
            transform: Coordinate transform (e.g., ccrs.PlateCarree())

        Returns:
            The Rectangle patch that was added
        """
        lon_min, lon_max = self.lon_range
        lat_min, lat_max = self.lat_range

        rect_kwargs = {
            'linewidth': linewidth,
            'linestyle': linestyle,
            'edgecolor': self.color,
            'facecolor': self.color if fill else 'none',
            'alpha': alpha if fill else 1.0,
        }

        if transform is not None:
            rect_kwargs['transform'] = transform

        rect = Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            **rect_kwargs
        )
        ax.add_patch(rect)

        if label:
            # Add label at center of region
            center_lon, center_lat = self.center
            text_kwargs = {'fontsize': 10, 'fontweight': 'bold', 'color': self.color,
                          'ha': 'center', 'va': 'center'}
            if transform is not None:
                text_kwargs['transform'] = transform
            ax.text(center_lon, center_lat, self.display_name, **text_kwargs)

        return rect

    def __repr__(self) -> str:
        parent_str = f", parent='{self.parent.name}'" if self.parent else ""
        return (
            f"Region(name='{self.name}', display_name='{self.display_name}', "
            f"bounds={self.bounds}{parent_str})"
        )


# =============================================================================
# California Region Definitions
# =============================================================================

# Parent region encompassing all of California coast
CALIFORNIA = Region(
    name="california",
    display_name="California",
    bounds={
        "lat": (32.0, 42.0),
        "lon": (-126.0, -117.0),
    },
    color="black",
)

# Subregions - bounds are placeholders to be refined
NORCAL = Region(
    name="norcal",
    display_name="Northern California",
    bounds={
        "lat": (38.5, 42.0),      # Oregon border to ~Point Reyes
        "lon": (-126.0, -122.0),
    },
    parent=CALIFORNIA,
    color="#E63946",  # Red
)

CENTRAL_CAL = Region(
    name="central",
    display_name="Central California",
    bounds={
        "lat": (34.5, 38.5),      # Point Conception to ~Point Reyes
        "lon": (-124.0, -120.0),
    },
    parent=CALIFORNIA,
    color="#2A9D8F",  # Teal
)

SOCAL = Region(
    name="socal",
    display_name="Southern California",
    bounds={
        "lat": (32.0, 34.5),      # Mexico border to Point Conception
        "lon": (-121.0, -117.0),
    },
    parent=CALIFORNIA,
    color="#E9C46A",  # Gold
    ww3_boundary_points=[
        # Western boundary WW3 grid points (0.25° resolution)
        # Extracted from: data/swan/ww3_endpoints/socal/ww3_boundary_west.json
        (-121.0, 32.0),
        (-121.0, 32.25),
        (-121.0, 32.5),
        (-121.0, 32.75),
        (-121.0, 33.0),
        (-121.0, 33.25),
        (-121.0, 33.5),
        (-121.0, 33.75),
        (-121.0, 34.0),
        (-121.0, 34.25),
        (-121.0, 34.5),
    ],
)

# Registry of all California subregions
CA_SUBREGIONS = [NORCAL, CENTRAL_CAL, SOCAL]

# Quick lookup by name
REGIONS = {
    "california": CALIFORNIA,
    "norcal": NORCAL,
    "central": CENTRAL_CAL,
    "socal": SOCAL,
}


def get_region(name: str) -> Region:
    """Get a region by name."""
    if name not in REGIONS:
        raise ValueError(f"Unknown region: {name}. Available: {list(REGIONS.keys())}")
    return REGIONS[name]
