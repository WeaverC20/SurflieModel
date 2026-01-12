"""
Region Configuration System

This module provides data structures and utilities for managing geographic regions
in the surf forecasting system. Each region encapsulates:
- Geographic bounds and sub-regions
- WW3 extraction points (cemented boundary conditions)
- SWAN domain configurations (outer, inner, spot-level)
- Bathymetry source priorities
- Surf spot definitions

Usage:
    from wave_forecast_common.regions import RegionRegistry, Region

    registry = RegionRegistry()
    california = registry.load_region("california")

    # Get boundary extraction points
    points = california.get_extraction_points()

    # Get outer domain configuration
    outer_domain = california.get_domain("outer")
"""

from .models import (
    GeoBounds,
    BoundaryPoint,
    WW3ExtractionPoints,
    DomainMesh,
    Domain,
    DomainType,
    GridType,
    BathymetrySource,
    SurfSpot,
    ChannelIsland,
    Region,
)
from .registry import RegionRegistry

__all__ = [
    "GeoBounds",
    "BoundaryPoint",
    "WW3ExtractionPoints",
    "DomainMesh",
    "Domain",
    "DomainType",
    "GridType",
    "BathymetrySource",
    "SurfSpot",
    "ChannelIsland",
    "Region",
    "RegionRegistry",
]
