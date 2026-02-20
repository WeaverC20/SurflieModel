"""
Configuration for ocean current tile generation

Defines regions, zoom levels, and variable resolution zones for optimized tile generation.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class RegionConfig:
    """Configuration for a geographic region"""
    name: str
    description: str
    bounds: Dict[str, float]  # min_lat, max_lat, min_lon, max_lon
    zoom_levels: range
    forecast_hours: range


@dataclass
class ZoneConfig:
    """Configuration for variable resolution zones"""
    name: str
    description: str
    buffer_miles: float
    zoom_levels: range
    priority: str


# California coastline points (for nearshore buffer calculation)
# Simplified representation from San Diego to Oregon border
CA_COASTLINE_POINTS = [
    (32.5, -117.1),   # San Diego
    (33.0, -117.3),   # Dana Point
    (33.6, -118.0),   # Huntington Beach
    (33.9, -118.4),   # Santa Monica
    (34.4, -119.7),   # Santa Barbara
    (34.9, -120.6),   # Point Conception
    (35.3, -120.8),   # San Luis Obispo
    (36.0, -121.4),   # Big Sur
    (36.6, -121.9),   # Santa Cruz
    (37.5, -122.4),   # Half Moon Bay
    (37.8, -122.5),   # San Francisco
    (38.3, -123.1),   # Bodega Bay
    (39.5, -123.8),   # Mendocino
    (40.0, -124.2),   # Humboldt
    (41.0, -124.4),   # Crescent City
    (42.0, -124.5),   # Oregon border
]


# Regional configurations
REGIONS: Dict[str, RegionConfig] = {
    "california": RegionConfig(
        name="california",
        description="California coast (San Diego to Oregon border)",
        bounds={
            "min_lat": 32.0,
            "max_lat": 42.0,
            "min_lon": -125.0,
            "max_lon": -117.0,
        },
        zoom_levels=range(0, 11),  # z0-z10
        forecast_hours=range(0, 73, 3),  # 0-72 hours, every 3 hours (24 time steps)
    ),
    "southern_california": RegionConfig(
        name="southern_california",
        description="Southern California (San Diego to Santa Barbara)",
        bounds={
            "min_lat": 32.0,
            "max_lat": 35.0,
            "min_lon": -121.0,
            "max_lon": -117.0,
        },
        zoom_levels=range(0, 11),  # z0-z10
        forecast_hours=range(0, 73, 3),  # 0-72 hours, every 3 hours
    ),
}


# Variable resolution zones for storage optimization
ZONES: Dict[str, ZoneConfig] = {
    "nearshore": ZoneConfig(
        name="nearshore",
        description="0-5 miles from California coastline (high detail)",
        buffer_miles=5.0,
        zoom_levels=range(0, 11),  # z0-z10 (full detail)
        priority="high"
    ),
    "offshore": ZoneConfig(
        name="offshore",
        description=">5 miles from coast (coarse detail)",
        buffer_miles=float('inf'),
        zoom_levels=range(0, 7),  # z0-z6 only (coarse)
        priority="low"
    ),
}


# Tile generation settings
TILE_SIZE = 256  # Standard tile size in pixels
TILE_FORMAT = "png"  # PNG for raster tiles
TILE_CACHE_DAYS = 2  # Keep tiles for 2 days

# NOAA model configurations

# DEPRECATED: RTOFS replaced by WCOFS (better resolution and accuracy for California coast)
RTOFS_CONFIG = {
    "model_name": "rtofs",
    "nomads_url": "https://nomads.ncep.noaa.gov/cgi-bin/filter_rtofs_global.pl",
    "update_time_utc": 0,
    "forecast_hours_max": 192,
    "resolution_km": 9,
    "variables": ["u_velocity", "v_velocity", "water_temp"],
    "depth_level": "surface",
    "_deprecated": "Use WCOFS_CONFIG instead. RTOFS is 9km global; WCOFS is 4km regional with better nearshore accuracy.",
}

WCOFS_CONFIG = {
    "model_name": "wcofs",
    "thredds_base": "https://opendap.co-ops.nos.noaa.gov/thredds",
    "update_times_utc": [3, 9, 15, 21],  # 4x daily cycles
    "forecast_hours_max": 72,  # 3-day forecast
    "nowcast_hours": 24,
    "resolution_km": 4,
    "variables": ["u_sur_eastward", "v_sur_northward"],
    "file_types": {
        "2ds": "2D surface fields (hourly)",
        "fields": "3D fields (3-hourly)",
        "regulargrid": "Interpolated to regular grid",
    },
    "depth_level": "surface",
}

HFRADAR_CONFIG = {
    "model_name": "hfradar",
    "erddap_base": "https://coastwatch.pfeg.noaa.gov/erddap/griddap",
    "datasets": {
        "6km": "ucsdHfrW6",
        "2km": "ucsdHfrW2",
        "1km": "ucsdHfrW1",
    },
    "default_resolution": "6km",
    "update_frequency": "hourly",
    "variables": ["water_u", "water_v"],
    "type": "observation",  # No forecast capability
}


# Current speed to color mapping (for visualization)
CURRENT_SPEED_COLORS = {
    # Speed in knots -> RGB color
    0.0: (0, 100, 255),      # Blue - very slow
    0.5: (0, 200, 150),      # Cyan - slow
    1.0: (100, 255, 100),    # Green - moderate
    1.5: (255, 255, 0),      # Yellow - fast
    2.0: (255, 150, 0),      # Orange - very fast
    2.5: (255, 50, 0),       # Red-orange - extreme
    3.0: (255, 0, 0),        # Red - extreme
}


# Cache directory structure
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CACHE_DIR = BASE_DIR / "data" / "cache" / "tiles"
GRIB_CACHE_DIR = BASE_DIR / "data" / "cache" / "grib"


def get_tile_path(model: str, variable: str, time: str, z: int, x: int, y: int) -> Path:
    """
    Get the filesystem path for a tile

    Args:
        model: Model name (rtofs, wcofs)
        variable: Variable name (currents, temp, etc.)
        time: ISO timestamp (2025-12-10T00)
        z, x, y: Tile coordinates

    Returns:
        Path to tile file
    """
    return CACHE_DIR / model / variable / time / str(z) / str(x) / f"{y}.{TILE_FORMAT}"


def get_tile_url(model: str, variable: str, time: str, z: int, x: int, y: int) -> str:
    """
    Get the API URL for a tile

    Args:
        model: Model name (rtofs, wcofs)
        variable: Variable name (currents, temp, etc.)
        time: ISO timestamp (2025-12-10T00)
        z, x, y: Tile coordinates

    Returns:
        URL path for tile
    """
    return f"/api/tiles/{model}/{variable}/{time}/{z}/{x}/{y}.{TILE_FORMAT}"


# Ensure cache directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
GRIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
