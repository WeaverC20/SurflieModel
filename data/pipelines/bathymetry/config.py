"""
Configuration for bathymetry data pipeline.

Defines regions, data sources, and SWAN domain specifications.
"""

from pathlib import Path
from typing import Dict, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_BATHYMETRY_DIR = DATA_DIR / "raw" / "bathymetry"
PROCESSED_BATHYMETRY_DIR = DATA_DIR / "processed" / "bathymetry"
CACHE_DIR = DATA_DIR / "cache" / "swan"

# Raw data directories
GEBCO_DIR = RAW_BATHYMETRY_DIR / "gebco_2024"
NCEI_DIR = RAW_BATHYMETRY_DIR / "ncei_crm"
USACE_DIR = RAW_BATHYMETRY_DIR / "USACE_CA_DEM_2009_9488"

# Processed data directories
STITCHED_DIR = PROCESSED_BATHYMETRY_DIR / "swan_stitched"
REGIONAL_DIR = PROCESSED_BATHYMETRY_DIR / "regional"
VIZ_DIR = PROCESSED_BATHYMETRY_DIR / "visualizations"

# Region definitions: (lat_min, lat_max, lon_min, lon_max)
REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    "california": (32.0, 42.0, -126.0, -117.0),
    "socal": (32.0, 35.0, -122.0, -117.0),
    "norcal": (36.0, 42.0, -126.0, -121.0),
    "central": (34.0, 38.0, -124.0, -119.0),
    "channel_islands": (33.0, 34.5, -121.0, -118.5),
    "la_area": (33.5, 34.2, -119.0, -117.5),
    "san_diego": (32.5, 33.3, -118.0, -117.0),
}

# SWAN domain configuration
SWAN_CONFIG = {
    # Outer domain: GEBCO data (25km → 3km offshore)
    "outer_domain": {
        "source": "GEBCO 2024",
        "resolution_m": 450,  # ~15 arc-seconds
        "offshore_start_km": 25,
        "offshore_end_km": 3,
    },
    # Inner domain: NCEI CRM data (3km → 500m offshore)
    "inner_domain": {
        "source": "NCEI Coastal Relief Model",
        "resolution_m": 90,  # ~3 arc-seconds
        "offshore_start_km": 3,
        "offshore_end_km": 0.5,
    },
    # Surf zone: existing USACE LiDAR (500m → beach)
    "surf_zone": {
        "source": "USACE NCMP LiDAR 2009",
        "resolution_m": 1,
        "offshore_start_km": 0.5,
        "offshore_end_km": 0,
        "note": "Not used in SWAN - separate surf zone physics model",
    },
    # Grid specifications
    "output_format": "swan",  # .grd format for SWAN
    "vertical_datum": "MSL",  # Mean Sea Level
    "coordinate_system": "WGS84",  # EPSG:4326
}

# Data source URLs
DATA_SOURCES = {
    "gebco": {
        "base_url": "https://www.gebco.net/data_and_products/gridded_bathymetry_data/",
        "download_url": "https://download.gebco.net/",
        "format": "netcdf",
        "resolution": "15 arc-second",
        "notes": "Requires manual region selection via web interface or use OpenDAP",
    },
    "ncei_crm": {
        "thredds_url": "https://www.ngdc.noaa.gov/thredds/catalog/crm/catalog.html",
        "opendap_base": "https://www.ngdc.noaa.gov/thredds/dodsC/crm/",
        "format": "netcdf",
        "resolution": "3 arc-second (~90m)",
        "tiles": {
            # California is covered by these CRM volumes
            "vol6": "crm_vol6.nc",  # Southern California
            "vol7": "crm_vol7.nc",  # Central California
            "vol8": "crm_vol8.nc",  # Northern California
        },
        "notes": "Can access via OPeNDAP for subsetting without full download",
    },
}

# Approximate distance to coast for key surf spots (km)
# Used for validation and visualization
SURF_SPOTS = {
    "huntington_pier": {"lat": 33.655, "lon": -117.999, "name": "Huntington Pier"},
    "trestles": {"lat": 33.382, "lon": -117.589, "name": "Trestles"},
    "malibu": {"lat": 34.035, "lon": -118.677, "name": "Malibu Surfrider"},
    "rincon": {"lat": 34.374, "lon": -119.476, "name": "Rincon"},
    "steamer_lane": {"lat": 36.951, "lon": -122.026, "name": "Steamer Lane"},
    "mavericks": {"lat": 37.495, "lon": -122.496, "name": "Mavericks"},
    "ocean_beach_sf": {"lat": 37.759, "lon": -122.510, "name": "Ocean Beach SF"},
}
