"""
Wind Forecast Router

Serves wind forecast data from GFS model for visualization.
Reads from local NetCDF files first, falls back to external API if unavailable.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Add data pipelines to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "data" / "pipelines"))

# Local data paths
LOCAL_DATA_DIR = project_root / "data" / "downloaded_weather_data"
LOCAL_WIND_DIR = LOCAL_DATA_DIR / "wind"

try:
    from wind.gfs_fetcher import GFSWindFetcher
except ImportError as e:
    logger.error(f"Could not import wind fetcher: {e}")
    GFSWindFetcher = None

router = APIRouter(prefix="/wind", tags=["wind"])


def get_latest_wind_file() -> Optional[Path]:
    """Get the most recent local wind NetCDF file."""
    if not LOCAL_WIND_DIR.exists():
        return None
    files = sorted(LOCAL_WIND_DIR.glob("gfs_*.nc"))
    return files[-1] if files else None


def read_local_wind_grid(
    forecast_hour: int,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> Optional[Dict[str, Any]]:
    """Read wind grid from local NetCDF file."""
    filepath = get_latest_wind_file()
    if not filepath:
        return None

    try:
        ds = xr.open_dataset(filepath)

        # Get cycle time and calculate forecast hours
        cycle_time = datetime.fromisoformat(ds.attrs.get("cycle_time", ""))
        times = pd.to_datetime(ds.time.values)
        forecast_hours = [(t - cycle_time).total_seconds() / 3600 for t in times]

        # Find closest forecast hour
        time_idx = min(
            range(len(forecast_hours)),
            key=lambda i: abs(forecast_hours[i] - forecast_hour)
        )

        # Subset by lat/lon bounds
        ds_subset = ds.sel(
            lat=slice(min_lat, max_lat),
            lon=slice(min_lon, max_lon)
        )

        lat = ds_subset.lat.values.tolist()
        lon = ds_subset.lon.values.tolist()

        def extract_var(var_name: str) -> List[List[float]]:
            """Extract 2D variable at time index."""
            if var_name not in ds_subset:
                return [[None] * len(lon) for _ in range(len(lat))]
            data = ds_subset[var_name].isel(time=time_idx).values
            return [
                [None if np.isnan(v) else float(v) for v in row]
                for row in data
            ]

        forecast_time = times[time_idx]

        result = {
            "lat": lat,
            "lon": lon,
            "u_wind": extract_var("u_wind"),
            "v_wind": extract_var("v_wind"),
            "wind_speed": extract_var("wind_speed"),
            "wind_direction": extract_var("wind_direction"),
            "forecast_time": forecast_time.isoformat(),
            "cycle_time": cycle_time.isoformat(),
            "forecast_hour": int(forecast_hours[time_idx]),
            "resolution_deg": 0.25,
            "model": "GFS",
            "source": "local",
            "units": {
                "wind_speed": "m/s",
                "wind_direction": "degrees",
                "u_wind": "m/s",
                "v_wind": "m/s",
                "lat": "degrees",
                "lon": "degrees"
            }
        }

        ds.close()
        logger.info(f"Loaded wind data from local file: {filepath.name}")
        return result

    except Exception as e:
        logger.error(f"Error reading local wind data: {e}")
        return None


@router.get("/grid")
async def get_wind_grid(
    min_lat: float = Query(32.0, description="Minimum latitude"),
    max_lat: float = Query(42.0, description="Maximum latitude"),
    min_lon: float = Query(-125.0, description="Minimum longitude"),
    max_lon: float = Query(-117.0, description="Maximum longitude"),
    forecast_hour: int = Query(0, description="Forecast hour (0, 3, 6, etc.)"),
):
    """
    Get wind forecast grid from GFS model.

    Reads from local NetCDF files first (fast), falls back to external API if unavailable.

    Returns a grid of wind forecast data including:
    - Wind speed and direction
    - U and V components
    - Forecast and cycle times
    """
    def serialize_array(arr):
        """Convert array to JSON-serializable list, handling NaN"""
        if isinstance(arr, list):
            if arr and isinstance(arr[0], list):
                return [
                    [None if (isinstance(val, float) and np.isnan(val)) else val for val in row]
                    for row in arr
                ]
            else:
                return [None if (isinstance(val, float) and np.isnan(val)) else val for val in arr]
        return arr

    # Try local data first
    grid_data = read_local_wind_grid(
        forecast_hour=forecast_hour,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
    )

    # Fall back to external API if local data unavailable
    if grid_data is None:
        if GFSWindFetcher is None:
            raise HTTPException(
                status_code=500,
                detail="No local wind data and wind fetcher not available."
            )

        try:
            logger.info("No local wind data, fetching from external API...")
            fetcher = GFSWindFetcher()
            grid_data = await fetcher.fetch_wind_grid(
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
                forecast_hour=forecast_hour
            )
        except Exception as e:
            logger.error(f"Error fetching wind grid: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch wind data: {str(e)}"
            )

    logger.info(
        f"Serving wind grid: {len(grid_data['lat'])} x {len(grid_data['lon'])} points"
    )

    serialized_data = {
        "lat": serialize_array(grid_data["lat"]),
        "lon": serialize_array(grid_data["lon"]),
        "u_wind": serialize_array(grid_data["u_wind"]),
        "v_wind": serialize_array(grid_data["v_wind"]),
        "wind_speed": serialize_array(grid_data["wind_speed"]),
        "wind_direction": serialize_array(grid_data["wind_direction"]),
        "forecast_time": grid_data["forecast_time"],
        "cycle_time": grid_data["cycle_time"],
        "forecast_hour": grid_data["forecast_hour"],
        "resolution_deg": grid_data["resolution_deg"],
        "model": grid_data["model"],
        "units": grid_data["units"]
    }

    return JSONResponse(content=serialized_data)


@router.get("/metadata")
async def get_wind_metadata():
    """
    Get metadata about wind forecast data

    Returns:
        JSON with wind forecast information
    """
    return JSONResponse(content={
        "model": "GFS (Global Forecast System)",
        "provider": "NOAA/NCEP",
        "resolution": "0.25 degrees (~25 km)",
        "update_frequency": "4 times daily (00, 06, 12, 18 UTC)",
        "forecast_range": "384 hours (16 days)",
        "variables": [
            "Wind speed at 10m (m/s)",
            "Wind direction (degrees)",
            "U and V components (m/s)"
        ],
        "source": "https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast"
    })
