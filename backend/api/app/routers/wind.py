"""
Wind Forecast Router

Serves wind forecast data from GFS model for visualization.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import numpy as np

logger = logging.getLogger(__name__)

# Add data pipelines to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "data" / "pipelines"))

try:
    from wind.gfs_fetcher import GFSWindFetcher
except ImportError as e:
    logger.error(f"Could not import wind fetcher: {e}")
    GFSWindFetcher = None

router = APIRouter(prefix="/wind", tags=["wind"])


@router.get("/grid")
async def get_wind_grid(
    min_lat: float = Query(32.0, description="Minimum latitude"),
    max_lat: float = Query(42.0, description="Maximum latitude"),
    min_lon: float = Query(-125.0, description="Minimum longitude"),
    max_lon: float = Query(-117.0, description="Maximum longitude"),
    forecast_hour: int = Query(0, description="Forecast hour (0, 3, 6, etc.)"),
):
    """
    Get wind forecast grid from GFS model

    Returns a grid of wind forecast data including:
    - Wind speed and direction
    - U and V components
    - Forecast and cycle times
    """
    if GFSWindFetcher is None:
        raise HTTPException(
            status_code=500,
            detail="Wind fetcher not available. Check server logs."
        )

    def serialize_array(arr):
        """Convert array to JSON-serializable list, handling NaN"""
        if isinstance(arr, list):
            # Already a list - check if it's 2D
            if arr and isinstance(arr[0], list):
                # 2D list
                return [
                    [None if (isinstance(val, float) and np.isnan(val)) else val for val in row]
                    for row in arr
                ]
            else:
                # 1D list
                return [None if (isinstance(val, float) and np.isnan(val)) else val for val in arr]
        return arr

    try:
        fetcher = GFSWindFetcher()
        grid_data = await fetcher.fetch_wind_grid(
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            forecast_hour=forecast_hour
        )

        logger.info(
            f"Serving wind grid: {len(grid_data['lat'])} x {len(grid_data['lon'])} points"
        )

        # Serialize arrays to handle NaN values
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

    except Exception as e:
        logger.error(f"Error fetching wind grid: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch wind data: {str(e)}"
        )


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
