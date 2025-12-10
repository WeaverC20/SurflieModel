"""
Wave Forecast Router

Serves wave forecast data from WaveWatch III model for visualization.
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
    from wave.wavewatch_fetcher import WaveWatchFetcher
except ImportError as e:
    logger.error(f"Could not import wave fetcher: {e}")
    WaveWatchFetcher = None

router = APIRouter(prefix="/waves", tags=["waves"])


@router.get("/grid")
async def get_wave_grid(
    min_lat: float = Query(32.0, description="Minimum latitude"),
    max_lat: float = Query(42.0, description="Maximum latitude"),
    min_lon: float = Query(-125.0, description="Minimum longitude"),
    max_lon: float = Query(-117.0, description="Maximum longitude"),
    forecast_hour: int = Query(0, description="Forecast hour (0, 3, 6, etc.)"),
):
    """
    Get wave forecast grid from WaveWatch III model

    Returns a grid of wave forecast data including:
    - Significant wave height
    - Peak wave period
    - Mean wave direction
    - Wind sea and swell components
    - Forecast and cycle times
    """
    if WaveWatchFetcher is None:
        raise HTTPException(
            status_code=500,
            detail="Wave fetcher not available. Check server logs."
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
        fetcher = WaveWatchFetcher()
        grid_data = await fetcher.fetch_wave_grid(
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            forecast_hour=forecast_hour
        )

        logger.info(
            f"Serving wave grid: {len(grid_data['lat'])} x {len(grid_data['lon'])} points"
        )

        # Serialize arrays to handle NaN values
        serialized_data = {
            "lat": serialize_array(grid_data["lat"]),
            "lon": serialize_array(grid_data["lon"]),
            "significant_wave_height": serialize_array(grid_data["significant_wave_height"]),
            "peak_wave_period": serialize_array(grid_data["peak_wave_period"]),
            "mean_wave_direction": serialize_array(grid_data["mean_wave_direction"]),
            "wind_sea_height": serialize_array(grid_data["wind_sea_height"]),
            "swell_height": serialize_array(grid_data["swell_height"]),
            "forecast_time": grid_data["forecast_time"],
            "cycle_time": grid_data["cycle_time"],
            "forecast_hour": grid_data["forecast_hour"],
            "resolution_deg": grid_data["resolution_deg"],
            "model": grid_data["model"],
            "units": grid_data["units"]
        }

        return JSONResponse(content=serialized_data)

    except Exception as e:
        logger.error(f"Error fetching wave grid: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch wave data: {str(e)}"
        )


@router.get("/metadata")
async def get_wave_metadata():
    """
    Get metadata about wave forecast data

    Returns:
        JSON with wave forecast information
    """
    return JSONResponse(content={
        "model": "WaveWatch III",
        "provider": "NOAA/NCEP",
        "resolution": "0.25 degrees (~25 km) regional",
        "update_frequency": "4 times daily (00, 06, 12, 18 UTC)",
        "forecast_range": "180 hours (7.5 days)",
        "variables": [
            "Significant wave height (m)",
            "Peak wave period (s)",
            "Mean wave direction (degrees)",
            "Wind sea height (m)",
            "Swell height (m)"
        ],
        "source": "https://polar.ncep.noaa.gov/waves/"
    })
