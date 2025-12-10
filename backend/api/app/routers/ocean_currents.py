"""
Ocean Currents Router

Serves raw ocean current grid data for simple visualization.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import numpy as np

logger = logging.getLogger(__name__)

# Add data pipelines to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "data" / "pipelines"))

try:
    from ocean_tiles.rtofs_fetcher import RTOFSFetcher
    from ocean_tiles.config import REGIONS
except ImportError as e:
    logger.error(f"Could not import ocean_tiles: {e}")
    RTOFSFetcher = None
    REGIONS = None

router = APIRouter(prefix="/ocean-currents", tags=["ocean-currents"])


@router.get("/grid")
async def get_ocean_current_grid(
    region: str = "california",
    forecast_hour: int = 0,
    model_date: Optional[str] = None
):
    """
    Get raw ocean current grid data for visualization

    Args:
        region: Region name (default: california)
        forecast_hour: Forecast hour (0-192, default: 0)
        model_date: Model run date in YYYY-MM-DD format (default: today)

    Returns:
        JSON with:
        - lats: Array of latitudes
        - lons: Array of longitudes
        - current_speed: 2D array of current speeds (m/s)
        - current_direction: 2D array of current directions (degrees)
        - metadata: Model run info
    """
    if RTOFSFetcher is None:
        raise HTTPException(
            status_code=500,
            detail="Ocean current fetcher not available. Check server logs."
        )

    if REGIONS is None or region not in REGIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid region: {region}. Available: {list(REGIONS.keys()) if REGIONS else []}"
        )

    # Parse model date
    if model_date:
        try:
            model_dt = datetime.strptime(model_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Use YYYY-MM-DD"
            )
    else:
        model_dt = None

    # Fetch data
    logger.info(f"Fetching ocean current grid for {region}, forecast hour {forecast_hour}")

    fetcher = RTOFSFetcher()
    data = await fetcher.fetch_current_grid(
        region_name=region,
        forecast_hour=forecast_hour,
        model_date=model_dt
    )

    if data is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch ocean current data"
        )

    # Convert numpy arrays to lists for JSON serialization
    # Handle NaN values by converting to None
    def serialize_array(arr):
        """Convert numpy array to JSON-serializable list, handling NaN"""
        if isinstance(arr, np.ndarray):
            # Replace NaN with None for JSON
            arr_list = arr.tolist()
            if arr.ndim == 2:
                # 2D array
                return [
                    [None if (isinstance(val, float) and np.isnan(val)) else val for val in row]
                    for row in arr_list
                ]
            else:
                # 1D array
                return [None if (isinstance(val, float) and np.isnan(val)) else val for val in arr_list]
        return arr

    response_data = {
        "lats": serialize_array(data.get("lats")),
        "lons": serialize_array(data.get("lons")),
        "current_speed": serialize_array(data.get("current_speed")),
        "current_direction": serialize_array(data.get("current_direction")),
        "metadata": data.get("metadata", {}),
        "bounds": {
            "min_lat": float(np.nanmin(data["lats"])),
            "max_lat": float(np.nanmax(data["lats"])),
            "min_lon": float(np.nanmin(data["lons"])),
            "max_lon": float(np.nanmax(data["lons"])),
        },
        "shape": {
            "height": data["current_speed"].shape[0],
            "width": data["current_speed"].shape[1]
        }
    }

    return JSONResponse(content=response_data)


@router.get("/metadata")
async def get_metadata():
    """
    Get metadata about available regions and forecast parameters

    Returns:
        JSON with available regions and their configurations
    """
    if REGIONS is None:
        raise HTTPException(
            status_code=500,
            detail="Region configuration not available"
        )

    regions_info = {}
    for name, config in REGIONS.items():
        regions_info[name] = {
            "name": config.name,
            "description": config.description,
            "bounds": config.bounds,
            "zoom_levels": [config.zoom_levels.start, config.zoom_levels.stop - 1],
            "forecast_hours": [config.forecast_hours.start, config.forecast_hours.stop - 1, config.forecast_hours.step]
        }

    return JSONResponse(content={
        "regions": regions_info,
        "model": "RTOFS (Real-Time Ocean Forecast System)",
        "resolution": "~9km (1/12 degree)",
        "update_frequency": "Daily at 00Z",
        "forecast_length": "192 hours (8 days)"
    })
