"""
Ocean Currents Router

Serves ocean current data from WCOFS (forecast) and HF Radar (real-time observations).

Data sources:
- WCOFS: ~4km resolution, 72hr forecast, 4x daily (ROMS with HF radar assimilation)
- HF Radar: ~6km resolution, hourly observations, ground truth surface currents
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
    from ocean_tiles.wcofs_fetcher import WCOFSFetcher
    from ocean_tiles.hfradar_fetcher import HFRadarFetcher
    from ocean_tiles.config import REGIONS
except ImportError as e:
    logger.error(f"Could not import ocean_tiles: {e}")
    WCOFSFetcher = None
    HFRadarFetcher = None
    REGIONS = None

router = APIRouter(prefix="/ocean-currents", tags=["ocean-currents"])


def serialize_array(arr):
    """Convert numpy array to JSON-serializable list, handling NaN."""
    if isinstance(arr, np.ndarray):
        arr_list = arr.tolist()
        if arr.ndim == 2:
            return [
                [None if (isinstance(val, float) and (val != val)) else val for val in row]
                for row in arr_list
            ]
        else:
            return [None if (isinstance(val, float) and (val != val)) else val for val in arr_list]
    return arr


@router.get("/grid")
async def get_ocean_current_grid(
    region: str = "california",
    forecast_hour: int = 0,
    model_date: Optional[str] = None
):
    """
    Get WCOFS ocean current forecast grid data for visualization.

    Args:
        region: Region name (default: california)
        forecast_hour: Forecast hour (0-72, default: 0)
        model_date: Model run date in YYYY-MM-DD format (default: today)

    Returns:
        JSON with lats, lons, current_speed, current_direction, metadata
    """
    if WCOFSFetcher is None:
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

    # Clamp forecast hour to WCOFS range
    forecast_hour = min(max(forecast_hour, 0), 72)

    logger.info(f"Fetching WCOFS current grid for {region}, forecast hour {forecast_hour}")

    fetcher = WCOFSFetcher()
    data = await fetcher.fetch_current_grid(
        region_name=region,
        forecast_hour=forecast_hour,
        model_date=model_dt
    )

    if data is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch WCOFS ocean current data"
        )

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


@router.get("/hfradar")
async def get_hfradar_grid(
    region: str = "california",
    resolution: str = "6km",
    observation_time: Optional[str] = None
):
    """
    Get real-time HF Radar surface current observations.

    Args:
        region: Region name (default: california)
        resolution: Data resolution - '6km', '2km', or '1km' (default: 6km)
        observation_time: Specific time in YYYY-MM-DDTHH:00:00Z format (default: latest)

    Returns:
        JSON with lats, lons, current_speed, current_direction, metadata
    """
    if HFRadarFetcher is None:
        raise HTTPException(
            status_code=500,
            detail="HF Radar fetcher not available. Check server logs."
        )

    if REGIONS is None or region not in REGIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid region: {region}. Available: {list(REGIONS.keys()) if REGIONS else []}"
        )

    # Parse observation time
    obs_dt = None
    if observation_time:
        try:
            obs_dt = datetime.fromisoformat(observation_time.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid time format. Use YYYY-MM-DDTHH:00:00Z"
            )

    logger.info(f"Fetching HF Radar ({resolution}) data for {region}")

    try:
        fetcher = HFRadarFetcher(resolution=resolution)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    data = await fetcher.fetch_current_grid(
        region_name=region,
        observation_time=obs_dt
    )

    if data is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch HF Radar data"
        )

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
    Get metadata about available current data sources and parameters.

    Returns:
        JSON with available regions, model info, and data source details.
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
        "sources": {
            "wcofs": {
                "name": "WCOFS (West Coast Operational Forecast System)",
                "type": "forecast",
                "resolution": "~4km",
                "update_frequency": "4x daily (03Z, 09Z, 15Z, 21Z)",
                "forecast_length": "72 hours (3 days)",
                "model_basis": "ROMS with 4DVAR data assimilation",
                "assimilated_data": ["HF radar surface currents", "satellite SST", "satellite altimetry"],
                "endpoint": "/api/ocean-currents/grid",
            },
            "hfradar": {
                "name": "HF Radar Surface Currents",
                "type": "observation",
                "resolutions": {"6km": "Broad coverage", "2km": "Higher resolution", "1km": "Highest resolution"},
                "update_frequency": "Hourly",
                "forecast_length": "None (real-time observations only)",
                "source": "SCCOOS + CeNCOOS (62 radars along California coast)",
                "endpoint": "/api/ocean-currents/hfradar",
            },
        },
    })
