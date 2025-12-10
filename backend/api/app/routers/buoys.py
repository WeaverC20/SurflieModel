"""
Buoys Router

Serves real-time buoy observation data for visualization.
"""

import sys
import logging
import asyncio
from pathlib import Path
from typing import List, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Add data pipelines to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "data" / "pipelines"))

try:
    from buoy.fetcher import NDBCBuoyFetcher
except ImportError as e:
    logger.error(f"Could not import buoy fetcher: {e}")
    NDBCBuoyFetcher = None

router = APIRouter(prefix="/buoys", tags=["buoys"])


@router.get("/california")
async def get_california_buoys():
    """
    Get all California buoys with their latest observations

    Returns:
        JSON with array of buoys including:
        - station_id: Buoy ID
        - name: Buoy name
        - lat: Latitude
        - lon: Longitude
        - wave_height_m: Significant wave height in meters
        - wave_height_ft: Significant wave height in feet
        - Other observation data
    """
    if NDBCBuoyFetcher is None:
        raise HTTPException(
            status_code=500,
            detail="Buoy fetcher not available. Check server logs."
        )

    # Get list of all California buoys
    # Using the known_buoys dict from the fetcher
    known_buoys = {
        "46256": {"name": "Long Beach Channel", "lat": 33.700, "lon": -118.201},
        "46237": {"name": "San Pedro", "lat": 33.218, "lon": -118.315},
        "46221": {"name": "Santa Monica Basin", "lat": 33.855, "lon": -119.048},
        "46025": {"name": "Santa Monica", "lat": 33.749, "lon": -119.053},
        "46011": {"name": "Santa Maria", "lat": 34.878, "lon": -120.867},
        "46054": {"name": "Point Conception", "lat": 34.274, "lon": -120.477},
        "46026": {"name": "San Francisco", "lat": 37.759, "lon": -122.833},
        "46012": {"name": "Half Moon Bay", "lat": 37.361, "lon": -122.879},
        "46042": {"name": "Monterey Bay", "lat": 36.785, "lon": -122.421},
        "46240": {"name": "Harvest Platform", "lat": 34.455, "lon": -120.537},
        "46028": {"name": "Cape San Martin", "lat": 35.741, "lon": -121.884},
        "46014": {"name": "Point Arena", "lat": 39.230, "lon": -123.968},
        "46013": {"name": "Bodega Bay", "lat": 38.242, "lon": -123.316},
        "46218": {"name": "Eel River", "lat": 40.267, "lon": -124.549},
        "46022": {"name": "Eel River Canyon", "lat": 40.719, "lon": -124.524},
    }

    logger.info(f"Fetching observations from {len(known_buoys)} California buoys")

    fetcher = NDBCBuoyFetcher()
    buoys_data = []

    # Fetch data from all buoys concurrently
    async def fetch_buoy_data(station_id: str, info: Dict) -> Dict:
        """Fetch data for a single buoy"""
        try:
            obs = await fetcher.fetch_latest_observation(station_id)
            return {
                "station_id": station_id,
                "name": info["name"],
                "lat": info["lat"],
                "lon": info["lon"],
                "observation": obs,
                "status": "success"
            }
        except Exception as e:
            logger.warning(f"Failed to fetch data for buoy {station_id}: {e}")
            return {
                "station_id": station_id,
                "name": info["name"],
                "lat": info["lat"],
                "lon": info["lon"],
                "observation": None,
                "status": "error",
                "error": str(e)
            }

    # Fetch all buoy data concurrently
    tasks = [fetch_buoy_data(station_id, info) for station_id, info in known_buoys.items()]
    buoys_data = await asyncio.gather(*tasks)

    # Calculate bounds
    lats = [b["lat"] for b in buoys_data]
    lons = [b["lon"] for b in buoys_data]

    response = {
        "buoys": buoys_data,
        "count": len(buoys_data),
        "successful": sum(1 for b in buoys_data if b["status"] == "success"),
        "failed": sum(1 for b in buoys_data if b["status"] == "error"),
        "bounds": {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons),
        }
    }

    return JSONResponse(content=response)


@router.get("/metadata")
async def get_buoys_metadata():
    """
    Get metadata about available buoys

    Returns:
        JSON with buoy network information
    """
    return JSONResponse(content={
        "network": "NDBC (National Data Buoy Center)",
        "region": "California Pacific Coast",
        "update_frequency": "Real-time (hourly updates)",
        "data_types": [
            "Wave height and period",
            "Wave direction",
            "Wind speed and direction",
            "Air and water temperature",
            "Atmospheric pressure"
        ],
        "source": "https://www.ndbc.noaa.gov/"
    })
