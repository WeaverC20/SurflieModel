"""
Buoys Router

Serves real-time buoy observation data for visualization.
Supports both NDBC and CDIP buoy networks with spectral/partitioned swell data.
Reads from local CSV files first, falls back to external API if unavailable.
"""

import sys
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import pandas as pd

logger = logging.getLogger(__name__)

# Add data pipelines to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "data" / "pipelines"))

# Local data paths
LOCAL_DATA_DIR = project_root / "data" / "downloaded_weather_data"
LOCAL_BUOYS_DIR = LOCAL_DATA_DIR / "buoys"

try:
    from buoy.fetcher import NDBCBuoyFetcher, CDIPBuoyFetcher
except ImportError as e:
    logger.error(f"Could not import buoy fetcher: {e}")
    NDBCBuoyFetcher = None
    CDIPBuoyFetcher = None

router = APIRouter(prefix="/buoys", tags=["buoys"])


# Known NDBC California buoys
NDBC_CALIFORNIA_BUOYS = {
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


def read_local_buoy_data() -> Optional[Dict[str, Any]]:
    """Read buoy data from local CSV files."""
    if not LOCAL_BUOYS_DIR.exists():
        return None

    buoys_data = []

    def safe_float(val):
        if pd.isna(val):
            return None
        return float(val)

    # Load NDBC data
    ndbc_files = sorted(LOCAL_BUOYS_DIR.glob("ndbc_*.csv"))
    if ndbc_files:
        ndbc_file = ndbc_files[-1]
        try:
            df = pd.read_csv(ndbc_file)
            logger.info(f"Loaded NDBC data from: {ndbc_file.name}")

            for _, row in df.iterrows():
                station_id = str(row.get("station_id", ""))
                info = NDBC_CALIFORNIA_BUOYS.get(station_id, {})

                buoy_entry = {
                    "station_id": station_id,
                    "name": info.get("name", f"NDBC {station_id}"),
                    "network": "NDBC",
                    "lat": info.get("lat", 0),
                    "lon": info.get("lon", 0),
                    "observation": {
                        "timestamp": row.get("timestamp"),
                        "wave_height_m": safe_float(row.get("wave_height_m")),
                        "dominant_wave_period_s": safe_float(row.get("dominant_period_s")),
                        "mean_wave_direction_deg": safe_float(row.get("mean_direction_deg")),
                        "water_temp_c": safe_float(row.get("water_temp_c")),
                        "wind_speed_ms": safe_float(row.get("wind_speed_ms")),
                        "wind_direction_deg": safe_float(row.get("wind_direction_deg")),
                    },
                    "spectral_data": None,
                    "status": "success"
                }

                if pd.notna(row.get("swell_height_m")):
                    buoy_entry["spectral_data"] = {
                        "swell": {
                            "height_m": safe_float(row.get("swell_height_m")),
                            "period_s": safe_float(row.get("swell_period_s")),
                            "direction_deg": safe_float(row.get("swell_direction_deg")),
                        }
                    }

                buoys_data.append(buoy_entry)

        except Exception as e:
            logger.error(f"Error reading NDBC file: {e}")

    # Load CDIP data
    cdip_files = sorted(LOCAL_BUOYS_DIR.glob("cdip_*.csv"))
    if cdip_files:
        cdip_file = cdip_files[-1]
        try:
            df = pd.read_csv(cdip_file)
            logger.info(f"Loaded CDIP data from: {cdip_file.name}")

            for _, row in df.iterrows():
                station_id = str(row.get("station_id", ""))

                buoy_entry = {
                    "station_id": f"CDIP-{station_id}",
                    "name": f"CDIP {station_id}",
                    "network": "CDIP",
                    "lat": safe_float(row.get("lat", 0)) or 0,
                    "lon": safe_float(row.get("lon", 0)) or 0,
                    "observation": {
                        "timestamp": row.get("timestamp"),
                        "wave_height_m": safe_float(row.get("wave_height_m")),
                        "dominant_period_s": safe_float(row.get("dominant_period_s")),
                        "peak_direction_deg": safe_float(row.get("peak_direction_deg")),
                        "mean_direction_deg": safe_float(row.get("mean_direction_deg")),
                        "water_temp_c": safe_float(row.get("water_temp_c")),
                    },
                    "spectral_data": None,
                    "status": "success"
                }

                buoys_data.append(buoy_entry)

        except Exception as e:
            logger.error(f"Error reading CDIP file: {e}")

    if not buoys_data:
        return None

    lats = [b["lat"] for b in buoys_data if b["lat"]]
    lons = [b["lon"] for b in buoys_data if b["lon"]]

    return {
        "buoys": buoys_data,
        "count": len(buoys_data),
        "successful": sum(1 for b in buoys_data if b["status"] == "success"),
        "failed": sum(1 for b in buoys_data if b["status"] == "error"),
        "networks": {
            "ndbc": sum(1 for b in buoys_data if b["network"] == "NDBC"),
            "cdip": sum(1 for b in buoys_data if b["network"] == "CDIP"),
        },
        "bounds": {
            "min_lat": min(lats) if lats else 32.0,
            "max_lat": max(lats) if lats else 42.0,
            "min_lon": min(lons) if lons else -125.0,
            "max_lon": max(lons) if lons else -117.0,
        },
        "source": "local"
    }


@router.get("/california")
async def get_california_buoys(
    include_cdip: bool = Query(True, description="Include CDIP buoys"),
    include_ndbc: bool = Query(True, description="Include NDBC buoys"),
    include_spectral: bool = Query(False, description="Include spectral/partitioned swell data"),
):
    """
    Get all California buoys with their latest observations from both NDBC and CDIP networks.

    Reads from local CSV files first (fast), falls back to external API if unavailable.

    Returns:
        JSON with array of buoys including:
        - station_id: Buoy ID
        - name: Buoy name
        - network: 'NDBC' or 'CDIP'
        - lat: Latitude
        - lon: Longitude
        - wave_height_m: Significant wave height in meters
        - wave_height_ft: Significant wave height in feet
        - spectral_data: (optional) Partitioned swell/wind wave components
        - Other observation data
    """
    # Try local data first (fast path - no spectral data support)
    if not include_spectral:
        local_data = read_local_buoy_data()
        if local_data:
            # Filter by network if needed
            if not include_ndbc:
                local_data["buoys"] = [b for b in local_data["buoys"] if b["network"] != "NDBC"]
            if not include_cdip:
                local_data["buoys"] = [b for b in local_data["buoys"] if b["network"] != "CDIP"]
            local_data["count"] = len(local_data["buoys"])
            local_data["networks"]["ndbc"] = sum(1 for b in local_data["buoys"] if b["network"] == "NDBC")
            local_data["networks"]["cdip"] = sum(1 for b in local_data["buoys"] if b["network"] == "CDIP")
            logger.info(f"Serving buoy data from local files: {local_data['count']} buoys")
            return JSONResponse(content=local_data)

    # Fall back to external API
    if NDBCBuoyFetcher is None and CDIPBuoyFetcher is None:
        raise HTTPException(
            status_code=500,
            detail="No local buoy data and buoy fetchers not available."
        )

    logger.info("Fetching buoy data from external API...")

    buoys_data = []
    tasks = []

    # NDBC fetcher and tasks
    ndbc_fetcher = NDBCBuoyFetcher() if NDBCBuoyFetcher and include_ndbc else None

    async def fetch_ndbc_buoy(station_id: str, info: Dict) -> Dict:
        """Fetch data for a single NDBC buoy"""
        try:
            obs = await ndbc_fetcher.fetch_latest_observation(station_id)

            # Optionally fetch spectral data
            spectral_data = None
            if include_spectral:
                try:
                    spectral_data = await ndbc_fetcher.fetch_spectral_wave_data(station_id)
                except Exception as e:
                    logger.warning(f"Could not fetch spectral data for NDBC {station_id}: {e}")

            return {
                "station_id": station_id,
                "name": info["name"],
                "network": "NDBC",
                "lat": info["lat"],
                "lon": info["lon"],
                "observation": obs,
                "spectral_data": spectral_data,
                "status": "success"
            }
        except Exception as e:
            logger.warning(f"Failed to fetch data for NDBC buoy {station_id}: {e}")
            return {
                "station_id": station_id,
                "name": info["name"],
                "network": "NDBC",
                "lat": info["lat"],
                "lon": info["lon"],
                "observation": None,
                "spectral_data": None,
                "status": "error",
                "error": str(e)
            }

    # Add NDBC tasks
    if ndbc_fetcher:
        for station_id, info in NDBC_CALIFORNIA_BUOYS.items():
            tasks.append(fetch_ndbc_buoy(station_id, info))

    # CDIP fetcher and tasks
    cdip_fetcher = CDIPBuoyFetcher() if CDIPBuoyFetcher and include_cdip else None

    async def fetch_cdip_buoy(station_id: str, info: Dict) -> Dict:
        """Fetch data for a single CDIP buoy"""
        try:
            obs = await cdip_fetcher.fetch_latest_observation(station_id)

            # Optionally fetch partitioned data
            spectral_data = None
            if include_spectral:
                try:
                    spectral_data = await cdip_fetcher.fetch_partitioned_wave_data(station_id)
                except Exception as e:
                    logger.warning(f"Could not fetch partitioned data for CDIP {station_id}: {e}")

            return {
                "station_id": f"CDIP-{station_id}",
                "name": info["name"],
                "network": "CDIP",
                "lat": info["lat"],
                "lon": info["lon"],
                "depth_m": info.get("depth_m"),
                "observation": obs,
                "spectral_data": spectral_data,
                "status": "success" if obs.get("status") == "success" else "error",
                "error": obs.get("error") if obs.get("status") != "success" else None
            }
        except Exception as e:
            logger.warning(f"Failed to fetch data for CDIP buoy {station_id}: {e}")
            return {
                "station_id": f"CDIP-{station_id}",
                "name": info["name"],
                "network": "CDIP",
                "lat": info["lat"],
                "lon": info["lon"],
                "depth_m": info.get("depth_m"),
                "observation": None,
                "spectral_data": None,
                "status": "error",
                "error": str(e)
            }

    # Add CDIP tasks
    if cdip_fetcher:
        cdip_ca_buoys = cdip_fetcher.get_california_buoys()
        logger.info(f"Found {len(cdip_ca_buoys)} California CDIP buoys")
        for station_id, info in cdip_ca_buoys.items():
            tasks.append(fetch_cdip_buoy(station_id, info))

    # Fetch all buoy data concurrently
    logger.info(f"Fetching observations from {len(tasks)} total buoys")
    buoys_data = await asyncio.gather(*tasks)

    # Calculate bounds
    lats = [b["lat"] for b in buoys_data]
    lons = [b["lon"] for b in buoys_data]

    response = {
        "buoys": buoys_data,
        "count": len(buoys_data),
        "successful": sum(1 for b in buoys_data if b["status"] == "success"),
        "failed": sum(1 for b in buoys_data if b["status"] == "error"),
        "networks": {
            "ndbc": sum(1 for b in buoys_data if b["network"] == "NDBC"),
            "cdip": sum(1 for b in buoys_data if b["network"] == "CDIP"),
        },
        "bounds": {
            "min_lat": min(lats) if lats else 32.0,
            "max_lat": max(lats) if lats else 42.0,
            "min_lon": min(lons) if lons else -125.0,
            "max_lon": max(lons) if lons else -117.0,
        }
    }

    return JSONResponse(content=response)


@router.get("/metadata")
async def get_buoys_metadata():
    """
    Get metadata about available buoys and networks

    Returns:
        JSON with buoy network information
    """
    return JSONResponse(content={
        "networks": [
            {
                "name": "NDBC",
                "full_name": "National Data Buoy Center",
                "source": "https://www.ndbc.noaa.gov/",
                "update_frequency": "Real-time (hourly updates)",
                "spectral_data": True,
                "partitioned_swell": True,
                "description": "NOAA's network of moored buoys providing meteorological and oceanographic data"
            },
            {
                "name": "CDIP",
                "full_name": "Coastal Data Information Program",
                "source": "https://cdip.ucsd.edu/",
                "update_frequency": "Real-time (30-minute updates)",
                "spectral_data": True,
                "partitioned_swell": True,
                "description": "Scripps Institution of Oceanography network with high-quality directional wave data"
            }
        ],
        "region": "California Pacific Coast",
        "data_types": [
            "Wave height and period",
            "Wave direction (peak and mean)",
            "Partitioned swell components (height, period, direction)",
            "Wind wave components",
            "Spectral energy density",
            "Wind speed and direction",
            "Air and water temperature",
            "Atmospheric pressure"
        ],
        "spectral_parameters": {
            "swell": "Long-period waves from distant storms",
            "wind_waves": "Locally generated short-period waves",
            "partitions": "Multiple swell trains from different source regions (CDIP)",
        }
    })


@router.get("/{station_id}/spectral")
async def get_buoy_spectral_data(
    station_id: str,
    network: str = Query("ndbc", description="Buoy network: 'ndbc' or 'cdip'"),
):
    """
    Get partitioned spectral wave data for a specific buoy.

    For NDBC buoys: Returns swell vs wind wave separation
    For CDIP buoys: Returns multi-partition swell data from different source regions

    Args:
        station_id: Buoy station ID (e.g., '46237' for NDBC, '067' for CDIP)
        network: 'ndbc' or 'cdip'

    Returns:
        JSON with partitioned wave components including:
        - swell: Primary swell height, period, direction
        - wind_waves: Local wind wave components
        - partitions: (CDIP only) Multiple swell trains
    """
    network = network.lower()

    if network == "ndbc":
        if NDBCBuoyFetcher is None:
            raise HTTPException(status_code=500, detail="NDBC fetcher not available")

        fetcher = NDBCBuoyFetcher()
        try:
            spectral_data = await fetcher.fetch_spectral_wave_data(station_id)
            return JSONResponse(content=spectral_data)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Could not fetch spectral data: {e}")

    elif network == "cdip":
        if CDIPBuoyFetcher is None:
            raise HTTPException(status_code=500, detail="CDIP fetcher not available")

        # Remove CDIP- prefix if present
        if station_id.startswith("CDIP-"):
            station_id = station_id[5:]

        fetcher = CDIPBuoyFetcher()
        try:
            partition_data = await fetcher.fetch_partitioned_wave_data(station_id)
            return JSONResponse(content=partition_data)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Could not fetch partition data: {e}")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown network: {network}. Use 'ndbc' or 'cdip'.")


@router.get("/{station_id}/observation")
async def get_buoy_observation(
    station_id: str,
    network: str = Query("ndbc", description="Buoy network: 'ndbc' or 'cdip'"),
):
    """
    Get latest observation for a specific buoy.

    Args:
        station_id: Buoy station ID
        network: 'ndbc' or 'cdip'

    Returns:
        JSON with latest observation data
    """
    network = network.lower()

    if network == "ndbc":
        if NDBCBuoyFetcher is None:
            raise HTTPException(status_code=500, detail="NDBC fetcher not available")

        fetcher = NDBCBuoyFetcher()
        try:
            obs = await fetcher.fetch_latest_observation(station_id)
            return JSONResponse(content=obs)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Could not fetch observation: {e}")

    elif network == "cdip":
        if CDIPBuoyFetcher is None:
            raise HTTPException(status_code=500, detail="CDIP fetcher not available")

        if station_id.startswith("CDIP-"):
            station_id = station_id[5:]

        fetcher = CDIPBuoyFetcher()
        try:
            obs = await fetcher.fetch_latest_observation(station_id)
            return JSONResponse(content=obs)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Could not fetch observation: {e}")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown network: {network}")
