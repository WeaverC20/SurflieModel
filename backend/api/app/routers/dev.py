"""Development and testing endpoints

These endpoints are for development and testing only.
Should be disabled or protected in production.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Add parent directory to path to import data pipelines
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from data.pipelines.noaa import (
    NOAAFetcher,
    NOAATideFetcher,
    NOAAWaveWatch3Fetcher,
    NOAAWindFetcher,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Response models
class TideStationInfo(BaseModel):
    """Tide station information"""
    id: str
    name: str
    lat: float
    lon: float
    distance_km: float


class NOAATestResponse(BaseModel):
    """Response for NOAA test endpoints"""
    status: str
    location: dict
    data: dict
    timestamp: str
    note: Optional[str] = None


# Predefined test locations
TEST_LOCATIONS = {
    "huntington_beach": {"name": "Huntington Beach, CA", "lat": 33.6595, "lon": -118.0007},
    "ocean_beach_sf": {"name": "Ocean Beach, SF", "lat": 37.7594, "lon": -122.5107},
    "newport_beach": {"name": "Newport Beach, CA", "lat": 33.6, "lon": -117.9},
    "pipeline": {"name": "Pipeline, Oahu, HI", "lat": 21.6644, "lon": -158.0528},
    "mavericks": {"name": "Mavericks, Half Moon Bay", "lat": 37.4947, "lon": -122.4969},
}


@router.get("/test/locations", tags=["dev"])
async def get_test_locations():
    """Get list of predefined test locations"""
    return {
        "locations": TEST_LOCATIONS,
        "usage": "Use location name as 'location' query parameter in other endpoints"
    }


@router.get("/test/noaa/tide", response_model=NOAATestResponse, tags=["dev"])
async def test_noaa_tide(
    lat: Optional[float] = Query(None, description="Latitude in decimal degrees"),
    lon: Optional[float] = Query(None, description="Longitude in decimal degrees"),
    location: Optional[str] = Query(None, description="Predefined location name"),
    station_id: Optional[str] = Query(None, description="NOAA tide station ID"),
    hours: int = Query(48, ge=1, le=168, description="Forecast hours (1-168)")
):
    """Test NOAA CO-OPS tide fetcher

    Fetches tide predictions from NOAA CO-OPS.
    Either provide lat/lon or use a predefined location.
    """
    # Determine location
    if location and location in TEST_LOCATIONS:
        loc_data = TEST_LOCATIONS[location]
        lat = loc_data["lat"]
        lon = loc_data["lon"]
        location_name = loc_data["name"]
    elif lat is not None and lon is not None:
        location_name = f"Custom ({lat}, {lon})"
    else:
        # Default to Huntington Beach
        loc_data = TEST_LOCATIONS["huntington_beach"]
        lat = loc_data["lat"]
        lon = loc_data["lon"]
        location_name = loc_data["name"]

    logger.info(f"Testing NOAA tide fetcher for {location_name}")

    try:
        fetcher = NOAATideFetcher()

        # Find or use station
        if not station_id:
            station = await fetcher.find_nearest_station(lat, lon)
            if not station:
                raise HTTPException(
                    status_code=404,
                    detail="No tide station found within 100km of location"
                )
            station_id = station["id"]
            station_info = station
        else:
            station_info = {"id": station_id, "provided": True}

        # Fetch predictions
        begin_date = datetime.utcnow()
        end_date = begin_date + timedelta(hours=hours)

        tide_data = await fetcher.fetch_tide_predictions(
            station_id=station_id,
            begin_date=begin_date,
            end_date=end_date,
            interval='hilo'
        )

        return NOAATestResponse(
            status="success",
            location={
                "name": location_name,
                "lat": lat,
                "lon": lon,
                "station": station_info
            },
            data=tide_data,
            timestamp=datetime.utcnow().isoformat(),
            note="Tide predictions for high/low tides only"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing tide fetcher: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/noaa/wave", response_model=NOAATestResponse, tags=["dev"])
async def test_noaa_wave(
    lat: Optional[float] = Query(None, description="Latitude in decimal degrees"),
    lon: Optional[float] = Query(None, description="Longitude in decimal degrees"),
    location: Optional[str] = Query(None, description="Predefined location name"),
    forecast_hour: int = Query(0, ge=0, le=384, description="Forecast hour (0-384)")
):
    """Test NOAA Wave Watch 3 fetcher

    Fetches wave data from NOAA Wave Watch 3 model.
    Returns GRIB2 metadata (parsing not yet implemented).
    """
    # Determine location
    if location and location in TEST_LOCATIONS:
        loc_data = TEST_LOCATIONS[location]
        lat = loc_data["lat"]
        lon = loc_data["lon"]
        location_name = loc_data["name"]
    elif lat is not None and lon is not None:
        location_name = f"Custom ({lat}, {lon})"
    else:
        loc_data = TEST_LOCATIONS["ocean_beach_sf"]
        lat = loc_data["lat"]
        lon = loc_data["lon"]
        location_name = loc_data["name"]

    logger.info(f"Testing NOAA wave fetcher for {location_name}")

    try:
        fetcher = NOAAWaveWatch3Fetcher()

        wave_data = await fetcher.fetch_wave_spectrum(lat, lon, forecast_hour)

        return NOAATestResponse(
            status="success",
            location={
                "name": location_name,
                "lat": lat,
                "lon": lon
            },
            data=wave_data,
            timestamp=datetime.utcnow().isoformat(),
            note="GRIB2 data fetched successfully. Parsing with cfgrib/xarray required for actual values."
        )

    except Exception as e:
        logger.error(f"Error testing wave fetcher: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/noaa/wind", response_model=NOAATestResponse, tags=["dev"])
async def test_noaa_wind(
    lat: Optional[float] = Query(None, description="Latitude in decimal degrees"),
    lon: Optional[float] = Query(None, description="Longitude in decimal degrees"),
    location: Optional[str] = Query(None, description="Predefined location name"),
    hours: int = Query(24, ge=1, le=384, description="Forecast hours (1-384)")
):
    """Test NOAA GFS wind fetcher

    Fetches wind data from NOAA GFS model.
    Returns GRIB2 metadata (parsing not yet implemented).
    """
    # Determine location
    if location and location in TEST_LOCATIONS:
        loc_data = TEST_LOCATIONS[location]
        lat = loc_data["lat"]
        lon = loc_data["lon"]
        location_name = loc_data["name"]
    elif lat is not None and lon is not None:
        location_name = f"Custom ({lat}, {lon})"
    else:
        loc_data = TEST_LOCATIONS["ocean_beach_sf"]
        lat = loc_data["lat"]
        lon = loc_data["lon"]
        location_name = loc_data["name"]

    logger.info(f"Testing NOAA wind fetcher for {location_name}")

    try:
        fetcher = NOAAWindFetcher()

        wind_data = await fetcher.fetch_wind_timeseries(lat, lon, hours)

        return NOAATestResponse(
            status="success",
            location={
                "name": location_name,
                "lat": lat,
                "lon": lon
            },
            data=wind_data,
            timestamp=datetime.utcnow().isoformat(),
            note="GRIB2 data fetched successfully. Parsing with cfgrib/xarray required for actual values."
        )

    except Exception as e:
        logger.error(f"Error testing wind fetcher: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/noaa/complete", response_model=NOAATestResponse, tags=["dev"])
async def test_noaa_complete(
    lat: Optional[float] = Query(None, description="Latitude in decimal degrees"),
    lon: Optional[float] = Query(None, description="Longitude in decimal degrees"),
    location: Optional[str] = Query(None, description="Predefined location name"),
    hours: int = Query(48, ge=1, le=168, description="Forecast hours (1-168)")
):
    """Test complete NOAA forecast

    Fetches tide, wave, and wind data in one call.
    This is the recommended endpoint for getting all surf forecast data.
    """
    # Determine location
    if location and location in TEST_LOCATIONS:
        loc_data = TEST_LOCATIONS[location]
        lat = loc_data["lat"]
        lon = loc_data["lon"]
        location_name = loc_data["name"]
    elif lat is not None and lon is not None:
        location_name = f"Custom ({lat}, {lon})"
    else:
        loc_data = TEST_LOCATIONS["ocean_beach_sf"]
        lat = loc_data["lat"]
        lon = loc_data["lon"]
        location_name = loc_data["name"]

    logger.info(f"Testing complete NOAA forecast for {location_name}")

    try:
        fetcher = NOAAFetcher()

        forecast = await fetcher.fetch_complete_forecast(
            latitude=lat,
            longitude=lon,
            forecast_hours=hours
        )

        return NOAATestResponse(
            status="success",
            location={
                "name": location_name,
                "lat": lat,
                "lon": lon
            },
            data=forecast,
            timestamp=datetime.utcnow().isoformat(),
            note="Complete forecast with tide (parsed), wave (GRIB2), and wind (GRIB2) data"
        )

    except Exception as e:
        logger.error(f"Error testing complete forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))
