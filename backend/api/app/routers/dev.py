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
from data.pipelines.buoy import (
    NDBCBuoyFetcher,
    CDIPBuoyFetcher,
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


@router.get("/test/buoy/ndbc/{station_id}", tags=["dev", "buoy"])
async def test_ndbc_buoy(
    station_id: str,
    include_spectral: bool = Query(False, description="Include spectral wave data")
):
    """Test NDBC buoy data fetcher

    Fetches real-time observation data from NDBC buoys.
    Includes all available meteorological and oceanographic data.
    """
    logger.info(f"Testing NDBC buoy fetcher for station {station_id}")

    try:
        fetcher = NDBCBuoyFetcher()

        # Fetch standard meteorological data
        obs_data = await fetcher.fetch_latest_observation(station_id)

        # Optionally fetch spectral wave data
        spectral_data = None
        if include_spectral:
            try:
                spectral_data = await fetcher.fetch_spectral_wave_data(station_id)
            except Exception as e:
                logger.warning(f"Could not fetch spectral data: {e}")
                spectral_data = {"error": str(e)}

        return {
            "status": "success",
            "station_id": station_id,
            "observation": obs_data,
            "spectral": spectral_data,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error testing NDBC buoy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/buoy/cdip/{station_id}", tags=["dev", "buoy"])
async def test_cdip_buoy(station_id: str):
    """Test CDIP buoy data fetcher

    Note: CDIP data is in NetCDF format and requires additional libraries.
    This endpoint returns metadata and access information.
    """
    logger.info(f"Testing CDIP buoy fetcher for station {station_id}")

    try:
        fetcher = CDIPBuoyFetcher()
        data = await fetcher.fetch_latest_observation(station_id)

        return {
            "status": "success",
            "station_id": station_id,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "note": "CDIP requires netCDF4 library for full data parsing"
        }

    except Exception as e:
        logger.error(f"Error testing CDIP buoy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/buoy/nearby", tags=["dev", "buoy"])
async def find_nearby_buoys(
    lat: Optional[float] = Query(None, description="Latitude in decimal degrees"),
    lon: Optional[float] = Query(None, description="Longitude in decimal degrees"),
    location: Optional[str] = Query(None, description="Predefined location name"),
    max_distance: float = Query(100, ge=1, le=500, description="Maximum distance in km"),
    source: str = Query("ndbc", description="Buoy source: 'ndbc' or 'cdip' or 'both'")
):
    """Find nearby buoys for a location

    Returns NDBC and/or CDIP buoys within the specified distance.
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

    logger.info(f"Finding nearby buoys for {location_name}")

    try:
        result = {
            "status": "success",
            "location": {
                "name": location_name,
                "lat": lat,
                "lon": lon
            },
            "max_distance_km": max_distance,
            "buoys": {}
        }

        # Find NDBC buoys
        if source in ["ndbc", "both"]:
            ndbc_fetcher = NDBCBuoyFetcher()
            ndbc_buoys = ndbc_fetcher.get_nearby_buoys(lat, lon, max_distance)
            result["buoys"]["ndbc"] = ndbc_buoys

        # Find CDIP buoys
        if source in ["cdip", "both"]:
            cdip_fetcher = CDIPBuoyFetcher()
            cdip_buoys = cdip_fetcher.get_nearby_buoys(lat, lon, max_distance)
            result["buoys"]["cdip"] = cdip_buoys

        return result

    except Exception as e:
        logger.error(f"Error finding nearby buoys: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/buoy/multi", tags=["dev", "buoy"])
async def fetch_multiple_buoys(
    lat: Optional[float] = Query(None, description="Latitude in decimal degrees"),
    lon: Optional[float] = Query(None, description="Longitude in decimal degrees"),
    location: Optional[str] = Query(None, description="Predefined location name"),
    max_distance: float = Query(50, ge=1, le=500, description="Maximum distance in km"),
    max_buoys: int = Query(3, ge=1, le=10, description="Maximum number of buoys to fetch")
):
    """Fetch data from multiple nearby NDBC buoys

    Finds nearby buoys and fetches current observations from them.
    Returns data sorted by distance from the location.
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
        loc_data = TEST_LOCATIONS["huntington_beach"]
        lat = loc_data["lat"]
        lon = loc_data["lon"]
        location_name = loc_data["name"]

    logger.info(f"Fetching multiple buoys for {location_name}")

    try:
        fetcher = NDBCBuoyFetcher()

        # Find nearby buoys
        nearby = fetcher.get_nearby_buoys(lat, lon, max_distance)

        # Limit to max_buoys
        nearby = nearby[:max_buoys]

        # Fetch data from each buoy
        buoy_data = []
        for buoy_info in nearby:
            try:
                obs = await fetcher.fetch_latest_observation(buoy_info["station_id"])
                buoy_data.append({
                    "buoy_info": buoy_info,
                    "observation": obs,
                    "status": "success"
                })
            except Exception as e:
                logger.warning(f"Failed to fetch buoy {buoy_info['station_id']}: {e}")
                buoy_data.append({
                    "buoy_info": buoy_info,
                    "observation": None,
                    "status": "error",
                    "error": str(e)
                })

        return {
            "status": "success",
            "location": {
                "name": location_name,
                "lat": lat,
                "lon": lon
            },
            "buoys_found": len(nearby),
            "buoys_fetched": len([b for b in buoy_data if b["status"] == "success"]),
            "data": buoy_data,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching multiple buoys: {e}")
        raise HTTPException(status_code=500, detail=str(e))
