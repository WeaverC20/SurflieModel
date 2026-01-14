"""
Wave Forecast Router

Serves wave forecast data from WaveWatch III model for visualization.
Reads from local NetCDF files first, falls back to external API if unavailable.
"""

import json
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
LOCAL_WAVES_DIR = LOCAL_DATA_DIR / "waves"
LOCAL_INDEX_FILE = LOCAL_DATA_DIR / "index.json"

try:
    from wave.wavewatch_fetcher import WaveWatchFetcher
except ImportError as e:
    logger.error(f"Could not import wave fetcher: {e}")
    WaveWatchFetcher = None

router = APIRouter(prefix="/waves", tags=["waves"])


def get_latest_wave_file() -> Optional[Path]:
    """Get the most recent local wave NetCDF file."""
    if not LOCAL_WAVES_DIR.exists():
        return None
    files = sorted(LOCAL_WAVES_DIR.glob("ww3_*.nc"))
    return files[-1] if files else None


def read_local_wave_grid(
    forecast_hour: int,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> Optional[Dict[str, Any]]:
    """Read wave grid from local NetCDF file."""
    filepath = get_latest_wave_file()
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
            "significant_wave_height": extract_var("significant_wave_height"),
            "peak_wave_period": extract_var("peak_wave_period"),
            "mean_wave_direction": extract_var("mean_wave_direction"),
            "wind_wave_height": extract_var("wind_wave_height"),
            "wind_wave_period": extract_var("wind_wave_period"),
            "wind_wave_direction": extract_var("wind_wave_direction"),
            "primary_swell_height": extract_var("primary_swell_height"),
            "primary_swell_period": extract_var("primary_swell_period"),
            "primary_swell_direction": extract_var("primary_swell_direction"),
            "wind_sea_height": extract_var("wind_wave_height"),
            "swell_height": extract_var("primary_swell_height"),
            "forecast_time": forecast_time.isoformat(),
            "cycle_time": cycle_time.isoformat(),
            "forecast_hour": int(forecast_hours[time_idx]),
            "resolution_deg": 0.25,
            "model": "WaveWatch III",
            "source": "local",
            "units": {
                "significant_wave_height": "m",
                "peak_wave_period": "s",
                "mean_wave_direction": "degrees",
                "wind_wave_height": "m",
                "primary_swell_height": "m",
                "lat": "degrees",
                "lon": "degrees"
            }
        }

        ds.close()
        logger.info(f"Loaded wave data from local file: {filepath.name}")
        return result

    except Exception as e:
        logger.error(f"Error reading local wave data: {e}")
        return None


@router.get("/grid")
async def get_wave_grid(
    min_lat: float = Query(32.0, description="Minimum latitude"),
    max_lat: float = Query(42.0, description="Maximum latitude"),
    min_lon: float = Query(-125.0, description="Minimum longitude"),
    max_lon: float = Query(-117.0, description="Maximum longitude"),
    forecast_hour: int = Query(0, description="Forecast hour (0, 3, 6, etc.)"),
):
    """
    Get wave forecast grid from WaveWatch III model.

    Reads from local NetCDF files first (fast), falls back to external API if unavailable.

    Returns a grid of wave forecast data including:
    - Significant wave height
    - Peak wave period
    - Mean wave direction
    - Wind sea and swell components
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
    grid_data = read_local_wave_grid(
        forecast_hour=forecast_hour,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
    )

    # Fall back to external API if local data unavailable
    if grid_data is None:
        if WaveWatchFetcher is None:
            raise HTTPException(
                status_code=500,
                detail="No local wave data and wave fetcher not available."
            )

        try:
            logger.info("No local wave data, fetching from external API...")
            fetcher = WaveWatchFetcher()
            grid_data = await fetcher.fetch_wave_grid(
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
                forecast_hour=forecast_hour
            )
        except Exception as e:
            logger.error(f"Error fetching wave grid: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch wave data: {str(e)}"
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
        "wind_wave_height": serialize_array(grid_data.get("wind_wave_height", grid_data.get("wind_sea_height"))),
        "wind_wave_period": serialize_array(grid_data.get("wind_wave_period")),
        "wind_wave_direction": serialize_array(grid_data.get("wind_wave_direction")),
        "primary_swell_height": serialize_array(grid_data.get("primary_swell_height", grid_data.get("swell_height"))),
        "primary_swell_period": serialize_array(grid_data.get("primary_swell_period")),
        "primary_swell_direction": serialize_array(grid_data.get("primary_swell_direction")),
        "wind_sea_height": serialize_array(grid_data.get("wind_sea_height", grid_data.get("wind_wave_height"))),
        "swell_height": serialize_array(grid_data.get("swell_height", grid_data.get("primary_swell_height"))),
        "forecast_time": grid_data["forecast_time"],
        "cycle_time": grid_data["cycle_time"],
        "forecast_hour": grid_data["forecast_hour"],
        "resolution_deg": grid_data["resolution_deg"],
        "model": grid_data["model"],
        "units": grid_data["units"]
    }

    if "secondary_swell_height" in grid_data:
        serialized_data["secondary_swell_height"] = serialize_array(grid_data["secondary_swell_height"])
    if "secondary_swell_period" in grid_data:
        serialized_data["secondary_swell_period"] = serialize_array(grid_data["secondary_swell_period"])
    if "secondary_swell_direction" in grid_data:
        serialized_data["secondary_swell_direction"] = serialize_array(grid_data["secondary_swell_direction"])

    return JSONResponse(content=serialized_data)


def direction_to_compass(degrees: float) -> str:
    """Convert degrees to compass direction"""
    if degrees is None:
        return "N/A"
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(degrees / 22.5) % 16
    return directions[index]


def get_forecast_hours() -> list:
    """Get forecast hours: 3-hourly 0-48h, then 24-hourly 72h+"""
    # 3-hour intervals from 0 to 48 hours
    hours_3h = list(range(0, 49, 3))  # 0, 3, 6, ..., 48
    # 24-hour intervals from 72 hours onwards (up to 168h = 7 days)
    hours_24h = list(range(72, 169, 24))  # 72, 96, 120, 144, 168
    return hours_3h + hours_24h


@router.get("/forecast")
async def get_wave_forecast(
    lat: float = Query(33.66, description="Latitude of location"),
    lon: float = Query(-118.0, description="Longitude of location"),
):
    """
    Get multi-hour wave forecast for a specific point.

    Returns all wave spectrum components:
    - Combined sea state (height, period, direction)
    - Wind waves (height, period, direction)
    - Primary swell (height, period, direction)
    - Secondary swell (height, period, direction) when available

    Time intervals:
    - 3-hour intervals from 0-48 hours
    - 24-hour intervals from 72-168 hours
    """
    if WaveWatchFetcher is None:
        raise HTTPException(
            status_code=500,
            detail="Wave fetcher not available. Check server logs."
        )

    try:
        fetcher = WaveWatchFetcher()
        forecast_hours = get_forecast_hours()
        forecasts = []

        # Define small region around point for data extraction
        buffer = 0.5  # degrees
        min_lat = lat - buffer
        max_lat = lat + buffer
        min_lon = lon - buffer
        max_lon = lon + buffer

        logger.info(f"Fetching wave forecast for point ({lat}, {lon}) - {len(forecast_hours)} hours")

        for hour in forecast_hours:
            try:
                grid_data = await fetcher.fetch_wave_grid(
                    min_lat=min_lat,
                    max_lat=max_lat,
                    min_lon=min_lon,
                    max_lon=max_lon,
                    forecast_hour=hour
                )

                # Find nearest grid point to requested location
                lats = np.array(grid_data["lat"])
                lons = np.array(grid_data["lon"])

                lat_idx = np.argmin(np.abs(lats - lat))
                lon_idx = np.argmin(np.abs(lons - lon))

                def get_value(data_key):
                    """Extract value at point, handling NaN"""
                    if data_key not in grid_data:
                        return None
                    data = grid_data[data_key]
                    if isinstance(data, list) and len(data) > lat_idx:
                        if isinstance(data[lat_idx], list) and len(data[lat_idx]) > lon_idx:
                            val = data[lat_idx][lon_idx]
                            if val is None or (isinstance(val, float) and np.isnan(val)):
                                return None
                            return round(val, 2) if isinstance(val, (int, float)) else val
                    return None

                # Extract all wave components at point
                forecast_entry = {
                    "forecast_hour": hour,
                    "forecast_time": grid_data["forecast_time"],
                    # Combined sea state
                    "combined": {
                        "height_m": get_value("significant_wave_height"),
                        "period_s": get_value("peak_wave_period"),
                        "direction_deg": get_value("mean_wave_direction"),
                        "direction_compass": direction_to_compass(get_value("mean_wave_direction"))
                    },
                    # Wind waves
                    "wind_waves": {
                        "height_m": get_value("wind_wave_height"),
                        "period_s": get_value("wind_wave_period"),
                        "direction_deg": get_value("wind_wave_direction"),
                        "direction_compass": direction_to_compass(get_value("wind_wave_direction"))
                    },
                    # Primary swell
                    "primary_swell": {
                        "height_m": get_value("primary_swell_height"),
                        "period_s": get_value("primary_swell_period"),
                        "direction_deg": get_value("primary_swell_direction"),
                        "direction_compass": direction_to_compass(get_value("primary_swell_direction"))
                    }
                }

                # Add secondary swell if available
                sec_height = get_value("secondary_swell_height")
                if sec_height is not None:
                    forecast_entry["secondary_swell"] = {
                        "height_m": sec_height,
                        "period_s": get_value("secondary_swell_period"),
                        "direction_deg": get_value("secondary_swell_direction"),
                        "direction_compass": direction_to_compass(get_value("secondary_swell_direction"))
                    }

                forecasts.append(forecast_entry)

            except Exception as e:
                logger.warning(f"Failed to fetch hour {hour}: {e}")
                continue

        if not forecasts:
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch any forecast hours"
            )

        # Get cycle time from first forecast
        first_grid = await fetcher.fetch_wave_grid(
            min_lat=min_lat, max_lat=max_lat,
            min_lon=min_lon, max_lon=max_lon,
            forecast_hour=0
        )

        response = {
            "location": {
                "lat": lat,
                "lon": lon
            },
            "model": "WaveWatch III",
            "cycle_time": first_grid["cycle_time"],
            "time_intervals": {
                "0-48h": "3-hourly",
                "72h+": "24-hourly"
            },
            "forecasts": forecasts
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching wave forecast: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch wave forecast: {str(e)}"
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
        "forecast_range": "168 hours (7 days)",
        "time_intervals": {
            "0-48h": "3-hourly",
            "72h+": "24-hourly"
        },
        "variables": {
            "combined_sea_state": [
                "Significant wave height (m)",
                "Peak wave period (s)",
                "Mean wave direction (degrees)"
            ],
            "wind_waves": [
                "Wind wave height (m)",
                "Wind wave period (s)",
                "Wind wave direction (degrees)"
            ],
            "primary_swell": [
                "Primary swell height (m)",
                "Primary swell period (s)",
                "Primary swell direction (degrees)"
            ],
            "secondary_swell": [
                "Secondary swell height (m)",
                "Secondary swell period (s)",
                "Secondary swell direction (degrees)"
            ]
        },
        "source": "https://polar.ncep.noaa.gov/waves/"
    })
