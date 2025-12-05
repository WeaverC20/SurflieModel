"""Fetch NOAA forecast data from multiple sources

This module provides fetchers for:
- NOAA CO-OPS: Tide data and harmonic constituents (US only, no license required)
- NOAA Wave Watch 3: Swell height and direction (US-focused, open and free)
- NOAA GFS: Wind data (global coverage, open and free)

Future expansion planned for GEFS-WAVE and regional high-res models.
"""

import logging
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Try to import GRIB parsing libraries
try:
    import numpy as np
    import xarray as xr
    import cfgrib
    GRIB_PARSING_AVAILABLE = True
except ImportError:
    GRIB_PARSING_AVAILABLE = False
    logger.warning("cfgrib/xarray not available. GRIB2 data will not be parsed. Install with: pip install cfgrib xarray")


class NOAATideFetcher:
    """Fetches tide data from NOAA CO-OPS API

    NOAA CO-OPS (Center for Operational Oceanographic Products and Services)
    provides tide predictions and harmonic constituents.

    Documentation: https://api.tidesandcurrents.noaa.gov/api/prod/
    """

    def __init__(self):
        """Initialize NOAA CO-OPS tide fetcher"""
        self.base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        self.metadata_url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"

    async def fetch_tide_predictions(
        self,
        station_id: str,
        begin_date: datetime,
        end_date: datetime,
        datum: str = "MLLW",
        interval: str = "hilo",
        units: str = "metric"
    ) -> Dict:
        """Fetch tide predictions for a station

        Args:
            station_id: NOAA station ID (e.g., "9414290" for San Francisco)
            begin_date: Start date for predictions
            end_date: End date for predictions
            datum: Tidal datum (MLLW, MSL, MTL, etc.)
            interval: 'hilo' for high/low only, 'h' for hourly, or '6' for 6-minute
            units: 'metric' or 'english'

        Returns:
            Tide prediction data dictionary with timestamps and heights
        """
        logger.info(f"Fetching tide predictions for station {station_id}")

        params = {
            "station": station_id,
            "begin_date": begin_date.strftime("%Y%m%d %H:%M"),
            "end_date": end_date.strftime("%Y%m%d %H:%M"),
            "product": "predictions",
            "datum": datum,
            "interval": interval,
            "units": units,
            "time_zone": "gmt",
            "application": "wave_forecast",
            "format": "json"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()

    async def fetch_harmonic_constituents(self, station_id: str) -> Dict:
        """Fetch harmonic constituents for a tide station

        Harmonic constituents allow for accurate long-term tide predictions.

        Args:
            station_id: NOAA station ID

        Returns:
            Harmonic constituents data
        """
        logger.info(f"Fetching harmonic constituents for station {station_id}")

        params = {
            "station": station_id,
            "product": "harcon",
            "application": "wave_forecast",
            "format": "json"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()

    async def find_nearest_station(
        self,
        latitude: float,
        longitude: float,
        max_distance_km: float = 100.0
    ) -> Optional[Dict]:
        """Find nearest tide station to given coordinates

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            max_distance_km: Maximum search distance in kilometers

        Returns:
            Station info dict or None if no station found
        """
        logger.info(f"Finding nearest tide station to ({latitude}, {longitude})")

        # Fetch all stations
        url = f"{self.metadata_url}/stations.json"
        params = {"type": "tidepredictions"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        # Find nearest station
        stations = data.get("stations", [])
        nearest = None
        min_distance = float('inf')

        for station in stations:
            try:
                station_lat = float(station.get("lat", 0))
                station_lon = float(station.get("lng", 0))

                # Haversine distance (approximate)
                from math import radians, cos, sin, asin, sqrt

                lat1, lon1 = radians(latitude), radians(longitude)
                lat2, lon2 = radians(station_lat), radians(station_lon)

                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                distance_km = 6371 * c  # Earth radius in km

                if distance_km < min_distance and distance_km <= max_distance_km:
                    min_distance = distance_km
                    nearest = {
                        "id": station.get("id"),
                        "name": station.get("name"),
                        "lat": station_lat,
                        "lon": station_lon,
                        "distance_km": distance_km
                    }
            except (ValueError, TypeError, KeyError):
                continue

        if nearest:
            logger.info(f"Found station {nearest['id']} at {nearest['distance_km']:.2f} km")
        else:
            logger.warning(f"No tide station found within {max_distance_km} km")

        return nearest


def _parse_grib2_data(grib_bytes: bytes, latitude: float, longitude: float) -> Optional[Dict]:
    """Parse GRIB2 data and extract values at nearest point

    Args:
        grib_bytes: Raw GRIB2 data bytes
        latitude: Target latitude
        longitude: Target longitude (should be in 0-360 format)

    Returns:
        Dictionary of variable names to values, or None if parsing fails
    """
    if not GRIB_PARSING_AVAILABLE:
        return None

    try:
        # cfgrib requires a file path, so write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as tmp_file:
            tmp_file.write(grib_bytes)
            tmp_path = tmp_file.name

        try:
            # Open with xarray/cfgrib
            ds = xr.open_dataset(tmp_path, engine='cfgrib')

            # Extract values at nearest point
            # First, try simple nearest neighbor selection
            point_data = ds.sel(latitude=latitude, longitude=longitude, method='nearest')

            # Check if the selected point has NaN values
            first_var = list(ds.data_vars.keys())[0]
            selected_value = float(point_data[first_var].values)

            # If the selected point is NaN, find the nearest valid (non-NaN) point
            if np.isnan(selected_value):
                logger.info(f"Selected point has NaN values, searching for nearest valid point...")

                # Get the first variable's data array
                data_array = ds[first_var]

                # Create a mask of valid (non-NaN) points
                valid_mask = ~np.isnan(data_array.values)

                if valid_mask.any():
                    # Find indices of valid points
                    valid_indices = np.argwhere(valid_mask)

                    # Get coordinates of all valid points
                    valid_lats = ds.latitude.values[valid_indices[:, 0]]
                    valid_lons = ds.longitude.values[valid_indices[:, 1]]

                    # Calculate distances to all valid points (simple Euclidean in lat/lon space)
                    # For more accuracy, use haversine, but for small regions this is fine
                    distances = np.sqrt((valid_lats - latitude)**2 + (valid_lons - longitude)**2)

                    # Find the nearest valid point
                    nearest_idx = np.argmin(distances)
                    nearest_lat = valid_lats[nearest_idx]
                    nearest_lon = valid_lons[nearest_idx]

                    # Select data at the nearest valid point
                    point_data = ds.sel(latitude=nearest_lat, longitude=nearest_lon, method='nearest')

                    logger.info(f"Found valid point at ({nearest_lat:.2f}, {nearest_lon:.2f}), "
                               f"distance: {distances[nearest_idx]:.2f} degrees")
                else:
                    logger.warning("No valid (non-NaN) points found in the dataset")
                    return None

            # Convert to dictionary, handling all variables
            result = {}
            for var in ds.data_vars:
                try:
                    value = float(point_data[var].values)
                    result[var] = value
                except (ValueError, TypeError):
                    # Skip variables that can't be converted to float
                    continue

            # Log available variables for debugging
            logger.info(f"GRIB2 variables found: {list(ds.data_vars.keys())}")
            logger.info(f"Extracted values: {result}")

            # Also include coordinate information
            result['actual_latitude'] = float(point_data['latitude'].values)
            result['actual_longitude'] = float(point_data['longitude'].values)

            ds.close()
            return result

        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass

    except Exception as e:
        logger.warning(f"Failed to parse GRIB2 data: {e}")
        return None


class NOAAWaveWatch3Fetcher:
    """Fetches swell height and direction from NOAA Wave Watch 3 model

    Wave Watch 3 is NOAA's primary wave forecasting model.
    Provides wave height, period, and direction forecasts.

    Documentation: https://polar.ncep.noaa.gov/waves/
    Data access: https://nomads.ncep.noaa.gov/
    """

    def __init__(self):
        """Initialize Wave Watch 3 fetcher"""
        self.nomads_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfswave.pl"
        self.opendap_url = "https://nomads.ncep.noaa.gov/dods"

    async def fetch_wave_data(
        self,
        latitude: float,
        longitude: float,
        forecast_hour: int = 0,
        variables: Optional[List[str]] = None
    ) -> Dict:
        """Fetch Wave Watch 3 data for a point location

        Args:
            latitude: Latitude in decimal degrees (-90 to 90)
            longitude: Longitude in decimal degrees (0 to 360 or -180 to 180)
            forecast_hour: Forecast hour (0-384 for 16-day forecasts)
            variables: List of variables to fetch. Defaults to all wave variables.
                Common variables:
                - HTSGW: Significant height of combined wind waves and swell
                - PERPW: Primary wave mean period
                - DIRPW: Primary wave direction
                - WVHGT: Significant height of wind waves
                - SWELL: Significant height of swell waves
                - SWPER: Swell period
                - SWDIR: Swell direction

        Returns:
            Wave forecast data dictionary
        """
        logger.info(f"Fetching Wave Watch 3 data for ({latitude}, {longitude}) at hour {forecast_hour}")

        if variables is None:
            variables = [
                "HTSGW",  # Significant wave height
                "PERPW",  # Peak wave period
                "DIRPW",  # Wave direction
                "SWELL",  # Swell height
                "SWPER",  # Swell period
                "SWDIR"   # Swell direction
            ]

        # Normalize longitude to 0-360 for NOAA data
        if longitude < 0:
            longitude += 360

        # Get latest model run timestamp with fallback logic
        # NOAA has 3-4 hour delay, so try current run, then previous runs
        now = datetime.utcnow()
        current_hour = (now.hour // 6) * 6

        # Try up to 3 previous model runs (current, -6h, -12h)
        for run_offset in [0, 6, 12]:
            model_time = now.replace(hour=current_hour, minute=0, second=0, microsecond=0) - timedelta(hours=run_offset)
            model_hour = model_time.hour

            params = {
                "file": f"gfswave.t{model_hour:02d}z.global.0p25.f{forecast_hour:03d}.grib2",
                "lev_surface": "on",
                "var_HTSGW": "on" if "HTSGW" in variables else "off",
                "var_PERPW": "on" if "PERPW" in variables else "off",
                "var_DIRPW": "on" if "DIRPW" in variables else "off",
                "var_SWELL": "on" if "SWELL" in variables else "off",
                "var_SWPER": "on" if "SWPER" in variables else "off",
                "var_SWDIR": "on" if "SWDIR" in variables else "off",
                "subregion": "",
                "leftlon": longitude - 0.5,
                "rightlon": longitude + 0.5,
                "toplat": latitude + 0.5,
                "bottomlat": latitude - 0.5,
                "dir": f"/gfs.{model_time.strftime('%Y%m%d')}/{model_hour:02d}/wave/gridded"
            }

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.get(self.nomads_url, params=params)
                    response.raise_for_status()

                    # Success! Parse GRIB2 data
                    logger.info(f"Successfully fetched Wave Watch 3 data from {model_time.isoformat()} run")

                    result = {
                        "model_time": model_time.isoformat(),
                        "forecast_hour": forecast_hour,
                        "valid_time": (model_time + timedelta(hours=forecast_hour)).isoformat(),
                        "latitude": latitude,
                        "longitude": longitude,
                        "data_size_bytes": len(response.content),
                        "status": "success",
                        "fallback_hours": run_offset if run_offset > 0 else None
                    }

                    # Parse GRIB2 data to extract values
                    parsed_data = _parse_grib2_data(response.content, latitude, longitude)
                    if parsed_data:
                        result["values"] = parsed_data
                        result["parsed"] = True
                        # Extract common wave parameters for easy access
                        if "swh" in parsed_data:  # Significant wave height
                            result["wave_height_m"] = round(parsed_data["swh"], 2)
                        if "perpw" in parsed_data:  # Peak wave period
                            result["wave_period_s"] = round(parsed_data["perpw"], 1)
                        if "dirpw" in parsed_data:  # Wave direction
                            result["wave_direction_deg"] = round(parsed_data["dirpw"], 0)
                    else:
                        result["parsed"] = False
                        result["note"] = "GRIB2 parsing not available. Install cfgrib: brew install eccodes && pip install cfgrib xarray"

                    return result
            except httpx.HTTPError as e:
                if run_offset < 12:  # Try next older run
                    logger.warning(f"Model run {model_time.isoformat()} not available, trying previous run...")
                    continue
                else:  # Last attempt failed
                    logger.error(f"Failed to fetch Wave Watch 3 data after trying multiple runs: {e}")
                    return {
                        "status": "error",
                        "error": str(e),
                        "note": "Latest model run not yet available. Data typically has 3-4 hour delay."
                    }

    async def fetch_wave_spectrum(
        self,
        latitude: float,
        longitude: float,
        forecast_hour: int = 0
    ) -> Dict:
        """Fetch full wave spectrum data for detailed swell analysis

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            forecast_hour: Forecast hour (0-384)

        Returns:
            Wave spectrum data with multiple swell components
        """
        logger.info(f"Fetching wave spectrum for ({latitude}, {longitude})")

        # This would fetch partition data for multiple swell components
        # Wave Watch 3 can provide primary, secondary, and tertiary swell

        variables = [
            "HTSGW",  # Total significant height
            "PERPW",  # Peak period
            "DIRPW",  # Mean direction
            "WVHGT",  # Wind wave height
            "WVPER",  # Wind wave period
            "WVDIR",  # Wind wave direction
            "SWELL",  # Swell height (combined)
            "SWPER",  # Swell period
            "SWDIR"   # Swell direction
        ]

        return await self.fetch_wave_data(latitude, longitude, forecast_hour, variables)


class NOAAWindFetcher:
    """Fetches wind data from NOAA GFS model

    GFS (Global Forecast System) provides wind forecasts globally.
    Short-term: GFS (16-day deterministic)
    Future: GEFS (16-day ensemble for uncertainty)

    Documentation: https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
    Data access: https://nomads.ncep.noaa.gov/
    """

    def __init__(self):
        """Initialize GFS wind fetcher"""
        self.nomads_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        self.opendap_url = "https://nomads.ncep.noaa.gov/dods"

    async def fetch_wind_data(
        self,
        latitude: float,
        longitude: float,
        forecast_hour: int = 0,
        levels: Optional[List[str]] = None
    ) -> Dict:
        """Fetch GFS wind data for a point location

        Args:
            latitude: Latitude in decimal degrees (-90 to 90)
            longitude: Longitude in decimal degrees (0 to 360 or -180 to 180)
            forecast_hour: Forecast hour (0-384 for 16-day forecasts)
            levels: Atmospheric levels to fetch. Defaults to surface (10m) only.
                Options: "10_m_above_ground", "surface", etc.

        Returns:
            Wind forecast data dictionary with speed and direction
        """
        logger.info(f"Fetching GFS wind data for ({latitude}, {longitude}) at hour {forecast_hour}")

        if levels is None:
            levels = ["10_m_above_ground"]  # Standard 10m wind height

        # Normalize longitude to 0-360 for NOAA data
        if longitude < 0:
            longitude += 360

        # Get latest model run timestamp with fallback logic
        # NOAA has 3-4 hour delay, so try current run, then previous runs
        now = datetime.utcnow()
        current_hour = (now.hour // 6) * 6

        # Try up to 3 previous model runs (current, -6h, -12h)
        for run_offset in [0, 6, 12]:
            model_time = now.replace(hour=current_hour, minute=0, second=0, microsecond=0) - timedelta(hours=run_offset)
            model_hour = model_time.hour

            params = {
                "file": f"gfs.t{model_hour:02d}z.pgrb2.0p25.f{forecast_hour:03d}",
                "lev_10_m_above_ground": "on",
                "var_UGRD": "on",  # U-component of wind
                "var_VGRD": "on",  # V-component of wind
                "var_GUST": "on",  # Wind gust
                "subregion": "",
                "leftlon": longitude - 0.5,
                "rightlon": longitude + 0.5,
                "toplat": latitude + 0.5,
                "bottomlat": latitude - 0.5,
                "dir": f"/gfs.{model_time.strftime('%Y%m%d')}/{model_hour:02d}/atmos"
            }

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.get(self.nomads_url, params=params)
                    response.raise_for_status()

                    # Success! Parse GRIB2 data
                    logger.info(f"Successfully fetched GFS wind data from {model_time.isoformat()} run")

                    result = {
                        "model_time": model_time.isoformat(),
                        "forecast_hour": forecast_hour,
                        "valid_time": (model_time + timedelta(hours=forecast_hour)).isoformat(),
                        "latitude": latitude,
                        "longitude": longitude,
                        "data_size_bytes": len(response.content),
                        "status": "success",
                        "fallback_hours": run_offset if run_offset > 0 else None
                    }

                    # Parse GRIB2 data to extract values
                    parsed_data = _parse_grib2_data(response.content, latitude, longitude)
                    if parsed_data:
                        result["values"] = parsed_data
                        result["parsed"] = True
                        # Calculate wind speed and direction from U/V components
                        import math
                        u = parsed_data.get("u10", 0)  # U-component (east-west)
                        v = parsed_data.get("v10", 0)  # V-component (north-south)
                        # Wind speed (m/s)
                        wind_speed = math.sqrt(u**2 + v**2)
                        result["wind_speed_ms"] = round(wind_speed, 1)
                        result["wind_speed_kts"] = round(wind_speed * 1.94384, 1)  # Convert to knots
                        # Wind direction (meteorological convention: direction FROM which wind blows)
                        wind_dir = (270 - math.atan2(v, u) * 180 / math.pi) % 360
                        result["wind_direction_deg"] = round(wind_dir, 0)
                        # Wind gust if available
                        if "gust" in parsed_data:
                            result["wind_gust_ms"] = round(parsed_data["gust"], 1)
                            result["wind_gust_kts"] = round(parsed_data["gust"] * 1.94384, 1)
                    else:
                        result["parsed"] = False
                        result["note"] = "GRIB2 parsing not available. Install cfgrib: brew install eccodes && pip install cfgrib xarray"

                    return result
            except httpx.HTTPError as e:
                if run_offset < 12:  # Try next older run
                    logger.warning(f"Model run {model_time.isoformat()} not available, trying previous run...")
                    continue
                else:  # Last attempt failed
                    logger.error(f"Failed to fetch GFS wind data after trying multiple runs: {e}")
                    return {
                        "status": "error",
                        "error": str(e),
                        "note": "Latest model run not yet available. Data typically has 3-4 hour delay."
                    }

    async def fetch_wind_timeseries(
        self,
        latitude: float,
        longitude: float,
        hours: int = 168  # 7 days default
    ) -> Dict:
        """Fetch wind forecast time series

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            hours: Number of forecast hours (max 384)

        Returns:
            Time series of wind forecasts
        """
        logger.info(f"Fetching wind time series for ({latitude}, {longitude}), {hours} hours")

        # GFS has 3-hourly output for first 240 hours, then 12-hourly
        forecast_hours = []
        for h in range(0, min(hours, 240), 3):
            forecast_hours.append(h)
        for h in range(240, min(hours, 384), 12):
            forecast_hours.append(h)

        results = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for fhour in forecast_hours:
                try:
                    data = await self.fetch_wind_data(latitude, longitude, fhour)
                    results.append(data)
                except Exception as e:
                    logger.warning(f"Failed to fetch hour {fhour}: {e}")
                    continue

        return {
            "latitude": latitude,
            "longitude": longitude,
            "forecast_hours": forecast_hours,
            "forecasts": results
        }


class NOAAFetcher:
    """Main NOAA fetcher that combines all data sources

    This class provides a unified interface to fetch tide, wave, and wind data
    from various NOAA services.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize NOAA fetcher

        Args:
            api_key: Optional NOAA API key (not required for most services)
        """
        self.api_key = api_key
        self.base_url = "https://api.weather.gov"

        # Initialize specialized fetchers
        self.tide_fetcher = NOAATideFetcher()
        self.wave_fetcher = NOAAWaveWatch3Fetcher()
        self.wind_fetcher = NOAAWindFetcher()

    async def fetch_point_forecast(self, latitude: float, longitude: float) -> Dict:
        """Fetch point forecast for given coordinates

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees

        Returns:
            Forecast data dictionary
        """
        logger.info(f"Fetching NOAA forecast for ({latitude}, {longitude})")

        async with httpx.AsyncClient() as client:
            # Get grid point
            gridpoint_url = f"{self.base_url}/points/{latitude},{longitude}"
            response = await client.get(gridpoint_url)
            response.raise_for_status()

            gridpoint = response.json()

            # Get forecast
            forecast_url = gridpoint["properties"]["forecast"]
            response = await client.get(forecast_url)
            response.raise_for_status()

            return response.json()

    async def fetch_marine_forecast(self, zone_id: str) -> Dict:
        """Fetch marine forecast for a zone

        Args:
            zone_id: NOAA marine zone ID (e.g., "GMZ850")

        Returns:
            Marine forecast data
        """
        logger.info(f"Fetching marine forecast for zone {zone_id}")

        async with httpx.AsyncClient() as client:
            url = f"{self.base_url}/zones/forecast/{zone_id}/forecast"
            response = await client.get(url)
            response.raise_for_status()

            return response.json()

    async def fetch_complete_forecast(
        self,
        latitude: float,
        longitude: float,
        station_id: Optional[str] = None,
        forecast_hours: int = 168
    ) -> Dict:
        """Fetch complete surf forecast: tides, waves, and wind

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            station_id: Optional NOAA tide station ID (will auto-find if not provided)
            forecast_hours: Number of hours to forecast (default 7 days)

        Returns:
            Complete forecast dictionary with tide, wave, and wind data
        """
        logger.info(f"Fetching complete forecast for ({latitude}, {longitude})")

        results = {}

        # Fetch tide data
        try:
            if not station_id:
                nearest = await self.tide_fetcher.find_nearest_station(latitude, longitude)
                if nearest:
                    station_id = nearest["id"]
                    results["tide_station"] = nearest

            if station_id:
                begin_date = datetime.utcnow()
                end_date = begin_date + timedelta(hours=forecast_hours)
                results["tide"] = await self.tide_fetcher.fetch_tide_predictions(
                    station_id, begin_date, end_date
                )
        except Exception as e:
            logger.error(f"Failed to fetch tide data: {e}")
            results["tide_error"] = str(e)

        # Fetch wave data
        try:
            results["waves"] = await self.wave_fetcher.fetch_wave_spectrum(
                latitude, longitude, forecast_hour=0
            )
        except Exception as e:
            logger.error(f"Failed to fetch wave data: {e}")
            results["wave_error"] = str(e)

        # Fetch wind data
        try:
            results["wind"] = await self.wind_fetcher.fetch_wind_timeseries(
                latitude, longitude, hours=forecast_hours
            )
        except Exception as e:
            logger.error(f"Failed to fetch wind data: {e}")
            results["wind_error"] = str(e)

        return results
