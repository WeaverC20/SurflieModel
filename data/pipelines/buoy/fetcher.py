"""Fetch NDBC buoy data"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


class NDBCBuoyFetcher:
    """Fetches data from NDBC buoys (National Data Buoy Center)

    NDBC provides real-time buoy observations from stations across US waters.
    Data includes wave height, period, direction, wind, pressure, and temperature.

    Documentation: https://www.ndbc.noaa.gov/
    Data Access: https://www.ndbc.noaa.gov/faq/rt_data_access.shtml
    """

    def __init__(self):
        """Initialize NDBC buoy fetcher"""
        self.base_url = "https://www.ndbc.noaa.gov"

    def _parse_value(self, value: str, default=None):
        """Parse a value from NDBC text format

        Args:
            value: String value from NDBC data
            default: Default value if parsing fails or value is missing

        Returns:
            Parsed float value or default
        """
        if value in ["MM", "999", "9999", "999.0", "9999.0"]:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    async def fetch_latest_observation(self, station_id: str) -> Dict:
        """Fetch latest buoy observation from standard meteorological data

        Args:
            station_id: NDBC station ID (e.g., "46237", "46026")

        Returns:
            Latest observation data with all available fields
        """
        logger.info(f"Fetching latest observation for NDBC buoy {station_id}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{self.base_url}/data/realtime2/{station_id}.txt"
            response = await client.get(url)
            response.raise_for_status()

            # Parse fixed-width format
            lines = response.text.strip().split("\n")
            if len(lines) < 3:
                raise ValueError("Insufficient data from buoy")

            # First line is headers, second is units, third is latest data
            headers = lines[0].split()
            units = lines[1].split()
            data = lines[2].split()

            # Parse timestamp
            try:
                timestamp = datetime(
                    year=int(data[0]),
                    month=int(data[1]),
                    day=int(data[2]),
                    hour=int(data[3]),
                    minute=int(data[4]),
                    tzinfo=timezone.utc
                )
                timestamp_str = timestamp.isoformat()
            except (ValueError, IndexError):
                timestamp_str = None

            # Build result dictionary with all available fields
            result = {
                "station_id": station_id,
                "timestamp": timestamp_str,
                "data_source": "NDBC",
                "raw_data": {}
            }

            # Map common NDBC field names to values
            # Based on NDBC standard meteorological format
            field_mapping = {
                "WDIR": ("wind_direction_deg", "Wind direction (degrees clockwise from true N)"),
                "WSPD": ("wind_speed_ms", "Wind speed (m/s)"),
                "GST": ("wind_gust_ms", "Wind gust (m/s)"),
                "WVHT": ("wave_height_m", "Significant wave height (m)"),
                "DPD": ("dominant_wave_period_s", "Dominant wave period (s)"),
                "APD": ("average_wave_period_s", "Average wave period (s)"),
                "MWD": ("mean_wave_direction_deg", "Mean wave direction (degrees)"),
                "PRES": ("air_pressure_hpa", "Sea level pressure (hPa)"),
                "ATMP": ("air_temp_c", "Air temperature (°C)"),
                "WTMP": ("water_temp_c", "Sea surface temperature (°C)"),
                "DEWP": ("dewpoint_temp_c", "Dewpoint temperature (°C)"),
                "VIS": ("visibility_nmi", "Visibility (nautical miles)"),
                "PTDY": ("pressure_tendency_hpa", "Pressure tendency (hPa)"),
                "TIDE": ("tide_ft", "Tide (ft)")
            }

            # Parse all available fields
            for i, header in enumerate(headers):
                if i < len(data) and header in field_mapping:
                    field_name, description = field_mapping[header]
                    value = self._parse_value(data[i])
                    result[field_name] = value
                    result["raw_data"][header] = {
                        "value": value,
                        "unit": units[i] if i < len(units) else None,
                        "description": description
                    }

            # Add derived fields
            if result.get("wind_speed_ms") is not None:
                result["wind_speed_kts"] = round(result["wind_speed_ms"] * 1.94384, 1)
            if result.get("wind_gust_ms") is not None:
                result["wind_gust_kts"] = round(result["wind_gust_ms"] * 1.94384, 1)
            if result.get("wave_height_m") is not None:
                result["wave_height_ft"] = round(result["wave_height_m"] * 3.28084, 1)

            return result

    async def fetch_spectral_wave_data(self, station_id: str) -> Dict:
        """Fetch latest spectral wave summary data

        Args:
            station_id: NDBC station ID

        Returns:
            Spectral wave data including swell components
        """
        logger.info(f"Fetching spectral wave data for NDBC buoy {station_id}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{self.base_url}/data/realtime2/{station_id}.spec"
            response = await client.get(url)
            response.raise_for_status()

            lines = response.text.strip().split("\n")
            if len(lines) < 2:
                return {"error": "Insufficient spectral data"}

            # Parse spectral summary format
            headers = lines[0].split()
            data = lines[1].split()

            result = {
                "station_id": station_id,
                "data_source": "NDBC Spectral",
                "spectral_data": {}
            }

            # Parse timestamp
            try:
                timestamp = datetime(
                    year=int(data[0]),
                    month=int(data[1]),
                    day=int(data[2]),
                    hour=int(data[3]),
                    minute=int(data[4]),
                    tzinfo=timezone.utc
                )
                result["timestamp"] = timestamp.isoformat()
            except (ValueError, IndexError):
                pass

            # Parse spectral wave components (varies by station)
            for i, header in enumerate(headers):
                if i < len(data):
                    value = self._parse_value(data[i])
                    result["spectral_data"][header] = value

            return result

    async def fetch_historical_data(
        self,
        station_id: str,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch historical buoy data

        Args:
            station_id: NDBC station ID
            year: Year (defaults to current year)
            month: Month (optional, for monthly data)

        Returns:
            DataFrame with historical observations
        """
        if year is None:
            year = datetime.now().year

        logger.info(f"Fetching historical data for buoy {station_id}, year {year}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            if month:
                url = f"{self.base_url}/data/stdmet/{month:02d}/{station_id}.txt"
            else:
                url = f"{self.base_url}/data/historical/stdmet/{station_id}h{year}.txt.gz"

            response = await client.get(url)
            response.raise_for_status()

            # TODO: Parse into DataFrame
            return pd.DataFrame()

    def get_nearby_buoys(
        self, latitude: float, longitude: float, max_distance_km: float = 100
    ) -> List[str]:
        """Find buoys near given coordinates

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            max_distance_km: Maximum distance in kilometers

        Returns:
            List of nearby buoy station IDs
        """
        logger.info(f"Finding buoys near ({latitude}, {longitude})")

        # Common California buoys with their locations
        # This is a simplified static list - could be expanded or fetched dynamically
        known_buoys = {
            "46237": {"name": "San Pedro", "lat": 33.218, "lon": -118.315},
            "46221": {"name": "Santa Monica Basin", "lat": 33.855, "lon": -119.048},
            "46025": {"name": "Santa Monica", "lat": 33.749, "lon": -119.053},
            "46011": {"name": "Santa Maria", "lat": 34.878, "lon": -120.867},
            "46054": {"name": "Point Conception", "lat": 34.274, "lon": -120.477},
            "46026": {"name": "San Francisco", "lat": 37.759, "lon": -122.833},
            "46012": {"name": "Half Moon Bay", "lat": 37.361, "lon": -122.879},
            "46042": {"name": "Monterey Bay", "lat": 36.785, "lon": -122.421},
            "46240": {"name": "Harvest Platform", "lat": 34.455, "lon": -120.537},
        }

        nearby = []
        from math import radians, cos, sin, asin, sqrt

        for station_id, info in known_buoys.items():
            # Haversine distance calculation
            lat1, lon1 = radians(latitude), radians(longitude)
            lat2, lon2 = radians(info["lat"]), radians(info["lon"])

            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            distance_km = 6371 * c

            if distance_km <= max_distance_km:
                nearby.append({
                    "station_id": station_id,
                    "name": info["name"],
                    "distance_km": round(distance_km, 1),
                    "lat": info["lat"],
                    "lon": info["lon"]
                })

        # Sort by distance
        nearby.sort(key=lambda x: x["distance_km"])
        return nearby


class CDIPBuoyFetcher:
    """Fetches data from CDIP buoys (Coastal Data Information Program)

    CDIP provides high-quality wave data from buoys primarily along US West Coast.
    Data is available via THREDDS server in NetCDF format.

    Documentation: https://cdip.ucsd.edu/
    Data Access: https://cdip.ucsd.edu/m/documents/data_access.html
    """

    def __init__(self):
        """Initialize CDIP buoy fetcher"""
        self.thredds_url = "https://thredds.cdip.ucsd.edu/thredds"
        self.base_url = "https://cdip.ucsd.edu"

    async def fetch_latest_observation(self, station_id: str) -> Dict:
        """Fetch latest observation from CDIP buoy

        Note: CDIP data is primarily in NetCDF format accessed via THREDDS.
        This is a simplified implementation that would need netCDF4 library
        for full functionality.

        Args:
            station_id: CDIP station ID (e.g., "067", "094")

        Returns:
            Latest observation data
        """
        logger.info(f"Fetching latest observation for CDIP buoy {station_id}")

        # CDIP provides data via THREDDS/OpenDAP
        # For now, return a placeholder indicating NetCDF parsing needed
        return {
            "station_id": station_id,
            "data_source": "CDIP",
            "note": "CDIP data requires netCDF4 library for full parsing",
            "thredds_url": f"{self.thredds_url}/dodsC/cdip/realtime/{station_id}p1_rt.nc",
            "status": "not_implemented"
        }

    def get_nearby_buoys(
        self, latitude: float, longitude: float, max_distance_km: float = 100
    ) -> List[str]:
        """Find CDIP buoys near given coordinates

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            max_distance_km: Maximum distance in kilometers

        Returns:
            List of nearby CDIP buoy station IDs with info
        """
        logger.info(f"Finding CDIP buoys near ({latitude}, {longitude})")

        # Common CDIP buoys (California)
        known_buoys = {
            "067": {"name": "San Diego South", "lat": 32.57, "lon": -117.17},
            "093": {"name": "Point Loma South", "lat": 32.53, "lon": -117.43},
            "191": {"name": "Santa Monica Bay", "lat": 33.89, "lon": -118.63},
            "111": {"name": "San Pedro", "lat": 33.61, "lon": -118.32},
            "094": {"name": "Oceanside Offshore", "lat": 33.22, "lon": -117.47},
            "100": {"name": "Torrey Pines Outer", "lat": 32.93, "lon": -117.39},
        }

        nearby = []
        from math import radians, cos, sin, asin, sqrt

        for station_id, info in known_buoys.items():
            lat1, lon1 = radians(latitude), radians(longitude)
            lat2, lon2 = radians(info["lat"]), radians(info["lon"])

            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            distance_km = 6371 * c

            if distance_km <= max_distance_km:
                nearby.append({
                    "station_id": station_id,
                    "name": info["name"],
                    "distance_km": round(distance_km, 1),
                    "lat": info["lat"],
                    "lon": info["lon"]
                })

        nearby.sort(key=lambda x: x["distance_km"])
        return nearby
