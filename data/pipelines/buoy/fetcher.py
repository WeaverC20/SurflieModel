"""Fetch NDBC and CDIP buoy data with spectral wave partitioning"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import netCDF4 for CDIP data access
try:
    from netCDF4 import Dataset as NetCDFDataset
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False
    logger.warning("netCDF4 not available - CDIP fetcher will have limited functionality")

# Try to import numpy for array operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("numpy not available - some calculations may be limited")


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
        """Fetch latest spectral wave summary data with partitioned swell components

        The .spec file contains partitioned wave data separating swell from wind waves:
        - WVHT: Significant wave height (m)
        - SwH: Swell height (m)
        - SwP: Swell period (s)
        - SwD: Swell direction (degrees, direction waves are coming FROM)
        - WWH: Wind wave height (m)
        - WWP: Wind wave period (s)
        - WWD: Wind wave direction (degrees)
        - STEEPNESS: Wave steepness
        - APD: Average wave period (s)
        - MWD: Mean wave direction (degrees)

        Args:
            station_id: NDBC station ID

        Returns:
            Spectral wave data including partitioned swell and wind wave components
        """
        logger.info(f"Fetching spectral wave data for NDBC buoy {station_id}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{self.base_url}/data/realtime2/{station_id}.spec"
            response = await client.get(url)
            response.raise_for_status()

            lines = response.text.strip().split("\n")
            if len(lines) < 3:
                return {"error": "Insufficient spectral data", "station_id": station_id}

            # Parse spectral summary format
            # Line 0: Headers
            # Line 1: Units
            # Line 2+: Data rows
            headers = lines[0].replace("#", "").split()
            units = lines[1].replace("#", "").split()
            data = lines[2].split()

            result = {
                "station_id": station_id,
                "data_source": "NDBC Spectral",
                "timestamp": None,
                "swell": None,
                "wind_waves": None,
                "combined": None,
                "raw_spectral": {}
            }

            # Parse timestamp (first 5 columns: YY MM DD hh mm)
            try:
                year = int(data[0])
                # Handle 2-digit years
                if year < 100:
                    year += 2000
                timestamp = datetime(
                    year=year,
                    month=int(data[1]),
                    day=int(data[2]),
                    hour=int(data[3]),
                    minute=int(data[4]),
                    tzinfo=timezone.utc
                )
                result["timestamp"] = timestamp.isoformat()
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse timestamp for {station_id}: {e}")

            # Map headers to values with proper field names
            field_mapping = {
                # Combined wave parameters
                "WVHT": ("significant_wave_height_m", "Significant wave height"),
                "APD": ("average_period_s", "Average wave period"),
                "MWD": ("mean_direction_deg", "Mean wave direction"),
                "STEEPNESS": ("steepness", "Wave steepness category"),
                # Swell components (long period waves from distant storms)
                "SwH": ("swell_height_m", "Swell wave height"),
                "SwP": ("swell_period_s", "Swell wave period"),
                "SwD": ("swell_direction_deg", "Swell direction (from)"),
                # Wind wave components (locally generated)
                "WWH": ("wind_wave_height_m", "Wind wave height"),
                "WWP": ("wind_wave_period_s", "Wind wave period"),
                "WWD": ("wind_wave_direction_deg", "Wind wave direction (from)"),
            }

            # Parse all values
            parsed_data = {}
            for i, header in enumerate(headers):
                if i < len(data):
                    value = self._parse_value(data[i])
                    if header in field_mapping:
                        field_name, description = field_mapping[header]
                        parsed_data[field_name] = value
                    result["raw_spectral"][header] = {
                        "value": value,
                        "unit": units[i] if i < len(units) else None
                    }

            # Organize into swell and wind wave components
            result["swell"] = {
                "height_m": parsed_data.get("swell_height_m"),
                "height_ft": round(parsed_data.get("swell_height_m", 0) * 3.28084, 1) if parsed_data.get("swell_height_m") else None,
                "period_s": parsed_data.get("swell_period_s"),
                "direction_deg": parsed_data.get("swell_direction_deg"),
                "direction_cardinal": self._deg_to_cardinal(parsed_data.get("swell_direction_deg")),
            }

            result["wind_waves"] = {
                "height_m": parsed_data.get("wind_wave_height_m"),
                "height_ft": round(parsed_data.get("wind_wave_height_m", 0) * 3.28084, 1) if parsed_data.get("wind_wave_height_m") else None,
                "period_s": parsed_data.get("wind_wave_period_s"),
                "direction_deg": parsed_data.get("wind_wave_direction_deg"),
                "direction_cardinal": self._deg_to_cardinal(parsed_data.get("wind_wave_direction_deg")),
            }

            result["combined"] = {
                "significant_height_m": parsed_data.get("significant_wave_height_m"),
                "significant_height_ft": round(parsed_data.get("significant_wave_height_m", 0) * 3.28084, 1) if parsed_data.get("significant_wave_height_m") else None,
                "average_period_s": parsed_data.get("average_period_s"),
                "mean_direction_deg": parsed_data.get("mean_direction_deg"),
                "mean_direction_cardinal": self._deg_to_cardinal(parsed_data.get("mean_direction_deg")),
                "steepness": parsed_data.get("steepness"),
            }

            return result

    def _deg_to_cardinal(self, degrees: Optional[float]) -> Optional[str]:
        """Convert degrees to cardinal direction

        Args:
            degrees: Direction in degrees (0-360, meteorological convention - direction FROM)

        Returns:
            Cardinal direction string (N, NE, E, etc.) or None
        """
        if degrees is None:
            return None

        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        index = round(degrees / 22.5) % 16
        return directions[index]

    async def fetch_directional_spectra(self, station_id: str) -> Dict:
        """Fetch full directional spectral data from multiple NDBC files

        This fetches the raw spectral density and directional parameters
        that can be used to compute custom wave partitioning.

        Files fetched:
        - .data_spec: Spectral energy density by frequency
        - .swdir (alpha1): Mean wave direction per frequency
        - .swdir2 (alpha2): Principal wave direction per frequency
        - .swr1 (r1): Directional spreading parameter
        - .swr2 (r2): Directional spreading parameter

        Args:
            station_id: NDBC station ID

        Returns:
            Dict with full directional spectral data
        """
        logger.info(f"Fetching directional spectra for NDBC buoy {station_id}")

        result = {
            "station_id": station_id,
            "data_source": "NDBC Directional Spectra",
            "timestamp": None,
            "frequencies": [],
            "energy_density": [],
            "alpha1": [],  # Mean direction
            "alpha2": [],  # Principal direction
            "r1": [],      # Spreading param 1
            "r2": [],      # Spreading param 2
            "available_files": [],
            "errors": []
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch spectral energy density
            try:
                url = f"{self.base_url}/data/realtime2/{station_id}.data_spec"
                response = await client.get(url)
                response.raise_for_status()

                lines = response.text.strip().split("\n")
                if len(lines) >= 3:
                    result["available_files"].append("data_spec")
                    # Parse frequency/energy pairs from the data line
                    data_line = lines[2].split()
                    # First 5 values are timestamp, then sep_freq, then alternating spec/freq
                    if len(data_line) > 6:
                        try:
                            year = int(data_line[0])
                            if year < 100:
                                year += 2000
                            result["timestamp"] = datetime(
                                year=year,
                                month=int(data_line[1]),
                                day=int(data_line[2]),
                                hour=int(data_line[3]),
                                minute=int(data_line[4]),
                                tzinfo=timezone.utc
                            ).isoformat()
                        except (ValueError, IndexError):
                            pass

                        # Parse separation frequency
                        result["separation_frequency"] = self._parse_value(data_line[5])

                        # Parse spectral values (format: spec_1 (freq_1) spec_2 (freq_2) ...)
                        i = 6
                        while i < len(data_line) - 1:
                            energy = self._parse_value(data_line[i])
                            freq_str = data_line[i + 1].strip("()")
                            freq = self._parse_value(freq_str)
                            if energy is not None and freq is not None:
                                result["energy_density"].append(energy)
                                result["frequencies"].append(freq)
                            i += 2
            except httpx.HTTPError as e:
                result["errors"].append(f"data_spec: {str(e)}")

            # Fetch directional parameters (alpha1, alpha2, r1, r2)
            dir_files = [
                ("swdir", "alpha1"),
                ("swdir2", "alpha2"),
                ("swr1", "r1"),
                ("swr2", "r2")
            ]

            for file_ext, param_name in dir_files:
                try:
                    url = f"{self.base_url}/data/realtime2/{station_id}.{file_ext}"
                    response = await client.get(url)
                    response.raise_for_status()

                    lines = response.text.strip().split("\n")
                    if len(lines) >= 3:
                        result["available_files"].append(file_ext)
                        data_line = lines[2].split()
                        # Parse values (format: value_1 (freq_1) value_2 (freq_2) ...)
                        i = 5  # Skip timestamp
                        values = []
                        while i < len(data_line) - 1:
                            value = self._parse_value(data_line[i])
                            if value is not None:
                                values.append(value)
                            i += 2
                        result[param_name] = values
                except httpx.HTTPError as e:
                    result["errors"].append(f"{file_ext}: {str(e)}")

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
    Data is available via THREDDS server in NetCDF format with OpenDAP access.

    Key features:
    - Real-time and historical wave data
    - Partitioned wave spectra (multiple swell trains identified)
    - 2D directional spectra using Maximum Entropy Method
    - Higher quality directional measurements than NDBC

    Documentation: https://cdip.ucsd.edu/
    Data Access: https://cdip.ucsd.edu/m/documents/data_access.html
    THREDDS: https://thredds.cdip.ucsd.edu/thredds/catalog.html
    """

    # Known California CDIP buoys with metadata
    CALIFORNIA_BUOYS = {
        "028": {"name": "Santa Monica Bay", "lat": 33.858, "lon": -118.633, "depth_m": 365},
        "045": {"name": "Oceanside Offshore", "lat": 33.178, "lon": -117.472, "depth_m": 215},
        "067": {"name": "San Diego", "lat": 32.570, "lon": -117.169, "depth_m": 27},
        "071": {"name": "Harvest", "lat": 34.451, "lon": -120.779, "depth_m": 549},
        "076": {"name": "Diablo Canyon", "lat": 35.203, "lon": -120.859, "depth_m": 26},
        "093": {"name": "Point Loma South", "lat": 32.530, "lon": -117.431, "depth_m": 305},
        "094": {"name": "Oceanside Harbor", "lat": 33.216, "lon": -117.435, "depth_m": 183},
        "096": {"name": "Goleta Point", "lat": 34.430, "lon": -119.872, "depth_m": 549},
        "100": {"name": "Torrey Pines Outer", "lat": 32.933, "lon": -117.391, "depth_m": 549},
        "101": {"name": "Torrey Pines Inner", "lat": 32.925, "lon": -117.279, "depth_m": 23},
        "106": {"name": "Cabrillo Point", "lat": 36.626, "lon": -121.907, "depth_m": 21},
        "107": {"name": "Monterey Canyon", "lat": 36.723, "lon": -121.996, "depth_m": 256},
        "111": {"name": "San Pedro Basin", "lat": 33.616, "lon": -118.316, "depth_m": 100},
        "132": {"name": "Point Reyes", "lat": 37.936, "lon": -123.463, "depth_m": 550},
        "139": {"name": "Point San Luis", "lat": 35.195, "lon": -120.741, "depth_m": 22},
        "142": {"name": "Point Sal", "lat": 34.903, "lon": -120.688, "depth_m": 550},
        "155": {"name": "Ipan, Guam", "lat": 13.354, "lon": 144.788, "depth_m": 200},  # Not CA but included
        "157": {"name": "Point Reyes Nearshore", "lat": 37.951, "lon": -122.978, "depth_m": 23},
        "158": {"name": "Del Mar Nearshore", "lat": 32.957, "lon": -117.280, "depth_m": 18},
        "168": {"name": "Rincon", "lat": 34.378, "lon": -119.458, "depth_m": 227},
        "179": {"name": "Astoria Canyon", "lat": 46.133, "lon": -124.644, "depth_m": 134},  # Oregon
        "181": {"name": "Grays Harbor", "lat": 46.857, "lon": -124.244, "depth_m": 42},  # Washington
        "191": {"name": "Huntington Beach", "lat": 33.632, "lon": -118.055, "depth_m": 21},
        "192": {"name": "San Nicholas Island", "lat": 33.219, "lon": -119.456, "depth_m": 295},
        "198": {"name": "Fort Point, SF", "lat": 37.810, "lon": -122.466, "depth_m": 24},
        "200": {"name": "Anacapa Passage", "lat": 34.055, "lon": -119.360, "depth_m": 265},
        "201": {"name": "Ipan Reef, Guam", "lat": 13.355, "lon": 144.783, "depth_m": 10},
        "203": {"name": "Point Mugu", "lat": 34.037, "lon": -119.241, "depth_m": 183},
        "204": {"name": "Cape Mendocino", "lat": 40.295, "lon": -124.724, "depth_m": 344},
        "213": {"name": "Point Loma Nearshore", "lat": 32.687, "lon": -117.293, "depth_m": 30},
        "214": {"name": "Santa Cruz Basin", "lat": 33.769, "lon": -119.565, "depth_m": 499},
        "215": {"name": "LA/Long Beach", "lat": 33.700, "lon": -118.258, "depth_m": 43},
        "217": {"name": "Ala Wai, Oahu", "lat": 21.270, "lon": -157.840, "depth_m": 10},  # Hawaii
        "220": {"name": "Wilmington Oil Field", "lat": 33.750, "lon": -118.230, "depth_m": 24},
        "221": {"name": "Hanalei, Kauai", "lat": 22.224, "lon": -159.575, "depth_m": 200},  # Hawaii
        "222": {"name": "Waimea Bay, Oahu", "lat": 21.673, "lon": -158.117, "depth_m": 200},  # Hawaii
        "224": {"name": "Santa Lucia Bank", "lat": 34.767, "lon": -121.500, "depth_m": 300},
        "226": {"name": "Palos Verdes Shelf", "lat": 33.700, "lon": -118.440, "depth_m": 80},
    }

    def __init__(self):
        """Initialize CDIP buoy fetcher"""
        self.thredds_url = "https://thredds.cdip.ucsd.edu/thredds"
        self.base_url = "https://cdip.ucsd.edu"

    async def fetch_latest_observation(self, station_id: str) -> Dict:
        """Fetch latest observation from CDIP buoy via THREDDS/OpenDAP

        Args:
            station_id: CDIP station ID (e.g., "067", "094")

        Returns:
            Latest observation data including wave parameters
        """
        logger.info(f"Fetching latest observation for CDIP buoy {station_id}")

        if not NETCDF4_AVAILABLE:
            return {
                "station_id": station_id,
                "data_source": "CDIP",
                "error": "netCDF4 library not available",
                "thredds_url": f"{self.thredds_url}/dodsC/cdip/realtime/{station_id}p1_rt.nc",
                "status": "unavailable"
            }

        try:
            # OpenDAP URL for realtime data
            url = f"{self.thredds_url}/dodsC/cdip/realtime/{station_id}p1_rt.nc"
            logger.debug(f"Opening CDIP dataset: {url}")

            # Open the NetCDF dataset via OpenDAP
            ds = NetCDFDataset(url)

            result = {
                "station_id": station_id,
                "data_source": "CDIP",
                "thredds_url": url,
                "status": "success",
                "timestamp": None,
                "wave_height_m": None,
                "wave_height_ft": None,
                "dominant_period_s": None,
                "average_period_s": None,
                "peak_direction_deg": None,
                "mean_direction_deg": None,
                "water_temp_c": None,
            }

            # Get the latest timestamp
            try:
                wave_time = ds.variables['waveTime'][:]
                if len(wave_time) > 0:
                    latest_idx = -1  # Get most recent
                    # CDIP times are Unix timestamps
                    latest_time = datetime.fromtimestamp(float(wave_time[latest_idx]), tz=timezone.utc)
                    result["timestamp"] = latest_time.isoformat()
            except (KeyError, IndexError) as e:
                logger.warning(f"Could not get timestamp for CDIP {station_id}: {e}")

            # Get wave parameters
            try:
                hs = ds.variables['waveHs'][:]
                if len(hs) > 0:
                    result["wave_height_m"] = float(hs[-1])
                    result["wave_height_ft"] = round(float(hs[-1]) * 3.28084, 1)
            except (KeyError, IndexError):
                pass

            try:
                tp = ds.variables['waveTp'][:]
                if len(tp) > 0:
                    result["dominant_period_s"] = float(tp[-1])
            except (KeyError, IndexError):
                pass

            try:
                ta = ds.variables['waveTa'][:]
                if len(ta) > 0:
                    result["average_period_s"] = float(ta[-1])
            except (KeyError, IndexError):
                pass

            try:
                dp = ds.variables['waveDp'][:]
                if len(dp) > 0:
                    result["peak_direction_deg"] = float(dp[-1])
            except (KeyError, IndexError):
                pass

            try:
                dm = ds.variables['waveDm'][:]  # Mean direction
                if len(dm) > 0:
                    result["mean_direction_deg"] = float(dm[-1])
            except (KeyError, IndexError):
                pass

            # Water temperature
            try:
                sst = ds.variables['sstSeaSurfaceTemperature'][:]
                if len(sst) > 0:
                    result["water_temp_c"] = float(sst[-1])
            except (KeyError, IndexError):
                pass

            ds.close()
            return result

        except Exception as e:
            logger.error(f"Error fetching CDIP data for {station_id}: {e}")
            return {
                "station_id": station_id,
                "data_source": "CDIP",
                "error": str(e),
                "status": "error"
            }

    async def fetch_partitioned_wave_data(self, station_id: str) -> Dict:
        """Fetch partitioned wave data from CDIP showing multiple swell components

        CDIP uses the Maximum Entropy Method (MEM) for 2D spectral estimation,
        then partitions using methods like Portilla et al. (2009) to identify
        individual wave fields (multiple swells from different source regions).

        Args:
            station_id: CDIP station ID

        Returns:
            Dict with partitioned swell data showing multiple wave trains
        """
        logger.info(f"Fetching partitioned wave data for CDIP buoy {station_id}")

        if not NETCDF4_AVAILABLE:
            return {
                "station_id": station_id,
                "data_source": "CDIP Partitioned",
                "error": "netCDF4 library not available",
                "status": "unavailable"
            }

        try:
            url = f"{self.thredds_url}/dodsC/cdip/realtime/{station_id}p1_rt.nc"
            ds = NetCDFDataset(url)

            result = {
                "station_id": station_id,
                "data_source": "CDIP Partitioned",
                "thredds_url": url,
                "partition_url": f"{self.base_url}/m/products/partition/?stn={station_id}p1",
                "status": "success",
                "timestamp": None,
                "partitions": [],
                "combined": None,
            }

            # Get timestamp
            try:
                wave_time = ds.variables['waveTime'][:]
                if len(wave_time) > 0:
                    latest_time = datetime.fromtimestamp(float(wave_time[-1]), tz=timezone.utc)
                    result["timestamp"] = latest_time.isoformat()
            except (KeyError, IndexError):
                pass

            # Get combined wave parameters
            try:
                hs = float(ds.variables['waveHs'][:][-1])
                tp = float(ds.variables['waveTp'][:][-1])
                dp = float(ds.variables['waveDp'][:][-1])

                result["combined"] = {
                    "significant_height_m": hs,
                    "significant_height_ft": round(hs * 3.28084, 1),
                    "peak_period_s": tp,
                    "peak_direction_deg": dp,
                }
            except (KeyError, IndexError):
                pass

            # Try to get partitioned wave data
            # CDIP provides wave source breakdown when available
            partition_vars = [
                ('waveSourceHs', 'height'),
                ('waveSourceTp', 'period'),
                ('waveSourceDp', 'direction'),
            ]

            # Check if partition data exists
            has_partitions = 'waveSourceHs' in ds.variables

            if has_partitions:
                try:
                    source_hs = ds.variables['waveSourceHs'][:]
                    source_tp = ds.variables['waveSourceTp'][:]
                    source_dp = ds.variables['waveSourceDp'][:]

                    # Get the latest values (last time index, all partitions)
                    if len(source_hs.shape) > 1:
                        latest_hs = source_hs[-1, :]
                        latest_tp = source_tp[-1, :]
                        latest_dp = source_dp[-1, :]

                        for i in range(len(latest_hs)):
                            hs_val = float(latest_hs[i])
                            if hs_val > 0.01:  # Filter out near-zero partitions
                                partition = {
                                    "partition_id": i + 1,
                                    "height_m": hs_val,
                                    "height_ft": round(hs_val * 3.28084, 1),
                                    "period_s": float(latest_tp[i]) if i < len(latest_tp) else None,
                                    "direction_deg": float(latest_dp[i]) if i < len(latest_dp) else None,
                                    "type": self._classify_wave_type(float(latest_tp[i]) if i < len(latest_tp) else 0),
                                }
                                result["partitions"].append(partition)
                except (KeyError, IndexError, TypeError) as e:
                    logger.warning(f"Could not parse partition data for {station_id}: {e}")

            # If no partitions available, try to derive from spectral data
            if not result["partitions"]:
                result["note"] = "Full partition data not available; consider using spectral decomposition"

            ds.close()
            return result

        except Exception as e:
            logger.error(f"Error fetching CDIP partition data for {station_id}: {e}")
            return {
                "station_id": station_id,
                "data_source": "CDIP Partitioned",
                "error": str(e),
                "status": "error"
            }

    def _classify_wave_type(self, period_s: float) -> str:
        """Classify wave type based on period

        Args:
            period_s: Wave period in seconds

        Returns:
            Wave type classification
        """
        if period_s >= 16:
            return "long_period_swell"  # Distant storm, e.g., Southern Hemisphere
        elif period_s >= 12:
            return "swell"  # Regional swell
        elif period_s >= 8:
            return "short_swell"  # Short period swell
        else:
            return "wind_waves"  # Locally generated

    async def fetch_spectral_data(self, station_id: str) -> Dict:
        """Fetch full spectral data from CDIP

        Args:
            station_id: CDIP station ID

        Returns:
            Dict with frequency spectrum and directional data
        """
        logger.info(f"Fetching spectral data for CDIP buoy {station_id}")

        if not NETCDF4_AVAILABLE:
            return {
                "station_id": station_id,
                "data_source": "CDIP Spectral",
                "error": "netCDF4 library not available",
                "status": "unavailable"
            }

        try:
            url = f"{self.thredds_url}/dodsC/cdip/realtime/{station_id}p1_rt.nc"
            ds = NetCDFDataset(url)

            result = {
                "station_id": station_id,
                "data_source": "CDIP Spectral",
                "thredds_url": url,
                "status": "success",
                "timestamp": None,
                "frequencies": [],
                "energy_density": [],
                "mean_direction": [],
                "directional_spread": [],
            }

            # Get timestamp
            try:
                wave_time = ds.variables['waveTime'][:]
                if len(wave_time) > 0:
                    latest_time = datetime.fromtimestamp(float(wave_time[-1]), tz=timezone.utc)
                    result["timestamp"] = latest_time.isoformat()
            except (KeyError, IndexError):
                pass

            # Get frequency bands
            try:
                freqs = ds.variables['waveFrequency'][:]
                result["frequencies"] = [float(f) for f in freqs]
            except (KeyError, IndexError):
                pass

            # Get spectral energy density
            try:
                energy = ds.variables['waveEnergyDensity'][:]
                if len(energy.shape) > 1:
                    result["energy_density"] = [float(e) for e in energy[-1, :]]
                else:
                    result["energy_density"] = [float(e) for e in energy]
            except (KeyError, IndexError):
                pass

            # Get mean direction per frequency
            try:
                mean_dir = ds.variables['waveMeanDirection'][:]
                if len(mean_dir.shape) > 1:
                    result["mean_direction"] = [float(d) for d in mean_dir[-1, :]]
                else:
                    result["mean_direction"] = [float(d) for d in mean_dir]
            except (KeyError, IndexError):
                pass

            # Get directional spread
            try:
                spread = ds.variables['waveSpread'][:]  # or waveA1, waveB1, etc.
                if len(spread.shape) > 1:
                    result["directional_spread"] = [float(s) for s in spread[-1, :]]
                else:
                    result["directional_spread"] = [float(s) for s in spread]
            except (KeyError, IndexError):
                pass

            ds.close()
            return result

        except Exception as e:
            logger.error(f"Error fetching CDIP spectral data for {station_id}: {e}")
            return {
                "station_id": station_id,
                "data_source": "CDIP Spectral",
                "error": str(e),
                "status": "error"
            }

    def get_california_buoys(self) -> Dict[str, Dict]:
        """Get all known California CDIP buoys

        Returns:
            Dict of station_id -> buoy info
        """
        # Filter to California region (roughly 32-42 lat, -125 to -117 lon)
        ca_buoys = {}
        for station_id, info in self.CALIFORNIA_BUOYS.items():
            if 32 <= info["lat"] <= 42 and -125 <= info["lon"] <= -117:
                ca_buoys[station_id] = info
        return ca_buoys

    def get_nearby_buoys(
        self, latitude: float, longitude: float, max_distance_km: float = 100
    ) -> List[Dict]:
        """Find CDIP buoys near given coordinates

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            max_distance_km: Maximum distance in kilometers

        Returns:
            List of nearby CDIP buoy station IDs with info
        """
        logger.info(f"Finding CDIP buoys near ({latitude}, {longitude})")

        nearby = []
        from math import radians, cos, sin, asin, sqrt

        for station_id, info in self.CALIFORNIA_BUOYS.items():
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
                    "lon": info["lon"],
                    "depth_m": info.get("depth_m"),
                })

        nearby.sort(key=lambda x: x["distance_km"])
        return nearby
