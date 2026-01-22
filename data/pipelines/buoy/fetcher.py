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

# Try to import scipy for peak finding in spectral partitioning
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - spectral partitioning will be limited")


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

    async def fetch_partitioned_spectral_data(self, station_id: str) -> Dict:
        """Fetch spectral data and partition into swell components with r1 confidence.

        This method fetches the full directional spectra and partitions it by
        grouping frequency bands with similar directions. Each partition includes
        an r1-based confidence indicator showing how reliable the direction estimate is.

        r1 interpretation:
        - r1 > 0.8: HIGH confidence - narrow, well-defined swell from single direction
        - r1 0.5-0.8: MEDIUM confidence - moderate spread
        - r1 < 0.5: LOW confidence - broad spread, possibly multiple overlapping swells

        Args:
            station_id: NDBC station ID

        Returns:
            Dict with partitioned wave data including r1 confidence
        """
        logger.info(f"Fetching partitioned spectral data for NDBC buoy {station_id}")

        # Fetch raw spectral data
        spectra = await self.fetch_directional_spectra(station_id)

        if spectra.get("errors") or not spectra.get("frequencies"):
            return {
                "station_id": station_id,
                "data_source": "NDBC Partitioned Spectral",
                "error": spectra.get("errors", ["No spectral data"])[0] if spectra.get("errors") else "No frequency data",
                "status": "error"
            }

        result = {
            "station_id": station_id,
            "data_source": "NDBC Partitioned Spectral",
            "timestamp": spectra.get("timestamp"),
            "status": "success",
            "partitions": [],
            "combined": None,
        }

        if not NUMPY_AVAILABLE:
            result["error"] = "numpy not available for partitioning"
            return result

        freqs = np.array(spectra["frequencies"])
        energy = np.array(spectra["energy_density"])
        alpha1 = np.array(spectra.get("alpha1", []))
        r1 = np.array(spectra.get("r1", []))

        # Ensure all arrays have same length
        min_len = min(len(freqs), len(energy), len(alpha1), len(r1)) if len(alpha1) > 0 and len(r1) > 0 else 0
        if min_len == 0:
            result["note"] = "Insufficient directional data for partitioning"
            return result

        freqs = freqs[:min_len]
        energy = energy[:min_len]
        alpha1 = alpha1[:min_len]
        r1 = r1[:min_len]

        # Calculate frequency bandwidth
        df = np.gradient(freqs)

        # Calculate total energy and combined parameters
        m0_total = np.sum(energy * df)
        if m0_total <= 0:
            result["note"] = "No significant wave energy"
            return result

        hs_total = 4 * np.sqrt(m0_total)
        peak_idx = np.argmax(energy)
        tp_total = 1 / freqs[peak_idx]
        dp_total = alpha1[peak_idx]

        result["combined"] = {
            "significant_height_m": round(float(hs_total), 2),
            "significant_height_ft": round(float(hs_total) * 3.28084, 1),
            "peak_period_s": round(float(tp_total), 1),
            "peak_direction_deg": round(float(dp_total), 0),
        }

        # Direction-based partitioning
        dir_threshold = 30  # degrees
        min_energy_pct = 5  # minimum % of total energy

        # Find significant frequency bands
        peak_energy = np.max(energy)
        significant = energy > 0.01 * peak_energy

        # Group consecutive bands with similar directions
        groups = []
        current_group = []

        def direction_distance(d1, d2):
            diff = abs(d1 - d2)
            return min(diff, 360 - diff)

        for i in range(len(freqs)):
            if not significant[i]:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                continue

            if not current_group:
                current_group = [i]
            else:
                # Check if direction is similar to group average
                group_dirs = [alpha1[j] for j in current_group]
                group_energies = [energy[j] for j in current_group]
                avg_dir = np.average(group_dirs, weights=group_energies)

                if direction_distance(alpha1[i], avg_dir) < dir_threshold:
                    current_group.append(i)
                else:
                    groups.append(current_group)
                    current_group = [i]

        if current_group:
            groups.append(current_group)

        # Calculate partition statistics
        partitions = []
        for indices in groups:
            g_energy = energy[indices]
            g_df = df[indices]
            g_dir = alpha1[indices]
            g_r1 = r1[indices]
            g_freq = freqs[indices]

            m0 = np.sum(g_energy * g_df)
            energy_pct = 100 * m0 / m0_total

            if energy_pct < min_energy_pct:
                continue

            hs = 4 * np.sqrt(m0)
            peak_idx = np.argmax(g_energy)
            tp = 1 / g_freq[peak_idx]
            mean_dir = np.average(g_dir, weights=g_energy)
            mean_r1 = np.average(g_r1, weights=g_energy)

            # Classify confidence based on r1
            if mean_r1 > 0.7:
                confidence = "HIGH"
            elif mean_r1 > 0.4:
                confidence = "MED"
            else:
                confidence = "LOW"

            # Classify wave type
            if tp >= 16:
                wave_type = "long_period_swell"
            elif tp >= 12:
                wave_type = "swell"
            elif tp >= 8:
                wave_type = "short_swell"
            else:
                wave_type = "wind_waves"

            partitions.append({
                "partition_id": len(partitions) + 1,
                "height_m": round(float(hs), 2),
                "height_ft": round(float(hs) * 3.28084, 1),
                "period_s": round(float(tp), 1),
                "direction_deg": round(float(mean_dir), 0),
                "type": wave_type,
                "energy_pct": round(float(energy_pct), 0),
                "r1": round(float(mean_r1), 2),
                "confidence": confidence,
            })

        # Sort by energy descending and renumber
        partitions.sort(key=lambda p: p.get("energy_pct", 0), reverse=True)
        for i, p in enumerate(partitions):
            p["partition_id"] = i + 1

        result["partitions"] = partitions
        return result


class CDIPBuoyFetcher:
    """Fetches data from CDIP buoys (Coastal Data Information Program)

    CDIP provides high-quality wave data from buoys primarily along US West Coast.

    Data access methods (in order of reliability):
    1. ERDDAP - REST API returning CSV with QC'd bulk parameters (Hs, Tp, Dp)
    2. ndar.cdip CGI - 9-band energy/direction breakdown by period
    3. THREDDS/OpenDAP - Full NetCDF access (fallback only)

    Key features:
    - Real-time and historical wave data
    - 9-band spectral breakdown (energy by period band with direction)
    - Higher quality directional measurements than NDBC

    Documentation: https://cdip.ucsd.edu/
    Data Access: https://cdip.ucsd.edu/m/documents/data_access.html
    ERDDAP: https://erddap.cdip.ucsd.edu/erddap/
    """

    # API endpoints
    ERDDAP_BASE = "https://erddap.cdip.ucsd.edu/erddap/tabledap"
    NDAR_BASE = "https://cdip.ucsd.edu/data_access/ndar.cdip"

    # 9-band period ranges (approximate center periods in seconds)
    # Based on CDIP frequency bands: 0.025-0.58 Hz divided into 9 bands
    BAND_PERIODS = {
        1: {"center": 22.0, "type": "long_period_swell"},  # ~22+ sec
        2: {"center": 20.0, "type": "long_period_swell"},  # ~20-22 sec
        3: {"center": 18.0, "type": "long_period_swell"},  # ~18-20 sec
        4: {"center": 16.0, "type": "swell"},              # ~16-18 sec
        5: {"center": 14.0, "type": "swell"},              # ~14-16 sec
        6: {"center": 12.0, "type": "swell"},              # ~12-14 sec
        7: {"center": 10.0, "type": "short_swell"},        # ~10-12 sec
        8: {"center": 8.0, "type": "short_swell"},         # ~8-10 sec
        9: {"center": 6.5, "type": "wind_waves"},          # ~6-8 sec
    }

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

        # Fix netCDF4 SSL certificate path bug (issue with paths containing spaces)
        # The bundled libcurl corrupts paths with spaces, so use system CA bundle
        if NETCDF4_AVAILABLE:
            try:
                from netCDF4 import rc_set
                rc_set('HTTP.SSL.CAINFO', '/etc/ssl/cert.pem')
            except Exception:
                pass  # If rc_set fails, continue anyway

    async def _fetch_erddap_latest(self, station_id: str) -> Optional[Dict]:
        """Fetch latest bulk parameters (Hs, Tp, Dp) from ERDDAP.

        Most reliable source - returns QC'd CSV data via simple HTTP.
        No netCDF4 dependency needed.

        Args:
            station_id: CDIP station ID (e.g., "100", "067")

        Returns:
            Dict with waveHs, waveTp, waveDp, timestamp, lat, lon or None if failed
        """
        try:
            # ERDDAP query for latest observation from specific station
            # orderByMax("time") ensures we get the most recent record
            url = (
                f"{self.ERDDAP_BASE}/wave_agg.csv?"
                f"station_id,time,waveHs,waveTp,waveDp,latitude,longitude"
                f'&station_id="{station_id}"'
                f'&orderByMax("time")'
            )

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Parse CSV response
                # ERDDAP returns: 1) headers, 2) units, 3) data
                lines = response.text.strip().split("\n")
                if len(lines) < 3:
                    logger.warning(f"ERDDAP returned no data for CDIP {station_id}")
                    return None

                # First line is headers, second is units, third is data
                headers = lines[0].split(",")
                values = lines[2].split(",")  # Skip units row (line 1)

                if len(values) != len(headers):
                    logger.warning(f"ERDDAP CSV parse error for CDIP {station_id}")
                    return None

                data = dict(zip(headers, values))

                # Parse values
                result = {
                    "station_id": data.get("station_id", "").strip('"'),
                    "timestamp": data.get("time", "").strip('"'),
                    "waveHs": self._parse_float(data.get("waveHs")),
                    "waveTp": self._parse_float(data.get("waveTp")),
                    "waveDp": self._parse_float(data.get("waveDp")),
                    "latitude": self._parse_float(data.get("latitude")),
                    "longitude": self._parse_float(data.get("longitude")),
                }

                return result

        except httpx.HTTPError as e:
            logger.warning(f"ERDDAP request failed for CDIP {station_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"ERDDAP parse error for CDIP {station_id}: {e}")
            return None

    async def _fetch_9band_data(self, station_id: str) -> Optional[List[Dict]]:
        """Fetch 9-band energy/direction breakdown from ndar.cdip CGI.

        Returns energy distribution across 9 period bands with direction.
        Format: https://cdip.ucsd.edu/data_access/ndar.cdip?{stn}+9c+1+h

        The 9c format returns combined energy (cm²) and direction (deg) for each band.
        Format is: YYYYMMDDHHMM E1 D1 E2 D2 E3 D3 E4 D4 E5 D5 E6 D6 E7 D7 E8 D8 E9 D9

        Args:
            station_id: CDIP station ID

        Returns:
            List of dicts with band_num, energy_m2, direction_deg, or None if failed
        """
        try:
            # 9c = combined 9-band energy + direction, 1 = last 1 day, h = include header
            url = f"{self.NDAR_BASE}?{station_id}+9c+1+h"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()

                lines = response.text.strip().split("\n")
                if len(lines) < 3:
                    logger.warning(f"ndar.cdip returned insufficient 9-band data for CDIP {station_id}")
                    return None

                # Skip header lines (first 2 lines are headers)
                # Data lines look like: 202601211400      2 283     12 264     70 258 ...
                # Find the last data line (skip any that start with letters or are too short)
                data_line = None
                for line in reversed(lines):
                    stripped = line.strip()
                    if stripped and stripped[0].isdigit():
                        data_line = stripped
                        break

                if not data_line:
                    logger.warning(f"No valid data line found for CDIP {station_id}")
                    return None

                # Split by whitespace - format is: YYYYMMDDHHMM E1 D1 E2 D2 ...
                fields = data_line.split()

                # Need at least: timestamp + 9 bands * 2 values = 19 fields
                if len(fields) < 19:
                    logger.warning(f"9-band data incomplete for CDIP {station_id}: {len(fields)} fields")
                    return None

                # Parse timestamp (YYYYMMDDHHMM format)
                timestamp = None
                try:
                    ts_str = fields[0]
                    if len(ts_str) == 12:
                        year = int(ts_str[0:4])
                        month = int(ts_str[4:6])
                        day = int(ts_str[6:8])
                        hour = int(ts_str[8:10])
                        minute = int(ts_str[10:12])
                        timestamp = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
                except (ValueError, IndexError):
                    pass

                # Parse the 9 band pairs (E1,D1 through E9,D9)
                # Fields 1-18 are the band data (0 is timestamp)
                bands = []
                for band_num in range(1, 10):
                    e_idx = 1 + (band_num - 1) * 2  # Energy field index
                    d_idx = e_idx + 1              # Direction field index

                    if e_idx < len(fields) and d_idx < len(fields):
                        # Energy is in cm², convert to m² (divide by 10000)
                        energy_cm2 = self._parse_float(fields[e_idx])
                        direction = self._parse_float(fields[d_idx])

                        # Skip if no valid energy
                        if energy_cm2 is None:
                            continue

                        # Convert cm² to m² for Hs calculation
                        energy_m2 = energy_cm2 / 10000.0

                        bands.append({
                            "band_num": band_num,
                            "energy_m2": energy_m2,
                            "energy_cm2": energy_cm2,  # Keep original for debugging
                            "direction_deg": direction if direction is not None and 0 <= direction <= 360 else None,
                            "period_s": self.BAND_PERIODS[band_num]["center"],
                            "wave_type": self.BAND_PERIODS[band_num]["type"],
                            "timestamp": timestamp.isoformat() if timestamp else None,
                        })

                return bands if bands else None

        except httpx.HTTPError as e:
            logger.warning(f"ndar.cdip request failed for CDIP {station_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"9-band parse error for CDIP {station_id}: {e}")
            return None

    def _9band_to_partitions(self, bands: List[Dict]) -> List[Dict]:
        """Convert 9-band data to partition format.

        Groups bands by wave type and calculates combined Hs for each group.
        Returns partitions sorted by energy (descending).

        Args:
            bands: List of band dicts from _fetch_9band_data()

        Returns:
            List of partition dicts with height_m, period_s, direction_deg, type, energy_pct
        """
        if not bands:
            return []

        # Group bands by wave type
        groups = {}
        for band in bands:
            wave_type = band.get("wave_type", "unknown")
            if wave_type not in groups:
                groups[wave_type] = []
            groups[wave_type].append(band)

        # Calculate total energy (sum of all band energies)
        total_energy = sum(b.get("energy_m2", 0) for b in bands)
        if total_energy <= 0:
            return []

        # Create partitions from groups
        partitions = []
        for wave_type, group_bands in groups.items():
            group_energy = sum(b.get("energy_m2", 0) for b in group_bands)
            if group_energy <= 0:
                continue

            # Hs = 4 * sqrt(m0), where m0 is the zeroth moment (energy)
            hs = 4 * (group_energy ** 0.5)

            # Energy-weighted average direction
            dir_sum = 0
            dir_weight = 0
            for b in group_bands:
                if b.get("direction_deg") is not None:
                    dir_sum += b["direction_deg"] * b.get("energy_m2", 0)
                    dir_weight += b.get("energy_m2", 0)
            mean_dir = dir_sum / dir_weight if dir_weight > 0 else None

            # Use the dominant band's period (highest energy in group)
            dominant_band = max(group_bands, key=lambda b: b.get("energy_m2", 0))
            period = dominant_band.get("period_s")

            energy_pct = 100 * group_energy / total_energy

            partitions.append({
                "partition_id": 0,  # Will be renumbered
                "height_m": round(hs, 2),
                "height_ft": round(hs * 3.28084, 1),
                "period_s": period,
                "direction_deg": round(mean_dir, 0) if mean_dir is not None else None,
                "type": wave_type,
                "energy_pct": round(energy_pct, 0),
            })

        # Sort by energy percentage descending and renumber
        partitions.sort(key=lambda p: p.get("energy_pct", 0), reverse=True)
        for i, p in enumerate(partitions):
            p["partition_id"] = i + 1

        return partitions

    def _parse_float(self, value: Any) -> Optional[float]:
        """Safely parse a float value, handling missing/invalid data."""
        if value is None:
            return None
        try:
            val = float(str(value).strip().strip('"'))
            # Common missing value indicators
            if val in (999.0, 9999.0, -999.0, -9999.0):
                return None
            return val
        except (ValueError, TypeError):
            return None

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
        """Fetch partitioned wave data from CDIP showing multiple swell components.

        Uses reliable data sources in order of preference:
        1. ERDDAP for bulk parameters (Hs, Tp, Dp) - most reliable, QC'd data
        2. ndar.cdip 9-band product for period-based energy breakdown
        3. THREDDS/OpenDAP as fallback (slower, requires netCDF4)

        Args:
            station_id: CDIP station ID (e.g., "100", "067")

        Returns:
            Dict with partitioned swell data showing wave energy by period band
        """
        logger.info(f"Fetching partitioned wave data for CDIP buoy {station_id}")

        result = {
            "station_id": station_id,
            "data_source": "CDIP",
            "partition_url": f"{self.base_url}/m/products/partition/?stn={station_id}p1",
            "status": "success",
            "timestamp": None,
            "partitions": [],
            "combined": None,
        }

        # Step 1: Get bulk parameters from ERDDAP (most reliable)
        erddap_data = await self._fetch_erddap_latest(station_id)
        if erddap_data and erddap_data.get("waveHs") is not None:
            result["timestamp"] = erddap_data.get("timestamp")
            result["combined"] = {
                "significant_height_m": erddap_data.get("waveHs"),
                "significant_height_ft": round(erddap_data.get("waveHs", 0) * 3.28084, 1),
                "peak_period_s": erddap_data.get("waveTp"),
                "peak_direction_deg": erddap_data.get("waveDp"),
            }
            result["data_source"] = "CDIP ERDDAP"

        # Step 2: Get 9-band breakdown for partitions
        bands = await self._fetch_9band_data(station_id)
        if bands:
            result["partitions"] = self._9band_to_partitions(bands)
            result["data_source"] = "CDIP 9-band"

            # Use 9-band timestamp if we don't have one yet
            if not result["timestamp"] and bands[0].get("timestamp"):
                result["timestamp"] = bands[0]["timestamp"]

        # Step 3: Fallback to THREDDS if we have no data
        if not result["combined"] and not result["partitions"]:
            logger.info(f"ERDDAP/9-band failed, trying THREDDS for CDIP {station_id}")
            thredds_data = await self._fetch_thredds_fallback(station_id)
            if thredds_data:
                result.update(thredds_data)
                result["data_source"] = "CDIP THREDDS"
            else:
                result["status"] = "error"
                result["error"] = f"No data available for station {station_id}"

        return result

    async def _fetch_thredds_fallback(self, station_id: str) -> Optional[Dict]:
        """Fallback to THREDDS/OpenDAP for bulk parameters only.

        Used when ERDDAP and 9-band endpoints fail.
        Only returns combined parameters, no partition data.
        """
        if not NETCDF4_AVAILABLE:
            return None

        try:
            url = f"{self.thredds_url}/dodsC/cdip/realtime/{station_id}p1_rt.nc"
            ds = NetCDFDataset(url)

            result = {
                "thredds_url": url,
                "timestamp": None,
                "combined": None,
                "partitions": [],
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

            ds.close()
            return result if result["combined"] else None

        except Exception as e:
            logger.warning(f"THREDDS fallback failed for CDIP {station_id}: {e}")
            return None

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

    def _direction_distance(self, d1: float, d2: float) -> float:
        """Calculate angular distance between two directions (0-360 degrees)."""
        diff = abs(d1 - d2)
        return min(diff, 360 - diff)

    def _partition_spectrum(self, ds) -> List[Dict]:
        """Partition wave spectrum by grouping frequency bands with similar directions.

        This approach identifies distinct wave trains (swell, wind waves) by looking
        for direction discontinuities in the spectrum. Frequency bands with similar
        directions are grouped together, and Hs is calculated for each group.

        Args:
            ds: Open netCDF4 Dataset with CDIP spectral data

        Returns:
            List of partition dicts with height_m, period_s, direction_deg, type
        """
        if not NUMPY_AVAILABLE:
            return []

        try:
            freq = ds.variables['waveFrequency'][:]
            energy = ds.variables['waveEnergyDensity'][:][-1, :]  # Latest time
            direction = ds.variables['waveMeanDirection'][:][-1, :]  # Latest time
            bandwidth = ds.variables['waveBandwidth'][:]

            # Total energy for reference
            m0_total = np.sum(energy * bandwidth)
            if m0_total <= 0:
                return []

            # Parameters for direction-based partitioning
            dir_threshold = 30  # Max direction difference to group together (degrees)
            min_energy_pct = 5  # Minimum % of total energy to be a partition

            # Find significant frequency bands (> 1% of peak energy)
            peak_energy = np.max(energy)
            significant = energy > 0.01 * peak_energy

            # Group consecutive bands with similar directions
            groups = []
            current_group = []

            for i in range(len(freq)):
                if not significant[i]:
                    # Save current group if exists
                    if current_group:
                        groups.append(current_group)
                        current_group = []
                    continue

                if not current_group:
                    current_group = [i]
                else:
                    # Check if direction is similar to group average
                    group_dirs = [direction[j] for j in current_group]
                    group_energies = [energy[j] for j in current_group]
                    avg_dir = np.average(group_dirs, weights=group_energies)

                    if self._direction_distance(direction[i], avg_dir) < dir_threshold:
                        current_group.append(i)
                    else:
                        # Direction changed significantly - start new group
                        groups.append(current_group)
                        current_group = [i]

            if current_group:
                groups.append(current_group)

            # Calculate statistics for each partition
            partitions = []
            for group_indices in groups:
                group_energy = energy[group_indices]
                group_bandwidth = bandwidth[group_indices]
                group_direction = direction[group_indices]
                group_freq = freq[group_indices]

                m0 = np.sum(group_energy * group_bandwidth)
                energy_pct = 100 * m0 / m0_total

                # Skip partitions with less than minimum energy
                if energy_pct < min_energy_pct:
                    continue

                hs = 4 * np.sqrt(m0)

                # Peak frequency in this group
                peak_idx = np.argmax(group_energy)
                tp = 1 / group_freq[peak_idx]

                # Energy-weighted mean direction
                mean_dir = np.average(group_direction, weights=group_energy)

                partitions.append({
                    "partition_id": len(partitions) + 1,
                    "height_m": round(float(hs), 2),
                    "height_ft": round(float(hs) * 3.28084, 1),
                    "period_s": round(float(tp), 1),
                    "direction_deg": round(float(mean_dir), 0),
                    "type": self._classify_wave_type(tp),
                    "energy_pct": round(float(energy_pct), 0),
                })

            # Sort by energy contribution descending
            partitions.sort(key=lambda p: p.get("energy_pct", 0), reverse=True)

            # Renumber partition IDs after sorting
            for i, p in enumerate(partitions):
                p["partition_id"] = i + 1

            return partitions

        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Spectral partitioning failed: {e}")
            return []

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
