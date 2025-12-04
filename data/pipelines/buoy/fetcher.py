"""Fetch NDBC buoy data"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


class BuoyFetcher:
    """Fetches data from NDBC buoys"""

    def __init__(self):
        """Initialize buoy fetcher"""
        self.base_url = "https://www.ndbc.noaa.gov"

    async def fetch_latest_observation(self, station_id: str) -> Dict:
        """Fetch latest buoy observation

        Args:
            station_id: NDBC station ID (e.g., "46237")

        Returns:
            Latest observation data
        """
        logger.info(f"Fetching latest observation for buoy {station_id}")

        async with httpx.AsyncClient() as client:
            url = f"{self.base_url}/data/realtime2/{station_id}.txt"
            response = await client.get(url)
            response.raise_for_status()

            # Parse fixed-width format
            lines = response.text.strip().split("\n")
            if len(lines) < 3:
                raise ValueError("Insufficient data from buoy")

            # Parse latest reading (third line, after headers)
            data = lines[2].split()

            return {
                "station_id": station_id,
                "timestamp": f"{data[0]}-{data[1]}-{data[2]} {data[3]}:{data[4]}",
                "wind_direction": float(data[5]) if data[5] != "MM" else None,
                "wind_speed": float(data[6]) if data[6] != "MM" else None,
                "wave_height": float(data[8]) if data[8] != "MM" else None,
                "dominant_period": float(data[9]) if data[9] != "MM" else None,
                "mean_wave_direction": float(data[11]) if data[11] != "MM" else None,
                "water_temperature": float(data[14]) if data[14] != "MM" else None,
            }

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

        async with httpx.AsyncClient() as client:
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

        # TODO: Implement buoy search
        # Could use a static list of buoy locations or fetch from NDBC

        return []
