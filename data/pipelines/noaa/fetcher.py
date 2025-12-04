"""Fetch NOAA forecast data"""

import logging
from datetime import datetime
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class NOAAFetcher:
    """Fetches data from NOAA APIs"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize NOAA fetcher

        Args:
            api_key: Optional NOAA API key (not always required)
        """
        self.api_key = api_key
        self.base_url = "https://api.weather.gov"
        self.wavewatch_url = "https://polar.ncep.noaa.gov/waves"

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

    async def fetch_wavewatch_data(
        self, latitude: float, longitude: float, timestamp: datetime
    ) -> Dict:
        """Fetch Wave Watch III model data

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            timestamp: Forecast timestamp

        Returns:
            Wave model data dictionary
        """
        logger.info(f"Fetching WaveWatch III data for ({latitude}, {longitude})")

        # TODO: Implement WaveWatch III data fetching
        # This requires parsing GRIB files or using specific NOAA APIs

        return {}

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
