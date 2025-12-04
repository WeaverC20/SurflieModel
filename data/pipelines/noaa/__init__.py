"""NOAA data fetching and parsing

This module provides access to NOAA data sources for surf forecasting:
- Tides: NOAA CO-OPS with harmonic constituents
- Waves: Wave Watch 3 for swell height and direction
- Wind: GFS for wind speed and direction

Example usage:
    from data.pipelines.noaa import NOAAFetcher

    fetcher = NOAAFetcher()
    forecast = await fetcher.fetch_complete_forecast(
        latitude=37.8,
        longitude=-122.4,
        forecast_hours=168
    )
"""

from .fetcher import (
    NOAAFetcher,
    NOAATideFetcher,
    NOAAWaveWatch3Fetcher,
    NOAAWindFetcher,
)

__all__ = [
    "NOAAFetcher",
    "NOAATideFetcher",
    "NOAAWaveWatch3Fetcher",
    "NOAAWindFetcher",
]
