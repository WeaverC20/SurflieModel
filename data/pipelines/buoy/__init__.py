"""Buoy data fetching from NDBC and CDIP

This module provides access to real-time buoy observations:
- NDBC: National Data Buoy Center (NOAA) buoys
- CDIP: Coastal Data Information Program buoys

Example usage:
    from data.pipelines.buoy import NDBCBuoyFetcher, CDIPBuoyFetcher

    # Fetch NDBC buoy data
    ndbc_fetcher = NDBCBuoyFetcher()
    data = await ndbc_fetcher.fetch_latest_observation("46237")

    # Find nearby buoys
    nearby = ndbc_fetcher.get_nearby_buoys(33.6595, -118.0007)
"""

from .fetcher import NDBCBuoyFetcher, CDIPBuoyFetcher

__all__ = [
    "NDBCBuoyFetcher",
    "CDIPBuoyFetcher",
]
