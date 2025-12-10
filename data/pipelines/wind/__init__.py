"""
Wind Forecast Pipeline

GFS wind data fetcher for wind visualization.
"""

from .gfs_fetcher import GFSWindFetcher

__all__ = [
    "GFSWindFetcher",
]
