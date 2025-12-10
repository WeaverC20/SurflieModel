"""
Wave Forecast Pipeline

WaveWatch III wave data fetcher for wave visualization.
"""

from .wavewatch_fetcher import WaveWatchFetcher

__all__ = [
    "WaveWatchFetcher",
]
