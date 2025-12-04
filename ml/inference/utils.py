"""Utility functions for inference"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


def preprocess_noaa_data(raw_data: Dict) -> Dict:
    """Preprocess NOAA data for model input

    Args:
        raw_data: Raw NOAA API response

    Returns:
        Preprocessed features dictionary
    """
    # TODO: Implement preprocessing
    logger.debug("Preprocessing NOAA data")
    return raw_data


def preprocess_buoy_data(raw_data: Dict) -> Dict:
    """Preprocess buoy data for model input

    Args:
        raw_data: Raw buoy data

    Returns:
        Preprocessed features dictionary
    """
    # TODO: Implement preprocessing
    logger.debug("Preprocessing buoy data")
    return raw_data


def calculate_surf_rating(
    wave_height: float,
    wave_period: float,
    wind_speed: float,
    spot_characteristics: Dict,
) -> int:
    """Calculate surf quality rating (1-10)

    Args:
        wave_height: Wave height in feet
        wave_period: Wave period in seconds
        wind_speed: Wind speed in mph
        spot_characteristics: Spot-specific parameters

    Returns:
        Rating from 1 (poor) to 10 (epic)
    """
    # TODO: Implement rating logic
    # Consider:
    # - Wave size relative to spot optimal size
    # - Wave period (longer = better)
    # - Wind (offshore = better, onshore = worse)
    # - Spot-specific factors

    return 5  # Placeholder
