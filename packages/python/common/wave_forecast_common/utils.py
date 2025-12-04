"""Shared utility functions"""

import math
from typing import Optional


def meters_to_feet(meters: float) -> float:
    """Convert meters to feet"""
    return meters * 3.28084


def feet_to_meters(feet: float) -> float:
    """Convert feet to meters"""
    return feet / 3.28084


def kph_to_mph(kph: float) -> float:
    """Convert km/h to mph"""
    return kph * 0.621371


def mps_to_mph(mps: float) -> float:
    """Convert m/s to mph"""
    return mps * 2.23694


def direction_to_compass(degrees: float) -> str:
    """Convert degrees to compass direction (N, NE, E, etc.)

    Args:
        degrees: Direction in degrees (0-360)

    Returns:
        Compass direction string
    """
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = round(degrees / 45) % 8
    return directions[index]


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate distance between two points using Haversine formula

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value between min and max"""
    return max(min_value, min(value, max_value))
