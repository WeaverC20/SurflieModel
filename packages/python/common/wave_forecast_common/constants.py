"""Shared constants"""

# Wave rating scale
MIN_RATING = 1
MAX_RATING = 10

# Default forecast horizon (days)
DEFAULT_FORECAST_DAYS = 7

# Confidence thresholds
LOW_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.8

# Unit conversions
METERS_TO_FEET = 3.28084
KPH_TO_MPH = 0.621371
MPS_TO_MPH = 2.23694

# Common surf spot characteristics
SPOT_TYPES = [
    "beach_break",
    "point_break",
    "reef_break",
    "river_mouth",
]

# Swell direction names
DIRECTION_NAMES = {
    0: "N",
    45: "NE",
    90: "E",
    135: "SE",
    180: "S",
    225: "SW",
    270: "W",
    315: "NW",
}
