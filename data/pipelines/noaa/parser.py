"""Parse NOAA data into structured format"""

import logging
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


def parse_forecast_periods(forecast_data: Dict) -> List[Dict]:
    """Parse NOAA forecast periods into structured format

    Args:
        forecast_data: Raw NOAA forecast response

    Returns:
        List of forecast period dictionaries
    """
    logger.debug("Parsing NOAA forecast periods")

    periods = []
    for period in forecast_data.get("properties", {}).get("periods", []):
        periods.append({
            "timestamp": period.get("startTime"),
            "temperature": period.get("temperature"),
            "wind_speed": period.get("windSpeed"),
            "wind_direction": period.get("windDirection"),
            "short_forecast": period.get("shortForecast"),
            "detailed_forecast": period.get("detailedForecast"),
        })

    return periods


def extract_wave_data(wavewatch_data: Dict) -> Dict:
    """Extract wave parameters from WaveWatch III data

    Args:
        wavewatch_data: Raw WaveWatch data

    Returns:
        Structured wave parameters
    """
    logger.debug("Extracting wave data from WaveWatch III")

    # TODO: Implement WaveWatch parsing
    return {
        "significant_wave_height": 0.0,
        "peak_wave_period": 0.0,
        "mean_wave_direction": 0.0,
        "primary_swell_height": 0.0,
        "primary_swell_period": 0.0,
        "primary_swell_direction": 0.0,
    }
