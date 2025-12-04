# wave-forecast-common

Shared Python utilities and constants used across the project.

## Installation

```bash
pip install -e .
```

## Usage

```python
from wave_forecast_common import meters_to_feet, haversine_distance
from wave_forecast_common.constants import MIN_RATING, MAX_RATING

# Convert units
height_ft = meters_to_feet(2.5)

# Calculate distance
distance_km = haversine_distance(37.7749, -122.4194, 37.8199, -122.4783)

# Use constants
rating = clamp(predicted_rating, MIN_RATING, MAX_RATING)
```

## What Goes Here

Shared code that's used by multiple parts of the project:

✅ **Include:**
- Constants and enums
- Unit conversion functions
- Common calculations (e.g., distance)
- Utility functions

❌ **Don't include:**
- Business logic (belongs in backend/api/services)
- Model code (belongs in ml/)
- Data fetching (belongs in data/pipelines)

## Integration

This package is installed in:
- `backend/api` - Uses constants and utils
- `backend/worker` - Uses constants and utils
- `ml/inference` - Uses unit conversions
- `ml/training` - Uses constants and calculations
- `data/pipelines` - Uses utils
