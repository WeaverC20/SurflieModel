# Wave Forecast Inference

Production inference code for wave forecasting models.

## Installation

```bash
pip install -e .
```

## Usage

```python
from wave_forecast_inference import WaveForecastPredictor

# Initialize predictor with model from S3
predictor = WaveForecastPredictor(model_url="s3://my-bucket/models/wave_model_v1.pt")

# Make prediction
prediction = predictor.predict(
    spot_id="ocean-beach-sf",
    timestamp=datetime.now(),
    noaa_data={"wind_speed": 10, "swell_height": 5, ...},
    buoy_data={"wave_height": 4.5, ...}
)

print(prediction)
# {
#     "wave_height": 5.2,
#     "wave_period": 14.0,
#     "rating": 8,
#     "confidence": 0.85
# }
```

## Integration with Backend

The backend API imports this module:

```python
# In backend/api/app/services/forecast_service.py
from wave_forecast_inference import WaveForecastPredictor

predictor = WaveForecastPredictor(model_url=settings.model_url)
forecast = predictor.predict(...)
```

## Model Storage

Models are stored in S3/GCS and loaded at runtime. Never commit model files to git.

Model URL format:
- S3: `s3://bucket-name/path/to/model.pt`
- GCS: `gs://bucket-name/path/to/model.pt`
- Local (dev only): `/path/to/local/model.pt`
