# ML Artifacts

This directory is for local development only.

**Production models are stored in S3/GCS, not in git.**

## Local Development

During development, you can save models here for testing:

```python
# Save model locally
torch.save(model.state_dict(), "ml/artifacts/my_model.pt")

# Load for testing
predictor = WaveForecastPredictor(model_url="ml/artifacts/my_model.pt")
```

## Production

Production models should be:
1. Trained with `ml/training/train.py`
2. Uploaded to S3: `aws s3 cp model.pt s3://my-bucket/models/`
3. Loaded via URL in production: `s3://my-bucket/models/model.pt`

Never commit model files to git (they're large and change frequently).
