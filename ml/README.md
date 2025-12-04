# ML - Wave Forecasting Models

Machine learning models for surf condition prediction.

## Structure

- **training/** - Model training scripts and experiments
- **inference/** - Production inference code (used by backend)
- **experiments/** - MLflow or W&B tracking data (gitignored)
- **artifacts/** - Local model storage (gitignored, production uses S3)

## Workflow

1. **Training** (`ml/training/`):
   - Develop and train models locally
   - Track experiments with MLflow/W&B
   - Save best models to S3/GCS

2. **Inference** (`ml/inference/`):
   - Simple API: `predict(spot_id, timestamp)`
   - Loads models from S3 on startup
   - Used by backend API and worker

3. **Deployment**:
   - Models stored in cloud (S3/GCS)
   - Backend loads model URLs from config
   - Never commit model files to git

## Key Principle

**Training code stays separate from production.** The `inference/` module is the only ML code deployed to production.
