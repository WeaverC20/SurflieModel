# Wave Forecast API

FastAPI backend service for wave forecasting.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
# or
pip install -r requirements.txt

# Install shared packages
pip install -e ../../packages/python/common
pip install -e ../../ml/inference
```

## Development

```bash
# Run development server
uvicorn app.main:app --reload

# Run with custom host/port
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

- `app/main.py` - Application entry point
- `app/routers/` - API route handlers
- `app/models/` - Pydantic models (request/response schemas)
- `app/services/` - Business logic
- `app/db/` - Database models and connection
- `app/config.py` - Configuration management
