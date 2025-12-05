"""FastAPI application entry point"""

import sys
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Add project root to path for data pipeline imports
# From backend/api/app/main.py, go up 4 levels to reach project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from data.pipelines.noaa import NOAAFetcher

app = FastAPI(
    title="Wave Forecast API",
    description="API for wave forecasting and surf condition predictions",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default location
DEFAULT_LOCATION = {
    "name": "Huntington Beach, CA",
    "lat": 33.6595,
    "lon": -118.0007,
}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - displays surf forecast for default location"""
    try:
        # Fetch complete forecast
        fetcher = NOAAFetcher()
        forecast = await fetcher.fetch_complete_forecast(
            latitude=DEFAULT_LOCATION["lat"],
            longitude=DEFAULT_LOCATION["lon"],
            forecast_hours=48
        )

        # Generate HTML
        html_content = generate_forecast_html(forecast, DEFAULT_LOCATION)
        return html_content

    except Exception as e:
        return f"""
        <html>
            <head><title>Wave Forecast API - Error</title></head>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <h1>Wave Forecast API</h1>
                <p style="color: red;">Error loading forecast: {str(e)}</p>
                <p><a href="/docs">View API Documentation</a></p>
            </body>
        </html>
        """


def generate_forecast_html(forecast: dict, location: dict) -> str:
    """Generate HTML display for forecast data"""

    # Extract tide data
    tide_html = ""
    if "tide_station" in forecast:
        station = forecast["tide_station"]
        tide_html += f"""
        <div style="background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
            <p><strong>Station:</strong> {station.get('name', 'N/A')} ({station.get('id', 'N/A')})</p>
            <p><strong>Distance:</strong> {station.get('distance_km', 0):.1f} km from spot</p>
        </div>
        """

    if "tide" in forecast and "predictions" in forecast["tide"]:
        predictions = forecast["tide"]["predictions"][:8]  # Show first 8 tides
        tide_html += "<table style='width: 100%; border-collapse: collapse;'>"
        tide_html += "<tr style='background: #2196F3; color: white;'><th style='padding: 8px;'>Time (UTC)</th><th style='padding: 8px;'>Height</th><th style='padding: 8px;'>Type</th></tr>"
        for pred in predictions:
            time = pred.get('t', 'N/A')
            height = pred.get('v', 'N/A')
            tide_type = pred.get('type', 'N/A')
            row_color = "#f5f5f5" if tide_type == "L" else "#fff"
            tide_html += f"<tr style='background: {row_color};'><td style='padding: 8px; border: 1px solid #ddd;'>{time}</td><td style='padding: 8px; border: 1px solid #ddd;'>{height}m</td><td style='padding: 8px; border: 1px solid #ddd;'>{'High' if tide_type == 'H' else 'Low'}</td></tr>"
        tide_html += "</table>"
    elif "tide_error" in forecast:
        tide_html += f"<p style='color: red;'>Error: {forecast['tide_error']}</p>"

    # Extract wave data
    wave_html = ""
    if "waves" in forecast:
        waves = forecast["waves"]
        if waves.get("status") == "success":
            wave_html += f"""
            <div style="background: #e8f5e9; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                <p><strong>Model Run:</strong> {waves.get('model_time', 'N/A')}</p>
                <p><strong>Valid Time:</strong> {waves.get('valid_time', 'N/A')}</p>
                <p><strong>Data Size:</strong> {waves.get('data_size_bytes', 0)} bytes</p>
                {f"<p><strong>Fallback:</strong> Used {waves.get('fallback_hours')}h old model run</p>" if waves.get('fallback_hours') else ""}
                <p style="color: #666; font-size: 0.9em;">{waves.get('note', '')}</p>
            </div>
            """
        else:
            wave_html += f"<p style='color: red;'>Error: {waves.get('error', 'Unknown error')}</p>"
    elif "wave_error" in forecast:
        wave_html += f"<p style='color: red;'>Error: {forecast['wave_error']}</p>"

    # Extract wind data
    wind_html = ""
    if "wind" in forecast:
        wind = forecast["wind"]
        if "forecasts" in wind:
            forecasts = wind["forecasts"]
            successful = [f for f in forecasts if f.get("status") == "success"]
            if successful:
                first = successful[0]
                wind_html += f"""
                <div style="background: #fff3e0; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                    <p><strong>Model Run:</strong> {first.get('model_time', 'N/A')}</p>
                    <p><strong>Forecast Hours:</strong> {len(successful)} hours available</p>
                    <p><strong>Data Size (per hour):</strong> {first.get('data_size_bytes', 0)} bytes</p>
                    {f"<p><strong>Fallback:</strong> Used {first.get('fallback_hours')}h old model run</p>" if first.get('fallback_hours') else ""}
                    <p style="color: #666; font-size: 0.9em;">{first.get('note', '')}</p>
                </div>
                """
            else:
                wind_html += "<p style='color: red;'>No successful wind forecasts</p>"
    elif "wind_error" in forecast:
        wind_html += f"<p style='color: red;'>Error: {forecast['wind_error']}</p>"

    # Build complete HTML
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Wave Forecast - {location['name']}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .section {{
                    background: white;
                    padding: 25px;
                    margin-bottom: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .section h2 {{
                    margin-top: 0;
                    color: #333;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 10px;
                }}
                .footer {{
                    text-align: center;
                    padding: 20px;
                    color: #666;
                }}
                .footer a {{
                    color: #667eea;
                    text-decoration: none;
                    margin: 0 10px;
                }}
                .footer a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌊 Wave Forecast</h1>
                <p style="font-size: 1.2em; margin: 10px 0;">{location['name']}</p>
                <p style="opacity: 0.9;">Lat: {location['lat']}, Lon: {location['lon']}</p>
                <p style="opacity: 0.8; font-size: 0.9em;">Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
            </div>

            <div class="section">
                <h2>🌊 Tide Forecast (48 hours)</h2>
                {tide_html}
            </div>

            <div class="section">
                <h2>🏄 Wave Forecast</h2>
                {wave_html}
            </div>

            <div class="section">
                <h2>💨 Wind Forecast</h2>
                {wind_html}
            </div>

            <div class="footer">
                <p>Wave Forecast API v0.1.0</p>
                <p>
                    <a href="/docs">API Documentation</a> |
                    <a href="/health">Health Check</a> |
                    <a href="/api/v1/dev/test/locations">Test Locations</a>
                </p>
                <p style="font-size: 0.9em; color: #999;">
                    Data sources: NOAA CO-OPS (Tide), Wave Watch 3 (Waves), GFS (Wind)
                </p>
            </div>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


# Include routers
from app.routers import dev

app.include_router(dev.router, prefix="/api/v1/dev", tags=["dev"])

# TODO: Add production routers
# from app.routers import forecast, spots, users
# app.include_router(forecast.router, prefix="/api/v1/forecast", tags=["forecast"])
# app.include_router(spots.router, prefix="/api/v1/spots", tags=["spots"])
# app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
