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
from data.pipelines.buoy import NDBCBuoyFetcher

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

# Default location (Huntington Beach Pier)
DEFAULT_LOCATION = {
    "name": "Huntington Beach, CA",
    "lat": 33.6556,
    "lon": -117.9999,
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

        # Fetch nearby buoy data
        buoy_fetcher = NDBCBuoyFetcher()
        nearby_buoys = buoy_fetcher.get_nearby_buoys(
            latitude=DEFAULT_LOCATION["lat"],
            longitude=DEFAULT_LOCATION["lon"],
            max_distance_km=100
        )

        # Fetch data from nearest buoys (up to 3)
        buoy_observations = []
        for buoy_info in nearby_buoys[:3]:
            try:
                obs = await buoy_fetcher.fetch_latest_observation(buoy_info["station_id"])
                buoy_observations.append({
                    "info": buoy_info,
                    "data": obs,
                    "status": "success"
                })
            except Exception as e:
                logger.warning(f"Could not fetch buoy {buoy_info['station_id']}: {e}")
                buoy_observations.append({
                    "info": buoy_info,
                    "data": None,
                    "status": "error",
                    "error": str(e)
                })

        # Generate HTML
        html_content = generate_forecast_html(forecast, DEFAULT_LOCATION, buoy_observations)
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


def generate_forecast_html(forecast: dict, location: dict, buoy_observations: list = None) -> str:
    """Generate HTML display for forecast data and buoy observations"""

    if buoy_observations is None:
        buoy_observations = []

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
        if "forecasts" in waves and len(waves["forecasts"]) > 0:
            # Get current conditions (first forecast)
            current = waves["forecasts"][0]

            if current.get("parsed") and current.get("values"):
                values = current["values"]

                # Current conditions - showing all components
                wave_html += f"""
                <div style="background: #e8f5e9; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                    <h3 style="margin-top: 0; color: #2e7d32;">Current Conditions</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="background: #c8e6c9;">
                            <th style="padding: 10px; border: 1px solid #a5d6a7; text-align: left;" colspan="2">Total Wave</th>
                        </tr>
                        <tr style="background: #f1f8f4;">
                            <td style="padding: 10px; border: 1px solid #a5d6a7; font-weight: bold;">Significant Height</td>
                            <td style="padding: 10px; border: 1px solid #a5d6a7; font-size: 1.3em; color: #1b5e20;">
                                {values.get('swh', current.get('wave_height_m', 'N/A'))} m ({values.get('swh', current.get('wave_height_m', 0)) * 3.28084:.1f} ft)
                            </td>
                        </tr>
                        <tr style="background: #fff;">
                            <td style="padding: 10px; border: 1px solid #a5d6a7; font-weight: bold;">Peak Period</td>
                            <td style="padding: 10px; border: 1px solid #a5d6a7;">{values.get('perpw', current.get('wave_period_s', 'N/A'))} s</td>
                        </tr>
                        <tr style="background: #f1f8f4;">
                            <td style="padding: 10px; border: 1px solid #a5d6a7; font-weight: bold;">Direction</td>
                            <td style="padding: 10px; border: 1px solid #a5d6a7;">{values.get('dirpw', current.get('wave_direction_deg', 'N/A'))}°</td>
                        </tr>
                """

                # Wind waves component
                if 'wvhgt' in values or 'wvper' in values or 'wvdir' in values:
                    wave_html += f"""
                        <tr style="background: #c8e6c9;">
                            <th style="padding: 10px; border: 1px solid #a5d6a7; text-align: left;" colspan="2">Wind Waves</th>
                        </tr>
                        <tr style="background: #fff;">
                            <td style="padding: 10px; border: 1px solid #a5d6a7; font-weight: bold;">Height</td>
                            <td style="padding: 10px; border: 1px solid #a5d6a7;">
                                {values.get('wvhgt', 'N/A')} m ({values.get('wvhgt', 0) * 3.28084:.1f} ft)
                            </td>
                        </tr>
                        <tr style="background: #f1f8f4;">
                            <td style="padding: 10px; border: 1px solid #a5d6a7; font-weight: bold;">Period</td>
                            <td style="padding: 10px; border: 1px solid #a5d6a7;">{values.get('wvper', 'N/A')} s</td>
                        </tr>
                        <tr style="background: #fff;">
                            <td style="padding: 10px; border: 1px solid #a5d6a7; font-weight: bold;">Direction</td>
                            <td style="padding: 10px; border: 1px solid #a5d6a7;">{values.get('wvdir', 'N/A')}°</td>
                        </tr>
                    """

                # Swell component
                if 'swell' in values or 'swper' in values or 'swdir' in values:
                    wave_html += f"""
                        <tr style="background: #c8e6c9;">
                            <th style="padding: 10px; border: 1px solid #a5d6a7; text-align: left;" colspan="2">Swell</th>
                        </tr>
                        <tr style="background: #fff;">
                            <td style="padding: 10px; border: 1px solid #a5d6a7; font-weight: bold;">Height</td>
                            <td style="padding: 10px; border: 1px solid #a5d6a7;">
                                {values.get('swell', 'N/A')} m ({values.get('swell', 0) * 3.28084:.1f} ft)
                            </td>
                        </tr>
                        <tr style="background: #f1f8f4;">
                            <td style="padding: 10px; border: 1px solid #a5d6a7; font-weight: bold;">Period</td>
                            <td style="padding: 10px; border: 1px solid #a5d6a7;">{values.get('swper', 'N/A')} s</td>
                        </tr>
                        <tr style="background: #fff;">
                            <td style="padding: 10px; border: 1px solid #a5d6a7; font-weight: bold;">Direction</td>
                            <td style="padding: 10px; border: 1px solid #a5d6a7;">{values.get('swdir', 'N/A')}°</td>
                        </tr>
                    """

                wave_html += """
                    </table>
                </div>
                """

                # Add metadata
                wave_html += f"""
                <div style="background: #fff; padding: 10px; border-radius: 5px; font-size: 0.9em; color: #666;">
                    <p style="margin: 5px 0;"><strong>Model Run:</strong> {current.get('model_time', 'N/A')}</p>
                    <p style="margin: 5px 0;"><strong>Valid Time:</strong> {current.get('valid_time', 'N/A')}</p>
                    {f"<p style='margin: 5px 0;'><strong>Note:</strong> Using {current.get('fallback_hours')}h old model run</p>" if current.get('fallback_hours') else ""}
                </div>
                """
            else:
                wave_html += f"""
                <div style="background: #fff3e0; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                    <p><strong>Model Run:</strong> {current.get('model_time', 'N/A')}</p>
                    <p><strong>Valid Time:</strong> {current.get('valid_time', 'N/A')}</p>
                    <p><strong>Data Size:</strong> {current.get('data_size_bytes', 0)} bytes</p>
                    {f"<p><strong>Fallback:</strong> Used {current.get('fallback_hours')}h old model run</p>" if current.get('fallback_hours') else ""}
                    <p style="color: #e65100; font-weight: bold;">{current.get('note', 'GRIB2 parsing not available')}</p>
                    <p style="font-size: 0.9em;">Install cfgrib to see parsed values: <code>pip install cfgrib xarray</code></p>
                </div>
                """
        else:
            wave_html += f"<p style='color: red;'>Error: {waves.get('error', 'No wave forecast data available')}</p>"
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

                # Show parsed wind values if available
                if first.get("parsed"):
                    wind_html += f"""
                    <div style="background: #fff3e0; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                        <h3 style="margin-top: 0; color: #e65100;">Current Conditions</h3>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr style="background: #fff8e1;">
                                <td style="padding: 10px; border: 1px solid #ffcc80; font-weight: bold;">Wind Speed</td>
                                <td style="padding: 10px; border: 1px solid #ffcc80; font-size: 1.3em; color: #e65100;">
                                    {first.get('wind_speed_kts', 'N/A')} kts ({first.get('wind_speed_ms', 'N/A')} m/s)
                                </td>
                            </tr>
                            <tr style="background: #fff;">
                                <td style="padding: 10px; border: 1px solid #ffcc80; font-weight: bold;">Wind Direction</td>
                                <td style="padding: 10px; border: 1px solid #ffcc80;">{first.get('wind_direction_deg', 'N/A')}°</td>
                            </tr>
                            {f'''<tr style="background: #fff8e1;">
                                <td style="padding: 10px; border: 1px solid #ffcc80; font-weight: bold;">Wind Gust</td>
                                <td style="padding: 10px; border: 1px solid #ffcc80;">{first.get('wind_gust_kts', 'N/A')} kts ({first.get('wind_gust_ms', 'N/A')} m/s)</td>
                            </tr>''' if 'wind_gust_kts' in first else ''}
                        </table>
                    </div>
                    <div style="background: #fff; padding: 10px; border-radius: 5px; font-size: 0.9em; color: #666;">
                        <p style="margin: 5px 0;"><strong>Model Run:</strong> {first.get('model_time', 'N/A')}</p>
                        <p style="margin: 5px 0;"><strong>Valid Time:</strong> {first.get('valid_time', 'N/A')}</p>
                        <p style="margin: 5px 0;"><strong>Forecast Hours Available:</strong> {len(successful)}</p>
                        {f"<p style='margin: 5px 0;'><strong>Note:</strong> Using {first.get('fallback_hours')}h old model run</p>" if first.get('fallback_hours') else ""}
                    </div>
                    """
                else:
                    wind_html += f"""
                    <div style="background: #fff3e0; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                        <p><strong>Model Run:</strong> {first.get('model_time', 'N/A')}</p>
                        <p><strong>Forecast Hours:</strong> {len(successful)} hours available</p>
                        <p><strong>Data Size (per hour):</strong> {first.get('data_size_bytes', 0)} bytes</p>
                        {f"<p><strong>Fallback:</strong> Used {first.get('fallback_hours')}h old model run</p>" if first.get('fallback_hours') else ""}
                        <p style="color: #e65100; font-weight: bold;">{first.get('note', 'GRIB2 parsing not available')}</p>
                        <p style="font-size: 0.9em;">Install cfgrib to see parsed values: <code>pip install cfgrib xarray</code></p>
                    </div>
                    """
            else:
                wind_html += "<p style='color: red;'>No successful wind forecasts</p>"
    elif "wind_error" in forecast:
        wind_html += f"<p style='color: red;'>Error: {forecast['wind_error']}</p>"

    # Build wave forecast table
    wave_forecast_html = ""
    if "waves" in forecast and "forecasts" in forecast["waves"]:
        wave_forecasts = forecast["waves"]["forecasts"]
        parsed_forecasts = [f for f in wave_forecasts if f.get("parsed") and f.get("values")]

        if len(parsed_forecasts) > 0:
            wave_forecast_html += """
            <div style="background: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 15px; overflow-x: auto;">
                <h3 style="margin-top: 0; color: #1565c0;">Wave Forecast</h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                    <tr style="background: #1976d2; color: white;">
                        <th style="padding: 8px; border: 1px solid #90caf9;">Time (UTC)</th>
                        <th style="padding: 8px; border: 1px solid #90caf9;">Total Height</th>
                        <th style="padding: 8px; border: 1px solid #90caf9;">Period</th>
                        <th style="padding: 8px; border: 1px solid #90caf9;">Direction</th>
                        <th style="padding: 8px; border: 1px solid #90caf9;">Wind Wave</th>
                        <th style="padding: 8px; border: 1px solid #90caf9;">Swell Height</th>
                        <th style="padding: 8px; border: 1px solid #90caf9;">Swell Period</th>
                        <th style="padding: 8px; border: 1px solid #90caf9;">Swell Dir</th>
                    </tr>
            """

            # Show up to 24 forecast periods (3 days at 3-hour intervals)
            for i, fc in enumerate(parsed_forecasts[:24]):
                values = fc.get("values", {})
                valid_time = fc.get("valid_time", "")
                # Parse and format time
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(valid_time.replace('Z', '+00:00'))
                    time_str = dt.strftime("%m/%d %H:%M")
                except:
                    time_str = valid_time[:16] if len(valid_time) > 16 else valid_time

                # Alternate row colors
                row_bg = "#e3f2fd" if i % 2 == 0 else "#fff"

                wave_forecast_html += f"""
                    <tr style="background: {row_bg};">
                        <td style="padding: 8px; border: 1px solid #90caf9;">{time_str}</td>
                        <td style="padding: 8px; border: 1px solid #90caf9; font-weight: bold;">
                            {values.get('swh', 'N/A')} m ({values.get('swh', 0) * 3.28084:.1f} ft)
                        </td>
                        <td style="padding: 8px; border: 1px solid #90caf9;">
                            {values.get('perpw', 'N/A')} s
                        </td>
                        <td style="padding: 8px; border: 1px solid #90caf9;">
                            {values.get('dirpw', 'N/A')}°
                        </td>
                        <td style="padding: 8px; border: 1px solid #90caf9;">
                            {values.get('wvhgt', 'N/A')} m ({values.get('wvhgt', 0) * 3.28084:.1f} ft)
                        </td>
                        <td style="padding: 8px; border: 1px solid #90caf9;">
                            {values.get('swell', 'N/A')} m ({values.get('swell', 0) * 3.28084:.1f} ft)
                        </td>
                        <td style="padding: 8px; border: 1px solid #90caf9;">
                            {values.get('swper', 'N/A')} s
                        </td>
                        <td style="padding: 8px; border: 1px solid #90caf9;">
                            {values.get('swdir', 'N/A')}°
                        </td>
                    </tr>
                """

            wave_forecast_html += """
                </table>
            </div>
            """

    # Build wind forecast table
    wind_forecast_html = ""
    if "wind" in forecast and "forecasts" in forecast["wind"]:
        wind_forecasts = forecast["wind"]["forecasts"]
        parsed_forecasts = [f for f in wind_forecasts if f.get("parsed") and f.get("status") == "success"]

        if len(parsed_forecasts) > 0:
            wind_forecast_html += """
            <div style="background: #fff3e0; padding: 15px; border-radius: 5px; margin-bottom: 15px; overflow-x: auto;">
                <h3 style="margin-top: 0; color: #e65100;">Wind Forecast</h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                    <tr style="background: #ff6f00; color: white;">
                        <th style="padding: 8px; border: 1px solid #ffcc80;">Time (UTC)</th>
                        <th style="padding: 8px; border: 1px solid #ffcc80;">Wind Speed</th>
                        <th style="padding: 8px; border: 1px solid #ffcc80;">Direction</th>
                        <th style="padding: 8px; border: 1px solid #ffcc80;">Wind Gust</th>
                    </tr>
            """

            # Show up to 24 forecast periods
            for i, fc in enumerate(parsed_forecasts[:24]):
                valid_time = fc.get("valid_time", "")
                # Parse and format time
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(valid_time.replace('Z', '+00:00'))
                    time_str = dt.strftime("%m/%d %H:%M")
                except:
                    time_str = valid_time[:16] if len(valid_time) > 16 else valid_time

                # Alternate row colors
                row_bg = "#fff3e0" if i % 2 == 0 else "#fff"

                wind_forecast_html += f"""
                    <tr style="background: {row_bg};">
                        <td style="padding: 8px; border: 1px solid #ffcc80;">{time_str}</td>
                        <td style="padding: 8px; border: 1px solid #ffcc80; font-weight: bold;">
                            {fc.get('wind_speed_kts', 'N/A')} kts ({fc.get('wind_speed_ms', 'N/A')} m/s)
                        </td>
                        <td style="padding: 8px; border: 1px solid #ffcc80;">
                            {fc.get('wind_direction_deg', 'N/A')}°
                        </td>
                        <td style="padding: 8px; border: 1px solid #ffcc80;">
                            {fc.get('wind_gust_kts', 'N/A') if 'wind_gust_kts' in fc else 'N/A'} kts
                        </td>
                    </tr>
                """

            wind_forecast_html += """
                </table>
            </div>
            """

    # Build buoy observations HTML
    buoy_html = ""
    if buoy_observations:
        buoy_html += """
        <div style="background: #e0f2f1; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
            <h3 style="margin-top: 0; color: #00695c;">Real-time Buoy Observations</h3>
        """

        for buoy in buoy_observations:
            info = buoy.get("info", {})
            data = buoy.get("data", {})
            status = buoy.get("status")

            if status == "success" and data:
                # Parse timestamp for data age
                timestamp_str = data.get("timestamp", "")
                data_age_str = ""
                try:
                    if timestamp_str:
                        from datetime import datetime
                        buoy_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        now = datetime.now(buoy_time.tzinfo)
                        age_minutes = int((now - buoy_time).total_seconds() / 60)
                        if age_minutes < 60:
                            data_age_str = f"{age_minutes} minutes ago"
                        else:
                            age_hours = age_minutes // 60
                            data_age_str = f"{age_hours} hour{'s' if age_hours > 1 else ''} ago"
                except:
                    pass

                buoy_html += f"""
                <div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #00897b;">
                    <h4 style="margin: 0 0 10px 0; color: #00695c;">
                        {info.get('name', 'Unknown')} (Station {info.get('station_id', 'N/A')})
                    </h4>
                    <p style="margin: 5px 0; font-size: 0.9em; color: #666;">
                        Distance: {info.get('distance_km', 'N/A')} km |
                        Last Updated: {timestamp_str[:19] if timestamp_str else 'N/A'} UTC
                        {f' ({data_age_str})' if data_age_str else ''}
                    </p>
                    <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                """

                # Wave data
                if data.get("wave_height_m") is not None or data.get("dominant_wave_period_s") is not None:
                    buoy_html += """
                        <tr style="background: #b2dfdb;">
                            <th colspan="2" style="padding: 8px; text-align: left; border: 1px solid #80cbc4;">Wave Conditions</th>
                        </tr>
                    """
                    if data.get("wave_height_m") is not None:
                        buoy_html += f"""
                        <tr style="background: #e0f2f1;">
                            <td style="padding: 8px; border: 1px solid #80cbc4; font-weight: bold;">Significant Wave Height</td>
                            <td style="padding: 8px; border: 1px solid #80cbc4; font-size: 1.2em; color: #00695c;">
                                {data.get('wave_height_m')} m ({data.get('wave_height_ft', 'N/A')} ft)
                            </td>
                        </tr>
                        """
                    if data.get("dominant_wave_period_s") is not None:
                        buoy_html += f"""
                        <tr style="background: white;">
                            <td style="padding: 8px; border: 1px solid #80cbc4; font-weight: bold;">Dominant Period</td>
                            <td style="padding: 8px; border: 1px solid #80cbc4;">{data.get('dominant_wave_period_s')} s</td>
                        </tr>
                        """
                    if data.get("average_wave_period_s") is not None:
                        buoy_html += f"""
                        <tr style="background: #e0f2f1;">
                            <td style="padding: 8px; border: 1px solid #80cbc4; font-weight: bold;">Average Period</td>
                            <td style="padding: 8px; border: 1px solid #80cbc4;">{data.get('average_wave_period_s')} s</td>
                        </tr>
                        """
                    if data.get("mean_wave_direction_deg") is not None:
                        buoy_html += f"""
                        <tr style="background: white;">
                            <td style="padding: 8px; border: 1px solid #80cbc4; font-weight: bold;">Mean Direction</td>
                            <td style="padding: 8px; border: 1px solid #80cbc4;">{data.get('mean_wave_direction_deg')}°</td>
                        </tr>
                        """

                # Wind data
                if data.get("wind_speed_ms") is not None or data.get("wind_direction_deg") is not None:
                    buoy_html += """
                        <tr style="background: #b2dfdb;">
                            <th colspan="2" style="padding: 8px; text-align: left; border: 1px solid #80cbc4;">Wind Conditions</th>
                        </tr>
                    """
                    if data.get("wind_speed_ms") is not None:
                        buoy_html += f"""
                        <tr style="background: #e0f2f1;">
                            <td style="padding: 8px; border: 1px solid #80cbc4; font-weight: bold;">Wind Speed</td>
                            <td style="padding: 8px; border: 1px solid #80cbc4;">
                                {data.get('wind_speed_kts', 'N/A')} kts ({data.get('wind_speed_ms')} m/s)
                            </td>
                        </tr>
                        """
                    if data.get("wind_direction_deg") is not None:
                        buoy_html += f"""
                        <tr style="background: white;">
                            <td style="padding: 8px; border: 1px solid #80cbc4; font-weight: bold;">Wind Direction</td>
                            <td style="padding: 8px; border: 1px solid #80cbc4;">{data.get('wind_direction_deg')}°</td>
                        </tr>
                        """
                    if data.get("wind_gust_ms") is not None:
                        buoy_html += f"""
                        <tr style="background: #e0f2f1;">
                            <td style="padding: 8px; border: 1px solid #80cbc4; font-weight: bold;">Wind Gust</td>
                            <td style="padding: 8px; border: 1px solid #80cbc4;">
                                {data.get('wind_gust_kts', 'N/A')} kts ({data.get('wind_gust_ms')} m/s)
                            </td>
                        </tr>
                        """

                # Temperature and pressure data
                if data.get("water_temp_c") is not None or data.get("air_temp_c") is not None or data.get("air_pressure_hpa") is not None:
                    buoy_html += """
                        <tr style="background: #b2dfdb;">
                            <th colspan="2" style="padding: 8px; text-align: left; border: 1px solid #80cbc4;">Temperature & Pressure</th>
                        </tr>
                    """
                    if data.get("water_temp_c") is not None:
                        water_temp_f = round(data.get("water_temp_c") * 9/5 + 32, 1)
                        buoy_html += f"""
                        <tr style="background: #e0f2f1;">
                            <td style="padding: 8px; border: 1px solid #80cbc4; font-weight: bold;">Water Temperature</td>
                            <td style="padding: 8px; border: 1px solid #80cbc4;">
                                {data.get('water_temp_c')}°C ({water_temp_f}°F)
                            </td>
                        </tr>
                        """
                    if data.get("air_temp_c") is not None:
                        air_temp_f = round(data.get("air_temp_c") * 9/5 + 32, 1)
                        buoy_html += f"""
                        <tr style="background: white;">
                            <td style="padding: 8px; border: 1px solid #80cbc4; font-weight: bold;">Air Temperature</td>
                            <td style="padding: 8px; border: 1px solid #80cbc4;">
                                {data.get('air_temp_c')}°C ({air_temp_f}°F)
                            </td>
                        </tr>
                        """
                    if data.get("air_pressure_hpa") is not None:
                        buoy_html += f"""
                        <tr style="background: #e0f2f1;">
                            <td style="padding: 8px; border: 1px solid #80cbc4; font-weight: bold;">Pressure</td>
                            <td style="padding: 8px; border: 1px solid #80cbc4;">{data.get('air_pressure_hpa')} hPa</td>
                        </tr>
                        """

                buoy_html += """
                    </table>
                </div>
                """
            else:
                # Error case
                buoy_html += f"""
                <div style="background: #ffebee; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #c62828;">
                    <h4 style="margin: 0 0 10px 0; color: #c62828;">
                        {info.get('name', 'Unknown')} (Station {info.get('station_id', 'N/A')})
                    </h4>
                    <p style="margin: 5px 0; color: #666;">
                        Distance: {info.get('distance_km', 'N/A')} km
                    </p>
                    <p style="color: #c62828; margin: 10px 0 0 0;">
                        Error fetching data: {buoy.get('error', 'Unknown error')}
                    </p>
                </div>
                """

        buoy_html += "</div>"

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

            {f'''<div class="section">
                <h2>🌊 Real-time Buoy Observations</h2>
                {buoy_html}
            </div>''' if buoy_html else ''}

            <div class="section">
                <h2>🏄 Current Wave Conditions</h2>
                {wave_html}
            </div>

            <div class="section">
                <h2>📊 Wave Forecast (Next 3 Days)</h2>
                {wave_forecast_html if wave_forecast_html else '<p>No forecast data available</p>'}
            </div>

            <div class="section">
                <h2>💨 Current Wind Conditions</h2>
                {wind_html}
            </div>

            <div class="section">
                <h2>🌬️ Wind Forecast (Next 3 Days)</h2>
                {wind_forecast_html if wind_forecast_html else '<p>No forecast data available</p>'}
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
from app.routers import dev, ocean_currents, buoys, wind, waves

app.include_router(dev.router, prefix="/api/v1/dev", tags=["dev"])
app.include_router(ocean_currents.router, prefix="/api", tags=["ocean-currents"])
app.include_router(buoys.router, prefix="/api", tags=["buoys"])
app.include_router(wind.router, prefix="/api", tags=["wind"])
app.include_router(waves.router, prefix="/api", tags=["waves"])

# TODO: Add production routers
# from app.routers import forecast, spots, users
# app.include_router(forecast.router, prefix="/api/v1/forecast", tags=["forecast"])
# app.include_router(spots.router, prefix="/api/v1/spots", tags=["spots"])
# app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
