"""SWAN model domain and output visualization API"""

import json
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
from fastapi import APIRouter, HTTPException, Query

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
DOMAINS_DIR = PROJECT_ROOT / "data" / "grids" / "swan_domains"
RUNS_DIR = PROJECT_ROOT / "data" / "swan_runs"


@router.get("/swan/domains")
async def list_swan_domains():
    """List available SWAN domains."""
    domains = []

    for config_path in DOMAINS_DIR.glob("*/config.json"):
        try:
            with open(config_path) as f:
                config = json.load(f)
                domains.append({
                    "name": config.get("name"),
                    "region": config.get("region"),
                    "resolution_m": config.get("resolution_m"),
                    "n_wet_cells": config.get("n_wet_cells"),
                    "lat_range": [config.get("lat_min"), config.get("lat_max")],
                    "lon_range": [config.get("lon_min"), config.get("lon_max")],
                })
        except Exception as e:
            logger.warning(f"Failed to load domain config {config_path}: {e}")

    return {"domains": domains}


@router.get("/swan/domain/{domain_name}")
async def get_swan_domain(
    domain_name: str,
    include_bathymetry: bool = Query(True, description="Include bathymetry grid data"),
    include_boundary: bool = Query(True, description="Include boundary condition points"),
    downsample: int = Query(1, ge=1, le=10, description="Downsample factor for large grids"),
):
    """
    Get SWAN domain data for visualization.

    Returns bathymetry grid and boundary condition points.
    """
    domain_dir = DOMAINS_DIR / domain_name
    config_path = domain_dir / "config.json"

    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain_name}")

    with open(config_path) as f:
        config = json.load(f)

    result = {
        "name": config.get("name"),
        "region": config.get("region"),
        "resolution_m": config.get("resolution_m"),
        "lat_min": config.get("lat_min"),
        "lat_max": config.get("lat_max"),
        "lon_min": config.get("lon_min"),
        "lon_max": config.get("lon_max"),
        "n_lat": config.get("n_lat"),
        "n_lon": config.get("n_lon"),
        "n_wet_cells": config.get("n_wet_cells"),
        "offshore_boundary_km": config.get("offshore_boundary_km"),
    }

    # Load bathymetry if requested
    if include_bathymetry and XARRAY_AVAILABLE:
        nc_path = Path(config.get("file_nc", ""))
        if nc_path.exists():
            try:
                ds = xr.open_dataset(nc_path)

                lats = ds["lat"].values
                lons = ds["lon"].values
                elevation = ds["elevation_masked"].values

                # Downsample if requested
                if downsample > 1:
                    lats = lats[::downsample]
                    lons = lons[::downsample]
                    elevation = elevation[::downsample, ::downsample]

                # Convert elevation to depth (positive) for visualization
                # NaN stays NaN, negative elevation becomes positive depth
                depth = np.where(np.isnan(elevation), np.nan, -elevation)

                result["bathymetry"] = {
                    "lat": lats.tolist(),
                    "lon": lons.tolist(),
                    "depth": np.where(np.isnan(depth), None, depth).tolist(),
                    "downsample_factor": downsample,
                }

                ds.close()

            except Exception as e:
                logger.error(f"Failed to load bathymetry: {e}")
                result["bathymetry_error"] = str(e)

    # Load boundary points if requested
    if include_boundary:
        boundary_path = domain_dir / "boundary" / "boundary_points.json"
        if boundary_path.exists():
            try:
                with open(boundary_path) as f:
                    boundary_data = json.load(f)
                    result["boundary_points"] = boundary_data.get("points", [])
            except Exception as e:
                logger.warning(f"Failed to load boundary points: {e}")

        # Also check for latest run boundary points
        latest_run = get_latest_run(domain_name)
        if latest_run:
            result["latest_run"] = latest_run

    return result


@router.get("/swan/runs/{domain_name}")
async def list_swan_runs(domain_name: str):
    """List SWAN runs for a domain."""
    runs_dir = RUNS_DIR / domain_name

    if not runs_dir.exists():
        return {"runs": []}

    runs = []
    for meta_path in sorted(runs_dir.glob("*/run_metadata.json"), reverse=True):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                runs.append({
                    "run_id": meta.get("run_id"),
                    "created_at": meta.get("created_at"),
                    "swan_completed": meta.get("swan_completed", False),
                    "n_boundary_points": meta.get("n_boundary_points"),
                    "ww3_time_range": meta.get("ww3_time_range"),
                })
        except Exception as e:
            logger.warning(f"Failed to load run metadata {meta_path}: {e}")

    return {"runs": runs}


@router.get("/swan/run/{domain_name}/{run_id}")
async def get_swan_run(domain_name: str, run_id: str):
    """Get details of a specific SWAN run."""
    run_dir = RUNS_DIR / domain_name / run_id
    meta_path = run_dir / "run_metadata.json"

    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {domain_name}/{run_id}")

    with open(meta_path) as f:
        meta = json.load(f)

    # Load boundary conditions from TPAR file
    tpar_path = run_dir / "boundary.tpar"
    boundary_conditions = []

    if tpar_path.exists():
        try:
            with open(tpar_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("$") or line == "TPAR" or not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        boundary_conditions.append({
                            "time": parts[0],
                            "hs": float(parts[1]),
                            "tp": float(parts[2]),
                            "dir": float(parts[3]),
                            "spreading": float(parts[4]),
                        })
        except Exception as e:
            logger.warning(f"Failed to parse TPAR file: {e}")

    return {
        **meta,
        "boundary_conditions": boundary_conditions,
    }


@router.get("/swan/output/{domain_name}")
async def get_swan_output(
    domain_name: str,
    run_id: Optional[str] = Query(None, description="Specific run ID, or latest if not specified"),
    lat: Optional[float] = Query(None, description="Latitude for point extraction"),
    lon: Optional[float] = Query(None, description="Longitude for point extraction"),
    downsample: int = Query(4, ge=1, le=20, description="Downsample factor for grid data"),
):
    """
    Get SWAN model output with partitioned swell components.

    Returns wave parameters for:
    - Combined sea state (total Hs, Tp, Dir)
    - Wind sea component (partition 0)
    - Swell 1-6: Up to 6 swell partitions sorted by energy (partition 1=largest)

    California typically sees multiple swell sources:
    - NW groundswell (Aleutian storms)
    - W groundswell (North Pacific)
    - SW swell (South Pacific)
    - S swell (Southern Hemisphere)
    - SSE swell (hurricanes/tropics)
    - Local wind waves

    Output format supports surf-style display:
    "3.2ft @ 14s WNW + 1.8ft @ 9s SSW + 0.8ft @ 16s S"
    """
    # Find the run directory
    if run_id:
        run_dir = RUNS_DIR / domain_name / run_id
    else:
        # Get latest run
        runs_dir = RUNS_DIR / domain_name
        if not runs_dir.exists():
            raise HTTPException(status_code=404, detail=f"No runs found for domain: {domain_name}")

        run_dirs = sorted(runs_dir.glob("*/run_metadata.json"), reverse=True)
        if not run_dirs:
            raise HTTPException(status_code=404, detail=f"No runs found for domain: {domain_name}")

        run_dir = run_dirs[0].parent

    # Check for SWAN output
    output_path = run_dir / "swan_output.json"
    meta_path = run_dir / "run_metadata.json"

    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Run not found")

    with open(meta_path) as f:
        meta = json.load(f)

    result = {
        "domain": domain_name,
        "run_id": meta.get("run_id"),
        "created_at": meta.get("created_at"),
        "swan_completed": meta.get("swan_completed", False),
        "ww3_time_range": meta.get("ww3_time_range"),
    }

    # If SWAN hasn't completed or no output file, return metadata only
    if not output_path.exists():
        result["status"] = "pending" if not meta.get("swan_completed") else "output_missing"
        result["message"] = "SWAN output not yet available"
        return result

    # Load SWAN output
    with open(output_path) as f:
        swan_data = json.load(f)

    # If point extraction requested
    if lat is not None and lon is not None:
        return extract_point_forecast(swan_data, lat, lon, result)

    # Return downsampled grid data
    result["grid"] = downsample_swan_output(swan_data, downsample)
    result["status"] = "complete"

    return result


@router.get("/swan/forecast")
async def get_swan_point_forecast(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    domain_name: str = Query("california_swan_14km", description="SWAN domain"),
):
    """
    Get surf-style wave forecast at a specific location.

    Returns up to 6 partitioned swell components sorted by energy, plus wind chop.
    California receives swells from multiple sources (NW, W, SW, S, SSE storms).

    Example response:
    {
        "location": {"lat": 33.66, "lon": -118.0},
        "swells": [
            {"height_ft": 3.2, "period_s": 14, "direction_deg": 285, "direction_compass": "WNW", "partition": "primary"},
            {"height_ft": 1.8, "period_s": 9, "direction_deg": 195, "direction_compass": "SSW", "partition": "secondary"},
            {"height_ft": 0.8, "period_s": 16, "direction_deg": 180, "direction_compass": "S", "partition": "tertiary"}
        ],
        "wind_chop": {"height_ft": 0.5, "period_s": 5, "direction_deg": 270},
        "summary": "3.2ft @ 14s WNW + 1.8ft @ 9s SSW + 0.8ft @ 16s S"
    }
    """
    # Find latest run
    runs_dir = RUNS_DIR / domain_name
    if not runs_dir.exists():
        raise HTTPException(status_code=404, detail=f"No runs for domain: {domain_name}")

    run_dirs = sorted(runs_dir.glob("*/run_metadata.json"), reverse=True)
    if not run_dirs:
        raise HTTPException(status_code=404, detail=f"No runs for domain: {domain_name}")

    run_dir = run_dirs[0].parent
    output_path = run_dir / "swan_output.json"

    if not output_path.exists():
        # If no SWAN output, fall back to WW3 partitioned data
        return await get_ww3_fallback_forecast(lat, lon)

    with open(output_path) as f:
        swan_data = json.load(f)

    return extract_point_forecast(swan_data, lat, lon, {"domain": domain_name})


def extract_point_forecast(swan_data: dict, lat: float, lon: float, base_result: dict) -> dict:
    """Extract wave forecast at a specific point."""
    lats = swan_data.get("lat", [])
    lons = swan_data.get("lon", [])

    if not lats or not lons:
        return {**base_result, "error": "No coordinate data in SWAN output"}

    # Find nearest grid point
    lat_idx = min(range(len(lats)), key=lambda i: abs(lats[i] - lat))
    lon_idx = min(range(len(lons)), key=lambda i: abs(lons[i] - lon))

    def get_value(component: str, var: str) -> Optional[float]:
        """Extract value at point from nested structure."""
        data = swan_data.get(component, {}).get(var)
        if data is None:
            return None
        if isinstance(data, list) and len(data) > lat_idx:
            row = data[lat_idx]
            if isinstance(row, list) and len(row) > lon_idx:
                val = row[lon_idx]
                return val if val is not None and not (isinstance(val, float) and np.isnan(val)) else None
        return None

    # Build surf-style forecast
    swells = []

    # Partition names for surf-style labeling
    partition_names = ["primary", "secondary", "tertiary", "quaternary", "quinary", "senary"]

    # Extract up to 6 swell partitions
    for i in range(1, 7):
        swell_key = f"swell{i}"
        hs = get_value(swell_key, "hs")
        if hs and hs > 0.1:  # Only include swells with significant height
            swells.append({
                "height_ft": round(hs * 3.28084, 1),
                "height_m": round(hs, 2),
                "period_s": get_value(swell_key, "tp"),
                "direction_deg": get_value(swell_key, "dir"),
                "direction_compass": direction_to_compass(get_value(swell_key, "dir")),
                "partition": partition_names[i - 1],
                "partition_number": i,
            })

    # Wind chop
    hs_wind = get_value("windsea", "hs")
    wind_chop = None
    if hs_wind and hs_wind > 0.05:
        wind_chop = {
            "height_ft": round(hs_wind * 3.28084, 1),
            "height_m": round(hs_wind, 2),
            "period_s": get_value("windsea", "tp"),
            "direction_deg": get_value("windsea", "dir"),
            "direction_compass": direction_to_compass(get_value("windsea", "dir")),
        }

    # Combined sea state
    combined = {
        "height_ft": round((get_value("combined", "hsig") or 0) * 3.28084, 1),
        "height_m": get_value("combined", "hsig"),
        "period_s": get_value("combined", "tpeak"),
        "direction_deg": get_value("combined", "dir"),
        "direction_compass": direction_to_compass(get_value("combined", "dir")),
    }

    # Generate surf-style summary string
    summary_parts = []
    for swell in swells:
        if swell.get("height_ft") and swell.get("period_s"):
            summary_parts.append(
                f"{swell['height_ft']}ft @ {int(swell['period_s'])}s {swell['direction_compass']}"
            )

    return {
        **base_result,
        "location": {"lat": lats[lat_idx], "lon": lons[lon_idx]},
        "combined": combined,
        "swells": swells,
        "wind_chop": wind_chop,
        "summary": " + ".join(summary_parts) if summary_parts else "Flat",
        "status": "complete",
    }


def downsample_swan_output(swan_data: dict, factor: int) -> dict:
    """Downsample SWAN grid output for efficient transfer."""
    result = {}

    # Downsample coordinates
    lats = swan_data.get("lat", [])
    lons = swan_data.get("lon", [])

    result["lat"] = lats[::factor]
    result["lon"] = lons[::factor]

    # Downsample each component (combined, windsea, and 6 swell partitions)
    for component in ["combined", "windsea", "swell1", "swell2", "swell3", "swell4", "swell5", "swell6"]:
        comp_data = swan_data.get(component, {})
        if comp_data:
            result[component] = {}
            for var in ["hs", "hsig", "tp", "tpeak", "dir", "depth"]:
                if var in comp_data:
                    data = comp_data[var]
                    if isinstance(data, list):
                        result[component][var] = [
                            row[::factor] if isinstance(row, list) else row
                            for row in data[::factor]
                        ]

    return result


async def get_ww3_fallback_forecast(lat: float, lon: float) -> dict:
    """Fall back to WW3 partitioned data when SWAN not available."""
    # This would query the WW3 zarr store directly
    # For now, return a placeholder
    return {
        "location": {"lat": lat, "lon": lon},
        "source": "WW3 (SWAN not available)",
        "swells": [],
        "wind_chop": None,
        "summary": "SWAN run required for nearshore forecast",
        "status": "fallback",
    }


def direction_to_compass(degrees: Optional[float]) -> str:
    """Convert degrees to compass direction."""
    if degrees is None:
        return "N/A"
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    index = round(degrees / 22.5) % 16
    return directions[index]


def get_latest_run(domain_name: str) -> Optional[dict]:
    """Get the latest SWAN run for a domain."""
    runs_dir = RUNS_DIR / domain_name

    if not runs_dir.exists():
        return None

    # Find most recent run
    run_dirs = sorted(runs_dir.glob("*/run_metadata.json"), reverse=True)
    if not run_dirs:
        return None

    try:
        with open(run_dirs[0]) as f:
            meta = json.load(f)
            return {
                "run_id": meta.get("run_id"),
                "created_at": meta.get("created_at"),
                "swan_completed": meta.get("swan_completed", False),
                "n_boundary_points": meta.get("n_boundary_points"),
            }
    except Exception:
        return None
