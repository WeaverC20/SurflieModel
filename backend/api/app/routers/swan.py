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
