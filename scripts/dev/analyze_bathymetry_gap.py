#!/usr/bin/env python3
"""
Bathymetry Gap Analysis

Analyzes the gap between:
1. SWAN model output points (coarse/fine mesh valid ocean cells)
2. High-resolution USACE lidar nearshore bathymetry
3. The actual shoreline/surf zone

Questions answered:
- How far are SWAN output points from the shoreline?
- Does GEBCO bathymetry cover the gap to shore?
- Where does USACE lidar coverage begin/end?
- What resolution is needed to bridge this gap?

Run from project root:
    python scripts/dev/analyze_bathymetry_gap.py
"""

import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# Data Paths
# =============================================================================

GEBCO_PATH = project_root / "data/raw/bathymetry/gebco_2024/gebco_2024_california.nc"
USACE_DIR = project_root / "data/raw/bathymetry/USACE_CA_DEM_2009_9488"
CRM_PATH = project_root / "data/raw/bathymetry/ncei_crm/crm_california.nc"
MESH_DIR = project_root / "data/meshes/socal"

# Surf spots to analyze
SURF_SPOTS = {
    "huntington_pier": {"lat": 33.6556, "lon": -117.9999, "display": "Huntington Beach Pier"},
    "trestles": {"lat": 33.3825, "lon": -117.5883, "display": "Trestles"},
    "blacks_beach": {"lat": 32.8894, "lon": -117.2528, "display": "Black's Beach"},
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in km."""
    R = 6371  # Earth's radius in km
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def load_gebco() -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load GEBCO bathymetry data."""
    print("Loading GEBCO data...")
    ds = xr.open_dataset(GEBCO_PATH)
    elevation = ds['elevation'].values
    lats = ds['lat'].values
    lons = ds['lon'].values
    ds.close()

    # Convert to depth (positive = ocean depth)
    depth = -elevation.astype(np.float64)
    depth[depth <= 0] = np.nan  # Mask land

    resolution_deg = np.abs(lats[1] - lats[0])
    resolution_m = resolution_deg * 111000 * np.cos(np.radians(np.mean(lats)))
    print(f"  Shape: {depth.shape}")
    print(f"  Resolution: {resolution_deg:.5f}° (~{resolution_m:.0f}m)")
    return depth, lats, lons, resolution_m


def load_mesh_bathymetry(mesh_name: str = "socal_fine") -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load SWAN mesh bathymetry and metadata."""
    mesh_type = mesh_name.split('_')[-1]
    mesh_dir = MESH_DIR / mesh_type

    # Load metadata
    json_path = mesh_dir / f"{mesh_name}.json"
    with open(json_path) as f:
        metadata = json.load(f)

    # Load .bot file (SWAN format)
    bot_path = mesh_dir / f"{mesh_name}.bot"
    depth = np.loadtxt(bot_path)

    # Build coordinates
    origin = metadata["origin"]
    nx, ny = metadata["nx"], metadata["ny"]
    dx, dy = metadata["dx"], metadata["dy"]

    lons = np.linspace(origin[0], origin[0] + nx * dx, nx + 1)
    lats = np.linspace(origin[1], origin[1] + ny * dy, ny + 1)

    # Mask land (exception value = -99)
    exception_val = metadata.get("exception_value", -99.0)
    depth = np.where(depth == exception_val, np.nan, depth)

    print(f"\nLoaded {mesh_name}:")
    print(f"  Grid: {depth.shape[1]} x {depth.shape[0]}")
    print(f"  Resolution: {metadata['resolution_km']} km")
    print(f"  Origin: ({origin[0]:.2f}, {origin[1]:.2f})")

    return depth, lats, lons, metadata


def analyze_spot_to_swan_distance(spot: dict, mesh_depth: np.ndarray,
                                   mesh_lats: np.ndarray, mesh_lons: np.ndarray) -> dict:
    """
    Analyze the distance from a surf spot to the nearest valid SWAN output point.
    Also finds distance to the shoreline (first land cell).
    """
    spot_lat, spot_lon = spot["lat"], spot["lon"]

    # Find nearest grid indices
    i_nearest = np.argmin(np.abs(mesh_lons - spot_lon))
    j_nearest = np.argmin(np.abs(mesh_lats - spot_lat))

    ny, nx = mesh_depth.shape

    # Find nearest valid (ocean) point by expanding search
    best_i, best_j = None, None
    best_distance = float('inf')

    for radius in range(0, 15):
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if abs(di) != radius and abs(dj) != radius and radius > 0:
                    continue

                i = i_nearest + di
                j = j_nearest + dj

                if 0 <= i < nx and 0 <= j < ny:
                    if not np.isnan(mesh_depth[j, i]):
                        dist = haversine_km(spot_lat, spot_lon, mesh_lats[j], mesh_lons[i])
                        if dist < best_distance:
                            best_distance = dist
                            best_i, best_j = i, j

        if best_i is not None:
            break

    # Calculate distance from SWAN point to shoreline (nearest land)
    shore_distance = float('inf')
    shore_lat, shore_lon = None, None

    if best_i is not None:
        for radius in range(1, 30):
            found_land = False
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if abs(di) != radius and abs(dj) != radius:
                        continue

                    i = best_i + di
                    j = best_j + dj

                    if 0 <= i < nx and 0 <= j < ny:
                        if np.isnan(mesh_depth[j, i]):  # Land cell
                            dist = haversine_km(mesh_lats[best_j], mesh_lons[best_i],
                                              mesh_lats[j], mesh_lons[i])
                            if dist < shore_distance:
                                shore_distance = dist
                                shore_lat, shore_lon = mesh_lats[j], mesh_lons[i]
                                found_land = True

            if found_land:
                break

    return {
        "spot_to_swan_km": best_distance if best_i else float('inf'),
        "swan_to_shore_km": shore_distance if shore_distance < float('inf') else None,
        "swan_point": (mesh_lons[best_i], mesh_lats[best_j]) if best_i else None,
        "swan_indices": (best_i, best_j) if best_i else None,
        "swan_depth_m": mesh_depth[best_j, best_i] if best_i else None,
        "nearest_shore": (shore_lon, shore_lat) if shore_lat else None,
    }


def analyze_gebco_coverage(spot: dict, gebco_depth: np.ndarray,
                           gebco_lats: np.ndarray, gebco_lons: np.ndarray,
                           swan_point: Tuple[float, float]) -> dict:
    """
    Analyze GEBCO bathymetry coverage between SWAN point and shore.
    Returns transect data.
    """
    spot_lat, spot_lon = spot["lat"], spot["lon"]
    swan_lon, swan_lat = swan_point

    # Sample GEBCO along a transect from SWAN point toward shore
    n_samples = 100
    transect_lons = np.linspace(swan_lon, spot_lon, n_samples)
    transect_lats = np.linspace(swan_lat, spot_lat, n_samples)

    depths = []
    for lon, lat in zip(transect_lons, transect_lats):
        i = np.argmin(np.abs(gebco_lons - lon))
        j = np.argmin(np.abs(gebco_lats - lat))
        depths.append(gebco_depth[j, i])

    depths = np.array(depths)
    valid_mask = ~np.isnan(depths)

    # Calculate distances along transect
    distances = np.array([
        haversine_km(swan_lat, swan_lon, lat, lon)
        for lat, lon in zip(transect_lats, transect_lons)
    ])

    # Find where GEBCO ocean data ends (hits land)
    if valid_mask.any():
        last_valid_idx = np.where(valid_mask)[0][-1]
        last_valid_lon = transect_lons[last_valid_idx]
        last_valid_lat = transect_lats[last_valid_idx]
        last_valid_depth = depths[last_valid_idx]

        # Distance from last valid GEBCO point to spot
        remaining_gap_km = haversine_km(last_valid_lat, last_valid_lon, spot_lat, spot_lon)
    else:
        last_valid_lon, last_valid_lat = None, None
        last_valid_depth = None
        remaining_gap_km = None

    return {
        "transect_depths": depths,
        "transect_distances": distances,
        "transect_lons": transect_lons,
        "transect_lats": transect_lats,
        "last_valid_point": (last_valid_lon, last_valid_lat),
        "last_valid_depth_m": last_valid_depth,
        "remaining_gap_km": remaining_gap_km,
    }


def find_shoreline_points(mesh_depth: np.ndarray, mesh_lats: np.ndarray,
                          mesh_lons: np.ndarray) -> List[Tuple[float, float]]:
    """Find approximate shoreline points (where ocean meets land)."""
    ny, nx = mesh_depth.shape
    shoreline = []

    for j in range(ny):
        for i in range(nx):
            if np.isnan(mesh_depth[j, i]):  # Land cell
                # Check if any neighbor is ocean
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nx and 0 <= nj < ny:
                        if not np.isnan(mesh_depth[nj, ni]):
                            shoreline.append((mesh_lons[i], mesh_lats[j]))
                            break

    return shoreline


def scan_usace_tiles() -> dict:
    """Scan USACE tiles to understand coverage."""
    print("\nScanning USACE lidar tiles...")
    tif_files = sorted(USACE_DIR.glob("*.tif"))
    print(f"  Found {len(tif_files)} tiles (1m resolution)")

    # The USACE lidar covers nearshore California in 1m resolution tiles
    # Total ~3.1 GB of data
    return {
        "n_tiles": len(tif_files),
        "resolution_m": 1,
        "coverage": "Nearshore California (topobathymetric lidar)",
    }


def create_analysis_figure(spot_name: str, spot: dict,
                          mesh_analysis: dict, gebco_analysis: dict,
                          gebco_depth: np.ndarray, gebco_lats: np.ndarray, gebco_lons: np.ndarray,
                          mesh_depth: np.ndarray, mesh_lats: np.ndarray, mesh_lons: np.ndarray,
                          gebco_resolution_m: float) -> go.Figure:
    """Create comprehensive figure showing the gap analysis for a single spot."""

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"GEBCO Bathymetry (~{gebco_resolution_m:.0f}m)",
            "SWAN Fine Mesh (1km) Grid",
            "Depth Transect: SWAN Point → Surf Spot",
            f"Gap Analysis Summary: {spot['display']}"
        ],
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "scatter"}, {"type": "table"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    spot_lat, spot_lon = spot["lat"], spot["lon"]

    # Define view extent
    view_extent_deg = 0.05
    lon_min, lon_max = spot_lon - view_extent_deg*2, spot_lon + view_extent_deg
    lat_min, lat_max = spot_lat - view_extent_deg, spot_lat + view_extent_deg

    # Panel 1: GEBCO bathymetry
    lat_mask = (gebco_lats >= lat_min) & (gebco_lats <= lat_max)
    lon_mask = (gebco_lons >= lon_min) & (gebco_lons <= lon_max)
    gebco_sub = gebco_depth[np.ix_(lat_mask, lon_mask)]

    fig.add_trace(go.Heatmap(
        z=gebco_sub,
        x=gebco_lons[lon_mask],
        y=gebco_lats[lat_mask],
        colorscale="Blues",
        zmin=0, zmax=200,
        colorbar=dict(title="Depth (m)", x=0.45),
        showscale=True,
    ), row=1, col=1)

    # Add spot and SWAN point markers
    fig.add_trace(go.Scatter(
        x=[spot_lon],
        y=[spot_lat],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Surf Spot',
        showlegend=False,
    ), row=1, col=1)

    if mesh_analysis["swan_point"]:
        swan_lon, swan_lat = mesh_analysis["swan_point"]
        fig.add_trace(go.Scatter(
            x=[swan_lon],
            y=[swan_lat],
            mode='markers',
            marker=dict(size=12, color='green', symbol='circle'),
            name='SWAN Point',
            showlegend=False,
        ), row=1, col=1)

    # Panel 2: SWAN mesh grid
    mesh_lat_mask = (mesh_lats >= lat_min) & (mesh_lats <= lat_max)
    mesh_lon_mask = (mesh_lons >= lon_min) & (mesh_lons <= lon_max)
    mesh_sub = mesh_depth[np.ix_(mesh_lat_mask, mesh_lon_mask)]

    fig.add_trace(go.Heatmap(
        z=mesh_sub,
        x=mesh_lons[mesh_lon_mask],
        y=mesh_lats[mesh_lat_mask],
        colorscale="Blues",
        zmin=0, zmax=200,
        showscale=False,
    ), row=1, col=2)

    # Add grid points as scatter overlay
    MLON, MLAT = np.meshgrid(mesh_lons[mesh_lon_mask], mesh_lats[mesh_lat_mask])
    valid_mask = ~np.isnan(mesh_sub)

    fig.add_trace(go.Scatter(
        x=MLON[valid_mask].flatten(),
        y=MLAT[valid_mask].flatten(),
        mode='markers',
        marker=dict(size=4, color='green', opacity=0.6),
        name='Ocean cells',
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=MLON[~valid_mask].flatten(),
        y=MLAT[~valid_mask].flatten(),
        mode='markers',
        marker=dict(size=4, color='brown', opacity=0.4),
        name='Land cells',
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[spot_lon],
        y=[spot_lat],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        showlegend=False,
    ), row=1, col=2)

    # Panel 3: Transect depth profile
    transect_depths = gebco_analysis["transect_depths"]
    transect_distances = gebco_analysis["transect_distances"]

    fig.add_trace(go.Scatter(
        x=transect_distances,
        y=transect_depths,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(65, 105, 225, 0.3)',
        line=dict(color='royalblue', width=2),
        name='GEBCO depth',
    ), row=2, col=1)

    # Mark SWAN point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[mesh_analysis["swan_depth_m"]],
        mode='markers',
        marker=dict(size=12, color='green', symbol='circle'),
        name='SWAN output point',
    ), row=2, col=1)

    # Mark where GEBCO ends
    if gebco_analysis["remaining_gap_km"]:
        total_dist = transect_distances[-1]
        gap_start = total_dist - gebco_analysis["remaining_gap_km"]
        fig.add_vline(x=gap_start, line=dict(color="red", dash="dash"), row=2, col=1)
        fig.add_annotation(
            x=gap_start, y=50,
            text=f"GEBCO ends<br>(gap: {gebco_analysis['remaining_gap_km']:.2f} km)",
            showarrow=True,
            arrowhead=2,
            row=2, col=1,
        )

    # Mark surf spot / shore
    fig.add_trace(go.Scatter(
        x=[transect_distances[-1]],
        y=[0],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Surf spot (shore)',
    ), row=2, col=1)

    # Panel 4: Summary table
    summary_data = [
        ["Metric", "Value"],
        ["Spot Location", f"({spot_lat:.4f}°N, {spot_lon:.4f}°W)"],
        ["Nearest SWAN Point", f"({mesh_analysis['swan_point'][1]:.4f}°N, {mesh_analysis['swan_point'][0]:.4f}°W)" if mesh_analysis['swan_point'] else "N/A"],
        ["Spot → SWAN Distance", f"{mesh_analysis['spot_to_swan_km']:.2f} km"],
        ["SWAN Point Depth", f"{mesh_analysis['swan_depth_m']:.1f} m"],
        ["SWAN → Shore Distance", f"{mesh_analysis['swan_to_shore_km']:.2f} km" if mesh_analysis['swan_to_shore_km'] else "N/A"],
        ["GEBCO → Shore Gap", f"{gebco_analysis['remaining_gap_km']:.2f} km" if gebco_analysis['remaining_gap_km'] else "N/A"],
        ["Last GEBCO Depth", f"{gebco_analysis['last_valid_depth_m']:.1f} m" if gebco_analysis['last_valid_depth_m'] else "N/A"],
    ]

    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Value</b>"],
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12),
        ),
        cells=dict(
            values=[[row[0] for row in summary_data[1:]], [row[1] for row in summary_data[1:]]],
            fill_color='lavender',
            align='left',
            font=dict(size=11),
        ),
    ), row=2, col=2)

    # Update axes
    fig.update_xaxes(title_text="Longitude", row=1, col=1)
    fig.update_yaxes(title_text="Latitude", row=1, col=1)
    fig.update_xaxes(title_text="Longitude", row=1, col=2)
    fig.update_yaxes(title_text="Latitude", row=1, col=2)
    fig.update_xaxes(title_text="Distance from SWAN point (km)", row=2, col=1)
    fig.update_yaxes(title_text="Depth (m)", autorange="reversed", row=2, col=1)

    fig.update_layout(
        title=f"Bathymetry Gap Analysis: {spot['display']}",
        height=900,
        width=1200,
        showlegend=True,
        legend=dict(x=0.5, y=-0.05, orientation="h"),
    )

    return fig


def main():
    """Main analysis function."""
    print("="*60)
    print("BATHYMETRY GAP ANALYSIS")
    print("Analyzing the space between SWAN output and the surf zone")
    print("="*60)

    # Load data
    gebco_depth, gebco_lats, gebco_lons, gebco_res_m = load_gebco()
    mesh_depth, mesh_lats, mesh_lons, mesh_meta = load_mesh_bathymetry("socal_fine")

    # Scan USACE tiles
    usace_info = scan_usace_tiles()

    print("\n" + "="*60)
    print("RESOLUTION COMPARISON")
    print("="*60)
    print(f"  WW3 (boundary):     ~25 km (0.25°)")
    print(f"  SWAN Coarse:        5 km")
    print(f"  SWAN Medium:        2.5 km")
    print(f"  SWAN Fine:          1 km")
    print(f"  GEBCO:              ~{gebco_res_m:.0f} m")
    print(f"  NCEI CRM:           ~92 m (3 arc-seconds)")
    print(f"  USACE Lidar:        1 m ({usace_info['n_tiles']} tiles)")
    print("="*60)

    # Analyze each spot
    all_results = {}
    figures = []

    for spot_name, spot in SURF_SPOTS.items():
        print(f"\n>>> Analyzing: {spot['display']}")

        # Distance analysis
        mesh_analysis = analyze_spot_to_swan_distance(spot, mesh_depth, mesh_lats, mesh_lons)
        print(f"    Spot to nearest SWAN point: {mesh_analysis['spot_to_swan_km']:.2f} km")
        print(f"    SWAN point depth: {mesh_analysis['swan_depth_m']:.1f} m")
        if mesh_analysis['swan_to_shore_km']:
            print(f"    SWAN point to shore: {mesh_analysis['swan_to_shore_km']:.2f} km")

        # GEBCO coverage
        if mesh_analysis["swan_point"]:
            gebco_analysis = analyze_gebco_coverage(spot, gebco_depth, gebco_lats, gebco_lons,
                                                    mesh_analysis["swan_point"])
            if gebco_analysis['remaining_gap_km']:
                print(f"    GEBCO coverage extends to: {gebco_analysis['remaining_gap_km']:.2f} km from spot")
        else:
            gebco_analysis = None

        all_results[spot_name] = {
            "spot": spot,
            "mesh_analysis": mesh_analysis,
            "gebco_analysis": gebco_analysis,
        }

        # Create figure
        if mesh_analysis["swan_point"] and gebco_analysis:
            fig = create_analysis_figure(
                spot_name, spot, mesh_analysis, gebco_analysis,
                gebco_depth, gebco_lats, gebco_lons,
                mesh_depth, mesh_lats, mesh_lons,
                gebco_res_m
            )
            figures.append((spot_name, fig))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: THE GAP PROBLEM AND SOLUTIONS")
    print("="*60)

    print("""
╔════════════════════════════════════════════════════════════════════╗
║                     THE NEARSHORE GAP                               ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  Current Setup:                                                     ║
║  • SWAN fine mesh outputs at 1 km resolution                       ║
║  • Nearest valid ocean points are typically 0.5-2 km from spots    ║
║  • These points are in 20-50+ meter depth water                    ║
║                                                                     ║
║  The Gap:                                                          ║
║  • From SWAN output points to actual surf zone (breaking waves)    ║
║  • Breaking waves occur in 1-5m depth water                        ║
║  • This gap spans approximately 1-3 km                             ║
║                                                                     ║
╠════════════════════════════════════════════════════════════════════╣
║                   BATHYMETRY COVERAGE                               ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  GEBCO (~462m resolution):                                         ║
║  • Covers most of the gap from SWAN to shore                       ║
║  • Typically extends to within 0.5-1 km of shoreline               ║
║  • Good for depths > 10-20m                                        ║
║                                                                     ║
║  NCEI CRM (~92m resolution):                                       ║
║  • Available but NOT YET INTEGRATED                                ║
║  • Could bridge gap between GEBCO and USACE lidar                  ║
║  • Better nearshore coverage than GEBCO                            ║
║                                                                     ║
║  USACE Lidar (1m resolution):                                      ║
║  • Ultra-high resolution for final nearshore segment               ║
║  • Covers 0-10km from coast (topobathymetric)                      ║
║  • Available but NOT YET INTEGRATED                                ║
║                                                                     ║
╠════════════════════════════════════════════════════════════════════╣
║                   BRIDGING OPTIONS                                  ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  Option 1: Nested SWAN Modeling (Most Accurate)                    ║
║  ─────────────────────────────────────────────────────             ║
║  • Run additional nested SWAN at 100-200m resolution               ║
║  • Use GEBCO/CRM for intermediate depths (20-100m)                 ║
║  • Use USACE lidar for final nearshore (<20m depth)                ║
║  • Computationally expensive but captures full physics             ║
║                                                                     ║
║  Option 2: Analytical Shoaling (Fast, Approximate)                 ║
║  ─────────────────────────────────────────────────────             ║
║  • Extract swell parameters from SWAN output points                ║
║  • Apply linear wave theory shoaling: H2 = H1 * sqrt(Cg1/Cg2)     ║
║  • Use high-res bathymetry transects from SWAN point to shore      ║
║  • Fast computation but ignores refraction/diffraction             ║
║                                                                     ║
║  Option 3: Ray Tracing (Medium Complexity)                         ║
║  ─────────────────────────────────────────────────────             ║
║  • Trace wave rays from SWAN output point to shore                 ║
║  • Use Snell's law for refraction over bathymetry                  ║
║  • Calculate spreading and shoaling along rays                     ║
║  • Captures refraction without full spectral model                 ║
║                                                                     ║
║  Recommended Approach:                                             ║
║  • Start with Option 2 (analytical) for quick results              ║
║  • Validate against buoy data at intermediate depths               ║
║  • Add nested SWAN for high-priority spots if needed               ║
║                                                                     ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    # Save figures
    output_dir = project_root / "scripts/dev/outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    html_paths = []
    for spot_name, fig in figures:
        output_path = output_dir / f"bathymetry_gap_{spot_name}.html"
        fig.write_html(str(output_path))
        html_paths.append(output_path)
        print(f"Saved: {output_path}")

    # Create combined summary figure
    if figures:
        print("\nOpening first analysis in browser...")
        webbrowser.open(f"file://{html_paths[0]}")

    return all_results


if __name__ == "__main__":
    results = main()
