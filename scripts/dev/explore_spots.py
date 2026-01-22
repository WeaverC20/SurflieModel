#!/usr/bin/env python3
"""
Interactive Spot Explorer

Plotly-based visualization for comparing SWAN outputs across different meshes.
Hover over any grid point to see partition details.

Run from project root:
    python scripts/dev/explore_spots.py

Opens an HTML file in your browser with interactive visualization.
"""

import asyncio
import contextlib
import os
import sys
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# Suppress libcurl verbose output from netCDF4
# =============================================================================

@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output (e.g., libcurl verbose logs)."""
    # Save original stderr
    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)

    # Open /dev/null and redirect stderr to it
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)

    try:
        yield
    finally:
        # Restore original stderr
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.pipelines.buoy.fetcher import NDBCBuoyFetcher, CDIPBuoyFetcher
from data.spots import SOCAL_SPOTS, SurfSpot
from data.swan.analysis.output_reader import SwanOutput, SwanOutputReader


# =============================================================================
# Buoy Configuration
# =============================================================================

# NDBC buoys relevant for Southern California with locations
# Using spectral data with r1 confidence for direction reliability
SOCAL_NDBC_BUOYS = {
    "46222": {"name": "San Pedro", "lat": 33.618, "lon": -118.317},
    "46025": {"name": "Santa Monica Basin", "lat": 33.749, "lon": -119.053},
    "46047": {"name": "Tanner Bank", "lat": 32.433, "lon": -119.533},
    "46086": {"name": "San Clemente Basin", "lat": 32.491, "lon": -118.034},
    "46069": {"name": "South Santa Rosa Island", "lat": 33.674, "lon": -120.212},
    "46053": {"name": "East Santa Barbara", "lat": 34.252, "lon": -119.841},
    "46219": {"name": "San Nicolas Island", "lat": 33.221, "lon": -119.882},
    "46011": {"name": "Santa Maria", "lat": 34.868, "lon": -120.857},
}

# CDIP buoys for Southern California - higher quality directional spectra
# NOTE: Station numbers get reassigned! These are current as of Jan 2026.
# Source: https://cdip.ucsd.edu/data_access/sccoos.cdip
SOCAL_CDIP_BUOYS = {
    # San Diego region
    "191": {"name": "Point Loma South", "lat": 32.518, "lon": -117.425, "depth_m": 1050},
    "220": {"name": "Mission Bay West", "lat": 32.749, "lon": -117.499, "depth_m": 563},
    "100": {"name": "Torrey Pines Outer", "lat": 32.930, "lon": -117.392, "depth_m": 572},
    # North San Diego / Camp Pendleton
    "045": {"name": "Oceanside Offshore", "lat": 33.181, "lon": -117.475, "depth_m": 223},
    "271": {"name": "Green Beach Offshore", "lat": 33.336, "lon": -117.659, "depth_m": 303},
    # Channel Islands / Offshore
    "067": {"name": "San Nicolas Island", "lat": 33.226, "lon": -119.892, "depth_m": 315},
    "203": {"name": "Santa Cruz Basin", "lat": 33.769, "lon": -119.563, "depth_m": 1890},
    # LA Basin
    "213": {"name": "San Pedro South", "lat": 33.578, "lon": -118.182, "depth_m": 66},
    "092": {"name": "San Pedro", "lat": 33.618, "lon": -118.317, "depth_m": 477},
    "028": {"name": "Santa Monica Bay", "lat": 33.860, "lon": -118.641, "depth_m": 387},
    # Santa Barbara Channel
    "071": {"name": "Harvest", "lat": 34.452, "lon": -120.780, "depth_m": 550},
    "222": {"name": "Santa Lucia Escarpment", "lat": 34.774, "lon": -121.491, "depth_m": 650},
}


@dataclass
class BuoyPartition:
    """A single wave partition from NDBC buoy spectral data."""
    partition_id: int
    height_m: float
    period_s: Optional[float] = None
    direction_deg: Optional[float] = None
    wave_type: Optional[str] = None  # e.g., "swell", "wind_waves", "long_period_swell"
    energy_pct: Optional[float] = None  # Percentage of total wave energy
    r1: Optional[float] = None  # Directional confidence (0-1)
    confidence: Optional[str] = None  # HIGH/MED/LOW based on r1


@dataclass
class BuoyData:
    """Container for buoy observation data with multiple partitions."""
    station_id: str
    name: str
    lat: float
    lon: float
    timestamp: Optional[str] = None
    # Multiple partitions from spectral analysis
    partitions: List[BuoyPartition] = field(default_factory=list)
    # Combined
    combined_height_m: Optional[float] = None
    combined_period_s: Optional[float] = None
    combined_direction_deg: Optional[float] = None
    error: Optional[str] = None

    def format_hover_text(self) -> str:
        """Generate HTML hover text for this buoy."""
        lines = [
            f"<b>NDBC Buoy {self.station_id}</b>",
            f"<b>{self.name}</b>",
            f"Location: ({self.lat:.3f}, {self.lon:.3f})",
        ]

        if self.error:
            lines.append(f"<span style='color: #ff6666'>Error: {self.error}</span>")
            return "<br>".join(lines)

        if self.timestamp:
            # Just show time portion
            time_part = self.timestamp.split("T")[1][:5] if "T" in self.timestamp else self.timestamp
            lines.append(f"Time: {time_part} UTC")

        lines.append("")

        # Combined
        if self.combined_height_m is not None:
            dir_str = f"{self.combined_direction_deg:.0f}°" if self.combined_direction_deg else "--"
            period_str = f"{self.combined_period_s:.1f}s" if self.combined_period_s else "--"
            lines.append(f"<b>Combined:</b> {self.combined_height_m:.2f}m, {period_str}, {dir_str}")

        lines.append("")
        lines.append("<b>Partitions:</b>")
        lines.append("<i>r1: directional confidence (HIGH=clean swell, LOW=mixed)</i>")

        if not self.partitions:
            lines.append("  No partition data available")
        else:
            # Partitions already sorted by energy in fetcher
            for p in self.partitions:
                dir_str = f"{p.direction_deg:.0f}°" if p.direction_deg else "--"
                period_str = f"{p.period_s:.1f}s" if p.period_s else "--"
                type_str = f" ({p.wave_type})" if p.wave_type else ""
                energy_str = f" [{p.energy_pct:.0f}%]" if p.energy_pct else ""
                # Show confidence with color coding
                if p.confidence == "HIGH":
                    conf_str = f" <span style='color: #66ff66'>r1={p.r1:.2f} HIGH</span>"
                elif p.confidence == "MED":
                    conf_str = f" <span style='color: #ffff66'>r1={p.r1:.2f} MED</span>"
                else:
                    conf_str = f" <span style='color: #ff6666'>r1={p.r1:.2f} LOW</span>" if p.r1 else ""
                lines.append(f"  #{p.partition_id}: {p.height_m:.2f}m, {period_str}, {dir_str}{type_str}{energy_str}{conf_str}")

        return "<br>".join(lines)


async def fetch_single_buoy(fetcher: NDBCBuoyFetcher, station_id: str, info: dict) -> BuoyData:
    """Fetch partitioned wave data for a single NDBC buoy."""
    buoy_data = BuoyData(
        station_id=station_id,
        name=info["name"],
        lat=info["lat"],
        lon=info["lon"],
    )

    try:
        result = await fetcher.fetch_partitioned_spectral_data(station_id)

        if result.get("status") == "error" or result.get("error"):
            buoy_data.error = result.get("error", "Unknown error")[:50]
            return buoy_data

        buoy_data.timestamp = result.get("timestamp")

        # Extract combined wave parameters
        combined = result.get("combined", {})
        if combined:
            buoy_data.combined_height_m = combined.get("significant_height_m")
            buoy_data.combined_period_s = combined.get("peak_period_s")
            buoy_data.combined_direction_deg = combined.get("peak_direction_deg")

        # Extract multiple partitions with r1 confidence
        partitions = result.get("partitions", [])
        for p in partitions:
            partition = BuoyPartition(
                partition_id=p.get("partition_id", 0),
                height_m=p.get("height_m", 0),
                period_s=p.get("period_s"),
                direction_deg=p.get("direction_deg"),
                wave_type=p.get("type"),
                energy_pct=p.get("energy_pct"),
                r1=p.get("r1"),
                confidence=p.get("confidence"),
            )
            buoy_data.partitions.append(partition)

    except Exception as e:
        buoy_data.error = str(e)[:50]

    return buoy_data


async def fetch_buoy_data(buoys: Dict[str, dict]) -> List[BuoyData]:
    """Fetch partitioned wave data for all NDBC buoys concurrently."""
    fetcher = NDBCBuoyFetcher()

    tasks = [
        fetch_single_buoy(fetcher, station_id, info)
        for station_id, info in buoys.items()
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and return valid BuoyData
    buoy_list = []
    for result in results:
        if isinstance(result, BuoyData):
            buoy_list.append(result)
        elif isinstance(result, Exception):
            print(f"  Warning: Buoy fetch failed: {result}")

    return buoy_list


def load_ndbc_buoy_data(buoys: Dict[str, dict] = SOCAL_NDBC_BUOYS) -> List[BuoyData]:
    """Load NDBC buoy data synchronously (wrapper for async fetch)."""
    return asyncio.run(fetch_buoy_data(buoys))


# =============================================================================
# CDIP Buoy Data
# =============================================================================

@dataclass
class CDIPPartition:
    """A single wave partition from CDIP buoy spectral data."""
    partition_id: int
    height_m: float
    period_s: Optional[float] = None
    direction_deg: Optional[float] = None
    wave_type: Optional[str] = None
    energy_pct: Optional[float] = None


@dataclass
class CDIPBuoyData:
    """Container for CDIP buoy observation data with multiple partitions."""
    station_id: str
    name: str
    lat: float
    lon: float
    depth_m: Optional[float] = None
    timestamp: Optional[str] = None
    partitions: List[CDIPPartition] = field(default_factory=list)
    combined_height_m: Optional[float] = None
    combined_period_s: Optional[float] = None
    combined_direction_deg: Optional[float] = None
    error: Optional[str] = None

    def format_hover_text(self) -> str:
        """Generate HTML hover text for this CDIP buoy."""
        lines = [
            f"<b>CDIP Buoy {self.station_id}</b>",
            f"<b>{self.name}</b>",
            f"Location: ({self.lat:.3f}, {self.lon:.3f})",
        ]

        if self.depth_m:
            lines.append(f"Depth: {self.depth_m}m")

        if self.error:
            lines.append(f"<span style='color: #ff6666'>Error: {self.error}</span>")
            return "<br>".join(lines)

        if self.timestamp:
            time_part = self.timestamp.split("T")[1][:5] if "T" in self.timestamp else self.timestamp
            lines.append(f"Time: {time_part} UTC")

        lines.append("")

        # Combined
        if self.combined_height_m is not None:
            dir_str = f"{self.combined_direction_deg:.0f}°" if self.combined_direction_deg else "--"
            period_str = f"{self.combined_period_s:.1f}s" if self.combined_period_s else "--"
            lines.append(f"<b>Combined:</b> {self.combined_height_m:.2f}m, {period_str}, {dir_str}")

        lines.append("")
        lines.append("<b>Partitions:</b>")
        lines.append("<i>(CDIP MEM spectral analysis)</i>")

        if not self.partitions:
            lines.append("  No partition data available")
        else:
            for p in self.partitions:
                dir_str = f"{p.direction_deg:.0f}°" if p.direction_deg else "--"
                period_str = f"{p.period_s:.1f}s" if p.period_s else "--"

                # Wave type with color
                wave_type = p.wave_type or "unknown"
                type_color = wave_type_color(wave_type)
                type_str = f"<span style='color: {type_color}'>{wave_type}</span>"

                energy_str = f" [{p.energy_pct:.0f}%]" if p.energy_pct else ""
                lines.append(f"  #{p.partition_id}: {p.height_m:.2f}m, {period_str}, {dir_str} ({type_str}){energy_str}")

        return "<br>".join(lines)


async def fetch_single_cdip_buoy(fetcher: CDIPBuoyFetcher, station_id: str, info: dict) -> CDIPBuoyData:
    """Fetch partitioned wave data for a single CDIP buoy."""
    buoy_data = CDIPBuoyData(
        station_id=station_id,
        name=info["name"],
        lat=info["lat"],
        lon=info["lon"],
        depth_m=info.get("depth_m"),
    )

    try:
        # Suppress libcurl verbose output during netCDF4 OpenDAP access
        with suppress_stderr():
            result = await fetcher.fetch_partitioned_wave_data(station_id)

        if result.get("status") == "error" or result.get("error"):
            buoy_data.error = result.get("error", "Unknown error")[:50]
            return buoy_data

        buoy_data.timestamp = result.get("timestamp")

        # Extract combined wave parameters
        combined = result.get("combined", {})
        if combined:
            buoy_data.combined_height_m = combined.get("significant_height_m")
            buoy_data.combined_period_s = combined.get("peak_period_s")
            buoy_data.combined_direction_deg = combined.get("peak_direction_deg")

        # Extract partitions
        partitions = result.get("partitions", [])
        for p in partitions:
            partition = CDIPPartition(
                partition_id=p.get("partition_id", 0),
                height_m=p.get("height_m", 0),
                period_s=p.get("period_s"),
                direction_deg=p.get("direction_deg"),
                wave_type=p.get("type"),
                energy_pct=p.get("energy_pct"),
            )
            buoy_data.partitions.append(partition)

    except Exception as e:
        buoy_data.error = str(e)[:50]

    return buoy_data


async def fetch_cdip_buoy_data(buoys: Dict[str, dict]) -> List[CDIPBuoyData]:
    """Fetch partitioned wave data for all CDIP buoys concurrently."""
    fetcher = CDIPBuoyFetcher()

    tasks = [
        fetch_single_cdip_buoy(fetcher, station_id, info)
        for station_id, info in buoys.items()
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    buoy_list = []
    for result in results:
        if isinstance(result, CDIPBuoyData):
            buoy_list.append(result)
        elif isinstance(result, Exception):
            print(f"  Warning: CDIP buoy fetch failed: {result}")

    return buoy_list


def load_cdip_buoy_data(buoys: Dict[str, dict] = SOCAL_CDIP_BUOYS) -> List[CDIPBuoyData]:
    """Load CDIP buoy data synchronously (wrapper for async fetch)."""
    return asyncio.run(fetch_cdip_buoy_data(buoys))


# =============================================================================
# Data Structures (extensible for future stats)
# =============================================================================

def classify_wave_type(period_s: float) -> str:
    """Classify wave type based on period (same as NDBC classification)."""
    if period_s >= 16:
        return "long_period_swell"
    elif period_s >= 12:
        return "swell"
    elif period_s >= 8:
        return "short_swell"
    else:
        return "wind_waves"


def wave_type_color(wave_type: str) -> str:
    """Get color for wave type display."""
    colors = {
        "long_period_swell": "#66ccff",  # Light blue - distant groundswell
        "swell": "#66ff66",               # Green - good swell
        "short_swell": "#ffff66",         # Yellow - short period
        "wind_waves": "#ff9966",          # Orange - wind chop
    }
    return colors.get(wave_type, "#ffffff")


@dataclass
class PointStats:
    """
    Statistics at a single grid point - extensible for future metrics.

    Add new fields here and they'll automatically appear in tooltips.
    """
    # Grid location
    lon: float
    lat: float
    grid_i: int
    grid_j: int

    # Core SWAN outputs
    hsig: float
    tps: float
    dir: float

    # Partition data: list of (hs, tp, dir, label)
    partitions: List[Tuple[float, float, float, str]]

    # === Future stats - add fields here ===
    # breaking_height: Optional[float] = None
    # iribarren_number: Optional[float] = None
    # surfability_score: Optional[float] = None

    def format_hover_text(self) -> str:
        """Generate HTML hover text for this point (matching NDBC format)."""
        lines = [
            f"<b>SWAN Model Output</b>",
            f"Location: ({self.lat:.4f}, {self.lon:.4f})",
            f"Grid: i={self.grid_i}, j={self.grid_j}",
            "",
        ]

        # Combined wave parameters
        dir_str = f"{self.dir:.0f}°" if not np.isnan(self.dir) else "--"
        period_str = f"{self.tps:.1f}s" if not np.isnan(self.tps) else "--"
        hs_str = f"{self.hsig:.2f}m" if not np.isnan(self.hsig) else "--"
        lines.append(f"<b>Combined:</b> {hs_str}, {period_str}, {dir_str}")

        lines.append("")
        lines.append("<b>Partitions:</b>")
        lines.append("<i>(wave type based on period)</i>")

        # Calculate total energy for percentage (energy ∝ Hs²)
        valid_partitions = [(hs, tp, d, label) for hs, tp, d, label in self.partitions
                           if not np.isnan(hs) and hs > 0]
        total_energy = sum(hs**2 for hs, _, _, _ in valid_partitions)

        if not valid_partitions:
            lines.append("  No partition data available")
        else:
            # Sort by energy (Hs²) descending
            sorted_partitions = sorted(valid_partitions, key=lambda x: x[0]**2, reverse=True)

            for i, (hs, tp, direction, label) in enumerate(sorted_partitions, 1):
                dir_str = f"{direction:.0f}°" if not np.isnan(direction) else "--"
                period_str = f"{tp:.1f}s" if not np.isnan(tp) else "--"

                # Wave type classification
                wave_type = classify_wave_type(tp) if not np.isnan(tp) else "unknown"
                type_color = wave_type_color(wave_type)

                # Energy percentage
                energy_pct = 100 * (hs**2) / total_energy if total_energy > 0 else 0

                # Format similar to NDBC, but include original SWAN label
                type_str = f"<span style='color: {type_color}'>{wave_type}</span>"
                lines.append(
                    f"  #{i} ({label}): {hs:.2f}m, {period_str}, {dir_str} "
                    f"({type_str}) [{energy_pct:.0f}%]"
                )

        return "<br>".join(lines)


@dataclass
class MeshData:
    """Container for a single mesh's data, ready for plotting."""
    name: str
    display_name: str
    swan_output: SwanOutput

    # Pre-computed for plotting
    hsig_masked: np.ndarray = field(default=None)
    hover_texts: np.ndarray = field(default=None)  # 2D array of hover strings

    def __post_init__(self):
        """Pre-compute masked data and hover texts."""
        self.hsig_masked, _, _ = self.swan_output.mask_land()
        self._build_hover_texts()

    def _build_hover_texts(self):
        """Build 2D array of hover text for every grid point."""
        output = self.swan_output
        ny, nx = output.hsig.shape
        self.hover_texts = np.empty((ny, nx), dtype=object)

        for j in range(ny):
            for i in range(nx):
                # Skip land points
                if np.isnan(self.hsig_masked[j, i]):
                    self.hover_texts[j, i] = "Land"
                    continue

                # Gather partition data at this point
                partitions = []
                for p in output.partitions:
                    hs = p.hs[j, i]
                    tp = p.tp[j, i]
                    direction = p.dir[j, i]
                    if hs == output.exception_value:
                        hs, tp, direction = np.nan, np.nan, np.nan
                    partitions.append((hs, tp, direction, p.label))

                # Create stats object
                stats = PointStats(
                    lon=output.lons[i],
                    lat=output.lats[j],
                    grid_i=i,
                    grid_j=j,
                    hsig=output.hsig[j, i],
                    tps=output.tps[j, i],
                    dir=output.dir[j, i],
                    partitions=partitions,
                )

                self.hover_texts[j, i] = stats.format_hover_text()


# =============================================================================
# Data Loading
# =============================================================================

def load_mesh_data(region: str = "socal") -> Dict[str, MeshData]:
    """
    Load SWAN output for all available meshes.

    Returns:
        Dict mapping mesh name to MeshData
    """
    runs_dir = project_root / "data" / "swan" / "runs" / region
    meshes = {}

    mesh_configs = [
        ("coarse", "Coarse (5km)"),
        ("medium", "Medium (2.5km)"),
        ("fine", "Fine (1km)"),
    ]

    for mesh_name, display_name in mesh_configs:
        run_dir = runs_dir / mesh_name / "latest"
        if not run_dir.exists():
            print(f"  Skipping {mesh_name} - no data at {run_dir}")
            continue

        try:
            print(f"  Loading {mesh_name}...")
            reader = SwanOutputReader(run_dir)
            swan_output = reader.read()
            meshes[mesh_name] = MeshData(
                name=mesh_name,
                display_name=display_name,
                swan_output=swan_output,
            )
            print(f"    Grid: {swan_output.hsig.shape}, {len(swan_output.partitions)} partitions")
        except Exception as e:
            print(f"  Error loading {mesh_name}: {e}")

    return meshes


# =============================================================================
# Visualization
# =============================================================================

def create_spot_markers(spots: List[SurfSpot], mesh_data: MeshData) -> go.Scatter:
    """Create scatter markers for surf spots with forecast hover text."""
    lons = [spot.lon for spot in spots]
    lats = [spot.lat for spot in spots]
    names = [spot.display_name for spot in spots]

    # Get forecast at each spot for hover text
    hover_texts = []
    for spot in spots:
        forecast = spot.get_forecast(mesh_data.swan_output)
        lines = [
            f"<b>{spot.display_name}</b>",
            f"<b>SWAN Forecast</b>",
            f"Grid point: ({forecast.grid_i}, {forecast.grid_j})",
            f"Distance to grid: {forecast.distance_km:.2f} km",
            "",
        ]

        # Get valid partitions and calculate total energy
        valid_partitions = [(hs, tp, d, label) for hs, tp, d, label in forecast.partitions
                           if not np.isnan(hs) and hs > 0]
        total_energy = sum(hs**2 for hs, _, _, _ in valid_partitions)

        # Combined parameters (sum of all partitions)
        if valid_partitions:
            combined_hs = np.sqrt(sum(hs**2 for hs, _, _, _ in valid_partitions))
            # Use dominant partition's period and direction
            dominant = max(valid_partitions, key=lambda x: x[0])
            lines.append(f"<b>Combined:</b> {combined_hs:.2f}m, {dominant[1]:.1f}s, {dominant[2]:.0f}°")
        lines.append("")
        lines.append("<b>Partitions:</b>")

        if not valid_partitions:
            lines.append("  No wave energy at this point")
        else:
            # Sort by energy descending
            sorted_partitions = sorted(valid_partitions, key=lambda x: x[0]**2, reverse=True)

            for i, (hs, tp, direction, label) in enumerate(sorted_partitions, 1):
                wave_type = classify_wave_type(tp) if not np.isnan(tp) else "unknown"
                type_color = wave_type_color(wave_type)
                energy_pct = 100 * (hs**2) / total_energy if total_energy > 0 else 0

                type_str = f"<span style='color: {type_color}'>{wave_type}</span>"
                dir_str = f"{direction:.0f}°" if not np.isnan(direction) else "--"
                lines.append(
                    f"  #{i} ({label}): {hs:.2f}m, {tp:.1f}s, {dir_str} "
                    f"({type_str}) [{energy_pct:.0f}%]"
                )

        hover_texts.append("<br>".join(lines))

    return go.Scatter(
        x=lons,
        y=lats,
        mode="markers+text",
        marker=dict(
            size=15,
            color="red",
            symbol="star",
            line=dict(width=2, color="white"),
        ),
        text=names,
        textposition="top center",
        textfont=dict(size=10, color="white"),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
        name="Surf Spots",
    )


def create_ndbc_buoy_markers(buoys: List[BuoyData]) -> go.Scatter:
    """Create scatter markers for NDBC buoys with partitioned swell hover text."""
    lons = [b.lon for b in buoys]
    lats = [b.lat for b in buoys]
    labels = [b.station_id for b in buoys]
    hover_texts = [b.format_hover_text() for b in buoys]

    return go.Scatter(
        x=lons,
        y=lats,
        mode="markers+text",
        marker=dict(
            size=12,
            color="cyan",
            symbol="diamond",
            line=dict(width=2, color="darkblue"),
        ),
        text=labels,
        textposition="bottom center",
        textfont=dict(size=9, color="cyan"),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
        name="NDBC Buoys",
    )


def create_cdip_buoy_markers(buoys: List[CDIPBuoyData]) -> go.Scatter:
    """Create scatter markers for CDIP buoys with partitioned swell hover text."""
    lons = [b.lon for b in buoys]
    lats = [b.lat for b in buoys]
    labels = [f"CDIP {b.station_id}" for b in buoys]
    hover_texts = [b.format_hover_text() for b in buoys]

    return go.Scatter(
        x=lons,
        y=lats,
        mode="markers+text",
        marker=dict(
            size=12,
            color="#ff66ff",  # Magenta to distinguish from NDBC (cyan)
            symbol="square",
            line=dict(width=2, color="darkmagenta"),
        ),
        text=labels,
        textposition="bottom center",
        textfont=dict(size=8, color="#ff66ff"),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
        name="CDIP Buoys",
    )


def create_heatmap(mesh_data: MeshData) -> go.Heatmap:
    """Create Hsig heatmap with hover text showing all partition details."""
    output = mesh_data.swan_output

    return go.Heatmap(
        z=mesh_data.hsig_masked,
        x=output.lons,
        y=output.lats,
        colorscale="YlOrRd",
        zmin=0,
        zmax=3,
        colorbar=dict(
            title=dict(text="Hsig (m)", side="right"),
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=mesh_data.hover_texts,
        name=mesh_data.display_name,
    )


def create_single_mesh_figure(
    mesh_data: MeshData,
    spots: List[SurfSpot],
    ndbc_buoys: Optional[List[BuoyData]] = None,
    cdip_buoys: Optional[List[CDIPBuoyData]] = None,
) -> go.Figure:
    """Create figure for a single mesh."""
    fig = go.Figure()

    # Add heatmap
    fig.add_trace(create_heatmap(mesh_data))

    # Add spot markers
    fig.add_trace(create_spot_markers(spots, mesh_data))

    # Add NDBC buoy markers
    if ndbc_buoys:
        fig.add_trace(create_ndbc_buoy_markers(ndbc_buoys))

    # Add CDIP buoy markers
    if cdip_buoys:
        fig.add_trace(create_cdip_buoy_markers(cdip_buoys))

    # Layout
    output = mesh_data.swan_output
    fig.update_layout(
        title=f"SWAN Output - {mesh_data.display_name}",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        xaxis=dict(
            range=[output.lons[0], output.lons[-1]],
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[output.lats[0], output.lats[-1]],
        ),
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.8)",
            font_size=12,
            font_family="monospace",
        ),
        width=900,
        height=700,
    )

    return fig


def create_comparison_figure(
    meshes: Dict[str, MeshData],
    spots: List[SurfSpot],
    ndbc_buoys: Optional[List[BuoyData]] = None,
    cdip_buoys: Optional[List[CDIPBuoyData]] = None,
) -> go.Figure:
    """Create side-by-side comparison figure with dropdown to select meshes."""
    mesh_list = list(meshes.values())

    if len(mesh_list) == 1:
        return create_single_mesh_figure(mesh_list[0], spots, ndbc_buoys, cdip_buoys)

    # Create figure with dropdown for mesh selection
    fig = go.Figure()

    # Add all meshes (only first visible initially)
    for i, mesh_data in enumerate(mesh_list):
        visible = (i == 0)

        # Heatmap
        heatmap = create_heatmap(mesh_data)
        heatmap.visible = visible
        fig.add_trace(heatmap)

        # Spot markers
        markers = create_spot_markers(spots, mesh_data)
        markers.visible = visible
        fig.add_trace(markers)

    # Add NDBC buoy markers (always visible, not tied to mesh selection)
    has_ndbc = ndbc_buoys is not None and len(ndbc_buoys) > 0
    if has_ndbc:
        ndbc_markers = create_ndbc_buoy_markers(ndbc_buoys)
        ndbc_markers.visible = True
        fig.add_trace(ndbc_markers)

    # Add CDIP buoy markers (always visible, not tied to mesh selection)
    has_cdip = cdip_buoys is not None and len(cdip_buoys) > 0
    if has_cdip:
        cdip_markers = create_cdip_buoy_markers(cdip_buoys)
        cdip_markers.visible = True
        fig.add_trace(cdip_markers)

    # Create dropdown buttons
    num_buoy_traces = (1 if has_ndbc else 0) + (1 if has_cdip else 0)
    buttons = []
    for i, mesh_data in enumerate(mesh_list):
        # Each mesh has 2 traces (heatmap + markers), plus buoy traces at the end
        num_mesh_traces = len(mesh_list) * 2
        visibility = [False] * num_mesh_traces
        visibility[i * 2] = True      # heatmap
        visibility[i * 2 + 1] = True  # markers

        # Add buoy visibility (always True)
        for _ in range(num_buoy_traces):
            visibility.append(True)

        buttons.append(dict(
            label=mesh_data.display_name,
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"SWAN Output - {mesh_data.display_name}"}
            ],
        ))

    # Use first mesh for initial extent
    first_output = mesh_list[0].swan_output

    fig.update_layout(
        title=f"SWAN Output - {mesh_list[0].display_name}",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        xaxis=dict(
            range=[first_output.lons[0], first_output.lons[-1]],
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[first_output.lats[0], first_output.lats[-1]],
        ),
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.8)",
            font_size=12,
            font_family="monospace",
        ),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.0,
            y=1.15,
            showactive=True,
            buttons=buttons,
            bgcolor="white",
            bordercolor="gray",
        )],
        autosize=True,
        margin=dict(l=60, r=60, t=80, b=60),
    )

    return fig


def create_side_by_side_figure(
    meshes: Dict[str, MeshData],
    spots: List[SurfSpot],
    mesh_names: Tuple[str, str] = None,
) -> go.Figure:
    """Create side-by-side comparison of two meshes."""
    mesh_list = list(meshes.values())

    if mesh_names:
        mesh_a = meshes.get(mesh_names[0])
        mesh_b = meshes.get(mesh_names[1])
    else:
        # Default to first two available
        mesh_a = mesh_list[0] if len(mesh_list) > 0 else None
        mesh_b = mesh_list[1] if len(mesh_list) > 1 else None

    if not mesh_a or not mesh_b:
        print("Need at least 2 meshes for side-by-side comparison")
        return create_comparison_figure(meshes, spots)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[mesh_a.display_name, mesh_b.display_name],
        horizontal_spacing=0.08,
    )

    # Left panel - mesh A
    fig.add_trace(create_heatmap(mesh_a), row=1, col=1)
    fig.add_trace(create_spot_markers(spots, mesh_a), row=1, col=1)

    # Right panel - mesh B
    heatmap_b = create_heatmap(mesh_b)
    heatmap_b.showscale = False  # Only show one colorbar
    fig.add_trace(heatmap_b, row=1, col=2)
    fig.add_trace(create_spot_markers(spots, mesh_b), row=1, col=2)

    fig.update_layout(
        title="SWAN Mesh Comparison",
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.8)",
            font_size=12,
            font_family="monospace",
        ),
        width=1400,
        height=700,
    )

    # Set axis ranges
    for col, mesh in [(1, mesh_a), (2, mesh_b)]:
        output = mesh.swan_output
        fig.update_xaxes(
            range=[output.lons[0], output.lons[-1]],
            title="Longitude" if col == 1 else "",
            row=1, col=col,
        )
        fig.update_yaxes(
            range=[output.lats[0], output.lats[-1]],
            title="Latitude" if col == 1 else "",
            scaleanchor=f"x{col}" if col > 1 else "x",
            scaleratio=1,
            row=1, col=col,
        )

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("INTERACTIVE SPOT EXPLORER")
    print("=" * 60)

    # Load data
    print("\nLoading mesh data...")
    meshes = load_mesh_data()

    if not meshes:
        print("No mesh data found! Run SWAN simulations first.")
        return

    print(f"\nLoaded {len(meshes)} meshes: {list(meshes.keys())}")

    # Load NDBC buoy data
    print("\nFetching NDBC buoy spectral data with r1 confidence...")
    ndbc_buoys = load_ndbc_buoy_data()
    print(f"Loaded {len(ndbc_buoys)} NDBC buoys with spectral partitioning")

    for buoy in ndbc_buoys:
        if buoy.error:
            print(f"  {buoy.station_id} ({buoy.name}): Error - {buoy.error}")
        else:
            n_partitions = len(buoy.partitions)
            combined_str = f"{buoy.combined_height_m:.2f}m" if buoy.combined_height_m else "--"
            # Show confidence of primary partition
            conf_str = ""
            if buoy.partitions:
                p = buoy.partitions[0]
                conf_str = f", r1={p.r1:.2f} ({p.confidence})" if p.r1 else ""
            print(f"  {buoy.station_id} ({buoy.name}): {n_partitions} partitions, combined {combined_str}{conf_str}")

    # Load CDIP buoy data
    print("\nFetching CDIP buoy spectral data (MEM analysis)...")
    cdip_buoys = load_cdip_buoy_data()
    print(f"Loaded {len(cdip_buoys)} CDIP buoys with spectral partitioning")

    for buoy in cdip_buoys:
        if buoy.error:
            print(f"  CDIP {buoy.station_id} ({buoy.name}): Error - {buoy.error}")
        else:
            n_partitions = len(buoy.partitions)
            combined_str = f"{buoy.combined_height_m:.2f}m" if buoy.combined_height_m else "--"
            depth_str = f", depth={buoy.depth_m}m" if buoy.depth_m else ""
            print(f"  CDIP {buoy.station_id} ({buoy.name}): {n_partitions} partitions, combined {combined_str}{depth_str}")

    # Create visualization
    print("\nGenerating interactive visualization...")

    # Single panel with dropdown to switch between meshes
    fig = create_comparison_figure(meshes, SOCAL_SPOTS, ndbc_buoys, cdip_buoys)

    # Save and open - full page layout
    output_path = project_root / "scripts" / "dev" / "spot_explorer.html"
    fig.write_html(
        str(output_path),
        full_html=True,
        include_plotlyjs=True,
        config={"responsive": True},
        default_width="100%",
        default_height="100vh",
    )
    print(f"\nSaved to: {output_path}")

    # Open in browser
    print("Opening in browser...")
    webbrowser.open(f"file://{output_path}")


if __name__ == "__main__":
    main()
