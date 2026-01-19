#!/usr/bin/env python3
"""
Interactive Spot Explorer

Plotly-based visualization for comparing SWAN outputs across different meshes.
Hover over any grid point to see partition details.

Run from project root:
    python scripts/dev/explore_spots.py

Opens an HTML file in your browser with interactive visualization.
"""

import sys
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.spots import SOCAL_SPOTS, SurfSpot
from data.swan.analysis.output_reader import SwanOutput, SwanOutputReader


# =============================================================================
# Data Structures (extensible for future stats)
# =============================================================================

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
    # energy_percentages: Optional[List[float]] = None

    def format_hover_text(self) -> str:
        """Generate HTML hover text for this point."""
        lines = [
            f"<b>Location:</b> ({self.lat:.4f}, {self.lon:.4f})",
            f"<b>Grid:</b> i={self.grid_i}, j={self.grid_j}",
            "",
            f"<b>Total Hsig:</b> {self.hsig:.2f} m",
            f"<b>Peak Period:</b> {self.tps:.1f} s",
            f"<b>Direction:</b> {self.dir:.0f}°",
            "",
            "<b>Partitions:</b>",
        ]

        for hs, tp, direction, label in self.partitions:
            if np.isnan(hs) or hs <= 0:
                lines.append(f"  {label}: --")
            else:
                lines.append(f"  {label}: {hs:.2f}m, {tp:.1f}s, {direction:.0f}°")

        # === Add future stats here ===
        # if self.breaking_height is not None:
        #     lines.append(f"<b>Breaking Ht:</b> {self.breaking_height:.2f} m")

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
            f"Grid point: ({forecast.grid_i}, {forecast.grid_j})",
            f"Distance: {forecast.distance_km:.2f} km",
            "",
        ]
        for hs, tp, direction, label in forecast.partitions:
            if np.isnan(hs) or hs <= 0:
                lines.append(f"<b>{label}:</b> --")
            else:
                lines.append(f"<b>{label}:</b> {hs:.2f}m, {tp:.1f}s, {direction:.0f}°")
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


def create_single_mesh_figure(mesh_data: MeshData, spots: List[SurfSpot]) -> go.Figure:
    """Create figure for a single mesh."""
    fig = go.Figure()

    # Add heatmap
    fig.add_trace(create_heatmap(mesh_data))

    # Add spot markers
    fig.add_trace(create_spot_markers(spots, mesh_data))

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
    spots: List[SurfSpot]
) -> go.Figure:
    """Create side-by-side comparison figure with dropdown to select meshes."""
    mesh_list = list(meshes.values())

    if len(mesh_list) == 1:
        return create_single_mesh_figure(mesh_list[0], spots)

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

    # Create dropdown buttons
    buttons = []
    for i, mesh_data in enumerate(mesh_list):
        # Each mesh has 2 traces (heatmap + markers)
        visibility = [False] * (len(mesh_list) * 2)
        visibility[i * 2] = True      # heatmap
        visibility[i * 2 + 1] = True  # markers

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

    # Create visualization
    print("\nGenerating interactive visualization...")

    # Single panel with dropdown to switch between meshes
    fig = create_comparison_figure(meshes, SOCAL_SPOTS)

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
