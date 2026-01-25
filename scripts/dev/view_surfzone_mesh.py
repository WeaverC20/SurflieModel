#!/usr/bin/env python3
"""
Interactive Surf Zone Mesh Viewer

View surf zone meshes with interactive zooming and panning.
Uses Plotly for browser-based visualization.

Usage:
    python scripts/dev/view_surfzone_mesh.py socal
    python scripts/dev/view_surfzone_mesh.py socal --show-coastline
    python scripts/dev/view_surfzone_mesh.py socal --show-triangulation
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_mesh(region_name: str):
    """Load a surf zone mesh."""
    from data.surfzone.mesh import SurfZoneMesh

    mesh_dir = project_root / "data" / "surfzone" / "meshes" / region_name
    if not mesh_dir.exists():
        print(f"Error: Mesh not found at {mesh_dir}")
        print(f"Generate it first with: python scripts/generate_surfzone_mesh.py {region_name}")
        sys.exit(1)

    return SurfZoneMesh.load(mesh_dir)


def view_mesh(
    region_name: str,
    depth_max: float = 30.0,
    land_max: float = 10.0,
    show_coastline: bool = False,
    show_triangulation: bool = False,
    use_lonlat: bool = False,
    max_points: int = None,
):
    """Create interactive mesh viewer for point cloud data."""

    print(f"Loading mesh for {region_name}...")
    mesh = load_mesh(region_name)
    print(mesh.summary())
    print()

    # Get point cloud data
    x = mesh.points_x
    y = mesh.points_y
    elevation = mesh.elevation

    n_points = len(x)

    # Downsample if needed
    if max_points and n_points > max_points:
        print(f"Downsampling from {n_points:,} to {max_points:,} points...")
        indices = np.random.choice(n_points, max_points, replace=False)
        indices.sort()  # Keep spatial ordering
        x = x[indices]
        y = y[indices]
        elevation = elevation[indices]
        n_points = max_points

    print(f"Plotting {n_points:,} points...")

    # Convert to lon/lat if requested
    if use_lonlat:
        x, y = mesh.utm_to_lon_lat(x, y)
        x_label = "Longitude (°)"
        y_label = "Latitude (°)"
        hover_template = 'Lon: %{x:.4f}°<br>Lat: %{y:.4f}°<br>Elevation: %{customdata:.1f}m<extra></extra>'
    else:
        x_label = "UTM Easting (m)"
        y_label = "UTM Northing (m)"
        hover_template = 'X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Elevation: %{customdata:.1f}m<extra></extra>'

    # Separate ocean and land
    ocean_mask = elevation < 0
    land_mask = elevation >= 0

    print(f"  Ocean points: {np.sum(ocean_mask):,}")
    print(f"  Land points: {np.sum(land_mask):,}")

    # Create figure
    fig = go.Figure()

    # Plot ocean points (depth = -elevation, so positive depth)
    if np.any(ocean_mask):
        ocean_depth = -elevation[ocean_mask]  # Convert to positive depth
        fig.add_trace(
            go.Scattergl(
                x=x[ocean_mask],
                y=y[ocean_mask],
                mode='markers',
                marker=dict(
                    size=4,
                    color=ocean_depth,
                    colorscale='Blues',
                    cmin=0,
                    cmax=depth_max,
                    colorbar=dict(
                        title='Depth (m)',
                        x=1.0,
                        len=0.45,
                        y=0.75,
                    ),
                ),
                customdata=elevation[ocean_mask],
                hovertemplate=hover_template,
                name='Ocean',
            )
        )

    # Plot land points
    if np.any(land_mask):
        land_height = elevation[land_mask]
        fig.add_trace(
            go.Scattergl(
                x=x[land_mask],
                y=y[land_mask],
                mode='markers',
                marker=dict(
                    size=4,
                    color=land_height,
                    colorscale='YlGn',
                    cmin=0,
                    cmax=land_max,
                    colorbar=dict(
                        title='Land Height (m)',
                        x=1.0,
                        len=0.45,
                        y=0.25,
                    ),
                ),
                customdata=elevation[land_mask],
                hovertemplate=hover_template,
                name='Land',
            )
        )

    # Add coastline points if requested
    if show_coastline and mesh.coastline_x is not None:
        coast_x = mesh.coastline_x
        coast_y = mesh.coastline_y
        if use_lonlat:
            coast_x, coast_y = mesh.utm_to_lon_lat(coast_x, coast_y)

        fig.add_trace(
            go.Scattergl(
                x=coast_x,
                y=coast_y,
                mode='markers',
                marker=dict(size=3, color='red', symbol='x'),
                name='Coastline',
                hovertemplate='Coastline<extra></extra>',
            )
        )

        # Also show normals as lines if we have them
        if mesh.coastline_nx is not None and not use_lonlat:
            # Show normals every 10th point to avoid clutter
            step = max(1, len(coast_x) // 100)
            normal_scale = 200  # Length of normal arrows in meters

            for i in range(0, len(mesh.coastline_x), step):
                cx, cy = mesh.coastline_x[i], mesh.coastline_y[i]
                nx, ny = mesh.coastline_nx[i], mesh.coastline_ny[i]

                fig.add_trace(
                    go.Scattergl(
                        x=[cx, cx + nx * normal_scale],
                        y=[cy, cy + ny * normal_scale],
                        mode='lines',
                        line=dict(color='red', width=1),
                        showlegend=False,
                        hoverinfo='skip',
                    )
                )

    # Add triangulation edges if requested
    if show_triangulation:
        mesh._build_interpolator()
        if mesh._triangulation is not None:
            tri = mesh._triangulation
            # Get edges from triangulation
            edges_x = []
            edges_y = []

            # Sample subset of triangles to avoid overwhelming the plot
            n_triangles = len(tri.simplices)
            step = max(1, n_triangles // 5000)

            pts_x = mesh.points_x if not use_lonlat else x
            pts_y = mesh.points_y if not use_lonlat else y

            for i in range(0, n_triangles, step):
                simplex = tri.simplices[i]
                for j in range(3):
                    p1, p2 = simplex[j], simplex[(j + 1) % 3]
                    edges_x.extend([pts_x[p1], pts_x[p2], None])
                    edges_y.extend([pts_y[p1], pts_y[p2], None])

            fig.add_trace(
                go.Scattergl(
                    x=edges_x,
                    y=edges_y,
                    mode='lines',
                    line=dict(color='gray', width=0.5),
                    name='Triangulation',
                    hoverinfo='skip',
                )
            )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Surf Zone Mesh: {region_name.upper()}<br>'
                 f'<sub>Points: {n_points:,} | '
                 f'Ocean: {np.sum(ocean_mask):,} | Land: {np.sum(land_mask):,}</sub>',
            x=0.5,
        ),
        xaxis_title=x_label,
        yaxis_title=y_label,
        autosize=True,
        yaxis=dict(scaleanchor='x', scaleratio=1),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        # Enable better zooming
        dragmode='zoom',
    )

    # Show in browser with full screen config
    print("Opening in browser...")
    fig.show(config={
        'scrollZoom': True,  # Enable scroll wheel zoom
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['drawrect', 'eraseshape'],
    })

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Interactive surf zone mesh viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        'region',
        help="Region name (e.g., 'socal')"
    )

    parser.add_argument(
        '--depth-max',
        type=float,
        default=30.0,
        help="Maximum depth for color scale (default: 30)"
    )

    parser.add_argument(
        '--land-max',
        type=float,
        default=10.0,
        help="Maximum land elevation for color scale (default: 10)"
    )

    parser.add_argument(
        '--show-coastline',
        action='store_true',
        help="Show detected coastline points and normals"
    )

    parser.add_argument(
        '--show-triangulation',
        action='store_true',
        help="Show Delaunay triangulation edges"
    )

    parser.add_argument(
        '--lonlat',
        action='store_true',
        help="Use longitude/latitude coordinates instead of UTM"
    )

    parser.add_argument(
        '--max-points',
        type=int,
        default=100000,
        help="Maximum points to display (default: 100000, use 0 for all)"
    )

    args = parser.parse_args()

    max_pts = args.max_points if args.max_points > 0 else None

    view_mesh(
        args.region,
        depth_max=args.depth_max,
        land_max=args.land_max,
        show_coastline=args.show_coastline,
        show_triangulation=args.show_triangulation,
        use_lonlat=args.lonlat,
        max_points=max_pts,
    )


if __name__ == "__main__":
    main()
