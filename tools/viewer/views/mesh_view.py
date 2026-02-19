"""
Surfzone mesh view using Datashader + HoloViews.

Ported from scripts/dev/view_surfzone_mesh_ds.py. Shows ocean points
colored by depth and land points colored by elevation, with a coastline
overlay and click-to-inspect via PointInspector.
"""

import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import streams
from holoviews.operation.datashader import datashade, spread
import datashader as ds
import panel as pn

from tools.viewer.views.base import BaseView
from tools.viewer.config import (
    OCEAN_CMAP, LAND_CMAP, COASTLINE_COLOR, DARK_BG, SIDEBAR_BG,
    DEFAULT_DEPTH_MAX, DEFAULT_LAND_MAX,
)
from tools.viewer.components.colorbar import create_matplotlib_colorbar
from tools.viewer.components.point_inspector import PointInspector


class MeshView(BaseView):
    """Surfzone mesh view with Datashader rendering."""

    def __init__(self, data_manager, **params):
        super().__init__(data_manager, **params)
        self._inspector_pane = pn.pane.HTML("", width=280, sizing_mode='stretch_height')
        self._colorbar_col = pn.Column(width=140)

    def update(self, region: str, **kwargs):
        """Reload mesh data and rebuild the Datashader plot."""
        self.region = region
        use_lonlat = kwargs.get('use_lonlat', False)

        try:
            mesh = self.data_manager.get_mesh(region)
        except Exception as e:
            self._plot_pane.object = None
            self._summary_html.object = (
                f"<div style='color: #ff8888; padding: 10px;'>Error: {e}</div>"
            )
            return

        x = mesh.points_x.copy()
        y = mesh.points_y.copy()
        elevation = mesh.elevation.copy()
        n_points = len(x)

        if use_lonlat:
            x, y = mesh.utm_to_lon_lat(x, y)
            x_label = "Longitude"
            y_label = "Latitude"
        else:
            x_label = "UTM Easting (m)"
            y_label = "UTM Northing (m)"

        # Separate ocean and land
        ocean_mask = elevation < 0
        land_mask = elevation >= 0
        n_ocean = int(np.sum(ocean_mask))
        n_land = int(np.sum(land_mask))

        depth_max = DEFAULT_DEPTH_MAX
        land_max = DEFAULT_LAND_MAX

        ocean_df = pd.DataFrame({
            'x': x[ocean_mask],
            'y': y[ocean_mask],
            'depth': np.clip(-elevation[ocean_mask], 0, depth_max),
        })
        land_df = pd.DataFrame({
            'x': x[land_mask],
            'y': y[land_mask],
            'height': np.clip(elevation[land_mask], 0, land_max),
        })

        ocean_points = hv.Points(ocean_df, kdims=['x', 'y'], vdims=['depth'])
        land_points = hv.Points(land_df, kdims=['x', 'y'], vdims=['height'])

        # Datashade pipeline (exact pattern from mesh_ds.py)
        ocean_shaded = spread(
            datashade(
                ocean_points,
                aggregator=ds.mean('depth'),
                cmap=OCEAN_CMAP,
                cnorm='linear',
            ),
            px=3,
        )
        land_shaded = spread(
            datashade(
                land_points,
                aggregator=ds.mean('height'),
                cmap=LAND_CMAP,
                cnorm='linear',
            ),
            px=3,
        )

        plot = ocean_shaded * land_shaded

        # Coastline overlay
        if mesh.coastlines:
            coastline_paths = []
            for coastline in mesh.coastlines:
                if use_lonlat:
                    cl_x, cl_y = mesh.utm_to_lon_lat(coastline[:, 0], coastline[:, 1])
                else:
                    cl_x, cl_y = coastline[:, 0], coastline[:, 1]
                coastline_paths.append(np.column_stack([cl_x, cl_y]))

            coastline_overlay = hv.Path(coastline_paths).opts(
                color=COASTLINE_COLOR,
                line_width=2,
            )
            plot = plot * coastline_overlay

        # Click marker via SingleTap
        tap = streams.SingleTap(x=None, y=None)

        def click_marker(x, y):
            if x is None or y is None:
                return hv.Points([]).opts(size=0)
            return hv.Points([(x, y)]).opts(
                size=15, color='red', marker='circle',
                line_color='white', line_width=2,
            )

        marker_dmap = hv.DynamicMap(click_marker, streams=[tap])
        plot = plot * marker_dmap

        plot = plot.opts(
            width=1200,
            height=800,
            xlabel=x_label,
            ylabel=y_label,
            tools=['wheel_zoom', 'pan', 'reset', 'box_zoom', 'tap'],
            active_tools=['wheel_zoom', 'pan'],
            bgcolor=DARK_BG,
            data_aspect=1,
        )

        self._plot_pane.object = plot

        # Build PointInspector data
        coords = np.column_stack([x, y])

        # Build a DataFrame for the inspector showing useful mesh info
        # Need lon/lat for all points regardless of display mode
        if use_lonlat:
            all_lon, all_lat = x, y
            all_utm_x, all_utm_y = mesh.points_x, mesh.points_y
        else:
            all_utm_x, all_utm_y = x, y
            all_lon, all_lat = mesh.utm_to_lon_lat(mesh.points_x, mesh.points_y)

        coast_dist = np.zeros(n_points)
        if hasattr(mesh, 'coast_distance') and mesh.coast_distance is not None:
            coast_dist = mesh.coast_distance

        inspector_df = pd.DataFrame({
            'elevation': elevation,
            'depth': np.where(elevation < 0, -elevation, 0.0),
            'coast_distance': coast_dist,
            'utm_x': all_utm_x,
            'utm_y': all_utm_y,
            'lon': all_lon,
            'lat': all_lat,
        })

        def format_mesh_point(row, idx):
            elev = row['elevation']
            kind = "Ocean" if elev < 0 else "Land"
            return f"""
            <div style="color: white; font-size: 11px; padding: 10px;
                        background: {SIDEBAR_BG}; border-radius: 5px;">
                <b style="font-size: 13px;">Point #{idx:,} ({kind})</b><br>
                <hr style="border-color: #444;">
                <b>Location</b><br>
                Lat: {row['lat']:.5f}<br>
                Lon: {row['lon']:.5f}<br>
                UTM X: {row['utm_x']:,.0f} m<br>
                UTM Y: {row['utm_y']:,.0f} m<br>
                <br>
                <b>Properties</b><br>
                Elevation: {elev:.2f} m<br>
                Depth: {row['depth']:.2f} m<br>
                Coast distance: {row['coast_distance']:.1f} m<br>
            </div>
            """

        inspector = PointInspector(tap, coords, inspector_df, format_fn=format_mesh_point)
        self._inspector_pane = inspector.panel()

        # Colorbars
        ocean_cb = create_matplotlib_colorbar(0, depth_max, 'Ocean Depth (m)', OCEAN_CMAP, height=300)
        land_cb = create_matplotlib_colorbar(0, land_max, 'Land Elevation (m)', LAND_CMAP, height=150)
        self._colorbar_col = pn.Column(
            pn.pane.HTML(ocean_cb, width=120, height=350),
            pn.Spacer(height=20),
            pn.pane.HTML(land_cb, width=120, height=200),
            width=140,
        )

        # Summary stats
        depth_vals = -elevation[ocean_mask] if n_ocean > 0 else np.array([0])
        self._summary_html.object = f"""
        <div style="color: white; font-size: 11px; padding: 10px;
                    background: {SIDEBAR_BG}; border-radius: 5px;">
            <b>Mesh Summary</b><br><br>
            Region: {region}<br>
            Total points: {n_points:,}<br>
            Ocean points: {n_ocean:,}<br>
            Land points: {n_land:,}<br>
            Coastlines: {len(mesh.coastlines) if mesh.coastlines else 0}<br>
            <br>
            <b>Depth range</b><br>
            Min: {np.min(depth_vals):.1f} m<br>
            Max: {np.max(depth_vals):.1f} m<br>
            Mean: {np.mean(depth_vals):.1f} m<br>
        </div>
        """

    def panel(self):
        """Return the Panel layout."""
        return pn.Row(
            self._plot_pane,
            self._colorbar_col,
            pn.Column(
                pn.pane.Markdown("### Point Inspector"),
                self._inspector_pane,
                pn.Spacer(height=10),
                self._summary_html,
                width=300,
            ),
        )
