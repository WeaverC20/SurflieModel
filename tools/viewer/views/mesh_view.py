"""
Surfzone mesh view using Datashader + HoloViews.

Ported from scripts/dev/view_surfzone_mesh_ds.py. Shows ocean points
colored by depth and land points colored by elevation, with a coastline
overlay and click-to-inspect via PointInspector.

Click inspector shows SWAN swell partition data (Hs, Tp, Dir) at each
mesh point via nearest-neighbor lookup on the SWAN grid.
"""

import logging

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
    DEFAULT_DEPTH_MAX, DEFAULT_LAND_MAX, PARTITION_COLORS,
)
from tools.viewer.components.colorbar import create_matplotlib_colorbar
from tools.viewer.components.point_inspector import PointInspector

logger = logging.getLogger(__name__)


def _swan_nearest_indices(lons_query, lats_query, swan_lons, swan_lats):
    """Vectorized nearest-neighbor lookup on a regular SWAN grid.

    Returns (lon_indices, lat_indices) arrays matching the input query arrays.
    Points outside the SWAN grid extent are clipped to the boundary.
    """
    lon_idx = np.searchsorted(swan_lons, lons_query).clip(0, len(swan_lons) - 1)
    lat_idx = np.searchsorted(swan_lats, lats_query).clip(0, len(swan_lats) - 1)

    # searchsorted finds insertion point; check if left neighbor is closer
    lon_left = np.maximum(lon_idx - 1, 0)
    closer_left_lon = (
        np.abs(swan_lons[lon_left] - lons_query)
        < np.abs(swan_lons[lon_idx] - lons_query)
    )
    lon_idx = np.where(closer_left_lon, lon_left, lon_idx)

    lat_left = np.maximum(lat_idx - 1, 0)
    closer_left_lat = (
        np.abs(swan_lats[lat_left] - lats_query)
        < np.abs(swan_lats[lat_idx] - lats_query)
    )
    lat_idx = np.where(closer_left_lat, lat_left, lat_idx)

    return lon_idx, lat_idx


class MeshView(BaseView):
    """Surfzone mesh view with Datashader rendering."""

    def __init__(self, data_manager, **params):
        super().__init__(data_manager, **params)
        self._inspector_pane = pn.pane.HTML("", width=240, sizing_mode='stretch_height')
        self._colorbar_col = pn.Column(width=120)

    def update(self, region: str, **kwargs):
        """Reload mesh data and rebuild the Datashader plot."""
        self.region = region
        use_lonlat = kwargs.get('use_lonlat', False)

        # Capture own ranges if no cross-view ranges were set,
        # but only when region and coordinate system are unchanged.
        if self._pending_ranges is None:
            current = self.get_ranges()
            if (current is not None
                    and getattr(self, '_prev_region', None) == region
                    and getattr(self, '_prev_use_lonlat', None) == use_lonlat):
                self._pending_ranges = current
        self._prev_region = region
        self._prev_use_lonlat = use_lonlat

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

        # Click marker — handled at the Bokeh level via a hook so that
        # tapping never triggers a HoloViews/datashader range recalculation.
        tap = streams.SingleTap(x=None, y=None)

        def _add_tap_handler(plot_obj, element):
            from bokeh.events import Tap as BokehTap
            from bokeh.models import ColumnDataSource
            from bokeh.models.callbacks import CustomJS

            fig = plot_obj.state
            self._bokeh_fig = fig

            if self._pending_ranges is not None:
                fig.x_range.start = self._pending_ranges['x_start']
                fig.x_range.end = self._pending_ranges['x_end']
                fig.y_range.start = self._pending_ranges['y_start']
                fig.y_range.end = self._pending_ranges['y_end']
                self._pending_ranges = None

            marker_src = ColumnDataSource(data={'x': [], 'y': []})
            fig.scatter(
                'x', 'y', source=marker_src, size=15,
                fill_color='red', line_color='white', line_width=2,
                level='overlay',
            )
            fig.js_on_event('tap', CustomJS(
                args=dict(source=marker_src),
                code="source.data = {'x': [cb_obj.x], 'y': [cb_obj.y]};",
            ))
            fig.on_event(BokehTap, lambda event: tap.event(x=event.x, y=event.y))

        plot = plot.opts(
            width=1000,
            height=700,
            xlabel=x_label,
            ylabel=y_label,
            tools=['wheel_zoom', 'pan', 'reset', 'box_zoom'],
            active_tools=['wheel_zoom', 'pan'],
            bgcolor=DARK_BG,
            data_aspect=1,
            hooks=[_add_tap_handler],
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

        # Load SWAN data and pre-compute partition values at each mesh point
        swan = None
        swan_labels = []
        resolutions = self.data_manager.available_swan_resolutions(region)
        if resolutions:
            try:
                res = resolutions[0]  # Best available (typically 'coarse')
                swan = self.data_manager.get_swan_output(region, res)

                lon_idx, lat_idx = _swan_nearest_indices(
                    all_lon, all_lat, swan.lons, swan.lats,
                )

                # Combined values
                hsig_m, tps_m, dir_m = swan.mask_land()
                inspector_df['swan_hs'] = hsig_m[lat_idx, lon_idx]
                inspector_df['swan_tp'] = tps_m[lat_idx, lon_idx]
                inspector_df['swan_dir'] = dir_m[lat_idx, lon_idx]

                # Per-partition values
                for p in swan.partitions:
                    hs_m, tp_m, d_m = p.mask_invalid(swan.exception_value)
                    tag = p.label.lower().replace(' ', '_')
                    swan_labels.append((tag, p.label))
                    inspector_df[f'swan_{tag}_hs'] = hs_m[lat_idx, lon_idx]
                    inspector_df[f'swan_{tag}_tp'] = tp_m[lat_idx, lon_idx]
                    inspector_df[f'swan_{tag}_dir'] = d_m[lat_idx, lon_idx]

                logger.info(
                    f"Mapped SWAN ({res}) to {n_points:,} mesh points "
                    f"({len(swan.partitions)} partitions)"
                )
            except Exception as e:
                logger.warning(f"Could not load SWAN data for mesh inspector: {e}")
                swan = None

        # Capture for closure
        has_swan = swan is not None
        partition_labels = swan_labels
        swan_run_ts = swan.run_timestamp if swan else None

        def format_mesh_point(row, idx):
            elev = row['elevation']
            kind = "Ocean" if elev < 0 else "Land"
            html = f"""
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
            """

            # SWAN partition data (only for ocean points with valid data)
            if has_swan and elev < 0:
                hs = row.get('swan_hs', np.nan)
                if not np.isnan(hs):
                    tp = row.get('swan_tp', np.nan)
                    dr = row.get('swan_dir', np.nan)
                    html += f"""
                <br>
                <hr style="border-color: #444;">
                <b>SWAN Combined</b><br>
                Hs: {hs:.2f} m<br>
                Tp: {tp:.1f} s<br>
                Dir: {dr:.0f}&deg;<br>
                    """

                    if partition_labels:
                        html += "<br><b>SWAN Partitions</b><br>"
                        for tag, label in partition_labels:
                            p_hs = row.get(f'swan_{tag}_hs', np.nan)
                            if np.isnan(p_hs) or p_hs <= 0:
                                continue
                            p_tp = row.get(f'swan_{tag}_tp', np.nan)
                            p_dir = row.get(f'swan_{tag}_dir', np.nan)
                            color = PARTITION_COLORS.get(
                                tag, PARTITION_COLORS.get(tag.replace(' ', '_'), 'white')
                            )
                            html += (
                                f"<span style='color: {color}'>{label}</span>: "
                                f"{p_hs:.2f}m, {p_tp:.1f}s, {p_dir:.0f}&deg;<br>"
                            )

                    if swan_run_ts:
                        html += (
                            f"<br><span style='color: #888; font-size: 10px;'>"
                            f"SWAN run: {swan_run_ts.strftime('%Y-%m-%d %H:%M')} UTC"
                            f"</span><br>"
                        )

            html += "</div>"
            return html

        inspector = PointInspector(tap, coords, inspector_df, format_fn=format_mesh_point)
        self._inspector_pane = inspector.panel()

        # Colorbars
        ocean_cb = create_matplotlib_colorbar(0, depth_max, 'Ocean Depth (m)', OCEAN_CMAP, height=300)
        land_cb = create_matplotlib_colorbar(0, land_max, 'Land Elevation (m)', LAND_CMAP, height=150)
        self._colorbar_col = pn.Column(
            pn.pane.HTML(ocean_cb, width=100, height=350),
            pn.Spacer(height=20),
            pn.pane.HTML(land_cb, width=100, height=200),
            width=120,
        )

        # Summary stats
        depth_vals = -elevation[ocean_mask] if n_ocean > 0 else np.array([0])
        swan_summary = ""
        if has_swan:
            n_parts = len(partition_labels)
            swan_summary = f"""
            <br>
            <b>SWAN Data</b><br>
            Resolution: {res}<br>
            Partitions: {n_parts}<br>
            """
            if swan_run_ts:
                swan_summary += f"Run: {swan_run_ts.strftime('%Y-%m-%d %H:%M')} UTC<br>"
            swan_summary += "<i>Click ocean points to see swell partitions</i><br>"
        else:
            swan_summary = "<br><i>No SWAN data available for this region</i><br>"

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
            {swan_summary}
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
                width=260,
            ),
        )
