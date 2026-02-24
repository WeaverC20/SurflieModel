"""
Surfzone results view using Datashader + HoloViews.

Displays forward ray tracing results: gray uncovered points overlaid
with viridis-colored wave heights. Uses PointInspector for click-to-
inspect with per-partition details. Optional ray path overlay.

Ported from scripts/dev/view_surfzone_result.py (converted from Plotly
Scattergl to Datashader) with click patterns from view_statistics.py.
"""

import math
import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import streams
from holoviews.operation.datashader import datashade, spread
import datashader as ds
import panel as pn

from scipy.spatial import cKDTree

from tools.viewer.views.base import BaseView
from tools.viewer.config import (
    WAVE_CMAP, STATS_CMAP, NO_WAVE_COLOR, COASTLINE_COLOR, DARK_BG, SIDEBAR_BG,
    PARTITION_COLORS, PARTITION_LABELS,
    SPOT_BBOX_COLOR, SPOT_BBOX_DASH, SPOT_BBOX_WIDTH,
    SPOT_EDIT_COLOR, SPOT_EDIT_FILL_ALPHA, SPOT_DIMMED_COLOR, SPOT_DIMMED_WIDTH,
    STAT_LABELS,
)
from tools.viewer.components.colorbar import create_matplotlib_colorbar
from tools.viewer.components.point_inspector import PointInspector


class ResultView(BaseView):
    """Surfzone simulation result view with Datashader rendering."""

    def __init__(self, data_manager, **params):
        super().__init__(data_manager, **params)
        # Persistent tap stream and inspector — created once so the Panel layout
        # always holds the same pane object regardless of how many times
        # _rebuild_plot() is called (which happens on region/use_lonlat changes
        # without panel() being re-invoked).
        self._tap = streams.SingleTap(x=None, y=None)
        _dummy_coords = np.zeros((1, 2))
        _dummy_df = pd.DataFrame({
            'H_at_mesh': [0], 'energy': [0], 'ray_count': [0], 'depth': [0],
            'utm_x': [0], 'utm_y': [0], 'lon': [0], 'lat': [0],
        })
        self._inspector = PointInspector(self._tap, _dummy_coords, _dummy_df)
        self._inspector_pane = self._inspector.panel()
        self._colorbar_pane = pn.pane.HTML("", height=60, sizing_mode='stretch_width')
        # Variable selector for heatmap display
        self._variable_select = pn.widgets.Select(
            name='Display Variable',
            options=['Wave Height (Hs)'],
            value='Wave Height (Hs)',
            width=220,
        )
        self._variable_select.param.watch(self._on_variable_change, 'value')
        self._stats_column_map = {}  # display label → CSV column name
        self._show_rays = pn.widgets.Checkbox(name='Show ray paths', value=False)
        self._n_rays = pn.widgets.IntSlider(
            name='Rays to show', start=10, end=1000, step=10, value=100,
        )
        self._show_rays.param.watch(self._on_ray_toggle, 'value')
        self._n_rays.param.watch(self._on_ray_toggle, 'value')
        # Spot selector
        self._spot_selector = pn.widgets.Select(
            name='Surf Spot', options=['(none)'], value='(none)', width=220,
        )
        self._spot_stats_html = pn.pane.HTML("", width=240)
        self._spot_selector.param.watch(self._on_spot_change, 'value')
        self._spots_list = []  # Populated in update()
        self._selected_spot = None
        # Spot edit mode
        self._edit_mode = pn.widgets.Toggle(
            name='Edit Spots', value=False, button_type='warning', width=220,
        )
        self._save_spots_btn = pn.widgets.Button(
            name='Save Changes', button_type='success', width=220, visible=False,
        )
        self._edit_status_html = pn.pane.HTML("", width=240)
        self._box_stream = None
        self._has_unsaved_changes = False
        self._edit_mode.param.watch(self._on_edit_toggle, 'value')
        self._save_spots_btn.on_click(self._on_save_click)
        # Store state for re-renders
        self._current_use_lonlat = False

    def update(self, region: str, **kwargs):
        """Reload result data and rebuild the plot."""
        # Auto-save and exit edit mode on region change
        if self._edit_mode.value:
            if self._has_unsaved_changes and self._selected_spot is not None:
                self._save_edited_spot()
                self._do_save_to_json()
            self._edit_mode.value = False

        self.region = region
        self._current_use_lonlat = kwargs.get('use_lonlat', False)

        # Capture own ranges if no cross-view ranges were set,
        # but only when region and coordinate system are unchanged.
        if self._pending_ranges is None:
            current = self.get_ranges()
            if (current is not None
                    and getattr(self, '_prev_region', None) == region
                    and getattr(self, '_prev_use_lonlat', None) == self._current_use_lonlat):
                self._pending_ranges = current
        self._prev_region = region
        self._prev_use_lonlat = self._current_use_lonlat

        # Load spots for this region
        spots = self.data_manager.get_spots(region)
        self._spots_list = spots
        if spots:
            self._spot_selector.options = ['(none)'] + [s.display_name for s in spots]
            self._spot_selector.value = '(none)'
        else:
            self._spot_selector.options = ['(none)']
        self._selected_spot = None
        self._spot_stats_html.object = ""

        # Populate variable selector from available stats columns
        stats_df = self.data_manager.get_statistics(region)
        if stats_df is not None:
            col_map = {label: col for col, label in STAT_LABELS.items()
                       if col in stats_df.columns}
            self._stats_column_map = col_map
            options = ['Wave Height (Hs)'] + list(col_map.keys())
        else:
            self._stats_column_map = {}
            options = ['Wave Height (Hs)']
        current = self._variable_select.value
        self._variable_select.options = options
        if current not in options:
            self._variable_select.value = 'Wave Height (Hs)'

        self._rebuild_plot()

    def _on_ray_toggle(self, event):
        """Re-render when ray visibility changes."""
        self._rebuild_plot()

    def _on_variable_change(self, event):
        """Re-render when display variable changes."""
        self._rebuild_plot()

    def _on_spot_change(self, event):
        """Update stats panel and rebuild plot when spot selection changes."""
        # Auto-save if switching spots during edit mode with unsaved changes
        if self._edit_mode.value and self._has_unsaved_changes and self._selected_spot is not None:
            self._save_edited_spot()
            self._do_save_to_json()

        if event.new == '(none)':
            self._selected_spot = None
            self._spot_stats_html.object = ""
        else:
            # Find the spot by display_name
            spot = next((s for s in self._spots_list if s.display_name == event.new), None)
            self._selected_spot = spot
            if spot:
                self._update_spot_stats(spot)
        self._rebuild_plot()

    def _update_spot_stats(self, spot):
        """Compute and display aggregated statistics for a spot."""
        aggregator = self.data_manager.get_spot_aggregator(self.region)
        if aggregator is None:
            self._spot_stats_html.object = (
                f"<div style='color: #ff8888; padding: 10px; font-size: 11px;'>"
                f"Statistics not available. Run:<br>"
                f"<code>python data/surfzone/statistics/run_statistics.py --region {self.region}</code>"
                f"</div>"
            )
            return

        summary = aggregator.aggregate(spot)
        self._spot_stats_html.object = self._format_spot_stats_html(summary)

    def _on_edit_toggle(self, event):
        """Enter or exit spot edit mode."""
        if event.new and self._selected_spot is None:
            self._edit_mode.value = False
            self._edit_status_html.object = (
                "<div style='color: #ff8888; padding: 5px; font-size: 11px;'>"
                "Select a spot first</div>"
            )
            return
        self._save_spots_btn.visible = event.new
        if not event.new:
            self._edit_status_html.object = ""
            self._has_unsaved_changes = False
        self._rebuild_plot()

    def _on_box_edit(self, data):
        """Called when box geometry changes via drag/resize."""
        if data and data.get('x0'):
            self._has_unsaved_changes = True
            self._edit_status_html.object = (
                "<div style='color: #ffaa00; padding: 5px; font-size: 11px;'>"
                "Unsaved changes</div>"
            )

    def _save_edited_spot(self):
        """Read BoxEdit stream data and update the selected spot's bbox."""
        if self._box_stream is None or self._selected_spot is None:
            return
        data = self._box_stream.data
        if not data or not data.get('x0'):
            return

        disp_x0 = data['x0'][0]
        disp_y0 = data['y0'][0]
        disp_x1 = data['x1'][0]
        disp_y1 = data['y1'][0]

        # Normalize min/max (user may have swapped corners)
        disp_x_min = min(disp_x0, disp_x1)
        disp_x_max = max(disp_x0, disp_x1)
        disp_y_min = min(disp_y0, disp_y1)
        disp_y_max = max(disp_y0, disp_y1)

        if self._current_use_lonlat:
            lon_min, lon_max = disp_x_min, disp_x_max
            lat_min, lat_max = disp_y_min, disp_y_max
        else:
            # Convert from UTM to lon/lat
            mesh = self.data_manager.get_mesh(self.region)
            lons, lats = mesh.utm_to_lon_lat(
                np.array([disp_x_min, disp_x_max]),
                np.array([disp_y_min, disp_y_max]),
            )
            lon_min, lon_max = float(lons[0]), float(lons[1])
            lat_min, lat_max = float(lats[0]), float(lats[1])

        from data.spots.spot import BoundingBox
        self._selected_spot.bbox = BoundingBox(
            lat_min=round(lat_min, 6),
            lat_max=round(lat_max, 6),
            lon_min=round(lon_min, 6),
            lon_max=round(lon_max, 6),
        )

    def _do_save_to_json(self):
        """Write all spots for this region to JSON and invalidate caches."""
        from data.spots.spot import save_spots_config
        save_spots_config(self.region, self._spots_list)
        self.data_manager.invalidate_spots(self.region)
        self._has_unsaved_changes = False
        self._edit_status_html.object = (
            "<div style='color: #44ff44; padding: 5px; font-size: 11px;'>"
            "Saved!</div>"
        )

    def _on_save_click(self, event):
        """Save edited bounding box to JSON."""
        if self._selected_spot is None or not self._has_unsaved_changes:
            return
        self._save_edited_spot()
        self._do_save_to_json()
        # Refresh stats for the edited spot
        if self._selected_spot:
            self._update_spot_stats(self._selected_spot)

    @staticmethod
    def _breaker_type_label(val):
        """Convert breaker type code to label."""
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "N/A"
        labels = {0: "Spilling", 1: "Plunging", 2: "Collapsing", 3: "Surging"}
        return labels.get(int(val), "N/A")

    def _format_spot_stats_html(self, s):
        """Render SpotStatsSummary as styled HTML."""
        bb = s.spot.bbox

        def fmt(val, fmt_str=".2f", suffix=""):
            """Format a value, returning 'N/A' for NaN."""
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return "N/A"
            return f"{val:{fmt_str}}{suffix}"

        def fmt_time(seconds):
            """Format seconds as 'Xs (Y.Z min)' or just 'Xs'."""
            if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
                return "N/A"
            if seconds >= 60:
                return f"{seconds:.0f}s ({seconds/60:.1f} min)"
            return f"{seconds:.0f}s"

        # Groupiness interpretation
        gf = s.groupiness_factor_mean
        if isinstance(gf, float) and not math.isnan(gf):
            if gf < 0.7:
                g_label = "choppy/disorganized"
            elif gf < 1.0:
                g_label = "moderate grouping"
            elif gf < 1.2:
                g_label = "well-defined sets"
            else:
                g_label = "very clean swell"
        else:
            g_label = ""

        g_text = f"{fmt(gf)}" + (f" ({g_label})" if g_label else "")

        html = f"""
        <div style="color: white; font-size: 11px; padding: 6px;
                    background: {SIDEBAR_BG}; border-radius: 5px;
                    max-height: 300px; overflow-y: auto;">
            <b style="font-size: 13px;">{s.spot.display_name}</b><br>
            <span style="color: #888;">{bb.lat_min:.3f}-{bb.lat_max:.3f}°N, {abs(bb.lon_min):.3f}-{abs(bb.lon_max):.3f}°W</span><br>
            <span style="color: #888;">Points: {s.n_points:,} ({s.n_covered:,} covered)</span><br>
            <hr style="border-color: #444;">

            <b>Wave Height</b><br>
            Min: {fmt(s.hs_min)} m &nbsp; Max: {fmt(s.hs_max)} m<br>
            Mean: {fmt(s.hs_mean)} m &nbsp; Std: {fmt(s.hs_std)} m<br>

            <br><b>Sets &amp; Timing</b><br>
            Set period: {fmt_time(s.set_period_mean)}<br>
            Waves per set: {fmt(s.waves_per_set_mean, '.1f')}<br>
            Set duration: {fmt_time(s.set_duration_mean)}<br>
            Lull duration: {fmt_time(s.lull_duration_mean)}<br>

            <br><b>Wave Quality</b><br>
            Height amplification: {fmt(s.height_amplification_mean)}x<br>
            Groupiness: {g_text}<br>
            Steepness: {fmt(s.dominant_steepness, '.4f')}<br>
            Wavelength: {fmt(s.dominant_wavelength, '.1f')} m<br>

            <br><b>Breaking</b><br>
            Breaking: {fmt(s.breaking_fraction, '.0%') if hasattr(s, 'breaking_fraction') else 'N/A'} of points<br>
            Iribarren: {fmt(s.iribarren_mean) if hasattr(s, 'iribarren_mean') else 'N/A'}<br>
            Breaker type: {self._breaker_type_label(s.dominant_breaker_type) if hasattr(s, 'dominant_breaker_type') else 'N/A'}<br>
            Intensity: {fmt(s.breaking_intensity_mean) if hasattr(s, 'breaking_intensity_mean') else 'N/A'}<br>

            <br><b>Depth</b><br>
            Range: {fmt(s.depth_min)} - {fmt(s.depth_max)} m &nbsp; Mean: {fmt(s.depth_mean)} m<br>
        </div>
        """
        return html

    def _build_spot_overlay(self, spot, use_lonlat, mesh, dimmed=False):
        """Build HoloViews overlay for spot bounding box rectangle + label."""
        bb = spot.bbox
        if use_lonlat:
            x0, y0 = bb.lon_min, bb.lat_min
            x1, y1 = bb.lon_max, bb.lat_max
        else:
            corners_lon = np.array([bb.lon_min, bb.lon_max, bb.lon_max, bb.lon_min, bb.lon_min])
            corners_lat = np.array([bb.lat_min, bb.lat_min, bb.lat_max, bb.lat_max, bb.lat_min])
            cx, cy = mesh.lon_lat_to_utm(corners_lon, corners_lat)
            x0, y0 = cx[0], cy[0]
            x1, y1 = cx[2], cy[2]

        color = SPOT_DIMMED_COLOR if dimmed else SPOT_BBOX_COLOR
        width = SPOT_DIMMED_WIDTH if dimmed else SPOT_BBOX_WIDTH
        dash = 'dotted' if dimmed else SPOT_BBOX_DASH

        rect_path = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
        rect = hv.Path([rect_path]).opts(
            color=color, line_width=width, line_dash=dash,
        )
        label = hv.Text((x0 + x1) / 2, y1, spot.display_name).opts(
            color=color, text_font_size='9pt',
            text_align='center', text_baseline='bottom',
        )
        return rect * label

    def _rebuild_plot(self):
        """Build the full Datashader result plot."""
        # Capture current ranges before rebuilding (within-view persistence)
        if self._pending_ranges is None:
            current = self.get_ranges()
            if current is not None:
                self._pending_ranges = current

        use_lonlat = self._current_use_lonlat

        try:
            result = self.data_manager.get_result(self.region)
            mesh = self.data_manager.get_mesh(self.region)
        except Exception as e:
            self._plot_pane.object = None
            return

        n_points = result.n_points

        if use_lonlat:
            display_x, display_y = mesh.utm_to_lon_lat(result.mesh_x, result.mesh_y)
            x_label = "Longitude"
            y_label = "Latitude"
        else:
            display_x = result.mesh_x.copy()
            display_y = result.mesh_y.copy()
            x_label = "UTM Easting (m)"
            y_label = "UTM Northing (m)"

        # Determine display variable: Hs or a statistic from the CSV
        selected_var = self._variable_select.value
        if selected_var != 'Wave Height (Hs)' and selected_var in self._stats_column_map:
            col_name = self._stats_column_map[selected_var]
            stats_df = self.data_manager.get_statistics(self.region)
            stats_arr = np.full(result.n_points, np.nan)
            if stats_df is not None and col_name in stats_df.columns:
                point_ids = stats_df['point_id'].values.astype(int)
                stats_arr[point_ids] = stats_df[col_name].values
            color_values = stats_arr
            covered_mask = np.isfinite(color_values)
            cmap = STATS_CMAP
            cb_label = selected_var
        else:
            color_values = result.H_at_mesh
            covered_mask = result.ray_count > 0
            cmap = WAVE_CMAP
            cb_label = 'Hs (m)'

        not_covered_mask = ~covered_mask
        n_covered = int(np.sum(covered_mask))

        # Color range: 98th percentile for Hs, 2nd-98th for stats
        if n_covered > 0:
            if cmap is WAVE_CMAP:
                v_min = 0.0
                v_max = float(np.nanpercentile(color_values[covered_mask], 98))
                v_max = max(v_max, 0.5)
            else:
                v_min = float(np.nanpercentile(color_values[covered_mask], 2))
                v_max = float(np.nanpercentile(color_values[covered_mask], 98))
                v_max = max(v_max, v_min + 1e-6)
        else:
            v_min, v_max = 0.0, 1.0

        plot = None

        # Layer 1: Uncovered points (gray)
        n_not_covered = int(np.sum(not_covered_mask))
        if n_not_covered > 0:
            no_wave_df = pd.DataFrame({
                'x': display_x[not_covered_mask],
                'y': display_y[not_covered_mask],
                'val': np.ones(n_not_covered),
            })
            no_wave_points = hv.Points(no_wave_df, kdims=['x', 'y'], vdims=['val'])
            no_wave_shaded = spread(
                datashade(
                    no_wave_points,
                    aggregator=ds.mean('val'),
                    cmap=[NO_WAVE_COLOR, NO_WAVE_COLOR],
                ),
                px=2,
            )
            plot = no_wave_shaded

        # Layer 2: Covered points - colored by selected variable
        if n_covered > 0:
            wave_df = pd.DataFrame({
                'x': display_x[covered_mask],
                'y': display_y[covered_mask],
                'H': np.clip(color_values[covered_mask], v_min, v_max),
            })
            wave_points = hv.Points(wave_df, kdims=['x', 'y'], vdims=['H'])
            wave_shaded = spread(
                datashade(
                    wave_points,
                    aggregator=ds.mean('H'),
                    cmap=cmap,
                    cnorm='linear',
                ),
                px=4,
            )
            if plot is None:
                plot = wave_shaded
            else:
                plot = plot * wave_shaded
        else:
            if plot is None:
                plot = hv.Points([]).opts(size=0)

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

        # Optional ray path overlay
        if self._show_rays.value:
            ray_overlay = self._build_ray_overlay(use_lonlat, mesh)
            if ray_overlay is not None:
                plot = plot * ray_overlay

        # Spot bbox overlay — edit mode vs view mode
        editing = self._edit_mode.value and self._selected_spot is not None

        if editing:
            # In edit mode: editable rectangle for selected spot, dimmed overlays for others
            bb = self._selected_spot.bbox
            if use_lonlat:
                ex0, ey0 = bb.lon_min, bb.lat_min
                ex1, ey1 = bb.lon_max, bb.lat_max
            else:
                corners_lon = np.array([bb.lon_min, bb.lon_max])
                corners_lat = np.array([bb.lat_min, bb.lat_max])
                cx, cy = mesh.lon_lat_to_utm(corners_lon, corners_lat)
                ex0, ey0 = float(cx[0]), float(cy[0])
                ex1, ey1 = float(cx[1]), float(cy[1])

            edit_rects = hv.Rectangles([(ex0, ey0, ex1, ey1)]).opts(
                fill_alpha=SPOT_EDIT_FILL_ALPHA,
                fill_color=SPOT_EDIT_COLOR,
                line_color=SPOT_EDIT_COLOR,
                line_width=2,
                line_dash='solid',
            )
            self._box_stream = streams.BoxEdit(source=edit_rects, num_objects=1)
            self._box_stream.add_subscriber(self._on_box_edit)
            plot = plot * edit_rects

            # Dimmed overlays for non-selected spots
            for spot in self._spots_list:
                if spot is not self._selected_spot:
                    overlay = self._build_spot_overlay(spot, use_lonlat, mesh, dimmed=True)
                    plot = plot * overlay

            # Label for selected spot
            label = hv.Text((ex0 + ex1) / 2, ey1, self._selected_spot.display_name).opts(
                color=SPOT_EDIT_COLOR, text_font_size='9pt',
                text_align='center', text_baseline='bottom',
            )
            plot = plot * label
        else:
            self._box_stream = None
            if self._selected_spot is not None:
                spot_overlay = self._build_spot_overlay(self._selected_spot, use_lonlat, mesh)
                plot = plot * spot_overlay

        # Click marker — handled at the Bokeh level via a hook so that
        # tapping never triggers a HoloViews/datashader range recalculation.
        # Uses self._tap (persistent) so the Panel layout's inspector pane
        # is always bound to the same stream across region/use_lonlat changes.
        # Suppressed in edit mode to avoid conflicts with BoxEdit tool.

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

            if editing:
                # Activate BoxEditTool as the drag tool (overrides pan)
                from bokeh.models.tools import BoxEditTool
                for tool in fig.tools:
                    if isinstance(tool, BoxEditTool):
                        fig.toolbar.active_drag = tool
                        break
            else:
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
                fig.on_event(BokehTap, lambda event: self._tap.event(x=event.x, y=event.y))

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

        # Update the persistent PointInspector in-place.
        # self._inspector_pane is the same object in the Panel layout throughout
        # the session; only its underlying data and format function are swapped.
        coords = np.column_stack([display_x, display_y])

        if use_lonlat:
            all_lon, all_lat = display_x, display_y
            all_utm_x, all_utm_y = result.mesh_x, result.mesh_y
        else:
            all_utm_x, all_utm_y = display_x, display_y
            all_lon, all_lat = mesh.utm_to_lon_lat(result.mesh_x, result.mesh_y)

        inspector_df = pd.DataFrame({
            'H_at_mesh': result.H_at_mesh,
            'energy': result.energy,
            'ray_count': result.ray_count,
            'depth': result.mesh_depth,
            'utm_x': all_utm_x,
            'utm_y': all_utm_y,
            'lon': all_lon,
            'lat': all_lat,
        })

        # Merge per-point statistics into inspector_df
        _INSPECTOR_STAT_COLS = [
            'set_period', 'waves_per_set', 'groupiness_factor',
            'height_amplification', 'set_duration', 'lull_duration',
            'is_breaking', 'breaker_index', 'iribarren',
            'breaker_type', 'breaking_intensity',
        ]
        stats_df = self.data_manager.get_statistics(self.region)
        if stats_df is not None:
            for col in _INSPECTOR_STAT_COLS:
                if col in stats_df.columns:
                    arr = np.full(result.n_points, np.nan)
                    arr[stats_df['point_id'].values.astype(int)] = stats_df[col].values
                    inspector_df[col] = arr

        # Capture partition data reference for the format function
        partition_data = self.data_manager.get_partition_data(self.region)

        def format_result_point(row, idx):
            ray_count = int(row['ray_count'])
            if ray_count == 0:
                return f"""
                <div style="color: white; font-size: 11px; padding: 6px;
                            background: {SIDEBAR_BG}; border-radius: 5px;">
                    <b style="font-size: 13px;">Point #{idx:,}</b><br>
                    <hr style="border-color: #444;">
                    Lat: {row['lat']:.5f}<br>
                    Lon: {row['lon']:.5f}<br>
                    Depth: {row['depth']:.2f} m<br>
                    <br>
                    <i style="color: #888;">Not covered (no rays)</i>
                </div>
                """

            html = f"""
            <div style="color: white; font-size: 11px; padding: 6px;
                        background: {SIDEBAR_BG}; border-radius: 5px;
                        max-height: 400px; overflow-y: auto;">
                <b style="font-size: 13px;">Point #{idx:,}</b><br>
                <hr style="border-color: #444;">
                <b>Location</b><br>
                Lat: {row['lat']:.5f}<br>
                Lon: {row['lon']:.5f}<br>
                Depth: {row['depth']:.2f} m<br>
                <br>
                <b>Combined Wave Data</b><br>
                Hs: {row['H_at_mesh']:.2f} m<br>
                Energy: {row['energy']:.1f} J/m<br>
                Ray count: {ray_count}<br>
            """

            # Per-partition details
            if partition_data:
                html += "<hr style='border-color: #444;'><b>Per-Partition</b><br>"
                for pname, pdata in partition_data.items():
                    label = PARTITION_LABELS.get(pname, pname)
                    color = PARTITION_COLORS.get(pname, 'white')
                    if 'converged' in pdata and idx < len(pdata['converged']):
                        if pdata['converged'][idx]:
                            p_hs = pdata.get('boundary_Hs', pdata.get('H_at_mesh', np.zeros(1)))
                            p_tp = pdata.get('boundary_Tp', np.zeros(1))
                            p_dir = pdata.get('boundary_direction', np.zeros(1))
                            hs_val = p_hs[idx] if idx < len(p_hs) else 0
                            tp_val = p_tp[idx] if idx < len(p_tp) else 0
                            dir_val = p_dir[idx] if idx < len(p_dir) else 0
                            html += (
                                f"<span style='color: {color};'>"
                                f"<b>{label}:</b></span><br>"
                                f"&nbsp;&nbsp;Hs: {hs_val:.2f}m, "
                                f"Tp: {tp_val:.1f}s, "
                                f"Dir: {dir_val:.0f} deg<br>"
                            )

            # Wave Statistics section (from statistics CSV)
            stat_display = [
                ('set_period',          'Set period'),
                ('waves_per_set',       'Waves/set'),
                ('set_duration',        'Set duration'),
                ('lull_duration',       'Lull duration'),
                ('groupiness_factor',   'Groupiness'),
                ('height_amplification', 'Ht. amplif.'),
                ('is_breaking',         'Breaking'),
                ('breaker_index',       'Breaker idx'),
                ('iribarren',           'Iribarren'),
                ('breaker_type',        'Breaker type'),
                ('breaking_intensity',  'Break. intens.'),
            ]
            stat_rows = []
            for col, label in stat_display:
                if col in row.index:
                    val = row[col]
                    if not (isinstance(val, float) and math.isnan(val)):
                        if col in ('set_period', 'set_duration', 'lull_duration'):
                            fmt_val = (f"{val:.0f}s ({val/60:.1f} min)"
                                       if val >= 60 else f"{val:.0f}s")
                        elif col == 'height_amplification':
                            fmt_val = f"{val:.2f}x"
                        elif col == 'waves_per_set':
                            fmt_val = f"{val:.1f}"
                        elif col == 'is_breaking':
                            fmt_val = "Yes" if val >= 0.5 else "No"
                        elif col == 'breaker_type':
                            bt_labels = {0: "Spilling", 1: "Plunging", 2: "Collapsing", 3: "Surging"}
                            fmt_val = bt_labels.get(int(val), f"{val:.0f}")
                        else:
                            fmt_val = f"{val:.2f}"
                        stat_rows.append((label, fmt_val))
            if stat_rows:
                html += "<hr style='border-color: #444;'><b>Wave Statistics</b><br>"
                for label, fmt_val in stat_rows:
                    html += f"{label}: {fmt_val}<br>"

            html += "</div>"
            return html

        # Update inspector data in-place (keeps self._inspector_pane identity stable)
        self._inspector.tree = cKDTree(coords)
        self._inspector.data_df = inspector_df
        self._inspector.format_fn = format_result_point

        # Reset inspector display so stale data from a previous region isn't shown
        if self._tap.x is not None or self._tap.y is not None:
            self._tap.event(x=None, y=None)

        # Colorbar (horizontal, below plot) — update in-place
        cb_html = create_matplotlib_colorbar(
            v_min, v_max, cb_label, cmap,
            orientation='horizontal', width=800,
        )
        self._colorbar_pane.object = cb_html


    def _build_ray_overlay(self, use_lonlat: bool, mesh):
        """Build HoloViews Path overlay for sampled ray paths."""
        ray_data = self.data_manager.get_ray_paths(self.region)
        if ray_data is None:
            return None

        n_sampled = int(ray_data['n_rays_sampled'])
        if n_sampled == 0:
            return None

        n_to_show = min(self._n_rays.value, n_sampled)
        np.random.seed(42)
        indices = np.random.choice(n_sampled, size=n_to_show, replace=False)

        ray_start_idx = ray_data['ray_start_idx']
        ray_length = ray_data['ray_length']
        ray_partition = ray_data['ray_partition']
        path_x = ray_data['path_x']
        path_y = ray_data['path_y']

        partition_id_to_name = {
            0: 'wind_sea', 1: 'primary_swell',
            2: 'secondary_swell', 3: 'tertiary_swell',
        }

        overlays = []
        for pid in sorted(set(ray_partition[indices])):
            pid_indices = indices[ray_partition[indices] == pid]
            pname = partition_id_to_name.get(int(pid), 'wind_sea')
            color = PARTITION_COLORS.get(pname, 'gray')

            paths = []
            for idx in pid_indices:
                start = int(ray_start_idx[idx])
                length = int(ray_length[idx])
                rx = path_x[start:start + length]
                ry = path_y[start:start + length]
                if use_lonlat:
                    rx, ry = mesh.utm_to_lon_lat(rx, ry)
                paths.append(np.column_stack([rx, ry]))

            if paths:
                overlay = hv.Path(paths).opts(
                    color=color, line_width=1, alpha=0.6,
                )
                overlays.append(overlay)

        if not overlays:
            return None

        combined = overlays[0]
        for ov in overlays[1:]:
            combined = combined * ov
        return combined

    def panel(self):
        """Return the Panel layout."""
        display_controls = pn.Column(
            pn.pane.Markdown("**Display**", margin=(0, 0, 0, 0)),
            self._variable_select,
            width=240,
        )

        ray_controls = pn.Column(
            pn.pane.Markdown("**Ray Paths**"),
            self._show_rays,
            self._n_rays,
            width=220,
        )

        spot_controls = pn.Column(
            pn.pane.Markdown("**Surf Spots**", margin=(0, 0, 0, 0)),
            self._spot_selector,
            self._edit_mode,
            self._save_spots_btn,
            self._edit_status_html,
            self._spot_stats_html,
            width=240,
        )

        return pn.Row(
            pn.Column(
                self._plot_pane,
                self._colorbar_pane,
            ),
            pn.Column(
                display_controls,
                pn.Spacer(height=5),
                pn.pane.Markdown("**Point Inspector**", margin=(0, 0, 0, 0)),
                self._inspector_pane,
                pn.Spacer(height=5),
                spot_controls,
                pn.Spacer(height=5),
                ray_controls,
                width=260,
            ),
        )
