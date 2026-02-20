"""
Surfzone results view using Datashader + HoloViews.

Displays forward ray tracing results: gray uncovered points overlaid
with viridis-colored wave heights. Uses PointInspector for click-to-
inspect with per-partition details. Optional ray path overlay.

Ported from scripts/dev/view_surfzone_result.py (converted from Plotly
Scattergl to Datashader) with click patterns from view_statistics.py.
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
    WAVE_CMAP, NO_WAVE_COLOR, COASTLINE_COLOR, DARK_BG, SIDEBAR_BG,
    PARTITION_COLORS, PARTITION_LABELS,
    SPOT_BBOX_COLOR, SPOT_BBOX_DASH, SPOT_BBOX_WIDTH,
)
from tools.viewer.components.colorbar import create_matplotlib_colorbar
from tools.viewer.components.point_inspector import PointInspector


class ResultView(BaseView):
    """Surfzone simulation result view with Datashader rendering."""

    def __init__(self, data_manager, **params):
        super().__init__(data_manager, **params)
        self._inspector_pane = pn.pane.HTML("", width=240, sizing_mode='stretch_height')
        self._colorbar_col = pn.Column(width=120)
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
        # Store state for re-renders
        self._current_use_lonlat = False

    def update(self, region: str, **kwargs):
        """Reload result data and rebuild the plot."""
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

        self._rebuild_plot()

    def _on_ray_toggle(self, event):
        """Re-render when ray visibility changes."""
        self._rebuild_plot()

    def _on_spot_change(self, event):
        """Update stats panel and rebuild plot when spot selection changes."""
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

    def _format_spot_stats_html(self, s):
        """Render SpotStatsSummary as styled HTML."""
        import math

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
        <div style="color: white; font-size: 11px; padding: 10px;
                    background: {SIDEBAR_BG}; border-radius: 5px;
                    max-height: 400px; overflow-y: auto;">
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

            <br><b>Depth</b><br>
            Range: {fmt(s.depth_min)} - {fmt(s.depth_max)} m &nbsp; Mean: {fmt(s.depth_mean)} m<br>
        </div>
        """
        return html

    def _build_spot_overlay(self, spot, use_lonlat, mesh):
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

        rect_path = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
        rect = hv.Path([rect_path]).opts(
            color=SPOT_BBOX_COLOR, line_width=SPOT_BBOX_WIDTH, line_dash=SPOT_BBOX_DASH,
        )
        label = hv.Text((x0 + x1) / 2, y1, spot.display_name).opts(
            color=SPOT_BBOX_COLOR, text_font_size='9pt',
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
            self._summary_html.object = (
                f"<div style='color: #ff8888; padding: 10px;'>Error: {e}</div>"
            )
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

        covered_mask = result.ray_count > 0
        not_covered_mask = ~covered_mask
        n_covered = int(np.sum(covered_mask))

        # Auto wave height max
        if n_covered > 0:
            h_max = float(np.nanpercentile(result.H_at_mesh[covered_mask], 98))
            h_max = max(h_max, 0.5)
        else:
            h_max = 2.0

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

        # Layer 2: Covered points - colored by wave height
        if n_covered > 0:
            wave_df = pd.DataFrame({
                'x': display_x[covered_mask],
                'y': display_y[covered_mask],
                'H': np.clip(result.H_at_mesh[covered_mask], 0, h_max),
            })
            wave_points = hv.Points(wave_df, kdims=['x', 'y'], vdims=['H'])
            wave_shaded = spread(
                datashade(
                    wave_points,
                    aggregator=ds.mean('H'),
                    cmap=WAVE_CMAP,
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

        # Spot bbox overlay
        if self._selected_spot is not None:
            spot_overlay = self._build_spot_overlay(self._selected_spot, use_lonlat, mesh)
            plot = plot * spot_overlay

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

        # Build PointInspector
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

        # Capture partition data reference for the format function
        partition_data = self.data_manager.get_partition_data(self.region)

        def format_result_point(row, idx):
            ray_count = int(row['ray_count'])
            if ray_count == 0:
                return f"""
                <div style="color: white; font-size: 11px; padding: 10px;
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
            <div style="color: white; font-size: 11px; padding: 10px;
                        background: {SIDEBAR_BG}; border-radius: 5px;
                        max-height: 700px; overflow-y: auto;">
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

            html += "</div>"
            return html

        inspector = PointInspector(tap, coords, inspector_df, format_fn=format_result_point)
        self._inspector_pane = inspector.panel()

        # Colorbar (horizontal, below plot)
        wave_cb = create_matplotlib_colorbar(
            0, h_max, 'Hs (m)', WAVE_CMAP,
            orientation='horizontal', width=800,
        )
        self._colorbar_col = pn.Row(
            pn.pane.HTML(wave_cb, height=60, sizing_mode='stretch_width'),
            sizing_mode='stretch_width',
        )

        # Summary
        coverage_pct = 100 * result.coverage_rate
        summary = f"""
        <div style="color: white; font-size: 11px; padding: 10px;
                    background: {SIDEBAR_BG}; border-radius: 5px;">
            <b>Result Summary</b><br><br>
            Region: {result.region_name}<br>
            Partitions: {result.n_partitions}<br>
            Total points: {result.n_points:,}<br>
            Covered: {result.n_covered:,} ({coverage_pct:.1f}%)<br>
            Rays traced: {result.n_rays_total:,}<br>
        """
        if n_covered > 0:
            H_cov = result.H_at_mesh[covered_mask]
            ray_cov = result.ray_count[covered_mask]
            summary += f"""
            <hr style="border-color: #444;">
            <b>Wave Height (covered)</b><br>
            Min: {np.nanmin(H_cov):.2f} m<br>
            Max: {np.nanmax(H_cov):.2f} m<br>
            Mean: {np.nanmean(H_cov):.2f} m<br>
            <br>
            <b>Rays per point</b><br>
            Min: {ray_cov.min()}<br>
            Max: {ray_cov.max()}<br>
            Mean: {ray_cov.mean():.1f}<br>
            """
        summary += "</div>"
        self._summary_html.object = summary

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
        ray_controls = pn.Column(
            pn.pane.Markdown("**Ray Paths**"),
            self._show_rays,
            self._n_rays,
            width=220,
        )

        spot_controls = pn.Column(
            pn.pane.Markdown("### Surf Spots"),
            self._spot_selector,
            self._spot_stats_html,
            width=240,
        )

        return pn.Row(
            pn.Column(
                self._plot_pane,
                self._colorbar_col,
            ),
            pn.Column(
                spot_controls,
                pn.Spacer(height=10),
                pn.pane.Markdown("### Point Inspector"),
                self._inspector_pane,
                pn.Spacer(height=10),
                ray_controls,
                pn.Spacer(height=10),
                self._summary_html,
                width=260,
            ),
        )
