"""
California coast-wide statistics view using Datashader + HoloViews.

Displays merged surfzone statistics across all regions on a single
lon/lat plot. Supports variable selection (wave height + 8 statistics),
click-to-inspect, and spot bounding box overlays from all regions.

Data source: data/surfzone/output/california/statistics_merged.csv
(produced by scripts/merge_statistics.py).
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
    WAVE_CMAP, STATS_CMAP, NO_WAVE_COLOR, DARK_BG, SIDEBAR_BG,
    SPOT_BBOX_COLOR, SPOT_BBOX_DASH, SPOT_BBOX_WIDTH,
)
from tools.viewer.components.colorbar import create_matplotlib_colorbar
from tools.viewer.components.point_inspector import PointInspector


# Statistic column → display label
STAT_LABELS = {
    'set_period': 'Set Period (s)',
    'waves_per_set': 'Waves per Set',
    'groupiness_factor': 'Groupiness Factor',
    'height_amplification': 'Height Amplification',
    'set_duration': 'Set Duration (s)',
    'lull_duration': 'Lull Duration (s)',
    'set_height': 'Set Height (m)',
}

# Columns to show in the point inspector tooltip
INSPECTOR_STAT_COLS = [
    'set_period', 'waves_per_set', 'groupiness_factor',
    'height_amplification', 'set_duration', 'lull_duration',
    'is_breaking', 'breaker_index', 'iribarren',
    'breaker_type', 'breaking_intensity',
]


class CoastView(BaseView):
    """California coast-wide merged statistics view."""

    def __init__(self, data_manager, **params):
        super().__init__(data_manager, **params)
        self._tap = streams.SingleTap(x=None, y=None)
        _dummy_coords = np.zeros((1, 2))
        _dummy_df = pd.DataFrame({
            'lon': [0], 'lat': [0], 'depth': [0],
            'H_at_mesh': [0], 'ray_count': [0], 'region': [''],
        })
        self._inspector = PointInspector(self._tap, _dummy_coords, _dummy_df)
        self._inspector_pane = self._inspector.panel()
        self._colorbar_pane = pn.pane.HTML("", height=60, sizing_mode='stretch_width')

        # Variable selector
        self._variable_select = pn.widgets.Select(
            name='Display Variable',
            options=['Wave Height (Hs)'],
            value='Wave Height (Hs)',
            width=220,
        )
        self._variable_select.param.watch(self._on_variable_change, 'value')
        self._stats_column_map = {}  # display label → CSV column name

        # Spot selector
        self._spot_selector = pn.widgets.Select(
            name='Surf Spot', options=['(none)'], value='(none)', width=220,
        )
        self._spot_selector.param.watch(self._on_spot_change, 'value')
        self._spots_list = []
        self._selected_spot = None

    def update(self, region: str, **kwargs):
        """Load merged data and rebuild the plot. Region param is ignored."""
        df = self.data_manager.get_merged_statistics()
        if df is None:
            self._plot_pane.object = hv.Text(0, 0, "No merged statistics found.\nRun: python scripts/merge_statistics.py").opts(
                color='white', text_font_size='14pt',
            )
            return

        # Populate variable selector from available columns
        col_map = {label: col for col, label in STAT_LABELS.items()
                   if col in df.columns}
        self._stats_column_map = col_map
        options = ['Wave Height (Hs)'] + list(col_map.keys())
        current = self._variable_select.value
        self._variable_select.options = options
        if current not in options:
            self._variable_select.value = 'Wave Height (Hs)'

        # Load spots from all regions
        self._spots_list = self.data_manager.get_all_spots()
        if self._spots_list:
            self._spot_selector.options = ['(none)'] + [s.display_name for s in self._spots_list]
            self._spot_selector.value = '(none)'
        self._selected_spot = None

        self._rebuild_plot()

    def _on_variable_change(self, event):
        self._rebuild_plot()

    def _on_spot_change(self, event):
        if event.new == '(none)':
            self._selected_spot = None
        else:
            self._selected_spot = next(
                (s for s in self._spots_list if s.display_name == event.new), None
            )
        self._rebuild_plot()

    def _build_spot_overlay(self, spot):
        """Build HoloViews overlay for a spot bounding box (lon/lat)."""
        bb = spot.bbox
        x0, y0 = bb.lon_min, bb.lat_min
        x1, y1 = bb.lon_max, bb.lat_max
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
        """Build the full Datashader coast-wide plot."""
        if self._pending_ranges is None:
            current = self.get_ranges()
            if current is not None:
                self._pending_ranges = current

        df = self.data_manager.get_merged_statistics()
        if df is None:
            return

        n_points = len(df)
        display_x = df['lon'].values
        display_y = df['lat'].values

        # Determine display variable
        selected_var = self._variable_select.value
        if selected_var != 'Wave Height (Hs)' and selected_var in self._stats_column_map:
            col_name = self._stats_column_map[selected_var]
            color_values = df[col_name].values.copy()
            covered_mask = np.isfinite(color_values)
            cmap = STATS_CMAP
            cb_label = selected_var
        else:
            color_values = df['H_at_mesh'].values.copy()
            covered_mask = df['ray_count'].values > 0
            cmap = WAVE_CMAP
            cb_label = 'Hs (m)'

        not_covered_mask = ~covered_mask
        n_covered = int(np.sum(covered_mask))

        # Color range
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

        # Layer 2: Covered points colored by selected variable
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

        # Spot bbox overlay
        if self._selected_spot is not None:
            plot = plot * self._build_spot_overlay(self._selected_spot)

        # Tap handler for click-to-inspect
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
            fig.on_event(BokehTap, lambda event: self._tap.event(x=event.x, y=event.y))

        plot = plot.opts(
            width=1000,
            height=700,
            xlabel='Longitude',
            ylabel='Latitude',
            tools=['wheel_zoom', 'pan', 'reset', 'box_zoom'],
            active_tools=['wheel_zoom', 'pan'],
            bgcolor=DARK_BG,
            data_aspect=1,
            hooks=[_add_tap_handler],
        )

        self._plot_pane.object = plot

        # Update PointInspector
        coords = np.column_stack([display_x, display_y])
        inspector_df = df[['lon', 'lat', 'depth', 'H_at_mesh', 'ray_count', 'energy', 'region']].copy()
        for col in INSPECTOR_STAT_COLS:
            if col in df.columns:
                inspector_df[col] = df[col].values

        def format_coast_point(row, idx):
            ray_count = int(row['ray_count'])
            region = row.get('region', '?')

            if ray_count == 0:
                return f"""
                <div style="color: white; font-size: 11px; padding: 6px;
                            background: {SIDEBAR_BG}; border-radius: 5px;">
                    <b style="font-size: 13px;">Point #{idx:,}</b>
                    <span style="color: #888;">({region})</span><br>
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
                <b style="font-size: 13px;">Point #{idx:,}</b>
                <span style="color: #888;">({region})</span><br>
                <hr style="border-color: #444;">
                <b>Location</b><br>
                Lat: {row['lat']:.5f}<br>
                Lon: {row['lon']:.5f}<br>
                Depth: {row['depth']:.2f} m<br>
                <br>
                <b>Wave Data</b><br>
                Hs: {row['H_at_mesh']:.2f} m<br>
                Ray count: {ray_count}<br>
            """

            # Wave Statistics section
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

        self._inspector.tree = cKDTree(coords)
        self._inspector.data_df = inspector_df
        self._inspector.format_fn = format_coast_point

        if self._tap.x is not None or self._tap.y is not None:
            self._tap.event(x=None, y=None)

        # Colorbar
        cb_html = create_matplotlib_colorbar(
            v_min, v_max, cb_label, cmap,
            orientation='horizontal', width=800,
        )
        self._colorbar_pane.object = cb_html

        # Summary
        self._summary_html.object = f"""
        <div style="color: white; font-size: 11px; padding: 6px;
                    background: {SIDEBAR_BG}; border-radius: 5px;">
            <b>Coast-Wide Statistics</b><br>
            Points: {n_points:,}<br>
            Covered: {n_covered:,} ({100*n_covered/n_points:.1f}%)<br>
            Lat: {display_y.min():.2f} – {display_y.max():.2f}<br>
            Lon: {display_x.min():.2f} – {display_x.max():.2f}<br>
        </div>
        """

    def panel(self):
        """Return the Panel layout."""
        display_controls = pn.Column(
            pn.pane.Markdown("**Display**", margin=(0, 0, 0, 0)),
            self._variable_select,
            width=240,
        )

        spot_controls = pn.Column(
            pn.pane.Markdown("**Surf Spots**", margin=(0, 0, 0, 0)),
            self._spot_selector,
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
                width=260,
            ),
        )
