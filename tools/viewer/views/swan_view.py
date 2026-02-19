"""
SWAN data view using Plotly heatmap.

Displays SWAN model output (Hs, Tp, Dir) as interactive heatmaps
with partition selection and surf spot bounding box overlays.
"""

import numpy as np
import panel as pn
import param

from tools.viewer.views.base import BaseView
from tools.viewer.config import PARTITION_LABELS, SIDEBAR_BG


# Variable-specific colormaps
_VAR_COLORSCALES = {
    'Hs': 'Viridis',
    'Tp': 'Plasma',
    'Dir': 'HSV',
}

_VAR_UNITS = {
    'Hs': 'm',
    'Tp': 's',
    'Dir': 'deg',
}

# Partition selector choices (Combined + individual partitions)
_PARTITION_CHOICES = ['Combined'] + list(PARTITION_LABELS.values())
_PARTITION_NAME_MAP = {v: k for k, v in PARTITION_LABELS.items()}


class SwanView(BaseView):
    """SWAN output heatmap view (Plotly)."""

    variable = param.Selector(
        default='Hs', objects=['Hs', 'Tp', 'Dir'], label='Variable',
    )
    partition = param.Selector(
        default='Combined', objects=_PARTITION_CHOICES, label='Partition',
    )
    resolution = param.Selector(default='coarse', objects=['coarse'], label='Resolution')

    def __init__(self, data_manager, **params):
        super().__init__(data_manager, **params)
        self._plotly_pane = pn.pane.Plotly(None, sizing_mode='stretch_both', min_height=700)
        self._var_widget = pn.widgets.RadioButtonGroup.from_param(
            self.param.variable, button_type='primary',
        )
        self._partition_widget = pn.widgets.Select.from_param(self.param.partition)
        self._resolution_widget = pn.widgets.Select.from_param(self.param.resolution)

    def update(self, region: str, **kwargs):
        """Reload SWAN data and rebuild the heatmap."""
        self.region = region

        # Update available resolutions
        resolutions = self.data_manager.available_swan_resolutions(region)
        if not resolutions:
            self._plotly_pane.object = None
            self._summary_html.object = (
                f"<div style='color: white; padding: 10px;'>"
                f"No SWAN data found for <b>{region}</b>.</div>"
            )
            return

        self.param.resolution.objects = resolutions
        if self.resolution not in resolutions:
            self.resolution = resolutions[0]

        # Update available partitions based on data
        swan = self.data_manager.get_swan_output(region, self.resolution)
        choices = ['Combined']
        if swan.has_partitions:
            for p in swan.partitions:
                choices.append(p.label)
        self.param.partition.objects = choices
        if self.partition not in choices:
            self.partition = 'Combined'

        self._rebuild_plot()

    @param.depends('variable', 'partition', 'resolution', watch=True)
    def _on_param_change(self):
        """React to widget changes."""
        self._rebuild_plot()

    def _rebuild_plot(self):
        """Build the Plotly heatmap figure."""
        import plotly.graph_objects as go

        try:
            swan = self.data_manager.get_swan_output(self.region, self.resolution)
        except Exception as e:
            self._plotly_pane.object = None
            self._summary_html.object = (
                f"<div style='color: #ff8888; padding: 10px;'>Error: {e}</div>"
            )
            return

        var = self.variable
        partition_label = self.partition

        # Get data grid based on partition selection
        if partition_label == 'Combined':
            hsig_m, tps_m, dir_m = swan.mask_land()
            data_map = {'Hs': hsig_m, 'Tp': tps_m, 'Dir': dir_m}
            z = data_map[var]
            title_extra = "Combined"
        else:
            # Find partition by label
            part_name = _PARTITION_NAME_MAP.get(partition_label)
            partition_obj = None
            if part_name:
                for p in swan.partitions:
                    if p.label == partition_label:
                        partition_obj = p
                        break
            if partition_obj is None:
                self._plotly_pane.object = None
                return

            hs_m, tp_m, dir_m = partition_obj.mask_invalid(swan.exception_value)
            data_map = {'Hs': hs_m, 'Tp': tp_m, 'Dir': dir_m}
            z = data_map[var]
            title_extra = partition_label

        colorscale = _VAR_COLORSCALES[var]
        unit = _VAR_UNITS[var]

        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=z,
            x=swan.lons,
            y=swan.lats,
            colorscale=colorscale,
            colorbar=dict(title=f'{var} ({unit})'),
            zsmooth='best',
            hovertemplate=f'Lon: %{{x:.3f}}<br>Lat: %{{y:.3f}}<br>{var}: %{{z:.2f}} {unit}<extra></extra>',
        ))

        # Overlay surf spot bounding boxes
        spots = self.data_manager.get_spots(self.region)
        for spot in spots:
            bb = spot.bbox
            fig.add_shape(
                type='rect',
                x0=bb.lon_min, y0=bb.lat_min,
                x1=bb.lon_max, y1=bb.lat_max,
                line=dict(color='white', width=1.5, dash='dot'),
            )
            fig.add_annotation(
                x=(bb.lon_min + bb.lon_max) / 2,
                y=bb.lat_max,
                text=spot.display_name,
                showarrow=False,
                font=dict(color='white', size=9),
                yshift=8,
            )

        fig.update_layout(
            title=f'SWAN {var} - {self.region} ({title_extra}, {self.resolution})',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            template='plotly_dark',
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#1a1a2e',
            height=750,
            xaxis=dict(scaleanchor='y', scaleratio=1),
        )

        self._plotly_pane.object = fig

        # Summary
        valid = z[~np.isnan(z)]
        n_valid = len(valid)
        summary = f"""
        <div style="color: white; font-size: 11px; padding: 10px;
                    background: {SIDEBAR_BG}; border-radius: 5px;">
            <b>SWAN Summary</b><br><br>
            Region: {self.region}<br>
            Resolution: {self.resolution}<br>
            Grid: {swan.hsig.shape[1]} x {swan.hsig.shape[0]}<br>
            Partitions: {len(swan.partitions)}<br>
            <br>
            <b>{var} ({title_extra})</b><br>
            Valid points: {n_valid:,}<br>
        """
        if n_valid > 0:
            summary += f"""
            Min: {np.nanmin(valid):.2f} {unit}<br>
            Max: {np.nanmax(valid):.2f} {unit}<br>
            Mean: {np.nanmean(valid):.2f} {unit}<br>
            """
        summary += "</div>"
        self._summary_html.object = summary

    def panel(self):
        """Return the Panel layout."""
        controls = pn.Column(
            pn.pane.Markdown("**Variable**"),
            self._var_widget,
            pn.Spacer(height=5),
            self._resolution_widget,
            pn.Spacer(height=5),
            self._partition_widget,
            width=260,
        )
        return pn.Row(
            self._plotly_pane,
            pn.Column(controls, pn.Spacer(height=10), self._summary_html, width=280),
        )
