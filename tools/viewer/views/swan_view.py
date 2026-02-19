"""
SWAN data view using Plotly heatmap.

Displays SWAN model output (Hs, Tp, Dir) as interactive heatmaps
with partition selection, surf spot bounding box overlays,
NDBC/CDIP buoy markers with swell component hover tooltips,
and SWAN run date display.
"""

import logging

import numpy as np
import panel as pn
import param

from tools.viewer.views.base import BaseView
from tools.viewer.config import (
    PARTITION_LABELS, SIDEBAR_BG,
    NDBC_MARKER_COLOR, CDIP_MARKER_COLOR,
    NDBC_MARKER_SYMBOL, CDIP_MARKER_SYMBOL,
    BUOY_MARKER_SIZE, CONFIDENCE_COLORS, WAVE_TYPE_COLORS,
)

logger = logging.getLogger(__name__)

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


def _format_buoy_hover(buoy: dict) -> str:
    """Build HTML hover text for a buoy with swell partitions and r1 confidence.

    Format matches the previous explore_spots.py implementation.
    """
    network = buoy['network']
    sid = buoy['station_id']
    name = buoy['name']
    lat, lon = buoy['lat'], buoy['lon']

    lines = [
        f"<b>{network} Buoy {sid}</b>",
        f"<b>{name}</b>",
        f"Location: ({lat:.3f}, {lon:.3f})",
    ]

    # Observation timestamp
    ts = buoy.get('timestamp')
    if ts:
        # Extract HH:MM from ISO string
        if 'T' in str(ts):
            time_part = str(ts).split('T')[1][:5]
        else:
            time_part = str(ts)
        lines.append(f"Time: {time_part} UTC")

    lines.append("")

    # Combined parameters
    combined = buoy.get('combined')
    if combined:
        hs = combined.get('significant_height_m')
        tp = combined.get('peak_period_s')
        dp = combined.get('peak_direction_deg')
        hs_str = f"{hs:.2f}m" if hs is not None else "--"
        tp_str = f"{tp:.1f}s" if tp is not None else "--"
        dp_str = f"{dp:.0f}\u00b0" if dp is not None else "--"
        lines.append(f"<b>Combined:</b> {hs_str}, {tp_str}, {dp_str}")

    lines.append("")
    lines.append("<b>Partitions:</b>")
    lines.append("<i>r1: directional confidence (HIGH=clean swell, LOW=mixed)</i>")

    partitions = buoy.get('partitions', [])
    if not partitions:
        lines.append("  No partition data available")
    else:
        for p in partitions:
            pid = p.get('partition_id', '?')
            ht = p.get('height_m')
            per = p.get('period_s')
            d = p.get('direction_deg')
            wtype = p.get('type', '')
            epct = p.get('energy_pct')
            r1 = p.get('r1')
            conf = p.get('confidence', '')

            ht_str = f"{ht:.2f}m" if ht is not None else "--"
            per_str = f"{per:.1f}s" if per is not None else "--"
            dir_str = f"{d:.0f}\u00b0" if d is not None else "--"

            # Wave type with color
            type_color = WAVE_TYPE_COLORS.get(wtype, 'white')
            type_str = f" (<span style='color: {type_color}'>{wtype}</span>)" if wtype else ""

            energy_str = f" [{epct:.0f}%]" if epct is not None else ""

            # r1 confidence with color coding
            if r1 is not None and conf:
                conf_color = CONFIDENCE_COLORS.get(conf, 'white')
                conf_str = f" <span style='color: {conf_color}'>r1={r1:.2f} {conf}</span>"
            else:
                conf_str = ""

            lines.append(
                f"  #{pid}: {ht_str}, {per_str}, {dir_str}{type_str}{energy_str}{conf_str}"
            )

    return "<br>".join(lines)


class SwanView(BaseView):
    """SWAN output heatmap view (Plotly)."""

    variable = param.Selector(
        default='Hs', objects=['Hs', 'Tp', 'Dir'], label='Variable',
    )
    partition = param.Selector(
        default='Combined', objects=_PARTITION_CHOICES, label='Partition',
    )
    resolution = param.Selector(default='coarse', objects=['coarse'], label='Resolution')
    show_buoys = param.Boolean(default=True, label='Show Buoys')

    def __init__(self, data_manager, **params):
        super().__init__(data_manager, **params)
        self._plotly_pane = pn.pane.Plotly(None, sizing_mode='stretch_both', min_height=700)
        self._var_widget = pn.widgets.RadioButtonGroup.from_param(
            self.param.variable, button_type='primary',
        )
        self._partition_widget = pn.widgets.Select.from_param(self.param.partition)
        self._resolution_widget = pn.widgets.Select.from_param(self.param.resolution)
        self._buoy_widget = pn.widgets.Checkbox.from_param(self.param.show_buoys)
        self._buoy_status_html = pn.pane.HTML("", width=260)

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

    @param.depends('variable', 'partition', 'resolution', 'show_buoys', watch=True)
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

        # Overlay buoy markers
        buoy_count = 0
        if self.show_buoys:
            buoy_count = self._add_buoy_traces(fig)

        # Title with SWAN run date
        run_date_str = ""
        if swan.run_timestamp:
            run_date_str = f" \u2014 Run: {swan.run_timestamp.strftime('%Y-%m-%d %H:%M')} UTC"

        fig.update_layout(
            title=f'SWAN {var} - {self.region} ({title_extra}, {self.resolution}){run_date_str}',
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
        """
        if swan.run_timestamp:
            summary += f"Run: {swan.run_timestamp.strftime('%Y-%m-%d %H:%M')} UTC<br>"
        summary += f"""
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
        if self.show_buoys:
            summary += f"<br>Buoys displayed: {buoy_count}<br>"
        summary += "</div>"
        self._summary_html.object = summary

    def _add_buoy_traces(self, fig) -> int:
        """Add NDBC and CDIP buoy scatter markers to the Plotly figure.

        Returns the number of buoys added.
        """
        import plotly.graph_objects as go

        try:
            buoys = self.data_manager.get_buoy_data(self.region)
        except Exception as e:
            logger.warning(f"Failed to load buoy data: {e}")
            self._buoy_status_html.object = (
                f"<div style='color: #ff8888; font-size: 10px;'>Buoy fetch error: {e}</div>"
            )
            return 0

        if not buoys:
            self._buoy_status_html.object = (
                "<div style='color: #aaa; font-size: 10px;'>No buoys found for this region</div>"
            )
            return 0

        # Separate by network
        ndbc_buoys = [b for b in buoys if b['network'] == 'NDBC']
        cdip_buoys = [b for b in buoys if b['network'] == 'CDIP']

        # NDBC scatter trace
        if ndbc_buoys:
            fig.add_trace(go.Scatter(
                x=[b['lon'] for b in ndbc_buoys],
                y=[b['lat'] for b in ndbc_buoys],
                mode='markers+text',
                marker=dict(
                    symbol=NDBC_MARKER_SYMBOL,
                    size=BUOY_MARKER_SIZE,
                    color=NDBC_MARKER_COLOR,
                    line=dict(color='#004d66', width=1.5),
                ),
                text=[b['station_id'] for b in ndbc_buoys],
                textposition='bottom center',
                textfont=dict(color=NDBC_MARKER_COLOR, size=8),
                hovertext=[_format_buoy_hover(b) for b in ndbc_buoys],
                hoverinfo='text',
                name='NDBC Buoys',
                showlegend=False,
            ))

        # CDIP scatter trace
        if cdip_buoys:
            fig.add_trace(go.Scatter(
                x=[b['lon'] for b in cdip_buoys],
                y=[b['lat'] for b in cdip_buoys],
                mode='markers+text',
                marker=dict(
                    symbol=CDIP_MARKER_SYMBOL,
                    size=BUOY_MARKER_SIZE,
                    color=CDIP_MARKER_COLOR,
                    line=dict(color='#662244', width=1.5),
                ),
                text=[b['station_id'] for b in cdip_buoys],
                textposition='bottom center',
                textfont=dict(color=CDIP_MARKER_COLOR, size=8),
                hovertext=[_format_buoy_hover(b) for b in cdip_buoys],
                hoverinfo='text',
                name='CDIP Buoys',
                showlegend=False,
            ))

        n_total = len(ndbc_buoys) + len(cdip_buoys)
        self._buoy_status_html.object = (
            f"<div style='color: #aaa; font-size: 10px;'>"
            f"Buoys: {len(ndbc_buoys)} NDBC, {len(cdip_buoys)} CDIP</div>"
        )
        return n_total

    def panel(self):
        """Return the Panel layout."""
        controls = pn.Column(
            pn.pane.Markdown("**Variable**"),
            self._var_widget,
            pn.Spacer(height=5),
            self._resolution_widget,
            pn.Spacer(height=5),
            self._partition_widget,
            pn.Spacer(height=5),
            self._buoy_widget,
            self._buoy_status_html,
            width=260,
        )
        return pn.Row(
            self._plotly_pane,
            pn.Column(controls, pn.Spacer(height=10), self._summary_html, width=280),
        )
