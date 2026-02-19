"""
Main application for the SurflieModel Dev Viewer.

Ties together region/data-type selectors, sidebar, and view panels
into a Panel FastListTemplate served via a local web server.
"""

import sys
from pathlib import Path

import param
import panel as pn
import holoviews as hv

from tools.viewer.config import (
    AVAILABLE_REGIONS, DATA_TYPES, DARK_BG, SIDEBAR_BG, TEXT_COLOR,
)
from tools.viewer.data_manager import DataManager
from tools.viewer.views.swan_view import SwanView
from tools.viewer.views.mesh_view import MeshView
from tools.viewer.views.result_view import ResultView


class DevViewerApp(param.Parameterized):
    """Panel application with region/data-type selectors and swappable views."""

    region = param.Selector(default='socal', objects=AVAILABLE_REGIONS)
    data_type = param.Selector(default='Surfzone Results', objects=DATA_TYPES)
    use_lonlat = param.Boolean(default=False, label='Use Lon/Lat')

    def __init__(self, project_root, **params):
        super().__init__(**params)
        self.data_manager = DataManager(project_root)
        self.views = {
            'SWAN Data': SwanView(self.data_manager),
            'Surfzone Mesh': MeshView(self.data_manager),
            'Surfzone Results': ResultView(self.data_manager),
        }
        self._update_view()

    @param.depends('region', 'data_type', 'use_lonlat', watch=True)
    def _update_view(self):
        """Reload the active view when selector params change."""
        view = self.views[self.data_type]
        view.update(self.region, use_lonlat=self.use_lonlat)

    @param.depends('data_type')
    def main_area(self):
        """Return the currently active view's panel."""
        return self.views[self.data_type].panel()

    def _data_availability_html(self) -> str:
        """Build an HTML summary of which data types are available per region."""
        html = f"""
        <div style="color: {TEXT_COLOR}; font-size: 11px; padding: 10px;
                    background: {SIDEBAR_BG}; border-radius: 5px;">
            <b>Data Availability</b><br>
        """
        for r in AVAILABLE_REGIONS:
            html += f"<br><b>{r}</b><br>"
            for dt in DATA_TYPES:
                has = self.data_manager.has_data(dt, r)
                icon = "&#10003;" if has else "&#10007;"
                color = "#44ff44" if has else "#ff4444"
                html += f"&nbsp;&nbsp;<span style='color: {color};'>{icon}</span> {dt}<br>"
        html += "</div>"
        return html

    def sidebar_panel(self):
        """Build sidebar with selectors, data availability, and summary stats."""
        region_widget = pn.widgets.Select.from_param(self.param.region, name='Region')
        dtype_widget = pn.widgets.Select.from_param(self.param.data_type, name='Data Type')
        lonlat_widget = pn.widgets.Checkbox.from_param(self.param.use_lonlat)

        availability = pn.pane.HTML(self._data_availability_html(), width=200)

        # Active view's summary
        active_view = self.views[self.data_type]
        summary = active_view.summary_panel()

        legend_html = f"""
        <div style="color: {TEXT_COLOR}; font-size: 12px; padding: 10px;">
            <b>Controls</b><br><br>
            Scroll: Zoom in/out<br>
            Drag: Pan<br>
            Click: Inspect point<br>
        </div>
        """

        return pn.Column(
            pn.pane.Markdown("## SurflieModel\n### Dev Viewer"),
            pn.Spacer(height=5),
            region_widget,
            dtype_widget,
            lonlat_widget,
            pn.Spacer(height=10),
            availability,
            pn.Spacer(height=10),
            pn.pane.HTML(legend_html, width=200),
            width=230,
        )

    def servable(self):
        """Create and return the Panel FastListTemplate."""
        template = pn.template.FastListTemplate(
            title='SurflieModel Dev Viewer',
            sidebar=[self.sidebar_panel()],
            main=[self.main_area],
            theme='dark',
            accent_base_color='#0099cc',
        )
        return template


def main(
    region: str = 'socal',
    data_type: str = 'Surfzone Results',
    port: int = 5007,
    use_lonlat: bool = False,
):
    """
    Entry point: create the app and serve it.

    Args:
        region: Initial region to display.
        data_type: Initial data type view.
        port: Port for the Panel server.
        use_lonlat: If True, display in lon/lat instead of UTM.
    """
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    hv.extension('bokeh')
    pn.extension()

    app = DevViewerApp(
        project_root,
        region=region,
        data_type=data_type,
        use_lonlat=use_lonlat,
    )
    template = app.servable()
    pn.serve(template, port=port, show=True, title='SurflieModel Dev Viewer')
