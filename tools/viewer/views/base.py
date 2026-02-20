"""
Base view class for all viewer panels.
"""

from typing import Optional

import param
import panel as pn


class BaseView(param.Parameterized):
    """
    Base class for all viewer panels.

    Subclasses must implement update() and panel().
    """

    region = param.String(default='socal')
    loading = param.Boolean(default=False)

    def __init__(self, data_manager, **params):
        super().__init__(**params)
        self.data_manager = data_manager
        self._plot_pane = pn.pane.HoloViews(None, sizing_mode='fixed')
        self._summary_html = pn.pane.HTML("", width=240)
        self._bokeh_fig = None
        self._pending_ranges = None

    def update(self, region: str, **kwargs):
        """Reload data and rebuild the plot for a new region or settings."""
        raise NotImplementedError

    def panel(self):
        """Return the Panel layout for this view."""
        raise NotImplementedError

    def summary_panel(self) -> pn.pane.HTML:
        """Return the summary stats HTML pane."""
        return self._summary_html

    def get_ranges(self) -> Optional[dict]:
        """Read current x/y ranges from the stored Bokeh figure."""
        fig = self._bokeh_fig
        if fig is None:
            return None
        try:
            xr = fig.x_range
            yr = fig.y_range
            if xr.start is None or xr.end is None:
                return None
            if yr.start is None or yr.end is None:
                return None
            return {
                'x_start': xr.start, 'x_end': xr.end,
                'y_start': yr.start, 'y_end': yr.end,
            }
        except Exception:
            return None

    def set_pending_ranges(self, ranges: Optional[dict]):
        """Store ranges to be applied to the next Bokeh figure."""
        self._pending_ranges = ranges
