"""
Base view class for all viewer panels.
"""

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
        self._summary_html = pn.pane.HTML("", width=280)

    def update(self, region: str, **kwargs):
        """Reload data and rebuild the plot for a new region or settings."""
        raise NotImplementedError

    def panel(self):
        """Return the Panel layout for this view."""
        raise NotImplementedError

    def summary_panel(self) -> pn.pane.HTML:
        """Return the summary stats HTML pane."""
        return self._summary_html
