"""
Reusable click-to-inspect panel with KDTree lookup.

Ported from the SingleTap + KDTree + reactive HTML pattern
in scripts/dev/view_statistics.py:337-420.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd
import panel as pn
from scipy.spatial import cKDTree

from tools.viewer.config import SIDEBAR_BG


class PointInspector:
    """
    Reusable click-to-inspect sidebar panel with KDTree nearest-point lookup.

    Given a HoloViews SingleTap stream, a coordinate array, and a DataFrame
    of point data, this class produces a Panel HTML pane that updates on
    every click to show information about the nearest point.
    """

    def __init__(
        self,
        tap_stream,
        coords: np.ndarray,
        data_df: pd.DataFrame,
        format_fn: Optional[Callable] = None,
    ):
        """
        Args:
            tap_stream: HoloViews SingleTap stream instance.
            coords: Array of shape (N, 2) used to build a KDTree for
                    nearest-point lookup.
            data_df: DataFrame with one row per point. The row at the
                     nearest index is passed to format_fn.
            format_fn: Optional callable(row, idx) -> HTML string.
                       If None, a default formatter showing all columns
                       is used.
        """
        self.tree = cKDTree(coords)
        self.data_df = data_df
        self.format_fn = format_fn or self._default_format
        self.tap = tap_stream

    def _default_format(self, row: pd.Series, idx: int) -> str:
        """Build generic HTML showing all columns in the row."""
        html = f"""
        <div style="color: white; font-size: 11px; padding: 10px;
                    background: {SIDEBAR_BG}; border-radius: 5px;
                    max-height: 700px; overflow-y: auto;">
            <b style="font-size: 13px;">Point #{idx:,}</b><br>
            <hr style="border-color: #444;">
        """
        for col in row.index:
            val = row[col]
            if isinstance(val, float):
                if np.isnan(val):
                    formatted = "N/A"
                elif abs(val) > 1000:
                    formatted = f"{val:,.0f}"
                else:
                    formatted = f"{val:.4f}"
            else:
                formatted = str(val)
            display_name = col.replace('_', ' ').title()
            html += f"{display_name}: {formatted}<br>"

        html += "</div>"
        return html

    def _get_info_html(self, x, y) -> str:
        """Generate HTML for the nearest point at click coordinates."""
        if x is None or y is None:
            return (
                "<div style='color: white; padding: 10px;'>"
                "<b>Click on a point to inspect</b></div>"
            )

        dist, idx = self.tree.query([x, y])
        row = self.data_df.iloc[idx]
        return self.format_fn(row, idx)

    def panel(self) -> pn.pane.HTML:
        """Return a Panel pane that updates on click."""
        return pn.pane.HTML(
            pn.bind(self._get_info_html, self.tap.param.x, self.tap.param.y),
            width=240,
        )
