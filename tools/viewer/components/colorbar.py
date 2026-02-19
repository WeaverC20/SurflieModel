"""
Matplotlib-based colorbar generator for embedding in Panel layouts.

Ported from scripts/dev/view_surfzone_mesh_ds.py:49-83.
"""

import io
import base64

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tools.viewer.config import DARK_BG


def create_matplotlib_colorbar(
    vmin: float,
    vmax: float,
    label: str,
    cmap_colors: list,
    height: int = 400,
) -> str:
    """
    Create a vertical colorbar as a base64 PNG HTML image.

    Args:
        vmin: Minimum value for the color scale.
        vmax: Maximum value for the color scale.
        label: Label text displayed next to the colorbar.
        cmap_colors: List of hex color strings for the colormap.
        height: Height of the colorbar image in pixels.

    Returns:
        HTML string with an embedded base64 PNG image.
    """
    fig, ax = plt.subplots(figsize=(1.2, height / 100), dpi=100)

    cmap = mcolors.LinearSegmentedColormap.from_list('custom', cmap_colors, N=256)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='vertical',
    )
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=8)

    # Style for dark theme
    ax.tick_params(colors='white')
    cb.ax.yaxis.set_tick_params(color='white')
    cb.outline.set_edgecolor('white')
    cb.ax.yaxis.label.set_color('white')
    for tick_label in cb.ax.get_yticklabels():
        tick_label.set_color('white')

    buf = io.BytesIO()
    plt.savefig(
        buf, format='png', bbox_inches='tight', dpi=100,
        facecolor=DARK_BG, edgecolor='none',
    )
    plt.close(fig)
    buf.seek(0)

    img_data = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_data}" />'
