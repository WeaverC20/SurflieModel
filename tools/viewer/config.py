"""
Theme constants and configuration for the dev viewer.
"""

# Dark theme colors
DARK_BG = '#1a1a2e'
SIDEBAR_BG = '#2a2a3e'
TEXT_COLOR = 'white'

# Ocean depth colormap: cyan/teal shallow -> deep blue
OCEAN_CMAP = [
    '#00ffff', '#00e5e5', '#00cccc', '#00b3b3',
    '#0099cc', '#0080b3', '#006699', '#004d80', '#003366',
]

# Land elevation colormap: yellow/gold low -> orange/red high
LAND_CMAP = [
    '#ffff00', '#ffdd00', '#ffbb00', '#ff9900',
    '#ff7700', '#ff5500', '#e64400', '#cc3300', '#aa2200',
]

# Wave height colormap: blue -> cyan -> green -> yellow
WAVE_CMAP = [
    '#0044aa', '#0066cc', '#0088ee', '#00aaff', '#00cccc',
    '#00ee88', '#44ff44', '#aaff00', '#ffff00',
]

# Gray for uncovered points
NO_WAVE_COLOR = '#444455'

# Colormap for statistics variables (set_period, waves_per_set, etc.)
STATS_CMAP = 'viridis'

# Partition colors for ray paths and per-partition display
PARTITION_COLORS = {
    'wind_sea': 'cornflowerblue',
    'primary_swell': 'orange',
    'secondary_swell': 'limegreen',
    'tertiary_swell': 'crimson',
}

# Partition display labels
PARTITION_LABELS = {
    'wind_sea': 'Wind Sea',
    'primary_swell': 'Primary Swell',
    'secondary_swell': 'Secondary Swell',
    'tertiary_swell': 'Tertiary Swell',
}

# Coastline overlay color
COASTLINE_COLOR = '#ff00ff'

# Available regions
AVAILABLE_REGIONS = ['socal', 'central', 'norcal']

# Data types the viewer supports
DATA_TYPES = ['SWAN Data', 'Surfzone Mesh', 'Surfzone Results', 'California Coast']

# Default depth/elevation clamp values for mesh view
DEFAULT_DEPTH_MAX = 30.0
DEFAULT_LAND_MAX = 5.0

# Buoy marker styles
NDBC_MARKER_COLOR = '#00cccc'     # Cyan for NDBC
CDIP_MARKER_COLOR = '#ff66cc'     # Pink/magenta for CDIP
NDBC_MARKER_SYMBOL = 'diamond'
CDIP_MARKER_SYMBOL = 'square'
BUOY_MARKER_SIZE = 12

# Buoy hover text colors
CONFIDENCE_COLORS = {
    'HIGH': '#66ff66',   # Green — clean concentrated swell
    'MED': '#ffff66',    # Yellow — moderate spread
    'LOW': '#ff6666',    # Red — confused/mixed sea
}
WAVE_TYPE_COLORS = {
    'long_period_swell': '#66ccff',  # Light blue
    'swell': '#66ff66',               # Green
    'short_swell': '#ffff66',         # Yellow
    'wind_waves': '#ff9966',          # Orange
}

# Spot overlay styling
SPOT_BBOX_COLOR = 'white'
SPOT_BBOX_DASH = 'dashed'
SPOT_BBOX_WIDTH = 2

# Spot edit mode styling
SPOT_EDIT_COLOR = 'cyan'
SPOT_EDIT_FILL_ALPHA = 0.15
SPOT_DIMMED_COLOR = '#666666'
SPOT_DIMMED_WIDTH = 1
