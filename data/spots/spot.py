"""
Surf Spot Definitions

Defines surf spots as bounding boxes within regions. Spot definitions
are loaded from per-region JSON config files (e.g., data/spots/socal.json).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class BoundingBox:
    """Geographic bounding box in lat/lon (WGS84)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @property
    def center(self) -> Tuple[float, float]:
        """Center point as (lat, lon)."""
        return (
            (self.lat_min + self.lat_max) / 2,
            (self.lon_min + self.lon_max) / 2,
        )

    @classmethod
    def from_dict(cls, d: Dict) -> BoundingBox:
        return cls(
            lat_min=d["lat_min"],
            lat_max=d["lat_max"],
            lon_min=d["lon_min"],
            lon_max=d["lon_max"],
        )

    def to_dict(self) -> Dict:
        return {
            "lat_min": self.lat_min,
            "lat_max": self.lat_max,
            "lon_min": self.lon_min,
            "lon_max": self.lon_max,
        }


class SurfSpot:
    """
    A surf spot defined by a bounding box within a region.

    Loaded from JSON config files at data/spots/{region}.json.
    """

    def __init__(
        self,
        name: str,
        display_name: str,
        bbox: BoundingBox,
        region_name: str,
        description: str = "",
    ):
        self.name = name
        self.display_name = display_name
        self.bbox = bbox
        self.region_name = region_name
        self.description = description

    @classmethod
    def from_dict(cls, d: Dict, region_name: str) -> SurfSpot:
        """Create from a JSON-parsed dictionary."""
        return cls(
            name=d["name"],
            display_name=d["display_name"],
            bbox=BoundingBox.from_dict(d["bbox"]),
            region_name=region_name,
            description=d.get("description", ""),
        )

    def to_dict(self) -> Dict:
        """Serialize to dictionary for JSON storage."""
        d = {
            "name": self.name,
            "display_name": self.display_name,
            "bbox": self.bbox.to_dict(),
        }
        if self.description:
            d["description"] = self.description
        return d

    def __repr__(self) -> str:
        lat, lon = self.bbox.center
        return (
            f"SurfSpot(name='{self.name}', display_name='{self.display_name}', "
            f"center=({lat:.4f}, {lon:.4f}), region='{self.region_name}')"
        )


# Path to spot config directory
_SPOTS_DIR = Path(__file__).parent


def load_spots_config(region_name: str) -> List[SurfSpot]:
    """
    Load spot definitions for a region from its JSON config file.

    Args:
        region_name: Region identifier (e.g., "socal", "central", "norcal")

    Returns:
        List of SurfSpot objects
    """
    config_path = _SPOTS_DIR / f"{region_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No spot config found at {config_path}. "
            f"Create a JSON file with spot definitions."
        )

    with open(config_path) as f:
        config = json.load(f)

    region = config.get("region", region_name)
    return [SurfSpot.from_dict(d, region) for d in config["spots"]]
