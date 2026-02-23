"""
Surf Spot Statistics Aggregation

Maps surf spots to mesh points and computes aggregated wave statistics
(heights, set timing, quality metrics, dominant partition) per spot.

Handles the case where the statistics CSV and forward tracing result may
have different numbers of points (e.g., different mesh generations) by
filtering each data source independently via lat/lon bounding box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from data.spots.spot import SurfSpot


@dataclass
class SpotStatsSummary:
    """Aggregated statistics for a single surf spot."""

    spot: SurfSpot
    n_points: int       # Stats CSV points within bbox
    n_covered: int      # Result points with wave data (ray_count > 0) within bbox

    # Wave height (from ForwardTracingResult.H_at_mesh, covered points only)
    hs_min: float
    hs_max: float
    hs_mean: float
    hs_std: float

    # Set timing (nanmean over stats points within bbox)
    set_period_mean: float       # seconds
    set_duration_mean: float     # seconds
    lull_duration_mean: float    # seconds
    waves_per_set_mean: float    # count

    # Wave quality
    height_amplification_mean: float  # ratio 1.0 to sqrt(2)
    groupiness_factor_mean: float     # ratio 0.5 to 1.5

    # Peakiness — from the dominant (most energetic) partition
    dominant_steepness: float    # H/L ratio
    dominant_wavelength: float   # meters

    # Depth context
    depth_min: float
    depth_max: float
    depth_mean: float

    # Breaking
    breaking_fraction: float         # fraction of points where is_breaking == 1
    iribarren_mean: float            # mean Iribarren number (surf similarity)
    dominant_breaker_type: float     # most common breaker type code (mode), NaN if none
    breaking_intensity_mean: float   # mean H/(gamma_b*h), >1 = breaking


class SpotStatisticsAggregator:
    """Maps surf spots to mesh points and aggregates statistics.

    The statistics CSV and forward tracing result may have different numbers
    of points (different mesh generations). This class filters each data
    source independently by lat/lon bounding box.
    """

    _N_PARTITIONS = 4

    def __init__(self, stats_df: pd.DataFrame, result, mesh):
        """
        Args:
            stats_df: DataFrame from statistics_latest.csv. Expected columns:
                point_id, lat, lon, depth, set_period, waves_per_set,
                wavelength_0..3, steepness_0..3, height_amplification,
                groupiness_factor, set_duration, lull_duration
            result: ForwardTracingResult with attributes:
                H_at_mesh, ray_count, mesh_depth, mesh_x, mesh_y
            mesh: SurfZoneMesh with utm_to_lon_lat() for coordinate conversion
        """
        self._stats_df = stats_df
        self._result = result
        self._mesh = mesh
        self._stats_masks: Dict[str, np.ndarray] = {}
        self._result_masks: Dict[str, np.ndarray] = {}
        # Lazy-computed lat/lon for result points
        self._result_lats: Optional[np.ndarray] = None
        self._result_lons: Optional[np.ndarray] = None

    def _get_result_coords(self) -> Tuple[np.ndarray, np.ndarray]:
        """Lazy-compute lat/lon for result points (cached)."""
        if self._result_lats is None:
            self._result_lons, self._result_lats = self._mesh.utm_to_lon_lat(
                self._result.mesh_x, self._result.mesh_y
            )
        return self._result_lats, self._result_lons

    def _get_stats_mask(self, spot: SurfSpot) -> np.ndarray:
        """Boolean mask of stats_df rows within spot's bounding box."""
        if spot.name not in self._stats_masks:
            bb = spot.bbox
            lats = self._stats_df["lat"].values
            lons = self._stats_df["lon"].values
            self._stats_masks[spot.name] = (
                (lats >= bb.lat_min)
                & (lats <= bb.lat_max)
                & (lons >= bb.lon_min)
                & (lons <= bb.lon_max)
            )
        return self._stats_masks[spot.name]

    def _get_result_mask(self, spot: SurfSpot) -> np.ndarray:
        """Boolean mask of result points within spot's bounding box."""
        if spot.name not in self._result_masks:
            bb = spot.bbox
            lats, lons = self._get_result_coords()
            self._result_masks[spot.name] = (
                (lats >= bb.lat_min)
                & (lats <= bb.lat_max)
                & (lons >= bb.lon_min)
                & (lons <= bb.lon_max)
            )
        return self._result_masks[spot.name]

    def _find_dominant_partition(self, stats_mask: np.ndarray) -> tuple:
        """Find the dominant partition among masked stats points.

        Returns (steepness, wavelength) for the partition with the highest
        mean wavelength (most energetic swell reaching the spot).
        """
        best_wavelength = 0.0
        best_steepness = float("nan")
        best_result_wl = float("nan")

        for i in range(self._N_PARTITIONS):
            wl_col = f"wavelength_{i}"
            st_col = f"steepness_{i}"
            if wl_col not in self._stats_df.columns:
                continue

            wl_vals = self._stats_df[wl_col].values[stats_mask]
            st_vals = self._stats_df[st_col].values[stats_mask]

            if np.all(np.isnan(wl_vals)):
                continue
            mean_wl = np.nanmean(wl_vals)
            if np.isnan(mean_wl) or mean_wl <= 0.0:
                continue

            if mean_wl > best_wavelength:
                best_wavelength = mean_wl
                best_result_wl = mean_wl
                best_steepness = np.nanmean(st_vals)

        return best_steepness, best_result_wl

    def aggregate(self, spot: SurfSpot) -> SpotStatsSummary:
        """Compute aggregated statistics for a spot.

        Uses independent spatial filtering on the statistics CSV and
        forward tracing result to handle different point counts.
        """
        nan = float("nan")

        # Stats CSV filtering (for statistics columns)
        stats_mask = self._get_stats_mask(spot)
        n_points = int(stats_mask.sum())

        # Result filtering (for wave height and coverage)
        result_mask = self._get_result_mask(spot)
        result_covered = result_mask & (self._result.ray_count > 0)
        n_covered = int(result_covered.sum())

        # Wave height from forward tracing result
        if n_covered > 0:
            hs = self._result.H_at_mesh[result_covered]
            hs_min = float(np.nanmin(hs))
            hs_max = float(np.nanmax(hs))
            hs_mean = float(np.nanmean(hs))
            hs_std = float(np.nanstd(hs))
        else:
            hs_min = hs_max = hs_mean = hs_std = nan

        # Statistics from CSV (use stats_mask, not result_covered)
        if n_points == 0:
            return SpotStatsSummary(
                spot=spot,
                n_points=0,
                n_covered=n_covered,
                hs_min=hs_min,
                hs_max=hs_max,
                hs_mean=hs_mean,
                hs_std=hs_std,
                set_period_mean=nan,
                set_duration_mean=nan,
                lull_duration_mean=nan,
                waves_per_set_mean=nan,
                height_amplification_mean=nan,
                groupiness_factor_mean=nan,
                dominant_steepness=nan,
                dominant_wavelength=nan,
                depth_min=nan,
                depth_max=nan,
                depth_mean=nan,
                breaking_fraction=nan,
                iribarren_mean=nan,
                dominant_breaker_type=nan,
                breaking_intensity_mean=nan,
            )

        # Set timing — filter out inf values in set_period
        set_period_vals = self._stats_df["set_period"].values[stats_mask].copy()
        set_period_vals[~np.isfinite(set_period_vals)] = np.nan
        set_period_mean = float(np.nanmean(set_period_vals))

        set_duration_mean = float(
            np.nanmean(self._stats_df["set_duration"].values[stats_mask])
        )
        lull_duration_mean = float(
            np.nanmean(self._stats_df["lull_duration"].values[stats_mask])
        )
        waves_per_set_mean = float(
            np.nanmean(self._stats_df["waves_per_set"].values[stats_mask])
        )

        # Wave quality
        height_amplification_mean = float(
            np.nanmean(self._stats_df["height_amplification"].values[stats_mask])
        )
        groupiness_factor_mean = float(
            np.nanmean(self._stats_df["groupiness_factor"].values[stats_mask])
        )

        # Dominant partition
        dominant_steepness, dominant_wavelength = self._find_dominant_partition(
            stats_mask
        )

        # Depth context (from stats CSV which has depth column)
        depth = self._stats_df["depth"].values[stats_mask]
        depth_min = float(np.nanmin(depth))
        depth_max = float(np.nanmax(depth))
        depth_mean = float(np.nanmean(depth))

        # Breaking stats
        if "is_breaking" in self._stats_df.columns:
            breaking_vals = self._stats_df["is_breaking"].values[stats_mask]
            valid_breaking = breaking_vals[np.isfinite(breaking_vals)]
            breaking_fraction = (
                float(np.nanmean(valid_breaking)) if len(valid_breaking) > 0 else nan
            )

            iribarren_vals = self._stats_df["iribarren"].values[stats_mask]
            iribarren_mean = float(np.nanmean(iribarren_vals))

            breaker_vals = self._stats_df["breaker_type"].values[stats_mask]
            valid_breaker = breaker_vals[np.isfinite(breaker_vals)]
            if len(valid_breaker) > 0:
                codes, counts = np.unique(valid_breaker.astype(int), return_counts=True)
                dominant_breaker_type = float(codes[np.argmax(counts)])
            else:
                dominant_breaker_type = nan

            intensity_vals = self._stats_df["breaking_intensity"].values[stats_mask]
            breaking_intensity_mean = float(np.nanmean(intensity_vals))
        else:
            breaking_fraction = nan
            iribarren_mean = nan
            dominant_breaker_type = nan
            breaking_intensity_mean = nan

        return SpotStatsSummary(
            spot=spot,
            n_points=n_points,
            n_covered=n_covered,
            hs_min=hs_min,
            hs_max=hs_max,
            hs_mean=hs_mean,
            hs_std=hs_std,
            set_period_mean=set_period_mean,
            set_duration_mean=set_duration_mean,
            lull_duration_mean=lull_duration_mean,
            waves_per_set_mean=waves_per_set_mean,
            height_amplification_mean=height_amplification_mean,
            groupiness_factor_mean=groupiness_factor_mean,
            dominant_steepness=dominant_steepness,
            dominant_wavelength=dominant_wavelength,
            depth_min=depth_min,
            depth_max=depth_max,
            depth_mean=depth_mean,
            breaking_fraction=breaking_fraction,
            iribarren_mean=iribarren_mean,
            dominant_breaker_type=dominant_breaker_type,
            breaking_intensity_mean=breaking_intensity_mean,
        )
