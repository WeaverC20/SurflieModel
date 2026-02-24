#!/usr/bin/env python3
"""
Merge regional surfzone statistics into a single coast-wide dataset.

Loads statistics_latest.csv and forward_result.npz from each region,
applies latitude-based hard clips at overlap midpoints, and outputs
a single merged CSV for the California coast.

Overlap zones:
    socal ↔ central: lat 34.5–35.0 → clip at 34.75
    central ↔ norcal: lat 38.5–39.0 → clip at 38.75

Usage:
    python scripts/merge_statistics.py
    python scripts/merge_statistics.py --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "data" / "surfzone" / "output"

# Hard clip boundaries (midpoints of overlap zones)
CLIP_BOUNDARIES = {
    "socal_central": 34.75,
    "central_norcal": 38.75,
}

# Region latitude clip rules
REGION_CLIPS = {
    "socal":   {"lat_max": CLIP_BOUNDARIES["socal_central"]},
    "central": {"lat_min": CLIP_BOUNDARIES["socal_central"],
                "lat_max": CLIP_BOUNDARIES["central_norcal"]},
    "norcal":  {"lat_min": CLIP_BOUNDARIES["central_norcal"]},
}

REGIONS = ["socal", "central", "norcal"]


def load_region_data(region: str) -> pd.DataFrame | None:
    """Load statistics CSV and forward result for a region, merge into one DataFrame."""
    stats_path = OUTPUT_DIR / region / "statistics_latest.csv"
    result_path = OUTPUT_DIR / region / "forward_result.npz"

    if not stats_path.exists():
        logger.warning(f"No statistics CSV for {region}: {stats_path}")
        return None

    df = pd.read_csv(stats_path)
    logger.info(f"  {region}: loaded {len(df):,} rows from statistics CSV")

    # Add wave height and ray count from forward result
    if result_path.exists():
        result = np.load(result_path)
        n_result = len(result['H_at_mesh'])
        n_stats = len(df)

        if n_result == n_stats:
            df['H_at_mesh'] = result['H_at_mesh']
            df['ray_count'] = result['ray_count']
            df['energy'] = result['energy']
        else:
            # Different mesh generations — align by point_id
            logger.warning(
                f"  {region}: result has {n_result:,} points vs stats {n_stats:,}. "
                f"Aligning by point_id."
            )
            point_ids = df['point_id'].values.astype(int)
            H = np.full(n_stats, np.nan)
            rc = np.full(n_stats, 0, dtype=int)
            en = np.full(n_stats, np.nan)
            valid = point_ids < n_result
            H[valid] = result['H_at_mesh'][point_ids[valid]]
            rc[valid] = result['ray_count'][point_ids[valid]]
            en[valid] = result['energy'][point_ids[valid]]
            df['H_at_mesh'] = H
            df['ray_count'] = rc
            df['energy'] = en
    else:
        logger.warning(f"  {region}: no forward_result.npz, wave heights unavailable")
        df['H_at_mesh'] = np.nan
        df['ray_count'] = 0
        df['energy'] = np.nan

    df['region'] = region
    return df


def clip_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Apply latitude clip to a region's DataFrame."""
    clips = REGION_CLIPS[region]
    mask = np.ones(len(df), dtype=bool)

    if 'lat_min' in clips:
        mask &= df['lat'].values >= clips['lat_min']
    if 'lat_max' in clips:
        mask &= df['lat'].values < clips['lat_max']

    clipped = df[mask].copy()
    logger.info(f"  {region}: {len(df):,} → {len(clipped):,} after clip")
    return clipped


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where all statistic columns AND wave height are NaN/zero."""
    # Statistic columns = everything except base columns
    base_cols = {'point_id', 'lat', 'lon', 'depth', 'H_at_mesh', 'ray_count', 'energy', 'region'}
    stat_cols = [c for c in df.columns if c not in base_cols]

    # Keep row if: any statistic is non-NaN OR ray_count > 0 (has wave data)
    has_stats = df[stat_cols].notna().any(axis=1)
    has_waves = df['ray_count'] > 0
    keep = has_stats | has_waves

    filtered = df[keep].copy()
    logger.info(f"  Dropped {len(df) - len(filtered):,} empty rows → {len(filtered):,} remaining")
    return filtered


def merge_statistics(dry_run: bool = False):
    """Main merge pipeline."""
    logger.info("Merging regional statistics into coast-wide dataset")
    logger.info(f"Clip boundaries: {CLIP_BOUNDARIES}")

    # Load and clip each region
    frames = []
    region_counts = {}

    for region in REGIONS:
        logger.info(f"\nProcessing {region}...")
        df = load_region_data(region)
        if df is None:
            continue
        df = clip_region(df, region)
        region_counts[region] = len(df)
        frames.append(df)

    if not frames:
        logger.error("No data loaded from any region!")
        sys.exit(1)

    # Concatenate
    merged = pd.concat(frames, ignore_index=True)
    logger.info(f"\nConcatenated: {len(merged):,} total rows")

    # Drop empty rows
    merged = drop_empty_rows(merged)

    # Reset point_id
    merged['point_id'] = np.arange(len(merged))

    # Summary
    logger.info(f"\nFinal merged dataset:")
    logger.info(f"  Total points: {len(merged):,}")
    logger.info(f"  Lat range: {merged['lat'].min():.3f} – {merged['lat'].max():.3f}")
    logger.info(f"  Lon range: {merged['lon'].min():.3f} – {merged['lon'].max():.3f}")
    covered = (merged['ray_count'] > 0).sum()
    logger.info(f"  Points with wave data: {covered:,} ({100*covered/len(merged):.1f}%)")
    for region, count in region_counts.items():
        logger.info(f"  {region}: {count:,} points")

    if dry_run:
        logger.info("\nDry run — not saving.")
        return

    # Save
    out_dir = OUTPUT_DIR / "california"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "statistics_merged.csv"
    merged.to_csv(csv_path, index=False)
    logger.info(f"\nSaved CSV: {csv_path}")

    # Metadata
    meta = {
        "created": datetime.now().isoformat(),
        "clip_boundaries": CLIP_BOUNDARIES,
        "regions": REGIONS,
        "region_point_counts": region_counts,
        "total_points": len(merged),
        "points_with_wave_data": int(covered),
        "lat_range": [float(merged['lat'].min()), float(merged['lat'].max())],
        "lon_range": [float(merged['lon'].min()), float(merged['lon'].max())],
        "columns": list(merged.columns),
    }

    meta_path = out_dir / "statistics_merged_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge regional surfzone statistics into coast-wide dataset",
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be merged without saving',
    )
    args = parser.parse_args()
    merge_statistics(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
