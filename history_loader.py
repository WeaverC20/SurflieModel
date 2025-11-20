from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional

import pandas as pd

ForecastKind = Literal[
    "surf",
    "swells",
    "rating",
    "conditions",
    "sunlight",
    "tides",
    "wind",
]


# --------------------- filename / CSV helpers --------------------- #

def _extract_issue_date_from_name(path: Path) -> Optional[pd.Timestamp]:
    """
    Extract issue date from filenames like:
      spot_5842041f4e65fad6a7708827_surf_20240219.csv
    Returns a pandas.Timestamp (midnight UTC) or None if pattern not found.
    """
    m = re.search(r"(\d{8})", path.stem)
    if not m:
        return None
    datestr = m.group(1)
    dt = datetime.strptime(datestr, "%Y%m%d")
    return pd.Timestamp(dt).tz_localize("UTC")


def _read_single_forecast_csv(path: Path) -> pd.DataFrame:
    """
    Read a single forecast CSV and parse the 'timestamp' column as datetime
    (Surfline timestamps are unix seconds).
    """
    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        # Surfline timestamps are UNIX seconds since epoch (UTC)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    issue_date = _extract_issue_date_from_name(path)
    if issue_date is not None:
        df["issue_date"] = issue_date

    # Optional: ensure sorted by timestamp if present
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def load_kind_history(
    base_dir: str | Path,
    spot_id: str,
    kind: ForecastKind,
) -> pd.DataFrame:
    """
    Load ALL historical CSV files for a given spot + kind and return
    one concatenated DataFrame.

    Filenames are assumed to look like:
      spot_<spot_id>_<kind>_YYYYMMDD.csv

    Example:
        surf_df = load_kind_history(
            "forecasts/spot_5842041f4e65fad6a7708827",
            "5842041f4e65fad6a7708827",
            "surf",
        )
    """
    base_dir = Path(base_dir)

    pattern = f"spot_{spot_id}_{kind}_*.csv"
    files = sorted(base_dir.glob(pattern))

    if not files:
        # Return empty DataFrame with no error
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = _read_single_forecast_csv(f)
            frames.append(df)
        except Exception as e:
            print(f"[warn] failed to read {f}: {e}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # For safety: sort by (timestamp, issue_date) if both exist
    sort_cols = [c for c in ["timestamp", "issue_date"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


# ---------------------- Surf nested structure --------------------- #

@dataclass
class SurfPoint:
    """
    A single surf forecast point for a specific valid time, from a given issue_date.

    utcOffset is deliberately NOT stored here (you said to disregard it).
    """
    timestamp: pd.Timestamp          # valid time (UTC)
    issue_date: pd.Timestamp         # forecast run date (UTC midnight from filename)
    lead_hours: int                  # hours ahead from issue_date to timestamp
    surf_min: Optional[float]
    surf_max: Optional[float]
    surf_humanRelation: Optional[str]


SurfForecastTree = Dict[pd.Timestamp, Dict[int, SurfPoint]]
# outer key: issue_date (UTC)
# inner key: lead_hours (integer)
# value: SurfPoint


def load_surf_tree(
    base_dir: str | Path,
    spot_id: str,
) -> SurfForecastTree:
    """
    Load all surf forecasts for a spot and organize them as:

        surf_tree[issue_date][lead_hours] -> SurfPoint

    - issue_date comes from the filename.
    - lead_hours is computed as (timestamp - issue_date) in hours, rounded to int.
    """
    df = load_kind_history(base_dir, spot_id, "surf")
    if df.empty:
        return {}

    # Ensure we have proper dtypes
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["issue_date"] = pd.to_datetime(df["issue_date"], utc=True)

    # Compute lead time (hours ahead) and round to nearest int
    lead_hours_float = (df["timestamp"] - df["issue_date"]).dt.total_seconds() / 3600.0
    df["lead_hours"] = lead_hours_float.round().astype(int)

    surf_tree: SurfForecastTree = {}

    for _, row in df.iterrows():
        issue_ts: pd.Timestamp = row["issue_date"]
        lead_h: int = row["lead_hours"]

        # Pull forecast fields (ignore utcOffset even if present)
        surf_min = row.get("surf_min", None)
        surf_max = row.get("surf_max", None)
        surf_hr = row.get("surf_humanRelation", None)

        sp = SurfPoint(
            timestamp=row["timestamp"],
            issue_date=issue_ts,
            lead_hours=lead_h,
            surf_min=surf_min,
            surf_max=surf_max,
            surf_humanRelation=surf_hr,
        )

        if issue_ts not in surf_tree:
            surf_tree[issue_ts] = {}

        # If multiple rows map to the same lead_hours, last one wins.
        # If you want to keep all, we could make this a list instead.
        surf_tree[issue_ts][lead_h] = sp

    return surf_tree


# ------------------------- Spot container ------------------------- #

@dataclass
class SpotForecastHistory:
    """
    Convenience container for all the different forecast channels for a spot.

    - surf: nested dict structure:
        surf[issue_date][lead_hours] -> SurfPoint
    - others remain as DataFrames.

    Any DataFrame may be empty if those files don't exist.
    """
    surf: SurfForecastTree
    swells: pd.DataFrame
    rating: pd.DataFrame
    conditions: pd.DataFrame
    sunlight: pd.DataFrame
    tides: pd.DataFrame
    wind: pd.DataFrame


def load_spot_history(
    base_dir: str | Path,
    spot_id: str,
) -> SpotForecastHistory:
    """
    Load ALL historical forecast data (all kinds) for a given spot.

    Example:
        base = "forecasts/spot_5842041f4e65fad6a7708827"
        spot_id = "5842041f4e65fad6a7708827"

        history = load_spot_history(base, spot_id)

        # Surf access:
        surf_tree = history.surf
        for issue_date, runs in surf_tree.items():
            print(issue_date, list(runs.keys())[:10])  # lead_hours available

        # DataFrames:
        history.wind.head()
        history.swells.query("component_index == 1")
        history.rating["rating_value"].hist()
    """
    base_dir = Path(base_dir)

    return SpotForecastHistory(
        surf=load_surf_tree(base_dir, spot_id),
        swells=load_kind_history(base_dir, spot_id, "swells"),
        rating=load_kind_history(base_dir, spot_id, "rating"),
        conditions=load_kind_history(base_dir, spot_id, "conditions"),
        sunlight=load_kind_history(base_dir, spot_id, "sunlight"),
        tides=load_kind_history(base_dir, spot_id, "tides"),
        wind=load_kind_history(base_dir, spot_id, "wind"),
    )