#!/usr/bin/env python3
"""
Simple Surfline Forecast Accuracy Analysis

Calculates MSE vs forecast lead time for surf, wind, rating, etc.
Shows plots and saves them to the figures/ directory.

Usage:
    Set DATA_DIR below, then run: python analyze_forecast_accuracy.py
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - Set your data path here
# ============================================================================
DATA_DIR = "/Users/colinweaver/Documents/Personal Projects/SurflineFetcher/real_forecasts/spot_5842041f4e65fad6a7708827"
# ============================================================================


def load_forecast_csvs(data_dir, forecast_type):
    """
    Load all CSV files for a given forecast type

    Args:
        data_dir: Directory containing CSVs
        forecast_type: e.g., 'surf', 'wind', 'rating'

    Returns:
        DataFrame with all forecasts
    """
    pattern = re.compile(rf".*_{forecast_type}_(\d{{8}})\.csv")

    dfs = []
    for filepath in Path(data_dir).glob(f"*_{forecast_type}_*.csv"):
        match = pattern.match(filepath.name)
        if not match:
            continue

        # Parse forecast date from filename
        date_str = match.group(1)
        forecast_date = datetime.strptime(date_str, "%Y%m%d")

        # Load CSV
        df = pd.read_csv(filepath)
        df['forecast_date'] = forecast_date
        dfs.append(df)

    if not dfs:
        return None

    all_data = pd.concat(dfs, ignore_index=True)

    # Convert timestamp to datetime
    all_data['timestamp'] = pd.to_datetime(all_data['timestamp'], unit='s', utc=True)
    all_data['forecast_date'] = pd.to_datetime(all_data['forecast_date'], utc=True)

    # Calculate lead time in hours
    all_data['lead_time_hours'] = (
        (all_data['timestamp'] - all_data['forecast_date']).dt.total_seconds() / 3600
    )

    return all_data


def get_ground_truth(df):
    """
    Use the most recent forecast for each timestamp as ground truth
    """
    # Only keep forecasts with positive lead time
    df = df[df['lead_time_hours'] > 0].copy()

    # For each timestamp, get the forecast with minimum lead time (most recent)
    ground_truth = df.sort_values('lead_time_hours').groupby('timestamp', as_index=False).first()

    return ground_truth


def calculate_mse_by_lead_time(forecasts, ground_truth, value_col, lead_time_bins):
    """
    Calculate MSE for different lead time bins

    Returns:
        DataFrame with lead_time_hours, mse, n_samples
    """
    # Merge forecasts with ground truth
    merged = forecasts.merge(
        ground_truth[['timestamp', value_col]],
        on='timestamp',
        suffixes=('_forecast', '_actual')
    )

    # Create lead time bins
    merged['lead_time_bin'] = pd.cut(merged['lead_time_hours'], bins=lead_time_bins)

    results = []
    for bin_label, group in merged.groupby('lead_time_bin', observed=True):
        pred = group[f'{value_col}_forecast'].values
        actual = group[f'{value_col}_actual'].values

        # Remove NaN
        mask = ~(np.isnan(pred) | np.isnan(actual))
        pred = pred[mask]
        actual = actual[mask]

        if len(pred) == 0:
            continue

        mse = np.mean((pred - actual) ** 2)

        results.append({
            'lead_time_hours': group['lead_time_hours'].mean(),
            'mse': mse,
            'n_samples': len(pred)
        })

    return pd.DataFrame(results).sort_values('lead_time_hours')


def plot_mse_by_lead_time(results_dict, output_dir):
    """
    Create plots showing MSE vs lead time

    Args:
        results_dict: Dict mapping (forecast_type, variable) -> DataFrame with mse results
        output_dir: Where to save plots
    """
    # Group by forecast type
    by_type = {}
    for (ftype, var), df in results_dict.items():
        if ftype not in by_type:
            by_type[ftype] = {}
        by_type[ftype][var] = df

    # Create one plot per forecast type
    for ftype, vars_dict in by_type.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.tab10(range(len(vars_dict)))

        for i, (var, df) in enumerate(vars_dict.items()):
            ax.plot(
                df['lead_time_hours'],
                df['mse'],
                marker='o',
                linewidth=2,
                markersize=6,
                label=var,
                color=colors[i]
            )

        ax.set_xlabel('Forecast Lead Time (hours)', fontsize=12)
        ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
        ax.set_title(f'{ftype.upper()} Forecast Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Save plot
        output_path = Path(output_dir) / f'{ftype}_mse_vs_lead_time.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    # Show all plots
    plt.show()


def analyze_forecast_type(data_dir, forecast_type, value_columns, lead_time_bins):
    """
    Analyze one forecast type
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {forecast_type.upper()}")
    print(f"{'='*60}")

    # Load data
    print("Loading data...")
    forecasts = load_forecast_csvs(data_dir, forecast_type)

    if forecasts is None:
        print(f"  No {forecast_type} files found")
        return {}

    print(f"  Loaded {len(forecasts)} forecast records")

    # Get ground truth
    print("Calculating ground truth...")
    ground_truth = get_ground_truth(forecasts)
    print(f"  Ground truth: {len(ground_truth)} unique timestamps")

    # Calculate MSE for each variable
    results = {}
    for col in value_columns:
        if col not in forecasts.columns:
            continue

        print(f"  Calculating MSE for {col}...")
        mse_df = calculate_mse_by_lead_time(forecasts, ground_truth, col, lead_time_bins)

        if len(mse_df) > 0:
            results[(forecast_type, col)] = mse_df
            print(f"    MSE range: {mse_df['mse'].min():.3f} to {mse_df['mse'].max():.3f}")

    return results


def main():
    data_dir = DATA_DIR

    if not os.path.exists(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        print(f"\nPlease update DATA_DIR at the top of this script (line 25)")
        sys.exit(1)

    print("="*60)
    print("SURFLINE FORECAST ACCURACY ANALYSIS")
    print("="*60)
    print(f"Data directory: {data_dir}")

    # Lead time bins (in hours): 0-12h, 12-24h, 1-2d, 2-3d, 3-5d, 5-7d, 7-14d, 14-16d
    lead_time_bins = [0, 12, 24, 48, 72, 120, 168, 336, 384]

    # Analyze each forecast type
    all_results = {}

    # SURF
    surf_results = analyze_forecast_type(
        data_dir,
        'surf',
        ['surf_min', 'surf_max'],
        lead_time_bins
    )
    all_results.update(surf_results)

    # RATING
    rating_results = analyze_forecast_type(
        data_dir,
        'rating',
        ['rating_value'],  # Adjust based on your CSV columns
        lead_time_bins
    )
    all_results.update(rating_results)

    # WIND
    wind_results = analyze_forecast_type(
        data_dir,
        'wind',
        ['speed'],  # Adjust based on your CSV columns
        lead_time_bins
    )
    all_results.update(wind_results)

    # SWELLS
    swells_results = analyze_forecast_type(
        data_dir,
        'swells',
        ['swell_height'],  # Adjust based on your CSV columns
        lead_time_bins
    )
    all_results.update(swells_results)

    if not all_results:
        print("\nNo data to analyze!")
        sys.exit(1)

    # Create plots
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")

    # Create figures directory
    output_dir = Path(__file__).parent / "figures"
    output_dir.mkdir(exist_ok=True)

    plot_mse_by_lead_time(all_results, output_dir)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Plots saved to: {output_dir}")
    print("\nPlots are now displayed. Close the plot windows to exit.")


if __name__ == '__main__':
    main()
