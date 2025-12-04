# Private Analysis

**This directory is for your personal analysis only. Never deployed to production.**

## Surfline Forecast Accuracy Analysis

Simple script to analyze how Surfline's forecast accuracy degrades over time.

### Quick Start

```bash
# Install dependencies
pip install pandas numpy matplotlib

# Edit the script to set your data path (line 25)
# Then run:
python analyze_forecast_accuracy.py
```

### What It Does

1. Loads your Surfline forecast CSV files
2. Uses the most recent forecast as "ground truth"
3. Calculates MSE (Mean Squared Error) for forecasts at different lead times
4. Shows graphs of MSE vs lead time for surf, wind, rating, swells
5. Saves the graphs as PNG files in this directory

### Output

- Shows interactive plots (close windows to exit)
- Saves to `figures/` directory: `surf_mse_vs_lead_time.png`, `wind_mse_vs_lead_time.png`, etc.

### Adjusting Variables

Edit the script to change which columns are analyzed:

```python
# Around line 180
surf_results = analyze_forecast_type(
    data_dir,
    'surf',
    ['surf_min', 'surf_max'],  # <- Change these
    lead_time_bins
)
```

## Ethics Note

If you're using Surfline data for comparison:
- Be respectful of their terms of service
- Use for personal learning and validation only
- Don't expose their data in your public app
- Don't overload their servers
