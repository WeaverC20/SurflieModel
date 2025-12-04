# Data Notebooks

Jupyter notebooks for data exploration, validation, and visualization.

## Organization

- **exploratory/** - Initial data exploration and EDA
- **validation/** - Data quality checks and validation

## Usage

```bash
# Start Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab
```

## Best Practices

1. **Name notebooks descriptively**: `01_explore_noaa_wavewatch.ipynb`
2. **Clear outputs before committing** (optional, or use nbstripout)
3. **Document findings** in markdown cells
4. **Export important visualizations** to share with team

## Difference from `data/analysis/`

- **notebooks/**: General data exploration, open to team
- **analysis/**: Private analysis (e.g., vs Surfline), just for you
