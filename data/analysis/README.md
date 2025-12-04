# Private Analysis

**This directory is for your personal analysis only. Never deployed to production.**

## What Goes Here

- Performance comparisons (your model vs Surfline)
- Private experiments
- Personal insights and research
- Anything using Surfline data

## Important

This code is **never imported by production code**:
- ❌ Not imported by `backend/api`
- ❌ Not imported by `ml/inference`
- ❌ Not deployed anywhere

This is your private workspace for understanding how your models perform compared to industry standards.

## Example Notebooks

```
data/analysis/
├── compare_surfline_vs_mine.ipynb
├── accuracy_over_time.ipynb
├── spot_specific_performance.ipynb
└── surfline_api_exploration.ipynb
```

## Ethics Note

If you're using Surfline data for comparison:
- Be respectful of their terms of service
- Use for personal learning and validation only
- Don't expose their data in your public app
- Don't overload their servers
