# Data Destruction Certificate

## Date of Destruction

February 24, 2026 at 6:15 PM EST

---

## Data Destroyed

- **Description:** Surf forecast data collected via automated daily scraping
  from Surfline.com, including spot-level wave height forecasts, condition
  ratings, and associated metadata.
- **Approximate date range of collection:** 11/2023 - 2/2026
- **Approximate volume:** ~200MB of CSV files
  covering 7 spots
- **Storage locations prior to destruction:**
  - [e.g., "Local directory: '/Users/colinweaver/Documents/Personal Projects/SurflineFetcher/forecasts' — archived on 2/24/26, data files
  removed. Repository preserved in read-only archive for record-keeping."]

---

## Collection Method

Data was collected via a manual script that was ran nearly daily, downloading premium forecast data from Surfline.com. The collection script and repository have been archived at WeaverC20/SurflineFetcher in read-only state for record-keeping purposes.

---

## Use of Data Prior to Destruction

During the period the data was retained, the following use occurred:

- **Accuracy analysis:** A script was developed within the main application that assessed the accuracy of surflien forecast data as a function of time for the purpose of evaluating forecast accuracy. This script and its outputs have been removed from the active codebase. The removal is reflected in the git history of the main repository.

- **No model training:** The Surfline data was NOT used as training data for any machine learning model or forecasting system.

- **No model calibration:** The Surfline data was NOT used to adjust, tune, or calibrate any model parameters, weights, or coefficients.

- **No redistribution:** The Surfline data was NOT shared with any third party, published, displayed to end users, or redistributed in any form.

---

## Method of Destruction

[Describe what you actually did. For example:]

- Surfline data files deleted from local storage using rm on 2/24/26
- Verified deletion by confirming files no longer accessible at listed paths
- Checked and cleared: Jupyter notebook outputs, cached files, downloads
  folder, cloud sync services, trash/recycle bin
- GitHub scraping repository archived in read-only state; collected data
  files removed from repository

---

## Related Repository Archive

The automated collection script is preserved in archived (read-only) GitHub
repository: WeaverC20/SurflineFetcher

This repository has been archived rather than deleted to maintain a
transparent record of what was collected and how.

---

## Certification

I, Colin Weaver, certify that the statements in this document
are true and accurate to the best of my knowledge.

Name: Colin Weaver
Date: 02/24/2026
