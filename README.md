# 🏃‍♂️ DuckRunnerAnalysis

**Solution to analyze running data from FIT files.**

DuckRunner is a Python-based Streamlit application designed to parse, analyze,
and visualize running metrics extracted from `.fit` files (Garmin, Strava, and
similar sources).
The project aims to provide an interactive experience for exploring pace,
heart rate, elevation, and lap data — with visualizations inspired by Strava’s
performance charts.

---

## 🚀 Features (planned)

- Upload and parse FIT files using the Garmin FIT SDK or `fitparse`.
- Transform raw data into structured `pandas` or `polars` DataFrames.
- Apply data cleaning, smoothing, and unit conversions (m/s ↔ min/km).
- Display interactive charts (pace, heart rate, elevation) using Plotly.
- Add reference lines for:
  - **Training zones** (horizontal)
  - **Lap boundaries** (vertical)
- Interactive tooltip with:
  - Point-specific metrics
  - Lap summary (min, max, mode, avg, distance, duration, etc.)

---
