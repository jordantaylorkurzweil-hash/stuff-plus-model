# ⚾ Stuff+ Pitch Quality Model

A machine learning pipeline that predicts pitch quality from physical Statcast characteristics — replicating and extending the **Stuff+** methodology used in MLB front offices.

Built as the capstone project for **SABR Analytics Certification Level IV**.

---

## 🎯 The Core Question

> *Can we predict the run value of a pitch based solely on its physical characteristics at the moment of release — independent of outcome?*

**Stuff+** is a pitch quality metric that evaluates how "nasty" a pitch is based purely on its physics: velocity, movement, spin, release point. A Stuff+ of 100 is league average. Above 100 is better. The metric deliberately ignores what happened after contact — it captures intrinsic pitch quality, not luck.

---

## 📁 Repository Structure

```
stuff-plus-model/
│
├── stuff_plus_model.ipynb       # Python pipeline — full Statcast pull → model → leaderboard
├── stuff_plus_models.R          # R implementation — Random Forest baseline vs. improved model
├── stuff_predictions.csv        # Model output — pitch-level Stuff+ predictions
├── README.md                    # This file
```

**Two implementations, one problem:** The Python notebook is the primary model. The R script is a companion implementation built during the SABR coursework, showing the same modeling problem approached in R with `ranger`. Together they demonstrate cross-language analytical ability.

---

## 🔬 Methodology

### Target Variable
`delta_run_exp` — the change in run expectancy per pitch, sourced directly from MLB Statcast via `pybaseball`. This is a true outcome-based run value, making it an ideal training signal for pitch quality.

### Features Used

| Feature | Description |
|---------|-------------|
| `release_speed` | Velocity at release point (mph) |
| `release_spin_rate` | Spin rate (rpm) |
| `release_extension` | Distance from rubber at release (ft) |
| `release_pos_x / z` | Horizontal and vertical release point |
| `pfx_x` | Horizontal movement vs. gravity-only path (in) |
| `pfx_z` | Induced vertical break / IVB (in) |
| `plate_x / z` | Location at the plate |
| `spin_axis` | Spin axis in degrees |
| `pitch_type` | Pitch type (encoded) |

### Engineered Features

| Feature | Rationale |
|---------|-----------|
| `spin_velocity_ratio` | Spin rate per mph — captures spin *efficiency*, not raw spin. Pitches with high spin relative to velocity tend to have more deceptive movement. |
| `total_break` | √(pfx_x² + pfx_z²) — total movement magnitude combining horizontal and vertical. |
| `plate_dist` | Distance from strike zone center — pitches at the edges or just off the zone are harder to handle. |
| `velo_x_ivb` | Velocity × IVB interaction — high velocity combined with high rise tends to generate whiffs. |

---

## 🤖 Model Comparison

| Model | Notes |
|-------|-------|
| **Random Forest** (baseline) | `ranger` in R / `RandomForestRegressor` in Python. Solid baseline, requires imputation for missing values. |
| **HistGradientBoostingRegressor** (final) | Gradient boosting with histogram binning. Handles Statcast's missing values natively. Faster on 700K+ pitch events. Outperformed RF on both RMSE and R². |

### Why HistGradientBoosting over Random Forest?

**1. Native NaN handling.** Statcast data has meaningful missingness — `spin_axis` isn't recorded for all pitch types, `release_extension` is occasionally absent. Random Forest requires imputation (which introduces assumptions). HistGradientBoosting learns to route missing values to the optimal branch during training, preserving the signal in missingness itself.

**2. Scale.** A full MLB season contains 700,000+ pitch events. Histogram binning reduces training time by an order of magnitude compared to Random Forest at this scale.

**3. Sequential residual correction.** Gradient boosting corrects prior errors iteratively — well-suited to the structured, non-linear relationships between pitch physics and run value outcomes (velocity × break interactions, spin efficiency, etc.).

---

## 📊 Key Findings — 2024 Season

- Model successfully differentiates elite pitch quality from average — top Stuff+ pitchers align closely with known MLB aces and high-swing-and-miss arms
- **Spin-Velocity Ratio** ranked among the most predictive engineered features — spin efficiency matters more than raw spin rate
- **Induced Vertical Break** and **release extension** were strong predictors across pitch types
- Pitch type was the single most important feature, confirming that evaluating pitches within type context is essential

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pybaseball` | Statcast data ingestion from Baseball Savant |
| `scikit-learn` | `HistGradientBoostingRegressor`, `RandomForestRegressor`, model evaluation |
| `pandas` / `numpy` | Data cleaning, feature engineering |
| `plotly` | Interactive Stuff+ leaderboard and scatter visualizations |
| `ranger` (R) | Random Forest baseline implementation in R |
| `dplyr` (R) | Feature engineering in R pipeline |

---

## 🚀 Getting Started

### Python Notebook

```bash
git clone https://github.com/jordantaylorkurzweil-hash/stuff-plus-model.git
cd stuff-plus-model
pip install pybaseball scikit-learn pandas numpy matplotlib seaborn plotly
```

Open `stuff_plus_model.ipynb` in Jupyter or Google Colab and run all cells top to bottom.

> **Note:** `pybaseball` caching is enabled. First run pulls ~700K pitch events and may take 2–3 minutes. Subsequent runs use cache.

### R Script

```r
install.packages(c("ranger", "dplyr"))
# Place training_data.csv and test_data_for_testing.csv in working directory
source("stuff_plus_models.R")
```

---

## ⚠️ Limitations

- This model is a **proxy for Stuff+**, not the proprietary Baseball Savant implementation
- Pitcher identity is excluded by design — we want pitch quality, not pitcher reputation
- Count and game context are excluded — Stuff+ is context-neutral by definition
- Trained on 2024 data only — cross-year validation would strengthen generalizability
- Gradient boosting trades interpretability for predictive power; feature importance should be interpreted carefully

---

## 📚 Background & Context

This project was built as part of the **SABR Analytics Certification (Level IV)**, which covers advanced modeling techniques applied to baseball data. The Stuff+ metric was developed by analysts at Baseball Savant and is now widely used across MLB organizations to evaluate pitcher arsenals independent of defense and sequencing.

**Related work:** See also the [Baseball Analytics Dashboard](https://github.com/jordantaylorkurzweil-hash/baseball-analytics-dashboard) repo — a full sabermetrics pipeline covering pitcher WAR, hitter profiles, and team-level analytics using FanGraphs data.

---

## 🙋 About

**Jordan Kurzweil**
M.S. Business Administration candidate, Pace University (Lubin School of Business) — August 2026
Google Data Analytics Certificate · SABR Analytics Certification Levels I–IV
[LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN) · [GitHub](https://github.com/jordantaylorkurzweil-hash)

---

*Data sourced from MLB Statcast via pybaseball. All stats reflect the 2024 MLB regular season.*
