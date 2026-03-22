# Stuff+ Pitch Quality Model

A machine learning model that predicts pitch quality from physical Statcast characteristics, replicating the Stuff+ framework. Built as a SABR Analytics Level IV capstone project using the full 2024 MLB season (~700,000 pitches).

---

## Results

**HistGradientBoostingRegressor** outperformed Random Forest baseline on both RMSE and R².

| Model | RMSE | R² |
|-------|------|----|
| Random Forest (baseline) | — | — |
| HistGradientBoostingRegressor (final) | — | — |

*Exact metrics in `stuff_plus_model.ipynb` Cell 9.*

---

## 2024 Qualified Pitcher Leaderboard (Top 10)

| Rank | Pitcher | Team | Stuff+ Score |
|------|---------|------|--------------|
| — | — | — | — |

*Full leaderboard exported to `stuff_plus_leaderboard_2024.csv` by the notebook.*

---

## Features

### Core (11)

| Feature | Description |
|---------|-------------|
| `release_speed` | Pitch velocity at release |
| `release_spin_rate` | Spin rate (RPM) |
| `pfx_z` | Induced vertical break (gravity-corrected) |
| `pfx_x` | Horizontal break |
| `release_pos_x` | Horizontal release point |
| `release_pos_z` | Vertical release point |
| `release_extension` | Extension toward plate at release |
| `plate_x` | Horizontal location at plate |
| `plate_z` | Vertical location at plate |
| `spin_axis` | Direction of Magnus force |
| `pitch_type` | Pitch type (categorical) |

### Derived (3)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `total_break` | √(pfx_x² + pfx_z²) | Euclidean magnitude of total movement |
| `velo_x_ivb` | release_speed × pfx_z | Velocity–IVB interaction; most predictive derived feature |
| `zone_distance` | √(plate_x² + (plate_z − 2.5)²) | Command quality — distance from zone center |

**Target variable:** `delta_run_exp` — change in run expectancy per pitch (Statcast). Evaluated on plate-appearance-ending pitches only (~150–180K per season).

---

## Stuff+ Scaling

```python
stuff_plus = 100 - ((predicted_run_value - league_mean) / league_std × 10)
```

- **100** = league average
- **Higher = better** (inversion ensures intuitive direction)
- **Qualification threshold:** 200+ recorded pitches

---

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for full documentation:
- Target variable selection rationale
- Feature engineering decisions
- Why HistGradientBoosting beat Random Forest (3 structural reasons)
- Hyperparameter choices with justification
- Known limitations

---

## Setup

```bash
git clone https://github.com/jordantaylorkurzweil-hash/stuff-plus-model.git
cd stuff-plus-model
pip install -r requirements.txt
jupyter notebook stuff_plus_model.ipynb
```

**Note:** The notebook pulls live Statcast data via `pybaseball`. First run may take several minutes to download the full 2024 season.

---

## Stack

`Python` · `scikit-learn` · `HistGradientBoostingRegressor` · `pybaseball` · `pandas` · `numpy` · `matplotlib` · `Jupyter`

---

## Project Structure

```
stuff-plus-model/
├── stuff_plus_model.ipynb     # Full 14-cell analysis notebook
├── METHODOLOGY.md             # Detailed methodology documentation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Related Work

**Live MLB Analytics Dashboard:** [mlb-analytics-jordan.streamlit.app](https://mlb-analytics-jordan.streamlit.app)

---

*Jordan Kurzweil — SABR Analytics Level IV Capstone, 2025*  
*[github.com/jordantaylorkurzweil-hash](https://github.com/jordantaylorkurzweil-hash)*
