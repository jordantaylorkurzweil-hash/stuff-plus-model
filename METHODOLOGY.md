# Methodology — Stuff+ Pitch Quality Model

## Overview

This document covers the full methodological decisions behind the Stuff+ pitch quality model — including target variable selection, feature engineering rationale, model selection, hyperparameter choices, and known limitations.

For a quick summary of results, see the [README](README.md).

---

## 1. Problem Framing

### Why delta_run_exp as the target variable?

The target variable is `delta_run_exp` — the change in run expectancy attributable to each pitch, as calculated by MLB Statcast.

Several target variable options were considered:

| Target | Pros | Cons |
|--------|------|------|
| `delta_run_exp` | Direct measure of run prevention value; Statcast-provided; accounts for count, baserunners, outs | Includes outcome luck (BABIP, HR/FB) |
| `strike_probability` | Clean binary; large signal | Doesn't capture swing-and-miss vs. called strike value difference |
| `whiff_rate` | Directly measures swing-and-miss | Ignores called strikes and weak contact |
| `xwOBA_against` | Expected outcome removes luck | Requires contact to have a value; misses strike-throwing ability |

`delta_run_exp` was selected because it is the most complete single measure of pitch value — it captures strikeouts, walks, and all contact outcomes in a unified run-value framework. The Baseball Savant implementation is well-documented and widely used in public research.

**Key limitation:** `delta_run_exp` is only recorded for pitches that end a plate appearance (the final pitch of each at-bat). This reduces the usable dataset from ~700K total pitches to roughly 150-180K plate-appearance-ending pitches per season. The tradeoff is accepted because predicting run value on plate-appearance-ending pitches is the correct framing — we want to know how much value a pitch generates when it matters most.

---

## 2. Feature Engineering

### Core features

All core features are physical pitch characteristics recorded at or near the moment of release. This is intentional — the goal is to evaluate pitch quality independent of batter reaction, umpire calls, or fielder positioning.

| Feature | Why included |
|---------|-------------|
| `release_speed` | Primary driver of pitch difficulty — velocity is the baseline |
| `release_spin_rate` | Higher spin generally correlates with more movement and harder contact |
| `pfx_z` (IVB) | Induced vertical break — the "rise" on a fastball or depth on a curve, gravity-corrected |
| `pfx_x` | Horizontal break — sweeping action, arm-side run |
| `release_pos_x` | Horizontal release point — affects deception and approach angle |
| `release_pos_z` | Vertical release point — higher release creates steeper approach angles |
| `release_extension` | Distance from rubber at release — effectively increases perceived velocity |
| `plate_x` | Horizontal location at plate — command component |
| `plate_z` | Vertical location at plate — command component |
| `spin_axis` | Determines the direction of Magnus force; gyro spin vs. transverse spin |
| `pitch_type` | Categorical — fastball vs. breaking ball physics differ fundamentally |

### Derived features

Three interaction and distance features were engineered to capture non-linear relationships:

**`total_break`** = √(pfx_x² + pfx_z²)

The Euclidean magnitude of total pitch movement. Captures the combined effect of both movement planes rather than treating them independently. A pitch with moderate horizontal AND vertical break may be harder to hit than one with extreme movement in a single dimension.

**`velo_x_ivb`** = release_speed × pfx_z

The velocity-IVB interaction. This is the most important derived feature in the final model. High velocity combined with high induced vertical break creates a "rising" effect that is disproportionately difficult for hitters — the pitch arrives faster than expected AND in a different vertical location than the swing plane projects. This interaction captures synergistic difficulty that neither feature captures alone.

**`zone_distance`** = √(plate_x² + (plate_z − 2.5)²)

Euclidean distance from the approximate center of the strike zone (2.5 feet off the ground). Captures command quality — a pitch can have elite physical characteristics but if it's consistently in the middle of the zone it is far easier to hit than the physical features suggest. This feature bridges the "stuff" and "command" components of pitch quality.

### Features explicitly excluded

Several features were considered and excluded by design:

- **Batter identity** — we want pitch quality independent of who is hitting
- **Pitcher identity** — we want pitch quality independent of pitcher reputation  
- **Count** — Stuff+ is defined as context-neutral; a 97mph fastball is equally impressive on 0-0 and 3-2
- **Game state** (inning, score, baserunners) — same reason as count
- **Launch angle, exit velocity, xwOBA** — these are outcome features that would leak information about the result of the pitch into the model

---

## 3. Model Selection

### Why HistGradientBoostingRegressor beat Random Forest

The model comparison was not close. HistGradientBoostingRegressor outperformed Random Forest on both RMSE and R² with the same training data. Three structural reasons explain this:

**Reason 1: Native NaN handling**

Statcast has meaningful missingness patterns. `spin_axis` is not recorded for all pitch types (certain fastball variants and some change-ups lack reliable spin axis readings). `release_extension` is occasionally missing for unusual arm slots. Random Forest handles NaN by requiring imputation — we chose median imputation as the baseline, but any imputation decision introduces noise and loses the signal contained in *why* a value is missing.

HistGradientBoostingRegressor learns during training to route NaN values to the optimal branch for prediction. It treats missingness as information rather than noise. On Statcast data where pitch-type-specific missingness is systematic (not random), this matters.

**Reason 2: Scale efficiency**

A full 2024 MLB season contains approximately 700,000 pitch events. HistGradientBoosting's core algorithmic innovation — binning continuous features into 256 discrete histogram bins before tree construction — reduces the computational complexity of node splitting from O(n × features) to O(bins × features). On a 700K pitch dataset this difference is an order of magnitude in training time. The accuracy cost of binning is negligible; the speed benefit is substantial.

**Reason 3: Sequential error correction**

Gradient boosting trains an ensemble of weak learners sequentially, with each tree fit to the residuals of the previous ensemble. For the structured non-linear relationships in pitch physics — where velocity × IVB interaction is more predictive than either feature alone — sequential correction better captures the compounding effects. Random Forest's parallel, independent tree approach treats each feature combination as equally likely to be important, which is a worse prior for sports physics data.

### Hyperparameter decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_iter` | 300 | Upper bound; early stopping active |
| `max_depth` | 6 | Prevents overfitting on 700K+ samples |
| `learning_rate` | 0.05 | Conservative shrinkage; allows more trees to contribute |
| `min_samples_leaf` | 20 | Each leaf must represent at least 20 pitches |
| `l2_regularization` | 0.1 | Mild weight penalty to reduce overfitting |
| `early_stopping` | True | Stops when validation loss doesn't improve for 20 rounds |
| `validation_fraction` | 0.1 | 10% of training data held for early stopping |

The `min_samples_leaf=20` constraint was important — without it, the model would learn highly specific rules for rare pitch-type/count combinations that don't generalize.

---

## 4. Stuff+ Scaling

The model outputs a predicted `delta_run_exp` per pitch. Lower values indicate better pitches (the pitcher prevented more runs). To convert to the Stuff+ scale where 100 = league average and higher = better:

```
stuff_plus = 100 - ((predicted_run_value - league_mean) / league_std × 10)
```

The inversion (subtracting from 100) ensures the intuitive direction: higher Stuff+ = better pitcher. The 10× scaling factor produces a distribution where roughly 68% of pitchers fall between 90 and 110 (one standard deviation from league average), consistent with how Baseball Savant's Stuff+ is distributed.

**Qualification threshold:** Pitchers must have 200+ recorded pitches to appear in the leaderboard. This eliminates small-sample noise for pitchers who threw only a handful of innings.

---

## 5. Known Limitations

**This is a proxy model, not the proprietary implementation**

Baseball Savant's Stuff+ uses proprietary methods developed by their R&D team. This model replicates the conceptual framework using publicly available Statcast data and open-source ML tools. Results will correlate with official Stuff+ scores but will not match them exactly.

**2024 data only — no cross-year validation**

The model was trained and evaluated on 2024 data only. Pitch mix evolution, rule changes (pitch clock effects on spin rate and extension), and year-to-year variation in Statcast calibration could affect generalizability. Cross-year validation (training on 2022-2023, testing on 2024) is the logical next step.

**Context-neutral by design**

Stuff+ is intentionally context-neutral — it evaluates pitch quality the same way regardless of count, score, or baserunner situation. This is a feature for evaluating pitcher arsenal quality but a limitation if you want to understand situational execution. A pitcher with elite Stuff+ who consistently throws it in poor locations is less valuable than the Stuff+ score suggests.

**Command is partially captured, not fully**

`plate_x` and `plate_z` capture where the pitch ends up, and `zone_distance` captures how far it is from the center of the zone — but the model doesn't have access to the intended target. A pitch that misses its spot by 8 inches looks identical to a pitch that was aimed at that location.

---

## 6. Future Work

- Cross-year validation: train on 2022-2023, test on 2024
- Pitch-level interactive predictor: input physical characteristics, get predicted Stuff+ score
- Per-pitch-type models: separate models for fastballs, breaking balls, off-speed (different physics warrant different features)
- Pitcher arsenal aggregation: weight individual pitch Stuff+ scores by usage rate to produce arsenal-level Stuff+

---

*Jordan Kurzweil — SABR Analytics Level IV Capstone, 2025*
*github.com/jordantaylorkurzweil-hash*
