---
# ⚾ Predicting MLB Free Agent Contracts with Machine Learning Using Sabermetrics
### DATA 606
---

## Project Summary
We build models to predict MLB player contract value (e.g., Average Annual Value / salary) using player performance metrics and sabermetrics (e.g., WAR, OPS+).

### Research Questions
1. Can player performance metrics predict salary/contract value?
2. Which features (WAR, OPS+, age, HR, etc.) are most important?
3. Which model performs best (Linear/Lasso/Random Forest/Gradient Boosting)?

### Data
Sources: FanGraphs 
- `data/raw/`: original downloaded files
- `data/processed/`: cleaned/merged datasets used for modeling

### Methods (Planned)
- Data cleaning + feature engineering 
- EDA (distributions, correlations)
- Modeling: two phase modeling
- Evaluation: R² with cross-validation, MAE

### Repo Structure
- `notebooks/`: EDA + modeling notebooks
- `src/`: helper scripts (data prep, modeling)
- `reports/figures/`: plots used in slides/report
- `slides/`: presentation files

### Team
Cliff, Siddika, Michael, Sai

---

## The Problem

Every November, after the World Series has been decided, player agents and team front offices begin the ritual of MLB free agency. It's a market where teams commit hundreds of millions of dollars to players based on a mix of statistics, gut instinct, medical reports, and negotiating leverage. Get it right and you can build a team that can contend for a World Series. Get it wrong and a bad contract can handcuff a franchise for half a decade.

The question that motivated this project: **can a machine learning pipeline predict what a player will earn using only publicly available statistics?**

---

## The Data

Two datasets form the backbone of this project:

- **FanGraphs player statistics (2016–2025):** One row per player-season, covering various performance metrics including WAR, wRC+, xwOBA, Barrel%, HardHit%, defensive runs (Def), and plate appearances (PA).
- **MLB free agent contracts (2017–2026):** 2,996 rows of contracts, most of which are minor league contracts. For guaranteed major league contracts signed with MLB teams, features include AAV (Average Annual Value), contract length, position, and MLBAM player IDs for joining to the stats dataset.

**One player was excluded from all analyses:** Shohei Ohtani. He is the only player in MLB who both pitches and bats at an elite level. This makes him incomparable to any other player in the dataset, and his contract would represent a market-setting anomaly with no historical precedent. Including him would corrupt every model given he is the only player paid to pitch and bat.

---

## The Central Insight: The Market Is Not One Market

The single most important design decision in this project wasn't a modeling choice; it was recognizing that free agent contracts don't follow a single distribution.

A fringe player (WAR < 0.5) earns near the MLB minimum regardless of anything else. A starting-calibre player (WAR ≥ 2.0) earns based on positional scarcity, market competition, projected future performance, and the specific landscape of a given offseason. A player being signed to fill in situationally or to cover for an injured or resting starter does not create the market demand that a game-changing player does. Trying to fit one model across both groups would not be a wise choice.

The solution: **three separate tiers, each with its own model.**

| Tier | WAR Range | n (train) | Deployed Model |
|------|-----------|-----------|----------------|
| **Fringe** | WAR < 0.5 | ~167 | Stacked Ensemble |
| **Role** | 0.5 ≤ WAR < 2.0 | ~219 | Random Forest |
| **Upper** | WAR ≥ 2.0 | ~114 | Comp Engine |

Contract **years** are predicted by a single Global GBM across all tiers (R² = +0.465).

---

## Five Engineering Problems Worth Solving

### 1. Injury Seasons Distort Simple Averages

Bo Bichette played 628 PA in 2023 (3.88 WAR), injured his knee in 2024 and played only 336 PA (0.29 WAR), then bounced back in 2025 with 628 PA (3.82 WAR). A simple 3-year average gives him 2.66 WAR. A **PA-weighted average** — where each season is weighted by its plate appearance count — gives him 3.10 WAR. That difference moved his comp matches from a Gleyber Torres tier to a shortstop-premium tier, reducing his prediction error by millions.

Weighting by playing time means injury-shortened seasons contribute proportionally less, without using a hard cutoff that would exclude part-time players who play fewer games by design.

### 2. Teams Are Paying for Future Production, Not Past

A 28-year-old signing a 5-year deal will be 33 when it ends. A model that ignores aging is predicting last year's performance on next year's contract. This project fits a **WAR aging curve** across all player-seasons in the dataset:

```
E[WAR | age] = β₀ + β₁ · age + β₂ · age²
```

The population peak age comes out at 27.0, cwhich closely follows broader literature on athletic performance. Projected WAR feeds into both the stacked model and the comparable contract engine.

### 3. Position Matters — But Only for Stars

A shortstop who hits well is rarer than a first baseman who hits well. The data bears this out: upper-tier players in defensively-minded positions who bat well historically earn more per WAR than upper-tier players in positions with limited defensive responsibilities where batting is the focus. We add a positional scarcity index, where the median $/WAR for upper-tier contracts at each position group in each year is normalized to the league median.

Four market-meaningful segments replace the raw position strings:

| Segment | Positions |
|---------|-----------|
| **INF** | 2B, 3B, SS |
| **OF** | LF, CF, RF |
| **BAT** | 1B, DH |
| **C** | Catcher |

A key point: this premium only applies to upper-tier players. A fringe shortstop and a fringe first baseman both sign near the league minimum. Because it's unlikely that teams are bidding against each other for fringe players, there's no positional premium applied to them.

### 4. Feature Selection Prevents Overfitting on Small Samples

With 240 total features and 114 upper-tier training rows, models would likely memorize noise and overfit the training date. Therefore, a two-stage feature selection is applied within each tier:

1. **Low-variance filter:** Remove features with std < 0.01 across that tier's training rows
2. **Importance-based selection:** Fit a fast shallow GBM and keep only features with above-mean importance

This reduces feature counts from 240 to 31–38 per tier. It also implicitly addresses multicollinearity.

### 5. Algorithm Selection Based on CV Competition

For each tier, four algorithms are cross-validated against each other:

- **Ridge regression** — regularized linear, lowest variance
- **Gradient Boosting (GBM)** — strong baseline
- **Random Forest** — lower variance than GBM via bagging
- **XGBoost** — newer and faster GBM variant commonly used in sports analytics 

The winner by 5-fold CV R² on log(AAV) becomes the production model. For fringe players, Ridge wins, as the salary distribution is nearly flat and regularization matters most. For role players, Random Forest wins. For upper-tier players, XGBoost or GBM wins depending on the training run.

---

## The Comparable Contract Engine

For upper-tier players, the biggest breakthrough was leaving traditional regression models behind and using a comparable contract (comp) engine. The engine searches a pool of 2017–2025 contracts for the most statistically similar historical players using weighted Euclidean distance on 9 features:

| Feature | Weight |
|---------|--------|
| PA-weighted mean WAR | 3.0× |
| PA-weighted mean xwOBA | 2.0× |
| Mean defensive runs (Def) | 1.5× |
| Mean age | 2.0× |
| Mean Barrel%, HardHit%, OBP, ISO, BsR | 1.0× each |

For upper-tier players, projected total WAR is also calculated over the contract length. Teams are buying future wins; the aging curve projection makes comp matching forward-looking rather than purely backward-looking.

Once the 5 closest comps are found, each are adjusted for inflation and positional scarcity. Closer comps receive a higher weight.

**Why does this beat regression for star players?** With only 114 training rows, and a salary range with incredibly high variance, regression models will struggle with the non-linear relationship between statistics and market outcomes. The comp engine bypasses this issue by looking directly at what similar players actually earned.

---

## The Stacked Ensemble

For fringe-tier players, the deployed model is a stacked model where a Level 1 model learns from Level 0 predictions:

**Level 0 — Base learners:**
- Global GBM trained on all 468 MLB training contracts
- Per-tier Ridge/GBM trained on fringe contracts only
- Comp engine output

**Level 1 — Meta-learner:** Ridge regression, trained on **out-of-fold predictions** from the Level-0 models. This is the key: by generating predictions on held-out folds before training the meta-learner, no information leaks from training labels into the stacking layer. The meta-learner learns which signal to trust in which situations.

In this setup, the Level-1 Ridge model assigns weights to the Level-0 predictions in different feature spaces to make its own predictions.

---

## Results

Evaluated on 54 held-out contracts signed after the 2025 season** — players the model never saw during training:

| Model | AAV R² | AAV MAE | Yrs R² |
|-------|--------|---------|--------|
| Global GBM | +0.639 | $4.1M | +0.465 |
| Tiered (Best) | +0.642 | $4.2M | +0.365 |
| Comp Only | +0.735 | $4.8M | +0.477 |
| Stacked v10 | +0.684 | $4.0M | +0.221 |
| **★ Hybrid Deployed** | **+0.808** | **$3.4M** | **+0.465** |

The hybrid achieves both the highest R² *and* the lowest MAE by combining the strengths of each approach:
- Stacked ensemble handles fringe players efficiently (R² = +0.513)
- Random Forest handles role players, though not well (R² = +0.239)
- Comp engine handles starting players well (R² = +0.519, MAE = $8.1M)
- Global GBM handles years across all tiers


## Future Work

**Player Agent Negotiating Features**: Some MLB player agents have a reputation for maximizing contract value for their players by leveraging policies and market dynamics. Incorporating player agent representation could further improve model performance.

**Pitcher Modeling**: Pitchers make up about half of all free agent contracts and require an entirely different feature set (ERA, FIP, K%, spin rate, stuff+). Though this would require additional data collection, feature engineering and selection, and modeling, incorporating pitchers can help us better understand the overall market.

**Team Roster & Payroll Dynamics**: Teams make decisions based on their unique (and changing) roster needs and their payroll situation. A team who loses a key player in free agency may overpay another player to make up for that gap. A team that has a specific need and sees a high-level player available would likely contribute to bidding wars for them. 

---

## Project Structure

```
├── notebooks/

│   └── MLB_Contract_Prediction_Final.ipynb   # Published explanatory guide
├── data/
│   ├── mlb_contracts_9.csv                # Free agent contracts 2017–2026
│   ├── mlb_player.csv                     # FanGraphs player statistics 2016–2025
│   └── team_interest_cache.csv            # MLBTR scraper output
├── scraper/
│   └── mlb_scraper_ner.py                 # MLBTR scraper + spaCy NER pipeline
```

---

## Running the Pipeline

All notebooks are designed to run in **Google Colab**. Upload 'mlb_contracts_9.csv', 'mlb_player.csv', and 'team_interest_cache' to '/content/' before running.

**Dependencies:**
```
pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib
```

*Built with Python, scikit-learn, XGBoost, React, and an unhealthy amount of baseball knowledge.*
