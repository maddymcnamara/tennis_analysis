# ATP Match Outcome Prediction

This project builds a predictive model for ATP tennis match outcomes, using a from-scratch Elo rating system, engineered form and head-to-head features, and multiple machine learning models — with a focus case study comparing Novak Djokovic and Carlos Alcaraz. Using full historical ATP match data (not just a single tournament slice), the pipeline cleans, engineers features for, models, and evaluates match outcomes against real betting-market baselines.

## Project Overview

**Dataset:** `atp_tennis.csv` — full multi-year ATP match history

**Goal:**
- Build a chronological, no-lookahead Elo rating system (overall + surface-specific)
- Engineer features representing relative strength: Elo, recent form, head-to-head record, and de-vigged implied win probability from betting odds
- Train and compare multiple ML models against real baselines (betting favorite, higher-Elo player)
- Evaluate not just accuracy but probability calibration (log-loss, Brier score)
- Predict outcomes for hypothetical matchups, including Djokovic vs. Alcaraz, using each player's live Elo/form snapshot

## Data Cleaning & Preparation

- Loaded the full ATP match dataset (not filtered to a single tournament/date range), so Elo and form features have enough history to be meaningful.
- Converted `Date` to datetime and sorted matches chronologically — critical, since every downstream feature depends on match order.
- Dropped rows with missing rankings or invalid odds (≤ 1).
- **Reframed the prediction target:** instead of an arbitrary `Player_1` vs. `Player_2` split, every match is reframed around the **market favorite** (lower odds) vs. the **underdog**. The target becomes `Favorite_won`, removing a source of noise the original column ordering introduced.

## What Is an Elo Rating? (and why it beats a static rank/seed)

Elo is a single number representing _how good is this player right now_, originally developed for ranking chess players and now used across chess, tennis, and esports matchmaking. Every player starts at a baseline rating (1500 here), and it updates after every match:

1. **Before a match**, the ratings gap predicts an expected outcome via a logistic formula:
   `Expected_A = 1 / (1 + 10^((R_B - R_A)/400))` — a 200-point rating edge translates to roughly a 76% win probability.
2. **After the match**, both ratings shift based on the gap between what actually happened and what was expected:
   `R_new = R_old + K * (actual - expected)`.
   An expected result barely moves the ratings; an upset moves them a lot, since it's more informative.

**A seed is a static snapshot used only to arrange a bracket. It doesn't account for *who* you beat, just *that* you won a given round. Elo, by contrast:
- Updates immediately after every match, not once per ranking cycle
- Rewards beating strong opponents more than weak ones (and penalizes losing to weak opponents more)
- Outputs an actual win probability, not just a relative order
- Can be computed **per surface** — this project tracks separate Hard/Clay/Grass Elo ratings, since a player's strength (e.g., Alcaraz on clay vs. Djokovic on hard) varies significantly by surface in a way a single rank number never captures

## Feature Engineering

Built a set of features that carry real predictive signal, computed causally (only using information available *before* each match):

- **`Elo_diff`** — overall Elo rating gap, from the custom Elo system above (K=32, starting rating 1500)
- **`SurfElo_diff`** — surface-specific Elo gap (Hard/Clay/Grass)
- **`Form_diff`** — rolling win rate over each player's last 10 matches
- **`H2H_fav_rate`** — head-to-head win rate between the two specific players, before the current match
- **`ImpProb_fav`** — de-vigged implied win probability from betting odds (`1/Odd`, normalized to remove the bookmaker's margin)
- **`Rank_diff`**, **`Pts_diff`** — ranking and ATP points gap, retained from the original approach

```python
matches["Elo_diff"] = matches["Elo_fav"] - matches["Elo_dog"]
matches["SurfElo_diff"] = matches["SurfElo_fav"] - matches["SurfElo_dog"]
matches["Form_diff"] = matches["Form_fav"] - matches["Form_dog"]
matches["ImpProb_fav"] = matches["ImpProb_fav_raw"] / (matches["ImpProb_fav_raw"] + matches["ImpProb_dog_raw"])
y = matches["Favorite_won"]
X = matches[["Elo_diff", "SurfElo_diff", "Form_diff", "H2H_fav_rate", "ImpProb_fav", "Rank_diff", "Pts_diff"]]
```

## Exploratory Data Analysis (EDA)

- **Feature separation check:** boxplots confirm `Elo_diff`, `SurfElo_diff`, and `ImpProb_fav` are visibly higher for matches the favorite won vs. lost — validating that the engineered features actually carry signal before any model is trained.
- **Player snapshot (Djokovic vs. Alcaraz):** Elo and surface Elo trajectories show Djokovic's sustained dominance across surfaces versus Alcaraz's rapid post-2021 rise, particularly on hard and clay courts.
- Visualizations include Elo-vs-outcome boxplots, feature importance rankings, and a model calibration curve.

## Modeling & Evaluation

- **Time-based train/test split** — trained on earlier matches, tested only on the most recent ~15% by date, avoiding the lookahead bias of a random split on time-ordered data.
- **Baselines established first:** always-pick-the-favorite (by odds) and always-pick-the-higher-Elo-player, so every model's accuracy is judged against a real reference point rather than in isolation.
- **Models trained:** Logistic Regression, Random Forest, and K-Nearest Neighbors, validated with 5-fold cross-validation before touching the test set.
- **Metrics reported:** accuracy, log-loss, Brier score, and ROC-AUC — because for a probabilistic sports prediction, how well-calibrated the predicted probabilities are matters as much as the raw win/loss call.
- **Feature importance** (Random Forest) and a **calibration curve** are used to sanity-check that Elo and implied probability are driving predictions, and that predicted probabilities track real-world outcome rates.

## Predicting Hypothetical Matchups

A generalized `predict_matchup()` function pulls each player's most recent Elo, surface Elo, recent form, and head-to-head record directly from the fitted pipeline — rather than relying on hand-typed historical averages — and outputs a win probability for any two players on any surface.

```python
predict_matchup("Djokovic N.", "Alcaraz C.", surface="Hard", model_name="Logistic Regression")
```

## Key Takeaways

- Elo built from scratch with no lookahead was the single biggest driver of improved predictive signal over raw rank/points/odds differences.
- Benchmarking against the betting-favorite baseline turns a standalone accuracy number into an actual measure of skill.
- Reporting log-loss and calibration alongside accuracy reflects that match prediction is fundamentally a probability estimation problem, not just binary classification.

**Next steps:** incorporate set/game-level scoring margins, add fatigue features (days since last match, tournament round), and test gradient-boosted models (XGBoost/LightGBM) with time-series cross-validation.
