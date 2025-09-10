# 🎾 US Open Tennis Match Outcome Prediction (2017–2023)

This project explores ATP tennis matches played at the US Open between 2017 and 2023, with a focus on comparing Novak Djokovic and Carlos Alcaraz. Using historical match data, we clean, analyze, and model match outcomes to predict winners.

---

## Project Overview

**Dataset:** `atp_tennis.csv`  
**Matches:** 761 US Open matches from 2017–2023  
**Goal:**  
- Compare player performance (Djokovic vs Alcaraz)  
- Engineer features representing relative strength  
- Train machine learning models to predict match outcomes  
- Make predictions for hypothetical matchups

---

## Data Cleaning & Preparation

- Filtered for US Open matches (2017–2023).  
- Converted `Date` to datetime format.  
- Selected relevant columns: players, rankings, ATP points, odds, and winner.  
- Created feature differences for predictive modeling:  
  - `Rank_diff = Rank_1 - Rank_2`  
  - `Pts_diff = Pts_1 - Pts_2`  
  - `Odds_diff = Odd_1 - Odd_2`  
- Target variable: `Winner == Player_1` (1 if Player 1 wins, 0 otherwise)

---

## Exploratory Data Analysis (EDA)

**Top Players:**  
- Djokovic, Nadal, Medvedev appear most frequently in matches.  

**Match Volume Over Time:**  
- Match counts consistent from 2017–2023, with a dip in 2020 due to the pandemic.  

**Player Statistics (Djokovic vs Alcaraz):**  

| Player       | Avg Rank | Avg Points | Avg Odds |
|--------------|----------|------------|----------|
| Djokovic N.  | 2.79     | 8792       | 1.09     |
| Alcaraz C.   | 25.13    | 3488       | 1.86     |

**Insights:**  
- Djokovic consistently dominated with high rankings and low odds.  
- Alcaraz shows rapid improvement from 2021 onward.  

**Win Distribution:**  
- A few top players dominate wins, but the field remains competitive.  

Visualizations include:
- Most frequent players
- Match counts per year
- Rank and odds distributions
- Win rate distribution

---

## Feature Engineering

To model match outcomes, we engineered **relative strength features**:

```python
us_open["Rank_diff"] = us_open["Rank_1"] - us_open["Rank_2"]
us_open["Pts_diff"] = us_open["Pts_1"] - us_open["Pts_2"]
us_open["Odds_diff"] = us_open["Odd_1"] - us_open["Odd_2"]
y = (us_open["Winner"] == us_open["Player_1"]).astype(int)
X = us_open[["Rank_diff", "Pts_diff", "Odds_diff"]]
