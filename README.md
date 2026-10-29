# Eclipse Analytics · Group E — NBA Player Points Predictor

Predict how many points an NBA player will score in an upcoming game, using
only information available before tipoff, and serve it through an interactive
website. Main model: gradient-boosted trees (XGBoost), compared against simple
rolling-average baselines.

**Research question.** How much do opponent matchup context, usage rate, and
teammate availability improve player points predictions beyond a player's
recent scoring form?

## Quick start

```bash
pip install -r requirements.txt
python src/get_data.py          # ~80 MB, 2010-11 → 2023-24 regular seasons
python src/clean.py             # → data/processed/player_game_logs.csv, team_game_logs.csv
python src/build_features.py    # → data/processed/model_table.csv (pre-game features only)
python src/train_baseline.py    # → reports/ablation_results.csv, feature_importance.csv
```

Newer seasons (run on your own laptop; NBA.com blocks most cloud servers):

```bash
python src/fetch_recent_seasons.py --seasons 2024-25 2025-26
python src/clean.py && python src/build_features.py
```

## Data

| File | Grain | Key columns |
|---|---|---|
| `player_game_logs.csv` | player × game (DNPs kept, `played`=0) | player_id, game_id, game_date, team, opp, home, starter, min, pts, fga, fta, usg, ts, comment |
| `team_game_logs.csv` | team × game | game_id, team_id, opp, poss, pace, ortg, drtg |
| `model_table.csv` | player × game played, ≥5 career games | 23 pre-game features + target `pts` |

Source: [NocturneBear/NBA-Data-2010-2024](https://github.com/NocturneBear/NBA-Data-2010-2024)
(NBA.com box scores, MIT license). Newer seasons via [`nba_api`](https://github.com/swar/nba_api).

Derived stats: possessions = FGA + 0.44·FTA − OREB + TOV; pace = possessions per
48 min; DRtg = points allowed per 100 possessions; USG% uses the
Basketball-Reference formula.

## Feature groups (for the ablation study)

| Group | Features |
|---|---|
| FORM | pts last 5/10, season-to-date pts, minutes last 5/10, FGA/FTA/3PA last 10, starter rate, games played |
| MATCHUP | opponent DRtg and pace (last 10), own team pace, home/away, rest days, back-to-back |
| USAGE | usage rate last 5/10 and season, true shooting last 10 |
| TEAMMATE | minutes / usage / count of regular teammates (≥20 mpg) who are out |

## The one rule

**Every feature must be computable before the game starts.** Rolling stats are
`groupby(player).shift(1).rolling(...)`, so the current game never feeds its own
prediction. Splits are chronological — never random:

| Split | Seasons |
|---|---|
| train | 2015-16 → 2021-22 |
| valid | 2022-23 (early stopping, tuning) |
| test | 2023-24 (final numbers only) |

## First results (2023-24 test set, MAE in points)

| Model | MAE |
|---|---|
| Last-5 average | 4.77 |
| Last-10 average | 4.67 |
| Season-to-date average | 4.64 |
| XGBoost A — form | 4.51 |
| XGBoost B — + matchup | 4.52 |
| XGBoost C — + usage | 4.50 |
| XGBoost D — + teammates out | **4.48** |

Matchup features add nothing yet in their current form — the opponent's overall
DRtg is too blunt. Worth testing opponent defense *by position* and pace
interactions. Teammate availability gives the clearest gain.

## Layout

```
src/        get_data · clean · build_features · train_baseline · fetch_recent_seasons
data/raw/   downloaded CSVs (gitignored)
data/processed/  generated tables (gitignored)
notebooks/  EDA
reports/    results tables, charts
app/        Streamlit site (Assignment 3)
```
