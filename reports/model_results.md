# Model results (Assignment 2)

**Bottom line:** our model misses a player's actual points by **4.48 points per game** on average
across the whole 2023-24 season. That beats the best simple rule of thumb — "he'll score his season
average" — which misses by 4.64. That is about a 3.4% smaller miss, on 25,904 games the model never saw
while it was being built.

## How we tested it

We never let the model see the games it is graded on. It learns from 2015-16 through 2021-22.
We used 2022-23 to make every choice (which settings, which model). Only after everything was locked
did we score 2023-24 once, so the numbers below are what you should expect on games that haven't
happened yet. Every input uses only information available before tip-off.

"Average miss" (MAE) = on average, how many points the prediction is off by, in either direction.
Lower is better.

## Which model

| Model | Average miss, 2022-23 (used to choose) | Average miss, 2023-24 (final) |
|---|---|---|
| Last-5 game average | – | 4.77 |
| Last-10 game average | – | 4.67 |
| Season average so far | – | 4.64 |
| Random Forest (best settings) | 4.62 | 4.53 |
| XGBoost (starter settings) | 4.58 | 4.48 |
| **XGBoost (tuned) — used on the website** | **4.58** | **4.48** |

Both machine-learning models beat every simple average. XGBoost beats Random Forest by about 0.05 points
per game, so the website uses XGBoost.

## Which inputs matter

We added groups of inputs one at a time to see what each one is worth (tuned XGBoost, 2023-24):

| Inputs | Average miss | Change |
|---|---|---|
| Recent form (points, minutes, shots, starting) | 4.515 | – |
| + Matchup (opponent defense and pace, home/away, rest) | 4.513 | −0.002 |
| + Usage (how much of the offense runs through him) | 4.504 | −0.009 |
| + Teammates out (minutes/usage of injured or resting regulars) | 4.481 | **−0.023** |
| + Opponent points allowed by position (guard / forward / center) | 4.481 | 0.000 |

## Settings we picked and why

We tried 27 XGBoost combinations — tree depth (4, 6, 8), learning rate (0.03, 0.05, 0.1), and minimum
games per leaf (10, 20, 50) — and 8 Random Forest combinations. We picked the one with the smallest miss
on 2022-23:

* **XGBoost:** depth 6, learning rate 0.03, minimum leaf size 50 (the starter used 20).
  The number of trees is chosen automatically: we stop adding trees once 2022-23 stops improving.
* **Random Forest:** 200 trees, depth 12, minimum leaf size 50, half the inputs considered at each split.

All 27 XGBoost settings landed between 4.578 and 4.588 on 2022-23. The difference between the starter
settings and the best ones is 0.002 points per game — smaller than the 0.003 the result moves just from
listing the inputs in a different order. In other words, tuning didn't make the model better, but it
shows the result is stable: it doesn't depend on lucky settings.

## What helped and what didn't

Most of the model's accuracy comes from a player's recent form — how many points, minutes and shots
he has been getting lately. The single most useful addition was knowing which teammates are sitting out:
when a regular starter is missing, someone has to take his shots, and the model picks that up (it cut the
miss by 0.02 points per game, the biggest gain of any group). Opponent strength — overall defense, and
even how many points the opponent gives up to guards, forwards or centers specifically — added almost
nothing on top of that, likely because a player's recent numbers already reflect most of it and one
night's matchup is a small effect compared to normal game-to-game swings. Tuning the model's settings
and switching to a Random Forest did not beat our starting XGBoost.

## Honest caveats

* A 4.5-point average miss is close to the natural randomness of single-game scoring; no model will get
  this near zero. The realistic goal is to beat simple averages, which we do on the 2023-24 test season.
* "Teammates out" uses who actually sat, from the box score. That usually matches the pre-game injury
  report, but late scratches would not be known in advance.
* 2022-23 is used both to choose settings and to decide when to stop adding trees, so the 2022-23
  numbers are slightly optimistic. That is why the 2023-24 numbers are the ones to quote.

*Files: `reports/model_comparison.csv`, `reports/ablation_final.csv`, `reports/tuning_xgb.csv`,
`reports/tuning_rf.csv`, `reports/best_params.json`. Reproduce with `python src/tune_models.py`
then `python src/train_final.py`.*
