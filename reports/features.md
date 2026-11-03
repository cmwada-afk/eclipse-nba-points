# Feature reference

Every feature is computed from games **strictly before** the one being predicted:
`groupby(player or team) → shift(1) → rolling/expanding mean`. Test: could it be
calculated the morning of the game? All rows below pass, with two caveats noted at the end.

Target: `pts` (points scored in the game). Rolling windows are over the player's previous
*played* games (player features) or the team's previous games (team features).

## FORM

| Feature | Calculation | Known pre-game because |
|---|---|---|
| `pts_l5`, `pts_l10` | mean pts, previous 5 / 10 played games | past box scores |
| `pts_szn` | mean pts, all earlier games this season | past box scores |
| `min_l5`, `min_l10` | mean minutes, previous 5 / 10 games | past box scores |
| `fga_l10`, `fta_l10`, `fg3a_l10` | mean FGA / FTA / 3PA, previous 10 games | past box scores |
| `starter_rate_l10` | share of previous 10 games started | past box scores |
| `games_played_szn` | number of games played earlier this season | count of past games |

## MATCHUP

| Feature | Calculation | Known pre-game because |
|---|---|---|
| `opp_drtg_l10` | opponent's points allowed per 100 possessions, previous 10 games (min 3) | opponent's past games |
| `opp_pace_l10` | opponent's possessions per 48 min, previous 10 games (min 3) | opponent's past games |
| `team_pace_l10` | player's own team pace, previous 10 games (min 3) | own team's past games |
| `home` | 1 if home game | schedule |
| `rest_days` | days since team's last game (cap 7; season opener = 7) | schedule |
| `b2b` | 1 if `rest_days == 1` | schedule |
| `opp_pts_allowed_G` / `_F` / `_C` | mean points scored by guards / forwards / centers against the opponent, previous 10 games (min 3) | opponent's past games |
| `opp_pts_allowed_pos` | the one of the three above that matches the player's own pre-game position (`pos_pre`) | as above |
| `opp_pts_allowed_pos_rel` | same, minus the league mean for that position over the previous 60 game days (shifted one day) | as above |

### Position rule (G / F / C)

NBA.com lists a position only for starters, so bench players need a rule.

* **Starters** are bucketed by their listed position (G, F or C) in the game being summed.
* **Bench players** inherit `pos_pre`: the most common listed position over their starts
  in the previous 40 games (current game excluded). A player with no starts in that window
  is treated as **F**.
* A player's own matching feature (`opp_pts_allowed_pos*`) always uses `pos_pre`, never the
  current game's listing.
* Points by position for a past game are summed per (game, scoring team), then attributed
  to the *defending* team, then lagged with `shift(1)` before the 10-game average.

## USAGE

| Feature | Calculation | Known pre-game because |
|---|---|---|
| `usg_l5`, `usg_l10` | mean usage rate, previous 5 / 10 games (games under 5 min ignored) | past box scores |
| `usg_szn` | mean usage rate, earlier games this season | past box scores |
| `ts_l10` | mean true shooting %, previous 10 games | past box scores |

## TEAMMATE

| Feature | Calculation | Known pre-game because |
|---|---|---|
| `tm_out_min` | summed last-10 minutes of teammates who are out and averaged ≥ 20 mpg | injury report / inactive list |
| `tm_out_usg` | summed last-10 usage of those teammates | injury report / inactive list |
| `tm_out_count` | number of such teammates out | injury report / inactive list |

## Caveats

1. **`tm_out_*`** takes who is out from the box score (`played == 0`). In practice this matches
   the pre-game injury report, but it also marks late scratches and coach's decisions made
   after the report. This is the one place the pipeline relies on an assumption.
2. **Rolling windows cross season boundaries** for team features (as before), so early-season
   values mix last season's games.
