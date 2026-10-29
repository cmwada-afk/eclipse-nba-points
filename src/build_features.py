"""Turn the cleaned logs into a modeling table with ONLY pre-game information.

Every rolling feature is computed on games strictly before the game being
predicted (groupby + shift(1) before rolling). Target = points in that game.

Feature groups (used for the ablation study):
  FORM      recent scoring / shooting / minutes
  MATCHUP   opponent defense + pace, home/away, rest
  USAGE     rolling usage rate and true shooting
  TEAMMATE  minutes and usage of regular teammates who are out (on/off proxy)

Output: data/processed/model_table.csv
Usage:  python src/build_features.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

FORM = ["pts_l5", "pts_l10", "pts_szn", "min_l5", "min_l10", "fga_l10", "fta_l10",
        "fg3a_l10", "starter_rate_l10", "games_played_szn"]
MATCHUP = ["opp_drtg_l10", "opp_pace_l10", "team_pace_l10", "home", "rest_days", "b2b"]
USAGE = ["usg_l5", "usg_l10", "ts_l10", "usg_szn"]
TEAMMATE = ["tm_out_min", "tm_out_usg", "tm_out_count"]
GROUPS = {"FORM": FORM, "MATCHUP": MATCHUP, "USAGE": USAGE, "TEAMMATE": TEAMMATE}


def lagged_roll(df, key, col, window, min_periods=1):
    """Mean of `col` over the previous `window` rows of each `key` group (excludes current row)."""
    return (df.groupby(key)[col]
              .transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean()))


def team_features(teams):
    t = teams.sort_values(["team_id", "game_date"]).copy()
    for c in ["drtg", "pace"]:
        t[f"{c}_l10"] = lagged_roll(t, "team_id", c, 10, min_periods=3)
    t["rest_days"] = t.groupby(["team_id", "season"])["game_date"].diff().dt.days.clip(upper=7)
    t["rest_days"] = t["rest_days"].fillna(7)            # season opener = fully rested
    t["b2b"] = (t["rest_days"] == 1).astype(int)
    return t[["game_id", "team_id", "drtg_l10", "pace_l10", "rest_days", "b2b"]]


def player_form(p):
    """Rolling stats over a player's previous *played* games."""
    g = p[p.played == 1].sort_values(["player_id", "game_date"]).copy()
    g.loc[g["min"] < 5, ["usg", "ts"]] = np.nan          # tiny-minute games are noise
    for col, wins in {"pts": [5, 10], "min": [5, 10], "fga": [10], "fta": [10],
                      "fg3a": [10], "usg": [5, 10], "ts": [10], "starter": [10]}.items():
        for w in wins:
            name = f"{col}_rate_l{w}" if col == "starter" else f"{col}_l{w}"
            g[name] = lagged_roll(g, "player_id", col, w)
    key = ["player_id", "season"]
    g["pts_szn"] = g.groupby(key)["pts"].transform(lambda s: s.shift(1).expanding().mean())
    g["usg_szn"] = g.groupby(key)["usg"].transform(lambda s: s.shift(1).expanding().mean())
    g["games_played_szn"] = g.groupby(key).cumcount()     # games before this one
    g["career_games"] = g.groupby("player_id").cumcount()
    return g


def teammate_out(p, form):
    """For each team-game: how much regular-rotation production is missing.

    A 'regular' is a player whose last-10 minutes were >= 20 heading into the game.
    Their status (played or not) comes from the box score's inactive/DNP list,
    which in practice is known from the injury report before tipoff.
    """
    # each player's most recent pre-game min_l10 / usg_l10, carried to games they missed
    last = form[["player_id", "game_date", "min", "usg"]].copy()
    allg = p[["game_id", "game_date", "team_id", "player_id", "played"]].sort_values("game_date")
    last = last.sort_values("game_date")
    # rolling means INCLUDING the most recent played game, as of the day before
    last["min_l10_incl"] = last.groupby("player_id")["min"].transform(lambda s: s.rolling(10, 1).mean())
    last["usg_l10_incl"] = last.groupby("player_id")["usg"].transform(lambda s: s.rolling(10, 1).mean())
    last["game_date"] = last["game_date"] + pd.Timedelta(days=1)   # known starting the next day
    m = pd.merge_asof(allg, last[["player_id", "game_date", "min_l10_incl", "usg_l10_incl"]],
                      on="game_date", by="player_id", direction="backward")
    out = m[(m.played == 0) & (m.min_l10_incl >= 20)]
    agg = out.groupby(["game_id", "team_id"]).agg(
        tm_out_min=("min_l10_incl", "sum"),
        tm_out_usg=("usg_l10_incl", "sum"),
        tm_out_count=("player_id", "size")).reset_index()
    return agg


def main():
    p = pd.read_csv(PROC / "player_game_logs.csv", parse_dates=["game_date"])
    teams = pd.read_csv(PROC / "team_game_logs.csv", parse_dates=["game_date"])

    tf = team_features(teams)
    form = player_form(p)
    tm_out = teammate_out(p, form)

    df = form.merge(tf.rename(columns={"drtg_l10": "team_drtg_l10", "pace_l10": "team_pace_l10"}),
                    on=["game_id", "team_id"], how="left")
    df = df.merge(tf[["game_id", "team_id", "drtg_l10", "pace_l10"]].rename(columns={
        "team_id": "opp_team_id", "drtg_l10": "opp_drtg_l10", "pace_l10": "opp_pace_l10"}),
        on=["game_id", "opp_team_id"], how="left")
    df = df.merge(tm_out, on=["game_id", "team_id"], how="left")
    df[TEAMMATE] = df[TEAMMATE].fillna(0)

    # keep players with some history so rolling features mean something
    df = df[df.career_games >= 5].copy()
    feats = [f for g in GROUPS.values() for f in g]
    keep = ["season", "game_date", "game_id", "player_id", "player", "team", "opp"] + feats + ["min", "pts"]
    df = df[keep].sort_values(["game_date", "game_id"]).reset_index(drop=True)
    df.to_csv(PROC / "model_table.csv", index=False)
    print(f"model table: {len(df):,} rows x {len(feats)} features")
    print("missing share per feature (top 5):")
    print(df[feats].isna().mean().sort_values(ascending=False).head(5).round(3).to_string())


if __name__ == "__main__":
    main()
