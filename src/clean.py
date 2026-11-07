"""Clean raw box scores into two tidy tables.

Outputs (data/processed/):
  player_game_logs.csv  one row per player per game (DNPs kept, flagged)
  team_game_logs.csv    one row per team per game, with possessions, pace,
                        offensive/defensive rating and the opponent's numbers

All per-game advanced stats here are POST-game facts. They are only allowed
into the model after build_features.py lags them (shift(1)) per player/team.

Usage:  python src/clean.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"


def to_minutes(s):
    """'34:12' -> 34.2 ; blanks -> 0."""
    s = s.fillna("").astype(str)
    parts = s.str.split(":", expand=True)
    mins = pd.to_numeric(parts[0], errors="coerce")
    secs = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else 0
    return (mins.fillna(0) + pd.Series(secs).fillna(0).values / 60).round(2)


def possessions(fga, fta, oreb, tov):
    return fga + 0.44 * fta - oreb + tov


def clean_teams():
    t = pd.concat([pd.read_csv(f) for f in sorted(RAW.glob("regular_season_totals_*.csv"))],
                  ignore_index=True)
    t = t.drop_duplicates(["GAME_ID", "TEAM_ID"])
    t = t.rename(columns={
        "SEASON_YEAR": "season", "TEAM_ID": "team_id", "TEAM_ABBREVIATION": "team",
        "GAME_ID": "game_id", "GAME_DATE": "game_date", "MATCHUP": "matchup",
        "WL": "wl", "MIN": "game_min", "PTS": "pts", "FGA": "fga", "FTA": "fta",
        "OREB": "oreb", "TOV": "tov", "FGM": "fgm", "FG3A": "fg3a",
    })[["season", "game_id", "game_date", "team_id", "team", "matchup", "wl",
        "game_min", "pts", "fgm", "fga", "fg3a", "fta", "oreb", "tov"]]
    t["game_date"] = pd.to_datetime(t["game_date"]).dt.normalize()
    # One abbreviation per franchise: use the most recent one for each team_id
    latest = t.sort_values("game_date").groupby("team_id")["team"].last()
    print("team rows relabeled:", (t["team"] != t["team_id"].map(latest)).sum())
    t["team"] = t["team_id"].map(latest)
    t["home"] = t["matchup"].str.contains("vs.").astype(int)
    t["poss"] = possessions(t.fga, t.fta, t.oreb, t.tov)

    opp = t[["game_id", "team_id", "team", "pts", "poss"]].rename(columns={
        "team_id": "opp_team_id", "team": "opp", "pts": "opp_pts", "poss": "opp_poss"})
    t = t.merge(opp, on="game_id")
    t = t[t.team_id != t.opp_team_id].copy()

    avg_poss = (t.poss + t.opp_poss) / 2
    t["pace"] = 48 * avg_poss / t.game_min          # possessions per 48 min
    t["ortg"] = 100 * t.pts / avg_poss              # points scored per 100 poss
    t["drtg"] = 100 * t.opp_pts / avg_poss          # points allowed per 100 poss
    return t.sort_values(["game_date", "game_id", "team"]).reset_index(drop=True)


def clean_players(teams):
    files = sorted(RAW.glob("regular_season_box_scores_*.csv"))
    p = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    p = p.rename(columns={
        "season_year": "season", "gameId": "game_id", "teamId": "team_id",
        "teamTricode": "team", "personId": "player_id", "personName": "player",
        "fieldGoalsMade": "fgm", "fieldGoalsAttempted": "fga",
        "threePointersMade": "fg3m", "threePointersAttempted": "fg3a",
        "freeThrowsMade": "ftm", "freeThrowsAttempted": "fta",
        "reboundsOffensive": "oreb", "reboundsTotal": "reb", "assists": "ast",
        "turnovers": "tov", "points": "pts", "plusMinusPoints": "plus_minus",
    })
    p["game_date"] = pd.to_datetime(p["game_date"]).dt.normalize()
    p["min"] = to_minutes(p["minutes"])
    p["played"] = (p["min"] > 0).astype(int)
    p["starter"] = p["position"].notna().astype(int)   # NBA.com lists a position only for starters
    p = p.drop_duplicates(["game_id", "player_id"])
    latest = teams.drop_duplicates("team_id").set_index("team_id")["team"]
    p["team"] = p["team_id"].map(latest)
    assert p["team"].notna().all(), "player rows with a team_id not in team logs"

    # team totals for usage rate
    tm = teams[["game_id", "team_id", "game_min", "fga", "fta", "tov", "home", "opp", "opp_team_id"]]
    tm = tm.rename(columns={"fga": "tm_fga", "fta": "tm_fta", "tov": "tm_tov"})
    p = p.merge(tm, on=["game_id", "team_id"], how="left")

    # Usage rate (Basketball-Reference formula), only defined when the player played
    tm_plays = p.tm_fga + 0.44 * p.tm_fta + p.tm_tov
    p["usg"] = np.where(
        p["min"] > 0,
        100 * (p.fga + 0.44 * p.fta + p.tov) * p.game_min / (p["min"] * tm_plays),
        np.nan,
    )
    p.loc[p["min"] < 1, "usg"] = np.nan          # garbage-time noise
    tsa = 2 * (p.fga + 0.44 * p.fta)
    p["ts"] = np.where(tsa > 0, p.pts / tsa, np.nan)

    cols = ["season", "game_date", "game_id", "player_id", "player", "team_id", "team",
            "opp", "opp_team_id", "home", "starter", "played", "comment", "min",
            "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "reb", "ast",
            "tov", "plus_minus", "usg", "ts"]
    return p[cols].sort_values(["game_date", "game_id", "team", "player"]).reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    teams = clean_teams()
    players = clean_players(teams)
    teams.to_csv(OUT / "team_game_logs.csv", index=False)
    players.to_csv(OUT / "player_game_logs.csv", index=False)
    print(f"teams:   {len(teams):,} rows, {teams.game_id.nunique():,} games")
    print(f"players: {len(players):,} rows, {players.played.sum():,} played, "
          f"{players.player_id.nunique():,} players")
    print("missing opponent on player rows:", players.opp.isna().sum())


if __name__ == "__main__":
    main()
