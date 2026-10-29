"""Pull newer seasons (2024-25 onward) from NBA.com with nba_api, saved in the
same raw format as the 2010-2024 files so clean.py picks them up unchanged.

Run on your own laptop (NBA.com often blocks cloud servers):
    pip install nba_api
    python src/fetch_recent_seasons.py --seasons 2024-25 2025-26

Limitation: LeagueGameLog only lists players who PLAYED, with no starter flag
or DNP/injury rows. So for these seasons tm_out_* and starter_rate_l10 will be
0 until someone adds a BoxScoreTraditionalV3 pass (one request per game;
~1,230 per season, sleep ~0.6s between calls). Good stretch task for Ethan.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HEADERS_TIMEOUT = 60


def fetch(season, kind):
    for attempt in range(4):
        try:
            return leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star="Regular Season",
                player_or_team_abbreviation=kind, timeout=HEADERS_TIMEOUT,
            ).get_data_frames()[0]
        except Exception as e:  # NBA.com is flaky; back off and retry
            wait = 5 * (attempt + 1)
            print(f"  {season} {kind}: {e.__class__.__name__}, retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"could not fetch {season} {kind}")


def to_box_format(df, season):
    return pd.DataFrame({
        "season_year": season,
        "game_date": df.GAME_DATE,
        "gameId": df.GAME_ID.astype(int),
        "matchup": df.MATCHUP,
        "teamId": df.TEAM_ID,
        "teamCity": "", "teamName": df.TEAM_NAME, "teamTricode": df.TEAM_ABBREVIATION,
        "teamSlug": "",
        "personId": df.PLAYER_ID, "personName": df.PLAYER_NAME,
        "position": pd.NA,            # starter info not available from this endpoint
        "comment": pd.NA, "jerseyNum": pd.NA,
        "minutes": df.MIN.astype(str),
        "fieldGoalsMade": df.FGM, "fieldGoalsAttempted": df.FGA, "fieldGoalsPercentage": df.FG_PCT,
        "threePointersMade": df.FG3M, "threePointersAttempted": df.FG3A,
        "threePointersPercentage": df.FG3_PCT,
        "freeThrowsMade": df.FTM, "freeThrowsAttempted": df.FTA, "freeThrowsPercentage": df.FT_PCT,
        "reboundsOffensive": df.OREB, "reboundsDefensive": df.DREB, "reboundsTotal": df.REB,
        "assists": df.AST, "steals": df.STL, "blocks": df.BLK, "turnovers": df.TOV,
        "foulsPersonal": df.PF, "points": df.PTS, "plusMinusPoints": df.PLUS_MINUS,
    })


def to_totals_format(df, season):
    out = df.rename(columns={"TEAM_ID": "TEAM_ID"}).copy()
    out.insert(0, "SEASON_YEAR", season)
    out["GAME_ID"] = out.GAME_ID.astype(int)
    out["MIN"] = out.MIN / 5            # this endpoint reports 240 team-minutes; source uses 48
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for s in args.seasons:
        tag = s.replace("-", "_")
        print(f"fetching {s} ...")
        players = fetch(s, "P"); time.sleep(1)
        teams = fetch(s, "T"); time.sleep(1)
        to_box_format(players, s).to_csv(RAW / f"regular_season_box_scores_{tag}.csv", index=False)
        to_totals_format(teams, s).to_csv(RAW / f"regular_season_totals_{tag}.csv", index=False)
        print(f"  {len(players):,} player rows, {len(teams):,} team rows")
    print("now run: python src/clean.py && python src/build_features.py")


if __name__ == "__main__":
    main()
