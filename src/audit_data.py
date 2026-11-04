import sys
import pandas as pd

D = "data/processed/"
P = sys.argv[1] if len(sys.argv) > 1 else D + "player_game_logs.csv"
T = sys.argv[2] if len(sys.argv) > 2 else D + "team_game_logs.csv"
M = sys.argv[3] if len(sys.argv) > 3 else D + "model_table.csv"

def show(title, obj, n=None):
    print(f"\n## {title}\n")
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if n is not None:
        obj = obj.head(n)
    print(obj.to_markdown() if len(obj) else "_none_")

def count(label, n):
    return pd.Series({label: int(n)})

df = pd.read_csv(P, parse_dates=["game_date"], dtype={"game_id": str})
df["game_id"] = df["game_id"].str.zfill(10)
played = df[df["played"] == 1]
print(f"# Raw audit output\n\nplayer logs: {df.shape}, "
      f"{df.game_date.min().date()} to {df.game_date.max().date()}")

# ---- 1. Rows per season ----
g = df.groupby("season")
show("Rows per season", pd.DataFrame({
    "rows": g.size(), "played": g["played"].sum(),
    "dnp": g["played"].apply(lambda s: (s == 0).sum()),
    "games": g["game_id"].nunique(), "players": g["player_id"].nunique()}))

# season label vs game_id (chars 3-4 = start year) and game type (char 2: 2 = regular season)
yr = 2000 + df["game_id"].str[3:5].astype(int)
mismatch = df[df["season"].astype(str).str[:4].astype(int) != yr]
show("Season label disagrees with game_id start year", count("rows", len(mismatch)))
show("Game type code (2 = regular season)", df["game_id"].str[2].value_counts())

# ---- 2. Missing values ----
miss = pd.DataFrame({"missing": df.isna().sum(),
                     "pct": (df.isna().mean() * 100).round(2),
                     "pct_among_played": (played.isna().mean() * 100).round(2)})
show("Missing values", miss[miss.missing > 0].sort_values("missing", ascending=False))
show("Top DNP comments", df.loc[df.played == 0, "comment"].value_counts(dropna=False), 15)
show("Missing starter/min among played, by season",
     played.assign(no_min=played["min"].isna(), no_usg=played["usg"].isna())
           .groupby("season")[["no_min", "no_usg"]].sum())

# ---- 3. Duplicates ----
show("Fully duplicated rows", count("count", df.duplicated().sum()))
dk = df[df.duplicated(["player_id", "game_id"], keep=False)]
show("Duplicate (player_id, game_id) rows", dk.sort_values(["player_id", "game_id"]), 20)
sd = df.groupby(["player_id", "game_date"])["team_id"].nunique()
show("Same player, same DATE, 2+ team_ids (impossible)", sd[sd > 1].rename("n_teams"), 20)
sg = df.groupby(["player_id", "game_id"])["team_id"].nunique()
show("Same player, same GAME, 2+ team_ids (impossible)", sg[sg > 1].rename("n_teams"), 20)

# ---- 4. Weird values ----
mx = played["usg"].max()
thr = 0.60 if mx <= 1.5 else 60
print(f"\nusg max = {mx} -> using threshold {thr}")
hi = played[played["usg"] > thr]
cols = ["season", "game_date", "player", "team", "min", "pts", "fga", "fta", "tov", "usg"]
show("Usage > 60%: count by season", hi.groupby("season").size().rename("rows"))
show("Usage > 60%: minutes distribution", hi["min"].describe())
show("Usage > 60% with >= 10 min (suspicious)",
     hi[hi["min"] >= 10][cols].sort_values("usg", ascending=False), 25)
show("Usage > 60% top 15", hi[cols].sort_values("usg", ascending=False), 15)
show("Usage > 40% with >= 20 min (real-but-extreme)",
     played[(played.usg > thr * 40 / 60) & (played["min"] >= 20)][cols]
     .sort_values("usg", ascending=False), 10)

tsmax = 1.5 if played["ts"].max() <= 3 else 150
show("Impossible / inconsistent values", pd.Series({
    "played=1, min missing or <=0": ((played["min"].isna()) | (played["min"] <= 0)).sum(),
    "min > 60": (played["min"] > 60).sum(),
    "negative pts/fga/usg": ((played[["pts", "fga", "usg"]] < 0).any(axis=1)).sum(),
    "fgm > fga": (played["fgm"] > played["fga"]).sum(),
    "fg3m > fg3a": (played["fg3m"] > played["fg3a"]).sum(),
    "ftm > fta": (played["ftm"] > played["fta"]).sum(),
    "fg3m > fgm": (played["fg3m"] > played["fgm"]).sum(),
    "pts != 2*fgm + fg3m + ftm": (played["pts"] != 2*played["fgm"] + played["fg3m"] + played["ftm"]).sum(),
    "reb != oreb+dreb impossible (oreb > reb)": (played["oreb"] > played["reb"]).sum(),
    "ts > cap": (played["ts"] > tsmax).sum(),
    "played=0 but min > 0": ((df.played == 0) & (df["min"] > 0)).sum(),
    "played=0 but pts > 0": ((df.played == 0) & (df["pts"] > 0)).sum(),
}, name="rows"))
show("Points distribution (played)", played["pts"].describe())
show("Minutes distribution (played)", played["min"].describe())

# ---- 5. Trades ----
tp = df.groupby(["player_id", "season"])["team_id"].nunique()
tr = tp[tp > 1]
show("Players with 2+ teams in a season (legit trades) by season", tr.groupby("season").size().rename("players"))
show("Most teams in one season", tr.rename("n_teams").sort_values(ascending=False), 8)
# team abbreviation -> team_id consistency (relocations, e.g. NJN->BKN)
ab = df.groupby("team_id")["team"].nunique()
show("team_ids with multiple abbreviations", ab[ab > 1].rename("n_abbrevs"))
show("team abbreviations spanning multiple team_ids", df.groupby("team")["team_id"].nunique().loc[lambda s: s > 1])

# ---- 6. Game / team-level consistency ----
pg = df.groupby("game_id")["team_id"].nunique()
show("Games without exactly 2 teams in player logs", pg[pg != 2].rename("n_teams"), 20)
game_teams = pd.MultiIndex.from_frame(df[["game_id", "team_id"]].drop_duplicates())
opp_pairs = pd.MultiIndex.from_frame(df[["game_id", "opp_team_id"]])
bad_opp = (~opp_pairs.isin(game_teams)) | (df["opp_team_id"] == df["team_id"])
show("opp_team_id isn't the other team in the game", count("rows", bad_opp.sum()))

t = pd.read_csv(T, parse_dates=["game_date"], dtype={"game_id": str})
t["game_id"] = t["game_id"].str.zfill(10)
print(f"\nteam logs: {t.shape}")
show("Team logs: games per season", t.groupby("season")["game_id"].nunique())
tg = t.groupby("game_id").size()
show("Team logs: games without exactly 2 rows", tg[tg != 2].rename("rows"), 20)
show("Team logs: dup (game_id, team_id)", count("count", t.duplicated(["game_id", "team_id"]).sum()))
show("Team logs: missing", t.isna().sum().rename("missing").loc[lambda s: s > 0])
show("Team logs: pace/ortg/drtg/poss", t[["poss", "pace", "ortg", "drtg"]].describe())
show("Games in player logs but not team logs (and vice versa)", pd.Series({
    "player_only": len(set(df.game_id) - set(t.game_id)),
    "team_only": len(set(t.game_id) - set(df.game_id))}))

# Reconcile: sum of player points per team-game vs team points
agg = df.groupby(["game_id", "team_id"]).agg(p_pts=("pts", "sum"), p_fga=("fga", "sum")).reset_index()
rec = agg.merge(t[["game_id", "team_id", "pts", "fga", "season"]], on=["game_id", "team_id"], how="outer", indicator=True)
show("Reconciliation merge result", rec["_merge"].value_counts())
both = rec[rec["_merge"] == "both"]
bad = both[(both.p_pts != both.pts) | (both.p_fga != both.fga)]
show("Team-games where player sums != team totals (pts or fga)", bad.groupby("season").size().rename("team_games"))
show("Examples", bad.head(15))

# ---- 7. Model table ----
m = pd.read_csv(M, parse_dates=["game_date"], dtype={"game_id": str})
m["game_id"] = m["game_id"].str.zfill(10)
print(f"\nmodel_table: {m.shape}")
show("Model table: rows per season", m.groupby("season").size().rename("rows"))
show("Model table: missing", pd.DataFrame({"missing": m.isna().sum(), "pct": (m.isna().mean()*100).round(2)})
     .loc[lambda x: x.missing > 0])
show("Model table: dup (player_id, game_id)", count("count", m.duplicated(["player_id", "game_id"]).sum()))
show("Model table rows not found among played=1 player-games", count("rows", (
    ~m.set_index(["player_id", "game_id"]).index.isin(played.set_index(["player_id", "game_id"]).index)).sum()))
show("Model table: rows with games_played_szn < 1 or min_l5 missing", count("rows", (
    (m.games_played_szn < 1) | m.min_l5.isna()).sum()))
show("Model table: feature summary", m.drop(columns=["season", "game_date", "game_id", "player_id", "player", "team", "opp"])
     .describe().T[["count", "min", "50%", "max"]])
