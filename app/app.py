"""Basic Streamlit page: pick a player, see his last 10 games.

Run from the main eclipse-nba-points folder (after build_features.py):
    streamlit run app/app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "processed" / "model_table.csv"
COLS = ["season", "game_date", "player", "team", "opp", "home", "min", "pts", "pts_l10", "pts_szn"]

st.set_page_config(page_title="NBA Points Predictor", layout="wide")


@st.cache_data
def load():
    return pd.read_csv(TABLE, usecols=COLS, parse_dates=["game_date"])


if not TABLE.exists():
    st.error("Can't find data/processed/model_table.csv. Run get_data.py, clean.py and "
             "build_features.py first (see the README).")
    st.stop()

df = load()

st.title("NBA Points Predictor")
st.caption("Group E · Eclipse Analytics. Predictions are coming in a later assignment. "
           "For now, pick a player to see his last 10 games.")

c1, c2 = st.columns([1, 2])
seasons = sorted(df.season.unique(), reverse=True)
season = c1.selectbox("Season", seasons)

# players in that season, best scorers first (players with 10+ games ahead of short stints)
in_season = df[df.season == season]
stats = in_season.groupby("player").pts.agg(["mean", "size"])
stats["regular"] = stats["size"] >= 10
order = stats.sort_values(["regular", "mean"], ascending=False).index.tolist()
player = c2.selectbox("Player", order, index=0)

games = in_season[in_season.player == player].sort_values("game_date")
last10 = games.tail(10).copy()
last10["matchup"] = last10.apply(lambda r: ("vs " if r.home == 1 else "@ ") + r.opp, axis=1)
last10["date"] = last10.game_date.apply(lambda d: f"{d:%b} {d.day}")
last10["label"] = last10["date"] + "  " + last10["matchup"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Last 10 avg", f"{last10.pts.mean():.1f}")
m2.metric("Season avg", f"{games.pts.mean():.1f}")
m3.metric("Minutes, last 10", f"{last10['min'].mean():.1f}")
m4.metric("Team", games.team.iloc[-1])

st.subheader(f"{player}: last 10 games of {season}")
order_x = last10["label"].tolist()
bars = alt.Chart(last10).mark_bar(color="#2a78d6", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
    x=alt.X("label:N", sort=order_x, title=None, axis=alt.Axis(labelAngle=0)),
    y=alt.Y("pts:Q", title="Points"),
    tooltip=[alt.Tooltip("date:N", title="Date"), alt.Tooltip("matchup:N", title="Game"),
             alt.Tooltip("pts:Q", title="Points"), alt.Tooltip("min:Q", title="Minutes", format=".1f")],
)
# value labels sit at the bottom of each bar so they never collide with the average line
labels = alt.Chart(last10).mark_text(baseline="bottom", dy=-6, color="white", fontWeight="bold").encode(
    x=alt.X("label:N", sort=order_x), y=alt.datum(0), text="pts:Q")
avg = alt.Chart(pd.DataFrame({"avg": [last10.pts.mean()]})).mark_rule(
    color="#52514e", strokeDash=[6, 4], size=2).encode(y="avg:Q")
st.altair_chart((bars + labels + avg).properties(height=340), width="stretch")
st.caption("Dashed line = his average over these 10 games.")

table = last10[["date", "matchup", "min", "pts"]].round({"min": 1}).rename(
    columns={"date": "Date", "matchup": "Game", "min": "Minutes", "pts": "Points"})
st.dataframe(table.iloc[::-1], hide_index=True)
