import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data"
TIERS = ["Bench", "Role player", "Starter", "Star"]

st.title("How accurate is it?")


@st.cache_data
def load():
    d = pd.read_csv(DATA / "test_predictions.csv")
    meta = json.loads((DATA / "meta.json").read_text())
    return d, meta


d, meta = load()
d["miss"] = (d.pts - d.pred).abs()
d["base_miss"] = (d.pts - d.pts_szn).abs()
# player type = pre-game season scoring average
d["type"] = pd.cut(d.pts_szn, [-1, 8, 15, 22, 100], labels=TIERS)

st.caption(f"Tested on {meta['test_season']} season")

# 1. Predicted vs actual
st.subheader("Predicted vs actual points")
s = d.sample(min(len(d), 4000), random_state=0)
dots = alt.Chart(s).mark_circle(size=20, opacity=0.25).encode(
    x=alt.X("pred:Q", title="Predicted points"),
    y=alt.Y("pts:Q", title="Actual points"),
    tooltip=[c for c in ["player", "game_date", "pred", "pts"] if c in s.columns])
diag = alt.Chart(pd.DataFrame({"v": [0, 60]})).mark_line(color="red", strokeDash=[4, 4]).encode(x="v:Q", y="v:Q")
st.altair_chart(dots + diag, use_container_width=True)
st.caption("Dashed line = perfect prediction. Random sample of 4,000 games shown. "
           "Scoring has a lot of game-to-game noise, so even good predictions scatter widely.")

# 2. Average miss by player type
st.subheader("Average miss by player type")
bt = (d.groupby("type", observed=True)
        .agg(games=("pts", "size"), model_miss=("miss", "mean"), season_avg_miss=("base_miss", "mean"))
        .reindex(TIERS).reset_index().round(2))
st.altair_chart(alt.Chart(bt).mark_bar().encode(
    x=alt.X("type:N", sort=TIERS, title=None),
    y=alt.Y("model_miss:Q", title="Average miss (points)"),
    tooltip=["type", "games", "model_miss", "season_avg_miss"]), use_container_width=True)
st.dataframe(bt, hide_index=True)
st.caption("Bench = under 8 ppg, Role player = 8-15, Starter = 15-22, Star = 22+ (season average before the game).")

# 3. vs season average
st.subheader("Compared to just using the season average")
m_mae, b_mae = d.miss.mean(), d.dropna(subset=["pts_szn"]).base_miss.mean()
c1, c2, c3 = st.columns(3)
c1.metric("Season average", f"{b_mae:.2f} pts")
c2.metric("Our model", f"{m_mae:.2f} pts", f"{m_mae - b_mae:+.2f}", delta_color="inverse")
c3.metric("Improvement", f"{(b_mae - m_mae) / b_mae:.1%}")
st.write(f"On average our prediction is off by {m_mae:.2f} points, "
         f"a {(b_mae - m_mae) / b_mae:.1%} improvement from just using "
         f"the player's season average.")
