"""Group E points predictor — starter website.

Run from the repo root:
    python src/train_final.py      # once, creates app/data/
    streamlit run app/app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import xgboost as xgb

DATA = Path(__file__).resolve().parent / "data"

LABELS = {
    "pts_l5": "Points, last 5 games", "pts_l10": "Points, last 10 games",
    "pts_szn": "Season scoring average", "min_l5": "Minutes, last 5", "min_l10": "Minutes, last 10",
    "fga_l10": "Shot attempts, last 10", "fta_l10": "Free throw attempts, last 10",
    "fg3a_l10": "3PA, last 10", "starter_rate_l10": "Starting lately",
    "games_played_szn": "Games played this season", "opp_drtg_l10": "Opponent defense",
    "opp_pace_l10": "Opponent pace", "team_pace_l10": "Own team pace", "home": "Home / away",
    "rest_days": "Days of rest", "b2b": "Back-to-back", "usg_l5": "Usage rate, last 5",
    "usg_l10": "Usage rate, last 10", "ts_l10": "Shooting efficiency, last 10",
    "usg_szn": "Season usage rate", "tm_out_min": "Teammates out (minutes)",
    "tm_out_usg": "Teammates out (usage)", "tm_out_count": "Teammates out (count)",
    "opp_pts_allowed_G": "Opponent pts allowed to guards", "opp_pts_allowed_F": "Opponent pts allowed to forwards",
    "opp_pts_allowed_C": "Opponent pts allowed to centers",
    "opp_pts_allowed_pos": "Opponent pts allowed to his position",
    "opp_pts_allowed_pos_rel": "Opponent vs league, his position",
}


@st.cache_resource
def load():
    meta = json.loads((DATA / "meta.json").read_text())
    booster = xgb.Booster()
    booster.load_model(DATA / "model.json")
    players = pd.read_csv(DATA / "player_state.csv")
    teams = pd.read_csv(DATA / "team_state.csv")
    return meta, booster, players, teams


def predict(meta, booster, row):
    X = pd.DataFrame([row]).reindex(columns=meta["features"])   # new features -> NaN, still runs
    dm = xgb.DMatrix(X)
    pred = float(booster.predict(dm)[0])
    contribs = booster.predict(dm, pred_contribs=True)[0][:-1]   # last value is the base score
    drivers = pd.Series(contribs, index=meta["features"])
    r = next((r for r in meta["ranges"] if r["lo"] <= pred < r["hi"]), meta["ranges"][-1])
    return pred, max(0, pred + r["q10"]), pred + r["q90"], drivers


st.set_page_config(page_title="NBA Points Predictor", page_icon="🏀", layout="centered")
meta, booster, players, teams = load()

st.title("NBA Points Predictor")
st.caption(f"Eclipse Analytics · Group E · data through {meta['data_through']} · "
           f"average miss on {meta['test_season']}: {meta['test_mae']} pts")

c1, c2 = st.columns(2)
player = c1.selectbox("Player", sorted(players.player), index=None, placeholder="Pick a player")
if not player:
    st.stop()
p = players[players.player == player].iloc[0]
opp = c2.selectbox("Opponent", sorted(teams.team[teams.team != p.team]))

c3, c4 = st.columns(2)
home = c3.radio("Where", ["Home", "Away"], horizontal=True) == "Home"
rest = c4.slider("Days of rest", 1, 5, 2)

mates = players[(players.team == p.team) & (players.player != player) & (players.min_l10 >= 20)]
out = st.multiselect("Teammates sitting out", sorted(mates.player))
out_df = mates[mates.player.isin(out)]

own, o = teams[teams.team == p.team].iloc[0], teams[teams.team == opp].iloc[0]
row = {**p.to_dict(), "opp_drtg_l10": o.drtg_l10, "opp_pace_l10": o.pace_l10,
       "team_pace_l10": own.pace_l10, "home": int(home), "rest_days": rest, "b2b": int(rest == 1),
       "tm_out_min": out_df.min_l10.sum(), "tm_out_usg": out_df.usg_l10.sum(),
       "tm_out_count": len(out_df),
       # points the opponent allows to guards / forwards / centers (+ the player's own position)
       **{f"opp_pts_allowed_{k}": o[f"pts_allowed_{k}"] for k in "GFC"},
       "opp_pts_allowed_pos": o[f"pts_allowed_{p.pos_pre}"],
       "opp_pts_allowed_pos_rel": o[f"pts_allowed_rel_{p.pos_pre}"]}

pred, lo, hi, drivers = predict(meta, booster, row)

st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Predicted points", f"{pred:.1f}")
m2.metric("Likely range (80%)", f"{lo:.0f}–{hi:.0f}")
m3.metric("Last 10 avg", f"{p.pts_l10:.1f}", f"{pred - p.pts_l10:+.1f} vs prediction", delta_color="off")

st.subheader("What's driving it")
top = drivers.reindex(drivers.abs().sort_values(ascending=False).index).head(6)
st.bar_chart(pd.DataFrame({"effect (pts)": top.values}, index=[LABELS.get(k, k) for k in top.index]),
             horizontal=True)
st.caption("Bars show how much each factor pushes the prediction up or down from the average player.")

# TODO (Quang): last-10-games chart, opponent defense comparison, model accuracy page
