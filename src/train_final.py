"""Train the model the website uses and export everything the app needs.

Trains XGBoost on 2015-16 .. 2022-23, measures error on 2023-24 (honest
prediction ranges), then writes small files into app/data/:

  model.json          the trained model
  meta.json           feature list, test MAE, error ranges by prediction size
  player_state.csv    each active player's form going into his NEXT game
  team_state.csv      each team's recent defense / pace going into its next game

Usage:  python src/train_final.py      (after build_features.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from build_features import GROUPS

ROOT = Path(__file__).resolve().parents[1]
PROC, APP = ROOT / "data" / "processed", ROOT / "app" / "data"

TRAIN = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"]
VALID, TEST = ["2022-23"], ["2023-24"]
FEATURES = [f for g in GROUPS.values() for f in g]


def train(d):
    tr, va, te = (d[d.season.isin(s)] for s in (TRAIN, VALID, TEST))
    m = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8,
                     colsample_bytree=0.8, min_child_weight=20, objective="reg:absoluteerror",
                     early_stopping_rounds=100, tree_method="hist", n_jobs=-1, random_state=42)
    m.fit(tr[FEATURES], tr.pts, eval_set=[(va[FEATURES], va.pts)], verbose=False)
    pred = m.predict(te[FEATURES])
    mae = mean_absolute_error(te.pts, pred)

    # 80% range: how far off the model was on 2023-24, grouped by prediction size
    res = pd.DataFrame({"pred": pred, "err": te.pts.values - pred})
    edges = [0, 5, 10, 15, 20, 25, 100]
    res["bin"] = pd.cut(res.pred, edges, right=False)
    ranges = [{"lo": float(b.left), "hi": float(b.right),
               "q10": float(g.err.quantile(0.10)), "q90": float(g.err.quantile(0.90))}
              for b, g in res.groupby("bin", observed=True)]
    return m, mae, ranges


def lastn(s, n):
    return s.tail(n).mean()


def player_state(p):
    """Rolling stats INCLUDING each player's most recent game = his form going into the next one."""
    season = p.season.max()
    g = p[(p.season == season) & (p.played == 1)].sort_values("game_date").copy()
    g.loc[g["min"] < 5, ["usg", "ts"]] = np.nan
    rows = []
    for pid, s in g.groupby("player_id"):
        if len(s) < 5:
            continue
        last = s.iloc[-1]
        rows.append({
            "player_id": pid, "player": last.player, "team": last.team, "team_id": last.team_id,
            "last_game": last.game_date.date().isoformat(),
            "pts_l5": lastn(s.pts, 5), "pts_l10": lastn(s.pts, 10), "pts_szn": s.pts.mean(),
            "min_l5": lastn(s["min"], 5), "min_l10": lastn(s["min"], 10),
            "fga_l10": lastn(s.fga, 10), "fta_l10": lastn(s.fta, 10), "fg3a_l10": lastn(s.fg3a, 10),
            "starter_rate_l10": lastn(s.starter, 10), "games_played_szn": len(s),
            "usg_l5": lastn(s.usg, 5), "usg_l10": lastn(s.usg, 10), "ts_l10": lastn(s.ts, 10),
            "usg_szn": s.usg.mean(),
        })
    return pd.DataFrame(rows).round(3)


def team_state(t):
    season = t.season.max()
    s = t[t.season == season].sort_values("game_date")
    out = s.groupby(["team_id", "team"]).agg(
        drtg_l10=("drtg", lambda x: x.tail(10).mean()),
        pace_l10=("pace", lambda x: x.tail(10).mean())).reset_index()
    return out.round(2)


def main():
    APP.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(PROC / "model_table.csv")
    m, mae, ranges = train(d)
    m.save_model(APP / "model.json")

    p = pd.read_csv(PROC / "player_game_logs.csv", parse_dates=["game_date"])
    t = pd.read_csv(PROC / "team_game_logs.csv", parse_dates=["game_date"])
    ps, ts = player_state(p), team_state(t)
    ps.to_csv(APP / "player_state.csv", index=False)
    ts.to_csv(APP / "team_state.csv", index=False)

    meta = {"features": FEATURES, "test_mae": round(float(mae), 3), "test_season": TEST[0],
            "data_through": str(p.game_date.max().date()), "ranges": ranges}
    (APP / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"test MAE {mae:.3f} | {len(ps)} players, {len(ts)} teams -> {APP}")


if __name__ == "__main__":
    main()
