"""Assignment 2 model work: tune XGBoost, compare a Random Forest, rerun the feature-group study.

All choices are made on the 2022-23 validation season. 2023-24 (test) is scored once per
final model at the end, never used to pick anything.

Writes:
  reports/tuning_xgb.csv         every XGBoost setting tried, with validation MAE
  reports/tuning_rf.csv          every Random Forest setting tried, with validation MAE
  reports/model_comparison.csv   baselines + best XGBoost + best Random Forest on 2023-24
  reports/ablation_final.csv     feature groups added one at a time, best XGBoost settings
  reports/best_params.json       the chosen XGBoost settings (train_final.py reads these)

Usage:  python src/tune_models.py         (after build_features.py; ~15 min on a laptop)
"""
import itertools
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from build_features import FORM, MATCHUP, POS_ALLOWED, TEAMMATE, USAGE

ROOT = Path(__file__).resolve().parents[1]
PROC, REP = ROOT / "data" / "processed", ROOT / "reports"

TRAIN = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"]
VALID, TEST = ["2022-23"], ["2023-24"]

# Ethan's position-matchup features get their own step instead of hiding inside MATCHUP
MATCHUP_BASE = [f for f in MATCHUP if f not in POS_ALLOWED]
GROUPS = {"form": FORM, "matchup": MATCHUP_BASE, "usage": USAGE,
          "teammates out": TEAMMATE, "position matchup": POS_ALLOWED}
ALL = [f for g in GROUPS.values() for f in g]

STARTER = {"max_depth": 6, "learning_rate": 0.03, "min_child_weight": 20}
XGB_GRID = {"max_depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1],
            "min_child_weight": [10, 20, 50]}
RF_GRID = {"max_depth": [12, 20], "min_samples_leaf": [20, 50], "max_features": [0.33, 0.5]}


def fit_xgb(tr, va, feats, max_depth, learning_rate, min_child_weight):
    m = XGBRegressor(n_estimators=3000, learning_rate=learning_rate, max_depth=max_depth,
                     min_child_weight=min_child_weight, subsample=0.8, colsample_bytree=0.8,
                     objective="reg:absoluteerror", early_stopping_rounds=100,
                     tree_method="hist", n_jobs=-1, random_state=42)
    m.fit(tr[feats], tr.pts, eval_set=[(va[feats], va.pts)], verbose=False)
    return m


def fit_rf(tr, feats, **kw):
    m = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42, **kw)
    m.fit(tr[feats], tr.pts)
    return m


def mae(m, df, feats):
    return mean_absolute_error(df.pts, m.predict(df[feats]))


def grid(g):
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*g.values())]


def main():
    REP.mkdir(exist_ok=True)
    d = pd.read_csv(PROC / "model_table.csv")
    tr, va, te = (d[d.season.isin(s)] for s in (TRAIN, VALID, TEST))
    print(f"train {len(tr):,}  valid {len(va):,}  test {len(te):,}  features {len(ALL)}")

    # 1. XGBoost tuning (validation season only)
    rows = []
    for prm in grid(XGB_GRID):
        t0 = time.time()
        m = fit_xgb(tr, va, ALL, **prm)
        rows.append({**prm, "valid_MAE": round(mae(m, va, ALL), 4), "trees": m.best_iteration + 1,
                     "seconds": round(time.time() - t0)})
        print("xgb", rows[-1], flush=True)
    xg = pd.DataFrame(rows).sort_values("valid_MAE")
    xg.to_csv(REP / "tuning_xgb.csv", index=False)
    best_x = xg.iloc[0][list(XGB_GRID)].to_dict()
    best_x = {"max_depth": int(best_x["max_depth"]), "learning_rate": float(best_x["learning_rate"]),
              "min_child_weight": int(best_x["min_child_weight"])}

    # 2. Random Forest tuning (validation season only)
    rows = []
    for prm in grid(RF_GRID):
        t0 = time.time()
        m = fit_rf(tr, ALL, **prm)
        rows.append({**prm, "valid_MAE": round(mae(m, va, ALL), 4), "seconds": round(time.time() - t0)})
        print("rf ", rows[-1], flush=True)
    rf = pd.DataFrame(rows).sort_values("valid_MAE")
    rf.to_csv(REP / "tuning_rf.csv", index=False)
    r0 = rf.iloc[0]
    best_r = {"max_depth": int(r0.max_depth), "min_samples_leaf": int(r0.min_samples_leaf),
              "max_features": float(r0.max_features)}

    # 3. Final scores on 2023-24 -- each model touches the test season exactly once
    res = []
    for name, col in [("Last-5 game average", "pts_l5"), ("Last-10 game average", "pts_l10"),
                      ("Season average so far", "pts_szn")]:
        res.append({"model": name, "test_MAE": mean_absolute_error(te.pts, te[col].fillna(te.pts_l10))})
    m_start = fit_xgb(tr, va, ALL, **STARTER)
    m_best = fit_xgb(tr, va, ALL, **best_x)
    m_rf = fit_rf(tr, ALL, **best_r)
    res += [
        {"model": "XGBoost, starter settings", "valid_MAE": mae(m_start, va, ALL),
         "test_MAE": mae(m_start, te, ALL), "settings": json.dumps(STARTER)},
        {"model": "XGBoost, tuned", "valid_MAE": mae(m_best, va, ALL),
         "test_MAE": mae(m_best, te, ALL), "settings": json.dumps(best_x)},
        {"model": "Random Forest, tuned", "valid_MAE": mae(m_rf, va, ALL),
         "test_MAE": mae(m_rf, te, ALL), "settings": json.dumps(best_r)},
    ]
    cmp_ = pd.DataFrame(res).round(3)
    cmp_.to_csv(REP / "model_comparison.csv", index=False)
    print("\n" + cmp_.to_string(index=False))

    # 4. Feature-group study with the tuned XGBoost settings
    rows, feats = [], []
    for i, (name, cols) in enumerate(GROUPS.items()):
        feats = feats + cols
        m = fit_xgb(tr, va, feats, **best_x)
        label = name if i == 0 else f"+ {name}"
        rows.append({"step": label, "n_features": len(feats),
                     "valid_MAE": mae(m, va, feats), "test_MAE": mae(m, te, feats)})
        print(rows[-1], flush=True)
    ab = pd.DataFrame(rows)
    ab["test_change"] = ab.test_MAE.diff()
    ab.round(3).to_csv(REP / "ablation_final.csv", index=False)
    print("\n" + ab.round(3).to_string(index=False))

    (REP / "best_params.json").write_text(json.dumps({"xgboost": best_x, "random_forest": best_r},
                                                     indent=2))


if __name__ == "__main__":
    main()
