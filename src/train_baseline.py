"""Baselines + XGBoost ablation on a chronological split.

  train : 2015-16 .. 2021-22
  valid : 2022-23   (early stopping / tuning only)
  test  : 2023-24   (touch only for final numbers)

Writes reports/ablation_results.csv and reports/feature_importance.csv.
Usage:  python src/train_baseline.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from build_features import GROUPS

ROOT = Path(__file__).resolve().parents[1]
PROC, REP = ROOT / "data" / "processed", ROOT / "reports"

TRAIN = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"]
VALID, TEST = ["2022-23"], ["2023-24"]

ABLATIONS = {
    "A. form":                      ["FORM"],
    "B. form + matchup":            ["FORM", "MATCHUP"],
    "C. form + matchup + usage":    ["FORM", "MATCHUP", "USAGE"],
    "D. all (+ teammates out)":     ["FORM", "MATCHUP", "USAGE", "TEAMMATE"],
}


def fit_xgb(Xtr, ytr, Xva, yva):
    m = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                     objective="reg:absoluteerror", early_stopping_rounds=100,
                     tree_method="hist", n_jobs=-1, random_state=42)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def main():
    REP.mkdir(exist_ok=True)
    d = pd.read_csv(PROC / "model_table.csv")
    tr, va, te = (d[d.season.isin(s)] for s in (TRAIN, VALID, TEST))
    print(f"train {len(tr):,}  valid {len(va):,}  test {len(te):,}")

    rows = []
    for name, col in [("Baseline: last-5 avg", "pts_l5"), ("Baseline: last-10 avg", "pts_l10"),
                      ("Baseline: season-to-date avg", "pts_szn")]:
        pred = te[col].fillna(te["pts_l10"])
        rows.append({"model": name, "test_MAE": mean_absolute_error(te.pts, pred)})

    best = None
    for name, groups in ABLATIONS.items():
        feats = [f for g in groups for f in GROUPS[g]]
        m = fit_xgb(tr[feats], tr.pts, va[feats], va.pts)
        mae = mean_absolute_error(te.pts, m.predict(te[feats]))
        rows.append({"model": f"XGBoost {name}", "test_MAE": mae, "n_features": len(feats),
                     "trees": m.best_iteration + 1})
        print(f"{name:32s} MAE {mae:.3f}")
        best = (m, feats)

    res = pd.DataFrame(rows)
    res["test_MAE"] = res["test_MAE"].round(3)
    res.to_csv(REP / "ablation_results.csv", index=False)
    print("\n" + res.to_string(index=False))

    m, feats = best
    imp = pd.Series(m.get_booster().get_score(importance_type="gain")).reindex(feats).fillna(0)
    (imp / imp.sum()).sort_values(ascending=False).round(4).rename("gain_share") \
        .to_csv(REP / "feature_importance.csv", header=True)

    # error by player role, for the write-up
    te = te.assign(pred=m.predict(te[feats]), err=lambda x: (x.pts - x.pred).abs())
    te["role"] = pd.cut(te.pts_l10, [-1, 8, 15, 22, 100], labels=["<8", "8-15", "15-22", "22+"])
    print("\nMAE by recent scoring tier (model D):")
    print(te.groupby("role", observed=True).err.agg(["mean", "size"]).round(2).to_string())


if __name__ == "__main__":
    main()
