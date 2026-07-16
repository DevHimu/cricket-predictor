"""Universal (format-agnostic) models + unsupervised match-state archetypes.

Why this exists: the base models were trained on absolute T20 quantities
(balls_bowled 0-120), so any other innings length is out-of-distribution.
Here every feature is NORMALIZED - progress fraction, wickets in hand,
per-ball rates - so one model applies to a 5-over gully game, a 6-over club
match, or a full T20.

Three artifacts:
  1. universal_score_lgbm.txt  - regression. Target = REMAINING runs per ball,
     not the final total. proj = current + rpb_hat * balls_left, which by
     construction can never fall below the current score and scales to any
     innings length.
  2. universal_win_lgbm.txt (+ isotonic calibrator) - chase win probability
     from normalized pressure features (required rate per ball, rate gap,
     resources), so a 6-over chase is just another point on the same scale.
  3. state_clusters.npz - UNSUPERVISED KMeans archetypes of match states, with
     the empirical outcome of each archetype (win rate for chases, remaining
     scoring rate for first innings). At inference the live state is assigned
     to its nearest archetype - "what usually happens from states like this" -
     which needs no labels and works for any format.

Run:  python train/train_universal.py
"""
import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")
DATA = "/mnt/user-data/outputs"
if not os.path.exists(os.path.join(DATA, "train_innings1_projected_score.csv")):
    DATA = os.path.join(ROOT, "train", "data")

PARAMS = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=60,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
              verbose=-1)
T20 = 120.0

# ---------------- normalized feature builders ----------------
# These mirror app/predictor.py::_universal_row exactly. Keep in sync.

U1 = ["progress", "wickets_remaining", "crr", "rpb_so_far",
      "mom_rpb_l30", "mom_wkts_l30", "mom_dot_rate_l30", "mom_bdry_rate_l30",
      "partnership_rpb"]

U2 = U1 + ["req_rpb", "rate_gap", "req_per_wkt"]


def u1_frame(df, total_balls=T20):
    f = pd.DataFrame()
    f["progress"] = df["balls_bowled"] / total_balls
    f["wickets_remaining"] = df["wickets_remaining"]
    f["crr"] = df["crr"]
    f["rpb_so_far"] = df["total_runs"] / df["balls_bowled"].clip(lower=1)
    f["mom_rpb_l30"] = df["mom_runs_l30"] / 30.0
    f["mom_wkts_l30"] = df["mom_wkts_l30"]
    f["mom_dot_rate_l30"] = df["mom_dot_rate_l30"]
    f["mom_bdry_rate_l30"] = df["mom_bdry_rate_l30"]
    f["partnership_rpb"] = df["partnership_runs"] / df["partnership_balls"].clip(lower=1)
    return f


def u2_frame(df, total_balls=T20):
    f = u1_frame(df, total_balls)
    br = df["balls_remaining"].clip(lower=1)
    f["req_rpb"] = df["runs_to_win"] / br
    f["rate_gap"] = df["crr"] / 6.0 - f["req_rpb"]
    f["req_per_wkt"] = df["runs_to_win"] / df["wickets_remaining"].clip(lower=1)
    return f


def main():
    os.makedirs(MODELS, exist_ok=True)
    meta = {}

    # ============ 1. UNIVERSAL PROJECTED SCORE (remaining rpb) ============
    d1 = pd.read_csv(os.path.join(DATA, "train_innings1_projected_score.csv"))
    d1 = d1[d1["balls_remaining"] > 0].reset_index(drop=True)
    X1 = u1_frame(d1)
    y1 = (d1["y_final_score"] - d1["total_runs"]) / d1["balls_remaining"]

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(d1))
    for tr, va in gkf.split(X1, y1, d1["match_no"]):
        m = lgb.train({**PARAMS, "objective": "l1"},
                      lgb.Dataset(X1.iloc[tr], y1.iloc[tr]), num_boost_round=400)
        oof[va] = m.predict(X1.iloc[va])
    proj_oof = d1["total_runs"] + np.clip(oof, 0, None) * d1["balls_remaining"]
    mae = float(mean_absolute_error(d1["y_final_score"], proj_oof))
    meta["universal_score"] = {"oof_mae_20over": round(mae, 2),
                               "target": "remaining_runs_per_ball",
                               "features": U1}
    lgb.train({**PARAMS, "objective": "l1"}, lgb.Dataset(X1, y1),
              num_boost_round=400).save_model(
        os.path.join(MODELS, "universal_score_lgbm.txt"))
    print(f"universal score model: OOF MAE (20-over) {mae:.2f}")

    # ============ 2. UNIVERSAL WIN PROBABILITY ============
    d2 = pd.read_csv(os.path.join(DATA, "train_innings2_win_prob.csv"))
    d2 = d2[d2["balls_remaining"] > 0].reset_index(drop=True)
    X2 = u2_frame(d2)
    y2 = d2["y_win"].values

    oof2 = np.zeros(len(d2))
    for tr, va in gkf.split(X2, y2, d2["match_no"]):
        m = lgb.train({**PARAMS, "objective": "binary"},
                      lgb.Dataset(X2.iloc[tr], y2[tr]), num_boost_round=400)
        oof2[va] = m.predict(X2.iloc[va])
    auc = float(roc_auc_score(y2, oof2))
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof2, y2)
    brier_raw = float(brier_score_loss(y2, oof2))
    brier_cal = float(brier_score_loss(y2, iso.predict(oof2)))
    meta["universal_win"] = {"oof_auc_20over": round(auc, 3),
                             "brier_raw": round(brier_raw, 3),
                             "brier_calibrated": round(brier_cal, 3),
                             "features": U2}
    lgb.train({**PARAMS, "objective": "binary"}, lgb.Dataset(X2, y2),
              num_boost_round=400).save_model(
        os.path.join(MODELS, "universal_win_lgbm.txt"))
    np.savez(os.path.join(MODELS, "universal_win_calibrator.npz"),
             x=iso.X_thresholds_, y=iso.y_thresholds_)
    print(f"universal win model: OOF AUC {auc:.3f}, Brier {brier_raw:.3f} -> {brier_cal:.3f}")

    # ============ 3. UNSUPERVISED MATCH-STATE ARCHETYPES ============
    # Chase archetypes: cluster normalized second-innings states. No labels are
    # used to FIT the clusters; the empirical win rate per cluster is attached
    # afterwards as "what historically happened from states like this".
    kk = 8
    Z2 = X2[["progress", "wickets_remaining", "req_rpb", "rate_gap",
             "mom_rpb_l30", "mom_dot_rate_l30"]].values.astype(float)
    mu2, sd2 = Z2.mean(0), Z2.std(0) + 1e-9
    km2 = KMeans(n_clusters=kk, n_init=10, random_state=42).fit((Z2 - mu2) / sd2)
    lab2 = km2.labels_
    stats2 = []
    for c in range(kk):
        mask = lab2 == c
        cen = Z2[mask].mean(0)
        stats2.append({"n": int(mask.sum()),
                       "win_rate": round(float(y2[mask].mean()), 3),
                       "centroid": [round(float(v), 3) for v in cen]})

    def name_chase(cen, wr):
        prog, wkts, req, gap, mom, dots = cen
        if wr >= 0.85: return "Cruising chase"
        if wr <= 0.15: return "Lost cause"
        if gap < -0.4 and wkts <= 5: return "Collapse under pressure"
        if gap < -0.25: return "Falling behind the rate"
        if gap > 0.15 and wkts >= 7: return "In control"
        if dots > 0.45: return "Pressure building"
        if prog > 0.75: return "Tense finish"
        return "Evenly poised"
    for s in stats2:
        s["label"] = name_chase(s["centroid"], s["win_rate"])

    # First-innings archetypes: attached outcome = remaining scoring rate.
    Z1 = X1[["progress", "wickets_remaining", "rpb_so_far",
             "mom_rpb_l30", "mom_dot_rate_l30", "mom_bdry_rate_l30"]].values.astype(float)
    mu1, sd1 = Z1.mean(0), Z1.std(0) + 1e-9
    km1 = KMeans(n_clusters=kk, n_init=10, random_state=42).fit((Z1 - mu1) / sd1)
    lab1 = km1.labels_
    y1v = y1.values
    stats1 = []
    for c in range(kk):
        mask = lab1 == c
        cen = Z1[mask].mean(0)
        stats1.append({"n": int(mask.sum()),
                       "rem_rpb": round(float(y1v[mask].mean()), 4),
                       "centroid": [round(float(v), 3) for v in cen]})

    # Name batting archetypes RELATIVE to each other (quantiles), not by absolute
    # thresholds - remaining rpb in T20 clusters tightly around ~1.4, so absolute
    # cutoffs collapse every cluster into one or two names.
    import statistics as _st
    rpbs = sorted(x["rem_rpb"] for x in stats1)
    dotsv = sorted(x["centroid"][4] for x in stats1)
    wktsv = sorted(x["centroid"][1] for x in stats1)
    def q(vals, v):
        return sum(1 for x in vals if x <= v) / len(vals)
    def name_bat(cen, rpb):
        prog, wkts, sofar, mom, dots, bdry = cen
        if q(wktsv, wkts) <= 0.25 and q(dotsv, dots) >= 0.6:
            return "Rebuilding after losses"
        if q(rpbs, rpb) >= 0.85: return "Launch mode"
        if q(rpbs, rpb) >= 0.6:  return "Accelerating"
        if q(dotsv, dots) >= 0.85: return "Bogged down"
        if prog < 0.3: return "Setting a platform"
        if q(rpbs, rpb) <= 0.25: return "Grinding it out"
        return "Steady accumulation"
    for s in stats1:
        s["label"] = name_bat(s["centroid"], s["rem_rpb"])

    np.savez(os.path.join(MODELS, "state_clusters.npz"),
             c1=km2.cluster_centers_, mu2=mu2, sd2=sd2,
             c0=km1.cluster_centers_, mu1=mu1, sd1=sd1)
    with open(os.path.join(MODELS, "state_clusters.json"), "w") as fh:
        json.dump({"chase": stats2, "batting": stats1,
                   "chase_features": ["progress", "wickets_remaining", "req_rpb",
                                      "rate_gap", "mom_rpb_l30", "mom_dot_rate_l30"],
                   "batting_features": ["progress", "wickets_remaining", "rpb_so_far",
                                        "mom_rpb_l30", "mom_dot_rate_l30",
                                        "mom_bdry_rate_l30"]}, fh, indent=1)
    print("archetypes: chase", [s["label"] for s in stats2])
    print("archetypes: batting", [s["label"] for s in stats1])

    with open(os.path.join(MODELS, "universal_metrics.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("saved to", MODELS)


if __name__ == "__main__":
    main()
