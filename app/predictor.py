"""Loads the trained models and turns a feature dict into predictions.
Win probabilities are isotonic-calibrated so they mean what they say."""
import os, json, numpy as np, pandas as pd, lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(os.path.dirname(HERE), "models")

_cfg = json.load(open(os.path.join(MODELS, "feature_config.json")))
_score_model = lgb.Booster(model_file=os.path.join(MODELS, "projected_score_lgbm.txt"))
_win_model = lgb.Booster(model_file=os.path.join(MODELS, "win_prob_lgbm.txt"))
_cal = np.load(os.path.join(MODELS, "win_calibrator.npz"))
_cal_x, _cal_y = _cal["x"], _cal["y"]

FEATS1 = _cfg["features_inn1"]
FEATS2 = _cfg["features_inn2"]
CAT = _cfg["cat_features"]

def _encode(feats, maps):
    row = dict(feats)
    for c in CAT:
        row[c] = maps[c].get(str(row.get(c)), 0)      # 0 = unseen
    return row

def _frame(feats, order):
    return pd.DataFrame([[feats[k] for k in order]], columns=order)

FULL_T20_BALLS = 120

def predict_projected_score(feats):
    row = _encode(feats, _cfg["maps_inn1"])
    val = float(_score_model.predict(_frame(row, FEATS1))[0])
    return round(val, 1)

def project_short_format(feats):
    """Rate-based projection for innings that are NOT 20 overs.

    Why not the ML model: it was trained only on 20-over IPL innings, where
    balls_bowled + balls_remaining is ALWAYS 120. A 6-over innings (36 balls)
    is a combination it never saw, and its training labels are 20-over totals,
    so it anchors to ~190 regardless of format. Using it here would be wrong.

    This projector is transparent instead: it extrapolates the team's own
    scoring rate over the balls actually left, adjusted for wickets in hand and
    end-of-innings acceleration. It works for any innings length.
    """
    runs = feats["total_runs"]
    bowled = max(feats["balls_bowled"], 1)
    left = feats["balls_remaining"]
    total = feats.get("total_balls") or FULL_T20_BALLS
    if left <= 0:
        return float(runs)

    # 1. Scoring rate: blend whole-innings rate with recent form (recent wins,
    #    because short formats swing fast). Window is scaled to the format.
    rpb_so_far = runs / bowled
    window = min(bowled, max(6, total // 4))
    recent_runs = feats.get("mom_runs_l30", 0)
    if bowled > 30:                       # momentum window caps at 30 balls
        window = min(window, 30)
    rpb_recent = recent_runs / max(min(window, 30), 1)
    base_rpb = 0.4 * rpb_so_far + 0.6 * rpb_recent if recent_runs else rpb_so_far

    # 2. Wickets in hand: fewer wickets -> the rate has to come down.
    wr = feats.get("wickets_remaining", 10)
    wkt_factor = 0.60 + 0.40 * (wr / 10.0)          # 10 wkts ->1.00, 5 ->0.80, 2 ->0.68

    # 3. Acceleration: remaining balls sit later in the innings, where teams hit
    #    harder. Mild, and scaled by how far through we already are.
    progress = bowled / total
    avg_remaining_progress = (progress + 1.0) / 2.0
    accel = 1.0 + 0.30 * avg_remaining_progress

    rpb = base_rpb * wkt_factor * accel
    proj = runs + left * rpb
    return round(max(proj, float(runs)), 1)

def predict_win_prob(feats):
    """Returns calibrated P(batting/chasing team wins), 0..1."""
    row = _encode(feats, _cfg["maps_inn2"])
    raw = float(_win_model.predict(_frame(row, FEATS2))[0])
    cal = float(np.interp(raw, _cal_x, _cal_y))
    return round(min(max(cal, 0.0), 1.0), 3)

def predict(feats, innings):
    """Route by innings. Returns a serving-ready dict."""
    if innings == 1:
        total = feats.get("total_balls") or FULL_T20_BALLS
        if total == FULL_T20_BALLS:
            proj, method = predict_projected_score(feats), "ml_model"
        else:
            proj, method = project_short_format(feats), "rate_based"
        return {"innings": 1, "projected_score": proj,
                "projection_method": method,
                "innings_balls": total,
                "current": f"{feats['total_runs']}/{feats['total_wickets']}",
                "balls_bowled": feats["balls_bowled"], "crr": feats["crr"]}
    p = predict_win_prob(feats)
    bat, bowl = feats["batting_team"], feats["bowling_team"]
    return {"innings": 2,
            "win_probability": {bat: p, bowl: round(1 - p, 3)},
            "current": f"{feats['total_runs']}/{feats['total_wickets']}",
            "target": feats.get("target"), "runs_to_win": feats.get("runs_to_win"),
            "balls_remaining": feats["balls_remaining"],
            "crr": feats["crr"], "rrr": feats.get("rrr")}
