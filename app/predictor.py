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

def predict_projected_score(feats):
    row = _encode(feats, _cfg["maps_inn1"])
    val = float(_score_model.predict(_frame(row, FEATS1))[0])
    return round(val, 1)

def predict_win_prob(feats):
    """Returns calibrated P(batting/chasing team wins), 0..1."""
    row = _encode(feats, _cfg["maps_inn2"])
    raw = float(_win_model.predict(_frame(row, FEATS2))[0])
    cal = float(np.interp(raw, _cal_x, _cal_y))
    return round(min(max(cal, 0.0), 1.0), 3)

def predict(feats, innings):
    """Route by innings. Returns a serving-ready dict."""
    if innings == 1:
        proj = predict_projected_score(feats)
        return {"innings": 1, "projected_score": proj,
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
