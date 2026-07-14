"""Stage 2b -- train the two models with honest match-level validation.
Run:  python train/train.py
Reads the two labelled tables from /mnt/user-data/outputs (built in Stage 2a)
and writes models + encoders + metrics into ../models/.
"""
import json, os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (mean_absolute_error, log_loss, roc_auc_score,
                             accuracy_score, brier_score_loss)
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, 'models')
DATA = '/mnt/user-data/outputs'          # falls back to bundled copy in the ZIP
if not os.path.exists(os.path.join(DATA, 'train_innings1_projected_score.csv')):
    DATA = os.path.join(ROOT, 'train', 'data')
CAT = ['batting_team', 'bowling_team', 'venue', 'phase']

PARAMS = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=60,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)

def encode(df, maps=None, fit=False):
    df = df.copy()
    if fit:
        maps = {}
    for c in CAT:
        if fit:
            vals = sorted(df[c].astype(str).unique())
            maps[c] = {v: i + 1 for i, v in enumerate(vals)}   # 0 = unknown/unseen
        df[c] = df[c].astype(str).map(maps[c]).fillna(0).astype(int)
    return df, maps

def cv(df, feats, target, objective):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(df)); X = df[feats]; y = df[target].values
    for tr, va in gkf.split(X, y, df['match_no']):
        d = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=CAT)
        m = lgb.train({**PARAMS, 'objective': objective}, d, num_boost_round=400)
        oof[va] = m.predict(X.iloc[va])
    return oof, y

# ---------------- INNINGS 1: projected final score (regression) ----------------
df1 = pd.read_csv(os.path.join(DATA, 'train_innings1_projected_score.csv'))
feat1 = ['batting_team','bowling_team','venue','phase','total_runs','total_wickets',
         'wickets_remaining','balls_bowled','balls_remaining','crr','mom_runs_l30',
         'mom_wkts_l30','mom_dot_rate_l30','mom_bdry_rate_l30','partnership_runs','partnership_balls']
df1e, maps1 = encode(df1, fit=True)
oof, y = cv(df1e, feat1, 'y_final_score', 'regression_l1')
mae_overall = mean_absolute_error(y, oof)
ph = df1['phase'].values
mae_phase = {p: round(mean_absolute_error(y[ph == p], oof[ph == p]), 1)
             for p in ['Powerplay','Middle','Death']}
lgb.train({**PARAMS,'objective':'regression_l1'},
          lgb.Dataset(df1e[feat1], df1e['y_final_score'], categorical_feature=CAT),
          num_boost_round=400).save_model(os.path.join(MODELS,'projected_score_lgbm.txt'))

# ---------------- INNINGS 2: win probability (classification) ----------------
df2 = pd.read_csv(os.path.join(DATA, 'train_innings2_win_prob.csv'))
feat2 = feat1 + ['target','runs_to_win','rrr','crr_minus_rrr']
df2e, maps2 = encode(df2, fit=True)
oofp, yw = cv(df2e, feat2, 'y_win', 'binary')
acc = accuracy_score(yw, (oofp > 0.5).astype(int)); ll = log_loss(yw, oofp)
auc = roc_auc_score(yw, oofp); brier = brier_score_loss(yw, oofp)
iso = IsotonicRegression(out_of_bounds='clip').fit(oofp, yw)   # calibrate on OOF
brier_cal = brier_score_loss(yw, iso.predict(oofp))
lgb.train({**PARAMS,'objective':'binary'},
          lgb.Dataset(df2e[feat2], df2e['y_win'], categorical_feature=CAT),
          num_boost_round=400).save_model(os.path.join(MODELS,'win_prob_lgbm.txt'))
np.savez(os.path.join(MODELS,'win_calibrator.npz'), x=iso.X_thresholds_, y=iso.y_thresholds_)

cal = iso.predict(oofp); bins = np.linspace(0, 1, 11); reliab = []
for i in range(10):
    m = (cal >= bins[i]) & (cal < bins[i + 1])
    if m.sum() > 0:
        reliab.append([round((bins[i]+bins[i+1])/2,2), round(float(yw[m].mean()),3), int(m.sum())])

json.dump({'cat_features':CAT,'maps_inn1':maps1,'maps_inn2':maps2,
           'features_inn1':feat1,'features_inn2':feat2,'momentum_window':30},
          open(os.path.join(MODELS,'feature_config.json'),'w'), indent=1)

metrics = {
 'innings1_projected_score': {'n_rows': int(len(df1)), 'n_matches': int(df1.match_no.nunique()),
    'MAE_overall_runs': round(mae_overall,1), 'MAE_by_phase_runs': mae_phase,
    'note': 'Avg error of the projected final score. Large early (little info), tight at the death.'},
 'innings2_win_prob': {'n_rows': int(len(df2)), 'n_matches': int(df2.match_no.nunique()),
    'accuracy': round(acc,3), 'log_loss': round(ll,3), 'roc_auc': round(auc,3),
    'brier': round(brier,3), 'brier_calibrated': round(brier_cal,3), 'reliability_curve': reliab,
    'note': 'Match-level GroupKFold (no ball leakage). Calibrated probabilities served at inference.'}}
json.dump(metrics, open(os.path.join(MODELS,'metrics.json'),'w'), indent=1)
print(json.dumps(metrics, indent=1))
