"""Single source of truth for features. The SAME function builds features for
training data and for live serving, which is what prevents train/serve skew.

A "ball" is a dict with:
    runs_of_bat : int   runs off the bat
    extras      : int   total extra runs on the delivery
    is_wicket   : 0/1   a valid dismissal happened
    is_legal    : 0/1   1 unless the delivery was a wide or no-ball
Derived internally: runs_this_ball = runs_of_bat + extras,
                    is_dot = runs_this_ball == 0,
                    is_boundary = runs_of_bat in (4, 6).
"""
MOMENTUM_WINDOW = 30           # deliveries (~5 overs)
INNINGS_BALLS = 120            # 20 overs

def phase_of(balls_bowled: int) -> str:
    if balls_bowled <= 0:
        return "Powerplay"
    over = (balls_bowled - 1) // 6 + 1     # 1..N
    if over <= 6:
        return "Powerplay"
    if over <= 15:
        return "Middle"
    return "Death"

def _rt(b):   # runs this ball
    return int(b["runs_of_bat"]) + int(b["extras"])

def build_features(timeline, *, innings, batting_team, bowling_team,
                   venue=None, target=None):
    """Build the feature dict for the CURRENT (latest) ball of an innings.

    timeline : ordered list of ball dicts for THIS innings up to and incl. now.
    Returns a flat dict whose keys are exactly the model's feature names
    (plus the chase-only keys when innings == 2).
    """
    if not timeline:
        raise ValueError("timeline is empty")

    total_runs = sum(_rt(b) for b in timeline)
    total_wickets = sum(int(b["is_wicket"]) for b in timeline)
    balls_bowled = sum(int(b["is_legal"]) for b in timeline)
    balls_remaining = max(INNINGS_BALLS - balls_bowled, 0)
    wickets_remaining = 10 - total_wickets
    crr = round(total_runs / (balls_bowled / 6), 2) if balls_bowled > 0 else 0.0

    win = timeline[-MOMENTUM_WINDOW:]
    n = len(win)
    mom_runs = sum(_rt(b) for b in win)
    mom_wkts = sum(int(b["is_wicket"]) for b in win)
    mom_dot = round(sum(1 for b in win if _rt(b) == 0) / n, 3)
    mom_bdry = round(sum(1 for b in win if int(b["runs_of_bat"]) in (4, 6)) / n, 3)

    # partnership: runs/balls since the last wicket, INCLUDING a wicket ball itself
    pr = pb = 0
    for b in timeline:
        pr += _rt(b); pb += 1
        if int(b["is_wicket"]) == 1:
            last_pr, last_pb = pr, pb
            pr = pb = 0
    partnership_runs = pr if timeline[-1]["is_wicket"] == 0 else last_pr
    partnership_balls = pb if timeline[-1]["is_wicket"] == 0 else last_pb

    feats = {
        "batting_team": batting_team, "bowling_team": bowling_team,
        "venue": venue if venue else "UNKNOWN", "phase": phase_of(balls_bowled),
        "total_runs": total_runs, "total_wickets": total_wickets,
        "wickets_remaining": wickets_remaining,
        "balls_bowled": balls_bowled, "balls_remaining": balls_remaining, "crr": crr,
        "mom_runs_l30": mom_runs, "mom_wkts_l30": mom_wkts,
        "mom_dot_rate_l30": mom_dot, "mom_bdry_rate_l30": mom_bdry,
        "partnership_runs": partnership_runs, "partnership_balls": partnership_balls,
    }
    if innings == 2:
        tg = int(target) if target else 0
        rtw = max(tg - total_runs, 0)
        rrr = round(rtw / (balls_remaining / 6), 2) if balls_remaining > 0 else 0.0
        feats.update(target=tg, runs_to_win=rtw, rrr=rrr,
                     crr_minus_rrr=round(crr - rrr, 2))
    return feats
