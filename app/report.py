"""Post-match analysis report. Consumes both innings' ball timelines and the
model, and produces a structured report + a readable text summary."""
from feature_builder import build_features, phase_of, _rt
import predictor
import analytics

def _phase_split(timeline):
    out = {"Powerplay": {"runs": 0, "wkts": 0}, "Middle": {"runs": 0, "wkts": 0},
           "Death": {"runs": 0, "wkts": 0}}
    bb = 0
    for b in timeline:
        bb += int(b["is_legal"])
        ph = phase_of(bb)
        out[ph]["runs"] += _rt(b); out[ph]["wkts"] += int(b["is_wicket"])
    return out

def _top_partnerships(timeline, k=3):
    parts = []; pr = pb = 0
    for b in timeline:
        pr += _rt(b); pb += 1
        if int(b["is_wicket"]) == 1:
            parts.append({"runs": pr, "balls": pb}); pr = pb = 0
    if pb:
        parts.append({"runs": pr, "balls": pb})
    return sorted(parts, key=lambda x: -x["runs"])[:k]

def _win_prob_trajectory(inn2, batting_team, bowling_team, venue, target):
    traj, tl = [], []
    for b in inn2:
        tl.append(b)
        f = build_features(tl, innings=2, batting_team=batting_team,
                           bowling_team=bowling_team, venue=venue, target=target)
        traj.append({"ball": len(tl), "over": round(f["balls_bowled"] / 6, 1),
                     "wp": predictor.predict_win_prob(f),
                     "score": f"{f['total_runs']}/{f['total_wickets']}"})
    return traj

def _turning_points(traj, k=3):
    swings = []
    for i in range(1, len(traj)):
        d = abs(traj[i]["wp"] - traj[i - 1]["wp"])
        swings.append((d, traj[i]))
    swings.sort(key=lambda x: -x[0])
    return [{"over": t["over"], "score": t["score"],
             "win_prob_after": t["wp"], "swing": round(d, 3)}
            for d, t in swings[:k]]

def generate_report(match):
    """match = {
         'meta': {'match_no','venue','batting_first','second'},
         'innings': {1:[balls], 2:[balls]}, 'target': int, 'result': str|None }
    """
    meta = match["meta"]; i1 = match["innings"][1]; i2 = match["innings"].get(2, [])
    venue = meta.get("venue"); teamA = meta["batting_first"]; teamB = meta["second"]
    target = match.get("target")

    r1 = sum(_rt(b) for b in i1); w1 = sum(int(b["is_wicket"]) for b in i1)
    r2 = sum(_rt(b) for b in i2); w2 = sum(int(b["is_wicket"]) for b in i2)

    report = {
        "match": meta.get("match_no"), "venue": venue,
        "first_innings": {"team": teamA, "score": f"{r1}/{w1}",
                          "phase_breakdown": _phase_split(i1),
                          "top_partnerships": _top_partnerships(i1),
                          "over_by_over": analytics.over_by_over(i1),
                          "batting": analytics.batting_scorecard(i1),
                          "bowling": analytics.bowling_scorecard(i1),
                          "strike_rotation": analytics.strike_rotation(i1)},
    }
    text = [f"Match {meta.get('match_no','')} at {venue or 'venue'}",
            f"{teamA} {r1}/{w1}"]

    if i2:
        traj = _win_prob_trajectory(i2, teamB, teamA, venue, target)
        report["second_innings"] = {
            "team": teamB, "score": f"{r2}/{w2}", "target": target,
            "phase_breakdown": _phase_split(i2),
            "top_partnerships": _top_partnerships(i2),
            "turning_points": _turning_points(traj),
            "win_prob_trajectory": traj,
            "over_by_over": analytics.over_by_over(i2),
            "batting": analytics.batting_scorecard(i2),
            "bowling": analytics.bowling_scorecard(i2),
            "strike_rotation": analytics.strike_rotation(i2)}
        balls_rem = 120 - sum(int(b["is_legal"]) for b in i2)
        tg = target or (r2 + 1)
        if r2 >= tg or (balls_rem > 0 and w2 < 10):
            # reached target, or finished below it with balls+wickets left
            # (a data-truncated chase win, same rule the training labels use)
            result = f"{teamB} won by {10 - w2} wickets"
        elif r2 == tg - 1:
            result = "Match tied"
        else:
            result = f"{teamA} won by {tg - 1 - r2} runs"
        report["result"] = match.get("result") or result
        text.append(f"{teamB} {r2}/{w2} (target {target}) -> {report['result']}")
        tp = report["second_innings"]["turning_points"]
        if tp:
            text.append("Turning points: " + "; ".join(
                f"{t['over']} ov ({t['score']}, WP->{int(t['win_prob_after']*100)}%)" for t in tp))
    report["text_summary"] = "\n".join(text)
    return report
