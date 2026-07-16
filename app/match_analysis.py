"""Builds the full match analysis from a normalized ball timeline.

Pure computation: given balls, return the numbers the dashboard charts. No ML,
no I/O. Scoring conventions follow official cricket rules:
  - balls faced by a batter exclude wides, but include no-balls
  - an over counts legal deliveries only
  - byes and leg-byes are NOT charged to the bowler; wides and no-balls are
  - run-outs are not credited to the bowler
"""
from typing import Any, Dict, List

WIDE = {"wd", "wide", "wides"}
NOBALL = {"nb", "noball", "no-ball", "noballs"}
BYES = {"b", "bye", "byes", "lb", "legbye", "legbyes", "leg-bye"}
BOWLER_WICKETS = {"bowled", "caught", "lbw", "stumped", "hit wicket",
                  "caught and bowled", "caught & bowled", "c&b", None, ""}


def _round(x, n=1):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return 0.0


def _pct(a, b):
    return _round((a / b * 100.0) if b else 0.0)


def build_innings(balls: List[Dict[str, Any]], number: int,
                  batting_team: str = "", bowling_team: str = "") -> Dict[str, Any]:
    bat: Dict[str, Dict[str, Any]] = {}
    bowl: Dict[str, Dict[str, Any]] = {}
    overs: List[Dict[str, Any]] = []

    total_runs = total_wkts = legal = 0
    team_dots = team_bdry_runs = 0
    cur_over = {"over": 1, "runs": 0, "wkts": 0}

    for b in balls:
        et = (b.get("extra_type") or "").lower()
        rob = int(b.get("runs_of_bat") or 0)
        ex = int(b.get("extras") or 0)
        is_legal = int(b.get("is_legal") or 0)
        is_wkt = int(b.get("is_wicket") or 0)
        ball_runs = rob + ex

        total_runs += ball_runs
        total_wkts += is_wkt

        # ---- batter ----
        sname = b.get("striker") or "Unknown"
        s = bat.setdefault(sname, {"name": sname, "runs": 0, "balls": 0, "fours": 0,
                                   "sixes": 0, "dots": 0, "out": False})
        if et not in WIDE:                       # wides are not a ball faced
            s["balls"] += 1
            if rob == 0:
                s["dots"] += 1
        s["runs"] += rob
        if rob == 4:
            s["fours"] += 1
        elif rob == 6:
            s["sixes"] += 1
        if is_wkt:
            s["out"] = True
        if rob in (4, 6):
            team_bdry_runs += rob

        # ---- bowler ----
        bname = b.get("bowler") or "Unknown"
        w = bowl.setdefault(bname, {"name": bname, "balls": 0, "runs": 0,
                                    "wickets": 0, "dots": 0})
        charged = rob + (ex if et in WIDE or et in NOBALL else 0)
        w["runs"] += charged
        if is_legal:
            w["balls"] += 1
            if ball_runs == 0:
                w["dots"] += 1
                team_dots += 1
        if is_wkt and (b.get("wicket_type") or "").lower() in BOWLER_WICKETS:
            w["wickets"] += 1

        # ---- over buckets ----
        cur_over["runs"] += ball_runs
        cur_over["wkts"] += is_wkt
        if is_legal:
            legal += 1
            if legal % 6 == 0:
                overs.append(dict(cur_over))
                cur_over = {"over": len(overs) + 1, "runs": 0, "wkts": 0}

    if cur_over["runs"] or cur_over["wkts"]:      # partial over in progress
        overs.append(dict(cur_over))

    cum = 0
    for o in overs:
        cum += o["runs"]
        o["cum"] = cum

    bat_list = []
    for s in bat.values():
        bdry = s["fours"] * 4 + s["sixes"] * 6
        bat_list.append({**s,
                         "sr": _pct(s["runs"], s["balls"]),
                         "dot_pct": _pct(s["dots"], s["balls"]),
                         "boundary_pct": _pct(bdry, s["runs"])})
    bat_list.sort(key=lambda x: (-x["runs"], x["balls"]))

    bowl_list = []
    for w in bowl.values():
        bowl_list.append({**w,
                          "overs": f"{w['balls'] // 6}.{w['balls'] % 6}",
                          "econ": _round(w["runs"] / (w["balls"] / 6.0)) if w["balls"] else 0.0,
                          "dot_pct": _pct(w["dots"], w["balls"])})
    bowl_list.sort(key=lambda x: (-x["wickets"], x["econ"]))

    return {
        "number": number,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "runs": total_runs,
        "wickets": total_wkts,
        "balls": legal,
        "overs_text": f"{legal // 6}.{legal % 6}",
        "batting": bat_list,
        "bowling": bowl_list,
        "overs": overs,
        "dot_pct": _pct(team_dots, legal),
        "boundary_pct": _pct(team_bdry_runs, total_runs),
    }


def build_analysis(timeline: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Full-match analysis. `meta` supplies team names, status and result."""
    inn_nums = sorted({int(b.get("innings") or 1) for b in timeline}) or [1]
    teams = meta.get("innings_teams") or {}
    innings = []
    for n in inn_nums:
        if n > 2:
            continue
        t = teams.get(n) or teams.get(str(n)) or {}
        innings.append(build_innings(
            [b for b in timeline if int(b.get("innings") or 1) == n], n,
            t.get("batting", ""), t.get("bowling", "")))

    # best batter: most runs in the match
    best_bat = None
    for i in innings:
        for b in i["batting"]:
            if b["name"] == "Unknown":
                continue
            if best_bat is None or b["runs"] > best_bat["runs"]:
                best_bat = {**b, "team": i["batting_team"], "innings": i["number"]}

    # best bowler: most wickets, tie-break on lowest economy ("run ratio")
    best_bowl = None
    for i in innings:
        for w in i["bowling"]:
            if w["name"] == "Unknown" or w["balls"] == 0:
                continue
            if best_bowl is None or (w["wickets"], -w["econ"]) > (best_bowl["wickets"], -best_bowl["econ"]):
                best_bowl = {**w, "team": i["bowling_team"], "innings": i["number"]}

    return {
        "match_id": meta.get("match_id"),
        "status": meta.get("status", "live"),
        "teamA": meta.get("teamA", ""),
        "teamB": meta.get("teamB", ""),
        "venue": meta.get("venue"),
        "result": meta.get("result"),
        "target": meta.get("target"),
        "innings": innings,
        "best_batter": best_bat,
        "best_bowler": best_bowl,
        "has_player_names": any(
            b["name"] != "Unknown" for i in innings for b in i["batting"]),
    }
