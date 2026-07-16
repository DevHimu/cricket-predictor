"""Builds the dashboard analysis from an innings block produced by timeline_adapter.

Design note: the scorer already aggregates every batter and bowler, so those are
treated as the authoritative scorecard - the numbers on our screen then match the
scorer's own screen exactly. The ball timeline is used only for the things those
aggregates do not carry: dot-ball counts and the over-by-over run/wicket graph.
"""
from typing import Any, Dict, List

WIDE = {"wd", "wide", "wides"}


def _round(x, n=1):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return 0.0


def _pct(a, b):
    return _round((a / b * 100.0) if b else 0.0)


def faced_by(balls: List[Dict[str, Any]]) -> List[Any]:
    """Work out who actually FACED each ball.

    The scorer records `strikerId` as the state AFTER the delivery, not the
    batter who faced it. Proof from real data: ball 1 scored 0 runs (no strike
    rotation), yet strikerId changes between ball 1 and ball 2 - impossible if
    it named the batter on strike for that ball.

    So: faced(N) = strikerId(N-1), except at an over boundary, where the strike
    rotates and faced(N) is the partner instead. Reconstructing it this way
    reproduces the scorer's own batting card exactly.

    (`bowlerId`, by contrast, IS the bowler of that ball - verified against the
    scorer's bowling card - so it needs no adjustment.)
    """
    n = len(balls)
    S = [(b.get("striker_id") or b.get("striker")) for b in balls]
    over_ends = [False] * n
    legal = 0
    for i, b in enumerate(balls):
        if int(b.get("is_legal") or 0):
            legal += 1
            if legal % 6 == 0:
                over_ends[i] = True

    # pair_at[i] = the last two distinct striker ids seen up to and including i
    pair_at: List[List[Any]] = []
    acc: List[Any] = []
    for i in range(n):
        sid = S[i]
        if sid is not None:
            if sid in acc:
                acc.remove(sid)
            acc.append(sid)
            del acc[:-2]
        pair_at.append(list(acc))

    faced: List[Any] = [None] * n
    for i in range(n):
        if i == 0:
            faced[0] = S[0]          # ball 1: even runs keep the same striker
            continue
        prev_wicket = int(balls[i - 1].get("is_wicket") or 0)
        if prev_wicket:
            # The wicket ball's strikerId may name the batter just dismissed, so
            # walking forward from it is unsafe. Infer from THIS ball's post-state
            # instead: even runs mean the striker did not change, so faced == S[i].
            runs_i = int(balls[i].get("runs_of_bat") or 0)
            if runs_i % 2 == 0:
                faced[i] = S[i]
            else:
                other = [p for p in pair_at[i] if p != S[i]]
                faced[i] = other[0] if other else S[i]
        elif over_ends[i - 1]:
            other = [p for p in pair_at[i - 1] if p != S[i - 1]]
            faced[i] = other[0] if other else S[i - 1]
        else:
            faced[i] = S[i - 1]
    return faced


def partnerships(balls: List[Dict[str, Any]], faced: List[Any],
                 names: Dict[str, str]) -> List[Dict[str, Any]]:
    """Partnerships, segmented by wickets.

    A partnership runs from when two batters come together until one is out, so
    the segments are simply the stretches of the innings between wickets. Each
    one reports both batters' individual contributions plus the stand total -
    the Cricbuzz layout: 44 (46) with 9 (16) one side and 31 (30) the other.
    """
    out: List[Dict[str, Any]] = []
    cur = {"ids": [], "runs": 0, "balls": 0, "per": {}}
    wkt = 0

    def close(c, n, unbroken):
        ids = c["ids"][:2]
        def side(i):
            if i >= len(ids):
                return None
            pid = ids[i]
            p = c["per"].get(pid, {"runs": 0, "balls": 0})
            return {"name": names.get(pid, pid), "runs": p["runs"], "balls": p["balls"]}
        return {"wicket": n, "unbroken": unbroken,
                "runs": c["runs"], "balls": c["balls"],
                "batter1": side(0), "batter2": side(1)}

    for i, b in enumerate(balls):
        rob = int(b.get("runs_of_bat") or 0)
        ex = int(b.get("extras") or 0)
        et = (b.get("extra_type") or "").lower()
        legal = int(b.get("is_legal") or 0)
        f = faced[i]

        if f and f not in cur["ids"]:
            cur["ids"].append(f)
        cur["runs"] += rob + ex
        if legal:
            cur["balls"] += 1
        if f:
            p = cur["per"].setdefault(f, {"runs": 0, "balls": 0})
            p["runs"] += rob
            if et not in WIDE:
                p["balls"] += 1

        if int(b.get("is_wicket") or 0):
            wkt += 1
            out.append(close(cur, wkt, False))
            cur = {"ids": [], "runs": 0, "balls": 0, "per": {}}

    if cur["balls"] or cur["runs"]:
        out.append(close(cur, wkt + 1, True))
    return out


def _dots_and_overs(balls: List[Dict[str, Any]]):
    """From the timeline: per-player dot counts, over buckets, team dot count."""
    bat_dots: Dict[str, int] = {}
    bat_faced: Dict[str, int] = {}
    bat_runs: Dict[str, int] = {}
    bowl_dots: Dict[str, int] = {}
    overs: List[Dict[str, Any]] = []
    team_dots = legal = 0
    cur = {"over": 1, "runs": 0, "wkts": 0}
    faced = faced_by(balls)

    for idx, b in enumerate(balls):
        et = (b.get("extra_type") or "").lower()
        rob = int(b.get("runs_of_bat") or 0)
        ex = int(b.get("extras") or 0)
        is_legal = int(b.get("is_legal") or 0)
        is_wkt = int(b.get("is_wicket") or 0)

        sk = faced[idx]
        if sk and et not in WIDE:               # a wide is not a ball faced
            bat_faced[sk] = bat_faced.get(sk, 0) + 1
            bat_runs[sk] = bat_runs.get(sk, 0) + rob
            if rob == 0:
                bat_dots[sk] = bat_dots.get(sk, 0) + 1

        bw = b.get("bowler_id") or b.get("bowler")
        if bw and is_legal and (rob + ex) == 0:
            bowl_dots[bw] = bowl_dots.get(bw, 0) + 1

        cur["runs"] += rob + ex
        cur["wkts"] += is_wkt
        if is_legal:
            legal += 1
            if (rob + ex) == 0:
                team_dots += 1
            if legal % 6 == 0:
                cur["balls"] = legal
                overs.append(dict(cur))
                cur = {"over": len(overs) + 1, "runs": 0, "wkts": 0}

    if cur["runs"] or cur["wkts"]:
        cur["balls"] = legal
        overs.append(dict(cur))

    cum = 0
    for o in overs:
        cum += o["runs"]
        o["cum"] = cum
        o["crr"] = _round(cum / (o["balls"] / 6.0)) if o.get("balls") else 0.0

    return bat_dots, bat_faced, bowl_dots, overs, team_dots, legal, bat_runs


def build_innings(inn: Dict[str, Any]) -> Dict[str, Any]:
    balls = inn.get("balls") or []
    bat_dots, bat_faced, bowl_dots, overs, team_dots, legal, bat_runs = _dots_and_overs(balls)
    names = {p.get("id") or p.get("name"): p["name"]
             for p in (inn.get("batting_card") or [])}
    stands = partnerships(balls, faced_by(balls), names)

    # ---- batting: scorer's card + dot% from the timeline ----
    batting = []
    bdry_runs = 0
    for p in inn.get("batting_card") or []:
        key = p.get("id") or p.get("name")
        dots = bat_dots.get(key, bat_dots.get(p.get("name"), 0))
        faced = p["balls"] or bat_faced.get(key, 0)
        b4, b6 = p.get("fours", 0), p.get("sixes", 0)
        bdry = b4 * 4 + b6 * 6
        bdry_runs += bdry
        if p["balls"] == 0 and not p.get("out"):
            continue                              # yet to bat
        batting.append({
            "name": p["name"], "runs": p["runs"], "balls": p["balls"],
            "fours": b4, "sixes": b6, "out": p.get("out", False),
            "how_out": p.get("how_out"),
            "sr": _pct(p["runs"], p["balls"]),
            "dots": dots,
            "dot_pct": _pct(dots, faced),
            "boundary_pct": _pct(bdry, p["runs"]),
        })
    batting.sort(key=lambda x: (-x["runs"], x["balls"]))

    # ---- bowling: scorer's card + dot% from the timeline ----
    bowling = []
    for p in inn.get("bowling_card") or []:
        if p["balls"] == 0:
            continue                              # has not bowled
        key = p.get("id") or p.get("name")
        dots = bowl_dots.get(key, bowl_dots.get(p.get("name"), 0))
        bowling.append({
            "name": p["name"], "balls": p["balls"], "runs": p["runs"],
            "wickets": p["wickets"], "maidens": p.get("maidens", 0),
            "overs": f"{p['balls'] // 6}.{p['balls'] % 6}",
            "econ": _round(p["runs"] / (p["balls"] / 6.0)) if p["balls"] else 0.0,
            "dots": dots,
            "dot_pct": _pct(dots, p["balls"]),
        })
    bowling.sort(key=lambda x: (-x["wickets"], x["econ"]))

    # Self-check: our reconstructed runs vs the scorer's own card. If these
    # ever diverge, the striker attribution model needs revisiting.
    check = []
    for p in inn.get("batting_card") or []:
        k = p.get("id") or p.get("name")
        got = bat_runs.get(k, 0)
        if p["balls"] or p.get("out"):
            check.append({"name": p["name"], "card_runs": p["runs"],
                          "derived_runs": got, "match": got == p["runs"]})

    total_runs = inn.get("runs", 0)
    lb = inn.get("legal_balls") or legal

    return {
        "number": inn["number"],
        "batting_team": inn.get("batting_team", ""),
        "bowling_team": inn.get("bowling_team", ""),
        "runs": total_runs,
        "wickets": inn.get("wickets", 0),
        "balls": lb,
        "overs_text": f"{lb // 6}.{lb % 6}",
        "extras": (inn.get("extras") or {}).get("total", 0),
        "target": inn.get("target"),
        "batting": batting,
        "bowling": bowling,
        "overs": overs,
        "partnerships": stands,
        "dot_pct": _pct(team_dots, legal or lb),
        "boundary_pct": _pct(bdry_runs, total_runs),
        "_attribution_check": check,
    }


def build_analysis(innings_blocks: List[Dict[str, Any]],
                   meta: Dict[str, Any]) -> Dict[str, Any]:
    innings = [build_innings(i) for i in innings_blocks if i.get("number") in (1, 2)]

    best_bat = None
    for i in innings:
        for b in i["batting"]:
            if best_bat is None or b["runs"] > best_bat["runs"]:
                best_bat = {**b, "team": i["batting_team"], "innings": i["number"]}

    # most wickets; ties broken by the lower economy ("run ratio")
    best_bowl = None
    for i in innings:
        for w in i["bowling"]:
            if best_bowl is None or (w["wickets"], -w["econ"]) > (best_bowl["wickets"], -best_bowl["econ"]):
                best_bowl = {**w, "team": i["bowling_team"], "innings": i["number"]}

    return {
        "match_id": meta.get("match_id"),
        "status": meta.get("status", "live"),
        "teamA": meta.get("teamA", ""),
        "teamB": meta.get("teamB", ""),
        "venue": meta.get("venue"),
        "result": meta.get("result"),
        "innings": innings,
        "best_batter": best_bat,
        "best_bowler": best_bowl,
        "has_player_names": any(b["name"] for i in innings for b in i["batting"]),
    }
