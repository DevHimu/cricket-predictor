"""Detailed analytics computed from a ball timeline. Works for past matches
(full detail from the master) and live matches (timeline accumulated from the
SSE snapshots). Each ball may carry: runs_of_bat, extras, wide, noballs, byes,
legbyes, is_wicket, is_legal, wicket_type, striker, bowler, dismissed."""

BOWLER_CREDIT = {"caught", "bowled", "lbw", "stumped", "hit wicket", "caught and bowled"}

def _g(b, k, d=0):
    return b.get(k, d) or 0

def over_by_over(balls):
    overs, cur, legal_in_over, over_no = [], {"runs": 0, "wkts": 0, "balls": 0}, 0, 0
    cum = 0
    for b in balls:
        cur["runs"] += _g(b, "runs_of_bat") + _g(b, "extras")
        cur["wkts"] += int(_g(b, "is_wicket"))
        if int(b.get("is_legal", 1)):
            legal_in_over += 1; cur["balls"] += 1
        if legal_in_over == 6:
            over_no += 1; cum += cur["runs"]
            overs.append({"over": over_no, "runs": cur["runs"], "wkts": cur["wkts"], "cum": cum})
            cur, legal_in_over = {"runs": 0, "wkts": 0, "balls": 0}, 0
    if cur["balls"] or cur["runs"]:
        over_no += 1; cum += cur["runs"]
        overs.append({"over": over_no, "runs": cur["runs"], "wkts": cur["wkts"], "cum": cum, "partial": True})
    return overs

def batting_scorecard(balls):
    order, sc = [], {}
    for b in balls:
        name = b.get("striker")
        if not name:
            continue
        if name not in sc:
            sc[name] = {"batter": name, "runs": 0, "balls": 0, "fours": 0, "sixes": 0, "out": False}
            order.append(name)
        r = int(_g(b, "runs_of_bat"))
        sc[name]["runs"] += r
        if int(b.get("is_legal", 1)) or int(_g(b, "noballs")):   # faced (not a wide)
            if not int(_g(b, "wide")):
                sc[name]["balls"] += 1
        if r == 4: sc[name]["fours"] += 1
        if r == 6: sc[name]["sixes"] += 1
        if int(_g(b, "is_wicket")) and b.get("dismissed", name) == name:
            sc[name]["out"] = True
    rows = []
    for n in order:
        s = sc[n]; bf = s["balls"]
        s["sr"] = round(s["runs"] / bf * 100, 1) if bf else 0.0
        bruns = s["fours"] * 4 + s["sixes"] * 6
        s["boundary_pct"] = round(bruns / s["runs"] * 100, 1) if s["runs"] else 0.0
        rows.append(s)
    return rows

def bowling_scorecard(balls):
    order, sc = [], {}
    for b in balls:
        name = b.get("bowler")
        if not name:
            continue
        if name not in sc:
            sc[name] = {"bowler": name, "balls": 0, "runs": 0, "wkts": 0, "dots": 0}
            order.append(name)
        legal = int(b.get("is_legal", 1))
        if legal:
            sc[name]["balls"] += 1
        sc[name]["runs"] += int(_g(b, "runs_of_bat")) + int(_g(b, "wide")) + int(_g(b, "noballs"))
        if legal and (_g(b, "runs_of_bat") + _g(b, "extras")) == 0:
            sc[name]["dots"] += 1
        if int(_g(b, "is_wicket")):
            wt = str(b.get("wicket_type", "")).lower()
            if (not wt) or wt in BOWLER_CREDIT:
                sc[name]["wkts"] += 1
    rows = []
    for n in order:
        s = sc[n]; ov = s["balls"] / 6
        s["overs"] = f"{s['balls'] // 6}.{s['balls'] % 6}"
        s["economy"] = round(s["runs"] / ov, 2) if ov else 0.0
        s["dot_pct"] = round(s["dots"] / s["balls"] * 100, 1) if s["balls"] else 0.0
        rows.append(s)
    return rows

def strike_rotation(balls):
    """Dot / rotated (1s & 2s & 3s) / boundary share of legal balls, per innings."""
    dot = rot = bdry = legal = 0
    for b in balls:
        if not int(b.get("is_legal", 1)):
            continue
        legal += 1
        tot = _g(b, "runs_of_bat") + _g(b, "extras")
        if tot == 0: dot += 1
        elif int(_g(b, "runs_of_bat")) in (4, 6): bdry += 1
        else: rot += 1
    p = lambda x: round(x / legal * 100, 1) if legal else 0.0
    return {"dot_pct": p(dot), "rotated_pct": p(rot), "boundary_pct": p(bdry), "legal_balls": legal}
