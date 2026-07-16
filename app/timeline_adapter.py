"""Reads the cricket-scorer's /api/matches/:id payload.

Written against the real shape (confirmed 2026-07):

  {
    id, status, overs, venue, teams:{A,B}, battingFirst, currentInnings, result,
    innings: [
      { number, battingSide, bowlingSide, battingTeamName, bowlingTeamName,
        runs, wickets, legalBalls, extras:{wides,noballs,byes,legbyes,total},
        target, strikerId, nonStrikerId, bowlerId,
        batters: { <playerId>: {id,name,runs,balls,fours,sixes,out,howOut} },
        bowlers: { <playerId>: {id,name,balls,runs,wickets,maidens} },
        timeline: [ {seq, over:"0.1", extra, runs, teamRuns, wicket,
                     strikerId, bowlerId, ts} ],
        fallOfWickets: [], closed }
    ]
  }

Two things matter here:
  1. The timeline references players by ID, not name. Names live in the
     `batters` / `bowlers` maps on the same innings block.
  2. The scorer already aggregates each batter and bowler. Those are the
     authoritative scorecard numbers, so we use them rather than recomputing;
     the timeline is used for what they do not carry (dot balls, over-by-over).

Alias lists are kept so a future scorer change degrades instead of breaking.
"""
from typing import Any, Dict, List, Optional

RUNS_KEYS = ("runs", "runs_of_bat", "runsOfBat", "batRuns", "r")
TEAMRUNS_KEYS = ("teamRuns", "totalRuns", "ballRuns")
EXTRA_KEYS = ("extra", "extras", "extraRuns")
ETYPE_KEYS = ("type", "extraType", "extra_type", "kind")
WICKET_KEYS = ("wicket", "isWicket", "is_wicket", "dismissal", "out")
STRIKER_ID_KEYS = ("strikerId", "batsmanId", "batterId", "striker_id")
BOWLER_ID_KEYS = ("bowlerId", "bowler_id")
STRIKER_KEYS = ("striker", "batsman", "batter", "strikerName")
BOWLER_KEYS = ("bowler", "bowlerName")
INN_NUM_KEYS = ("number", "innings", "inning", "inningsNumber", "innNo")
BALLLIST_KEYS = ("timeline", "balls", "deliveries", "events", "ballByBall", "log")

WIDE_T = {"wd", "wide", "wides"}
NOBALL_T = {"nb", "noball", "no-ball", "noballs"}
ILLEGAL = WIDE_T | NOBALL_T


def _first(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _name_of(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v.get("name") or v.get("playerName") or v.get("id")
    return str(v)


def player_lookup(inn: Dict[str, Any]) -> Dict[str, str]:
    """playerId -> display name, from the innings block's batters/bowlers maps."""
    out: Dict[str, str] = {}
    for grp in ("batters", "bowlers", "players"):
        blk = inn.get(grp)
        if isinstance(blk, dict):
            for pid, p in blk.items():
                if isinstance(p, dict) and p.get("name"):
                    out[str(pid)] = p["name"]
                elif isinstance(p, str):
                    out[str(pid)] = p
        elif isinstance(blk, list):
            for p in blk:
                if isinstance(p, dict) and p.get("id") and p.get("name"):
                    out[str(p["id"])] = p["name"]
    return out


def normalize_ball(ev: Dict[str, Any], innings: int,
                   names: Dict[str, str] = None) -> Dict[str, Any]:
    """One raw delivery -> canonical ball dict."""
    names = names or {}
    runs = _int(_first(ev, RUNS_KEYS, 0))

    # extras: prefer (teamRuns - runs), which is exact and shape-independent.
    team_runs = _first(ev, TEAMRUNS_KEYS, None)
    extra = _first(ev, EXTRA_KEYS, None)
    etype = ""
    if isinstance(extra, dict):
        etype = str(_first(extra, ETYPE_KEYS, "") or "").lower()
        ex_from_obj = _int(extra.get("runs", extra.get("value", 1)), 1)
    elif isinstance(extra, str):
        etype = extra.lower()
        ex_from_obj = 1
    elif isinstance(extra, (int, float)):
        etype = str(_first(ev, ETYPE_KEYS, "") or "").lower()
        ex_from_obj = int(extra)
    else:
        ex_from_obj = 0

    if team_runs is not None:
        extras = max(_int(team_runs) - runs, 0)
    else:
        extras = ex_from_obj

    is_legal = 0 if etype in ILLEGAL else 1

    w = _first(ev, WICKET_KEYS, None)
    if isinstance(w, dict):
        is_wicket = 1
        wtype = w.get("type") or w.get("kind") or w.get("how")
    elif isinstance(w, str) and w.lower() not in ("", "none", "false", "null"):
        is_wicket, wtype = 1, w
    elif w is True:
        is_wicket, wtype = 1, None
    else:
        is_wicket, wtype = 0, None

    sid = _first(ev, STRIKER_ID_KEYS)
    bid = _first(ev, BOWLER_ID_KEYS)
    striker = names.get(str(sid)) if sid is not None else None
    bowler = names.get(str(bid)) if bid is not None else None
    if striker is None:
        striker = _name_of(_first(ev, STRIKER_KEYS))
    if bowler is None:
        bowler = _name_of(_first(ev, BOWLER_KEYS))

    return {"innings": innings, "runs_of_bat": runs, "extras": extras,
            "extra_type": etype, "is_legal": is_legal, "is_wicket": is_wicket,
            "wicket_type": wtype, "striker": striker, "bowler": bowler,
            "striker_id": str(sid) if sid is not None else None,
            "bowler_id": str(bid) if bid is not None else None}


def _cards(inn: Dict[str, Any]):
    """The scorer's own aggregates -> (batting list, bowling list)."""
    bats, bowls = [], []
    b = inn.get("batters")
    if isinstance(b, dict):
        for pid, p in b.items():
            if not isinstance(p, dict):
                continue
            bats.append({"id": str(pid), "name": p.get("name") or str(pid),
                         "runs": _int(p.get("runs")), "balls": _int(p.get("balls")),
                         "fours": _int(p.get("fours")), "sixes": _int(p.get("sixes")),
                         "out": bool(p.get("out")), "how_out": p.get("howOut")})
    w = inn.get("bowlers")
    if isinstance(w, dict):
        for pid, p in w.items():
            if not isinstance(p, dict):
                continue
            bowls.append({"id": str(pid), "name": p.get("name") or str(pid),
                          "balls": _int(p.get("balls")), "runs": _int(p.get("runs")),
                          "wickets": _int(p.get("wickets")),
                          "maidens": _int(p.get("maidens"))})
    return bats, bowls


def _team_name(v) -> str:
    """A team may be a plain string, or the full object {id, name, players}.

    The /score endpoint sends strings; /api/matches/:id sends objects. Returning
    the object into React renders nothing and throws 'Objects are not valid as a
    React child', so everything is flattened to a name here, once.
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v.get("name") or v.get("teamName") or v.get("id") or ""
    return str(v)


def _result_text(v) -> Optional[str]:
    """Result may be a string, or an object like {winnerName, margin, text}."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if v.get("text"):
            return v["text"]
        if v.get("description"):
            return v["description"]
        win = v.get("winnerName") or _team_name(v.get("winner"))
        margin, mtype = v.get("margin"), v.get("marginType") or ""
        if win and margin is not None:
            return f"{win} won by {margin} {mtype}".strip()
        return win or None
    return str(v)


def extract_innings(payload: Any) -> List[Dict[str, Any]]:
    """Every innings, with its balls and the scorer's own scorecards."""
    if not isinstance(payload, dict):
        return []
    blocks = payload.get("innings")
    if not isinstance(blocks, list):
        return []

    teams = payload.get("teams") if isinstance(payload.get("teams"), dict) else {}
    out = []
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        num = _int(_first(blk, INN_NUM_KEYS, 1), 1)
        names = player_lookup(blk)
        raw = _first(blk, BALLLIST_KEYS, []) or []
        balls = [normalize_ball(e, num, names) for e in raw if isinstance(e, dict)]
        bats, bowls = _cards(blk)

        bat_team = (_team_name(blk.get("battingTeamName"))
                    or _team_name(blk.get("battingTeam"))
                    or _team_name(teams.get(blk.get("battingSide"))))
        bowl_team = (_team_name(blk.get("bowlingTeamName"))
                     or _team_name(blk.get("bowlingTeam"))
                     or _team_name(teams.get(blk.get("bowlingSide"))))

        out.append({
            "number": num,
            "batting_team": bat_team,
            "bowling_team": bowl_team,
            "runs": _int(blk.get("runs")),
            "wickets": _int(blk.get("wickets")),
            "legal_balls": _int(blk.get("legalBalls")),
            "extras": blk.get("extras") or {},
            "target": blk.get("target"),
            "balls": balls,
            "batting_card": bats,
            "bowling_card": bowls,
        })
    out.sort(key=lambda i: i["number"])
    return out


def normalize_timeline(payload: Any) -> List[Dict[str, Any]]:
    """Flat ordered ball list across all innings (used by the model)."""
    balls: List[Dict[str, Any]] = []
    for inn in extract_innings(payload):
        balls += inn["balls"]
    return balls


def match_meta(payload: Any, match_id: str = None) -> Dict[str, Any]:
    p = payload if isinstance(payload, dict) else {}
    teams = p.get("teams") if isinstance(p.get("teams"), dict) else {}
    return {"match_id": match_id or p.get("id"),
            "status": p.get("status") or "live",
            "teamA": _team_name(teams.get("A")) or _team_name(p.get("teamA")),
            "teamB": _team_name(teams.get("B")) or _team_name(p.get("teamB")),
            "venue": p.get("venue") if isinstance(p.get("venue"), (str, type(None))) else str(p.get("venue")),
            "result": _result_text(p.get("result")),
            "overs": p.get("overs")}


def describe_payload(payload: Any) -> Dict[str, Any]:
    """Diagnostic: what did we find, and can the charts be built?"""
    inns = extract_innings(payload)
    tl = [b for i in inns for b in i["balls"]]
    named = sum(1 for b in tl if b["striker"] and b["bowler"])
    return {
        "top_level_keys": list(payload.keys()) if isinstance(payload, dict) else str(type(payload)),
        "meta": match_meta(payload),
        "innings_found": [
            {"number": i["number"], "batting_team": i["batting_team"],
             "bowling_team": i["bowling_team"], "runs": i["runs"],
             "wickets": i["wickets"], "legal_balls": i["legal_balls"],
             "balls_in_timeline": len(i["balls"]),
             "batters_in_card": len(i["batting_card"]),
             "bowlers_in_card": len(i["bowling_card"])}
            for i in inns],
        "normalized_total": len(tl),
        "balls_with_player_names": named,
        "normalized_sample": tl[:3],
        "verdict": (
            "OK - timeline found and players attributed" if named and tl else
            "TIMELINE FOUND but no striker/bowler names - per-player charts will be empty"
            if tl else
            "NO TIMELINE FOUND - analysis cannot be built from this payload"),
    }
