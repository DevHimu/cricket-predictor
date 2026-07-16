"""Turns the scorer's /api/matches/:id payload into a flat, normalized ball list.

The scorer's exact timeline shape is the one thing this project does not control,
so this adapter is deliberately defensive: it searches the payload for the ball
list wherever it lives, and maps field names from a set of known aliases.

If your scorer uses different names, `describe_payload()` (exposed at
/debug/timeline/{id}) reports exactly what was found so the aliases below can be
extended in one place.
"""
from typing import Any, Dict, List, Optional

# field aliases -> canonical name
RUNS_KEYS    = ("runs_of_bat", "runsOfBat", "batRuns", "runs", "r")
EXTRA_KEYS   = ("extra", "extras", "extraRuns")
ETYPE_KEYS   = ("extraType", "extra_type", "type", "kind")
WICKET_KEYS  = ("wicket", "isWicket", "is_wicket", "dismissal", "out")
STRIKER_KEYS = ("striker", "batsman", "batter", "onStrike", "strikerName")
BOWLER_KEYS  = ("bowler", "bowlerName")
INN_KEYS     = ("innings", "inning", "inningsNumber", "innNo", "number")
BALLLIST_KEYS = ("timeline", "balls", "deliveries", "events", "ballByBall", "log")

ILLEGAL = {"wd", "wide", "wides", "nb", "noball", "no-ball", "noballs"}


def _first(d: Dict[str, Any], keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


def _name(v) -> Optional[str]:
    """Player field may be a string, or an object like {name, id}."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v.get("name") or v.get("playerName") or v.get("id")
    return str(v)


def find_ball_lists(payload: Any) -> List[Dict[str, Any]]:
    """Locate every plausible ball list anywhere in the payload.

    Returns a list of {path, count, sample} so we can pick the best one and so
    the diagnostic endpoint can show a human what exists.
    """
    found: List[Dict[str, Any]] = []

    def looks_like_ball(x) -> bool:
        if not isinstance(x, dict):
            return False
        hits = sum(1 for grp in (RUNS_KEYS, EXTRA_KEYS, WICKET_KEYS, STRIKER_KEYS, BOWLER_KEYS)
                   if _first(x, grp) is not None)
        return hits >= 2

    def walk(node, path):
        if isinstance(node, list):
            if node and looks_like_ball(node[0]):
                found.append({"path": path or "(root)", "count": len(node), "sample": node[0]})
            for i, v in enumerate(node[:3]):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)

    walk(payload, "")
    # prefer the longest list found under a known key name
    found.sort(key=lambda f: (any(k in f["path"] for k in BALLLIST_KEYS), f["count"]), reverse=True)
    return found


def normalize_ball(ev: Dict[str, Any], fallback_innings: int = 1) -> Dict[str, Any]:
    """One raw delivery -> canonical ball dict used by the analysis engine."""
    runs = _first(ev, RUNS_KEYS, 0)
    try:
        runs = int(runs)
    except (TypeError, ValueError):
        runs = 0

    extra = _first(ev, EXTRA_KEYS, 0)
    etype = ""
    eruns = 0
    if isinstance(extra, dict):
        etype = str(_first(extra, ETYPE_KEYS, "") or "").lower()
        try:
            eruns = int(extra.get("runs", extra.get("value", 1)))
        except (TypeError, ValueError):
            eruns = 1
    elif isinstance(extra, (int, float)):
        eruns = int(extra)
        etype = str(_first(ev, ETYPE_KEYS, "") or "").lower()
    elif isinstance(extra, str):
        etype = extra.lower()
        eruns = 1

    is_legal = 0 if etype in ILLEGAL else 1

    w = _first(ev, WICKET_KEYS, None)
    if isinstance(w, dict):
        is_wicket = 1 if (w.get("type") or w.get("kind") or w.get("player")) else 0
        wtype = w.get("type") or w.get("kind")
    elif isinstance(w, bool):
        is_wicket, wtype = (1 if w else 0), None
    elif isinstance(w, str):
        is_wicket, wtype = (1 if w and w.lower() not in ("none", "false", "") else 0), w
    else:
        is_wicket, wtype = (1 if w else 0), None

    inn = _first(ev, INN_KEYS, fallback_innings)
    try:
        inn = int(inn)
    except (TypeError, ValueError):
        inn = fallback_innings

    return {
        "innings": inn,
        "runs_of_bat": runs,
        "extras": eruns,
        "extra_type": etype,
        "is_legal": is_legal,
        "is_wicket": is_wicket,
        "wicket_type": wtype,
        "striker": _name(_first(ev, STRIKER_KEYS)),
        "bowler": _name(_first(ev, BOWLER_KEYS)),
    }


def normalize_timeline(payload: Any) -> List[Dict[str, Any]]:
    """Full match payload -> ordered list of canonical balls (all innings)."""
    lists = find_ball_lists(payload)
    if not lists:
        return []

    # If innings blocks each carry their own list, merge them with the right number.
    inns = payload.get("innings") if isinstance(payload, dict) else None
    if isinstance(inns, list) and len(inns) > 0 and any(
            isinstance(b, dict) and any(k in b for k in BALLLIST_KEYS) for b in inns):
        out: List[Dict[str, Any]] = []
        for blk in inns:
            num = _first(blk, INN_KEYS, 1)
            try:
                num = int(num)
            except (TypeError, ValueError):
                num = 1
            raw = _first(blk, BALLLIST_KEYS, []) or []
            out += [normalize_ball(e, num) for e in raw if isinstance(e, dict)]
        if out:
            return out

    best = lists[0]
    node: Any = payload
    for part in best["path"].replace("(root)", "").split("."):
        if not part:
            continue
        if "[" in part:
            k, idx = part.split("[")
            if k:
                node = node[k]
            node = node[int(idx.rstrip("]"))]
        else:
            node = node[part]
    return [normalize_ball(e) for e in node if isinstance(e, dict)]


def describe_payload(payload: Any) -> Dict[str, Any]:
    """Human-readable diagnostic: what does this payload actually contain?"""
    lists = find_ball_lists(payload)
    tl = normalize_timeline(payload)
    per_inn: Dict[int, int] = {}
    named = 0
    for b in tl:
        per_inn[b["innings"]] = per_inn.get(b["innings"], 0) + 1
        if b["striker"] and b["bowler"]:
            named += 1
    return {
        "top_level_keys": list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
        "ball_lists_found": [{"path": f["path"], "count": f["count"]} for f in lists[:5]],
        "raw_sample_ball": lists[0]["sample"] if lists else None,
        "normalized_total": len(tl),
        "normalized_per_innings": per_inn,
        "balls_with_player_names": named,
        "normalized_sample": tl[:3],
        "verdict": (
            "OK - timeline found and players attributed" if named > 0 else
            "TIMELINE FOUND but no striker/bowler names - per-player charts will be empty"
            if tl else
            "NO TIMELINE FOUND - analysis cannot be built from this payload"
        ),
    }
