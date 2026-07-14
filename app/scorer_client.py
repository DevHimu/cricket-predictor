"""Adapts the cricket-scorer API into the shapes feature_builder expects.

There is exactly ONE place you may need to touch when wiring this to your live
server: `ball_from_event` — confirm the field names your timeline / SSE uses.
Everything else is derived. The /score snapshot shape is already handled (it
matches the sample you provided)."""
import os, requests

BASE = os.environ.get("SCORER_BASE", "https://cricket-scorer-pk3j.onrender.com")

# ---- normalization: one delivery -> the 4 fields the model needs -------------
def ball_from_event(ev: dict) -> dict:
    """Map ONE delivery record from /api/matches/:id (timeline) to a model ball.
    Adjust the right-hand keys to your actual timeline field names if they differ.
    """
    runs_of_bat = int(ev.get("runs", ev.get("runs_of_bat", 0)))
    extra = ev.get("extra") or ev.get("extras") or 0
    # `extra` may be a number or an object like {"type":"wd","runs":1}
    if isinstance(extra, dict):
        etype = (extra.get("type") or "").lower()
        eruns = int(extra.get("runs", 1))
        is_legal = 0 if etype in ("wd", "wide", "nb", "noball", "no-ball") else 1
        extras = eruns
    else:
        extras = int(extra)
        etype = str(ev.get("extraType", "")).lower()
        is_legal = 0 if etype in ("wd", "wide", "nb", "noball", "no-ball") else 1
    is_wicket = 1 if ev.get("wicket") else 0
    return {"runs_of_bat": runs_of_bat, "extras": extras,
            "is_wicket": is_wicket, "is_legal": is_legal}

def _timeline_list(match_json: dict, innings: int):
    """Pull the ordered delivery list for one innings out of /api/matches/:id.
    Tries the common shapes; raise a clear error if yours differs."""
    for key in ("timeline", "balls", "deliveries"):
        if key in match_json:
            tl = match_json[key]
            return [b for b in tl if int(b.get("innings", innings)) == innings]
    inns = match_json.get("innings")
    if isinstance(inns, list):
        for blk in inns:
            if int(blk.get("number", blk.get("innings", 0))) == innings:
                return blk.get("timeline", blk.get("balls", []))
    raise ValueError("Could not locate the ball timeline in /matches/:id. "
                     "Point `_timeline_list` at the right field.")

# ---- public helpers ----------------------------------------------------------
def ball_from_snapshots(prev: dict, cur: dict) -> dict:
    """Reconstruct the delivery just bowled by diffing two SSE /score snapshots.
    The SSE stream pushes a full snapshot on every ball, so accumulating these
    rebuilds the whole timeline (with striker/bowler) for analytics + prediction.
    `prev` is the snapshot before this ball (None for the very first ball)."""
    pe = (prev or {}).get("extras", {}) if prev else {}
    ce = cur.get("extras", {})
    runs_delta = cur.get("runs", 0) - (prev.get("runs", 0) if prev else 0)
    wide = ce.get("wides", 0) - pe.get("wides", 0)
    nb = ce.get("noballs", 0) - pe.get("noballs", 0)
    byes = ce.get("byes", 0) - pe.get("byes", 0)
    legbyes = ce.get("legbyes", 0) - pe.get("legbyes", 0)
    extras_total = ce.get("total", 0) - (pe.get("total", 0) if prev else 0)
    runs_of_bat = max(runs_delta - extras_total, 0)
    is_legal = 0 if (wide > 0 or nb > 0) else 1
    is_wicket = 1 if cur.get("wickets", 0) > (prev.get("wickets", 0) if prev else 0) else 0
    facing = (prev or cur).get("striker", {})     # who was on strike for this ball
    bowling = (prev or cur).get("bowler", {})
    return {"runs_of_bat": runs_of_bat, "extras": extras_total, "wide": wide,
            "noballs": nb, "byes": byes, "legbyes": legbyes,
            "is_wicket": is_wicket, "is_legal": is_legal, "wicket_type": None,
            "striker": facing.get("name"), "bowler": bowling.get("name")}


def fetch_json(path: str, timeout: int = 30, retries: int = 1):
    """GET JSON from the scorer. Longer timeout + one retry so a free-tier
    cold start (scorer waking up) doesn't fail the first request."""
    import time
    last = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(f"{BASE}{path}", timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if attempt < retries:
                time.sleep(2)
    raise last

def state_from_match(match_id: str):
    """Fetch /matches/:id and return everything needed to predict the live ball:
    (innings, batting_team, bowling_team, venue, target, timeline)."""
    m = fetch_json(f"/api/matches/{match_id}")
    s = fetch_json(f"/api/matches/{match_id}/score")   # reliable current-state summary
    innings = int(s.get("innings", 1))
    timeline = [ball_from_event(e) for e in _timeline_list(m, innings)]
    return {"innings": innings, "batting_team": s.get("batting"),
            "bowling_team": s.get("bowling"), "venue": s.get("venue"),
            "target": s.get("target"), "timeline": timeline, "raw_score": s}
