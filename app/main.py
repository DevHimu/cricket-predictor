"""Cricket Realtime Predictor — combined service.

This ONE service serves BOTH the frontend webapp AND the prediction API, so you
deploy a single thing to Render. Because the page and the API share the same
origin, there are no CORS problems and no URL to configure in the frontend.

Routes:
  GET  /                 -> the webapp (index.html)
  GET  /health           -> liveness check
  POST /predict          -> stateless prediction from a timeline
  GET  /live/{id}        -> fetch scorer + predict the current ball
  POST /report           -> full post-match analysis
  GET  /api/matches      -> proxy to the scorer's match list (avoids browser CORS)
"""
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from feature_builder import build_features
import predictor
import report as report_mod

# webapp/index.html sits one level up from app/
HERE = os.path.dirname(os.path.abspath(__file__))
WEBAPP = os.path.join(os.path.dirname(HERE), "webapp", "index.html")

app = FastAPI(title="Cricket Realtime Predictor", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

class Ball(BaseModel):
    runs_of_bat: int = 0
    extras: int = 0
    is_wicket: int = 0
    is_legal: int = 1

class PredictReq(BaseModel):
    innings: int
    batting_team: str
    bowling_team: str
    venue: Optional[str] = None
    target: Optional[int] = None
    total_balls: Optional[int] = 120
    timeline: List[Ball]

class ReportReq(BaseModel):
    meta: dict
    innings1: List[Ball]
    innings2: List[Ball] = []
    target: Optional[int] = None
    result: Optional[str] = None

# ── Frontend ──────────────────────────────────────────────────────
@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return FileResponse(WEBAPP)

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)   # no icon; stops the 404 noise

# ── Health ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models": ["projected_score", "win_prob"]}

# ── Diagnostic: can THIS service reach the scorer? ────────────────
@app.get("/debug/scorer")
def debug_scorer():
    """Open this in your browser to see whether the Predictor can reach the
    Scorer, how long it takes, and how many matches it returns. Turns a vague
    'api error' into a concrete answer."""
    import scorer_client, time
    out = {"scorer_base": scorer_client.BASE}
    t = time.time()
    try:
        data = scorer_client.fetch_json("/api/matches")
        out["reachable"] = True
        out["elapsed_seconds"] = round(time.time() - t, 1)
        if isinstance(data, list):
            out["match_count"] = len(data)
            out["sample"] = data[:2]
        elif isinstance(data, dict):
            lst = data.get("matches", [])
            out["match_count"] = len(lst) if isinstance(lst, list) else "unknown"
            out["sample"] = data
        else:
            out["match_count"] = "unknown"
    except Exception as e:
        out["reachable"] = False
        out["elapsed_seconds"] = round(time.time() - t, 1)
        out["error"] = str(e)
        out["hint"] = ("If elapsed is ~15-30s, the scorer was asleep (free-tier "
                       "cold start). Open the scorer's own URL once to wake it, "
                       "then retry. If it says connection/DNS, check SCORER_BASE.")
    return out

# ── Prediction API ────────────────────────────────────────────────
@app.post("/predict")
def predict(req: PredictReq):
    tl = [b.model_dump() for b in req.timeline]
    if not tl:
        raise HTTPException(400, "timeline is empty")
    feats = build_features(tl, innings=req.innings, batting_team=req.batting_team,
                           bowling_team=req.bowling_team, venue=req.venue,
                           target=req.target, total_balls=req.total_balls or 120)
    return predictor.predict(feats, req.innings)

@app.get("/live/{match_id}")
def live(match_id: str):
    import scorer_client
    try:
        st = scorer_client.state_from_match(match_id)
    except Exception as e:
        raise HTTPException(502, f"scorer fetch/parse failed: {e}")
    if not st["timeline"]:
        return {"status": "no balls yet"}
    feats = build_features(st["timeline"], innings=st["innings"],
                           batting_team=st["batting_team"],
                           bowling_team=st["bowling_team"],
                           venue=st["venue"], target=st["target"])
    out = predictor.predict(feats, st["innings"])
    out["match_id"] = match_id
    out["runs"] = feats["total_runs"]
    out["wkts"] = feats["total_wickets"]
    out["balls_bowled"] = feats["balls_bowled"]
    out["batting_team"] = st["batting_team"]
    out["bowling_team"] = st["bowling_team"]
    out["venue"] = st.get("venue")
    out["status"] = (st.get("raw_score") or {}).get("status", "live")
    return out

@app.post("/report")
def make_report(req: ReportReq):
    match = {"meta": req.meta,
             "innings": {1: [b.model_dump() for b in req.innings1],
                         2: [b.model_dump() for b in req.innings2]},
             "target": req.target, "result": req.result}
    return report_mod.generate_report(match)

# ── Scorer proxy (so the browser only ever talks to THIS origin) ──
@app.get("/api/matches")
def matches_proxy():
    import scorer_client
    try:
        return scorer_client.fetch_json("/api/matches")
    except Exception as e:
        raise HTTPException(502, f"scorer fetch failed: {e}")

@app.get("/api/matches/{match_id}/score")
def score_proxy(match_id: str):
    import scorer_client
    try:
        return scorer_client.fetch_json(f"/api/matches/{match_id}/score")
    except Exception as e:
        raise HTTPException(502, f"scorer fetch failed: {e}")
