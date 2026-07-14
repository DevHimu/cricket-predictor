"""Prediction service. Your webapp calls these endpoints.

Recommended live wiring:
  frontend subscribes to the scorer SSE stream, and on every ball calls
  GET /live/{match_id} here -> gets the current projected score / win prob.

Endpoints:
  GET  /health
  POST /predict          stateless; you send the innings timeline, get a prediction
  GET  /live/{match_id}  server fetches the scorer API and predicts the current ball
  POST /report           full post-match analysis from both innings' timelines
"""
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from feature_builder import build_features
import predictor
import report as report_mod

app = FastAPI(title="Cricket Realtime Predictor", version="1.0")
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
    timeline: List[Ball]

class ReportReq(BaseModel):
    meta: dict                       # {match_no, venue, batting_first, second}
    innings1: List[Ball]
    innings2: List[Ball] = []
    target: Optional[int] = None
    result: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok", "models": ["projected_score", "win_prob"]}

@app.post("/predict")
def predict(req: PredictReq):
    tl = [b.model_dump() for b in req.timeline]
    if not tl:
        raise HTTPException(400, "timeline is empty")
    feats = build_features(tl, innings=req.innings, batting_team=req.batting_team,
                           bowling_team=req.bowling_team, venue=req.venue,
                           target=req.target)
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
    return out

@app.post("/report")
def make_report(req: ReportReq):
    match = {"meta": req.meta,
             "innings": {1: [b.model_dump() for b in req.innings1],
                         2: [b.model_dump() for b in req.innings2]},
             "target": req.target, "result": req.result}
    return report_mod.generate_report(match)
