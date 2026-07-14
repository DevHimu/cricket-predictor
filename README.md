# Cricket Realtime Predictor

Prediction + analysis service for the IPL-2026 scorer project. Trained on your
master ball-by-ball dataset (74 matches). Serves two things per ball and a
post-match report:

- **1st innings** → projected final score (regression)
- **2nd innings** → calibrated win probability (classification)
- **After the match** → analysis report (phase breakdown, top partnerships, win-probability turning points)

The webapp (Stage 3) calls this service over HTTP.

## What's inside
```
app/
  feature_builder.py   ONE feature function used by BOTH training and serving (no skew)
  predictor.py         loads models, returns projected score / calibrated win prob
  report.py            post-match analysis report
  scorer_client.py     adapts your scorer API (/score, /matches/:id) into model inputs
  main.py              FastAPI service (endpoints below)
models/                trained LightGBM models, encoders, calibrator, metrics.json
train/
  train.py             reproducible training (run to retrain as you log more matches)
  data/                the two labelled tables (so training runs offline)
simulate/
  replay_demo.py       offline end-to-end proof — no live server needed
  sample_matches.csv   a few real matches for the demo
Dockerfile, render.yaml, requirements.txt
```

## Quickstart (local)
```bash
pip install -r requirements.txt

# 1) See it work end-to-end, offline:
python simulate/replay_demo.py 7      # try 2, 5, 7

# 2) Run the API:
uvicorn main:app --app-dir app --reload
# -> http://127.0.0.1:8000/health   and interactive docs at /docs
```

## Endpoints
- `GET /health` — liveness.
- `POST /predict` — stateless. You send the innings timeline; get a prediction.
  ```json
  {"innings":2,"batting_team":"Mumbai Indians","bowling_team":"Chennai Super Kings",
   "venue":"Wankhede Stadium, Mumbai","target":180,
   "timeline":[{"runs_of_bat":1,"extras":0,"is_wicket":0,"is_legal":1}, ...]}
  ```
- `GET /live/{match_id}` — the service fetches your scorer API and predicts the current ball.
- `POST /report` — full post-match analysis from both innings' timelines.

## Wiring the webapp (Stage 3)
Simplest reliable pattern:
1. Frontend subscribes to the scorer SSE stream `/api/matches/:id/stream`.
2. On every ball event, the frontend calls **`GET /live/{match_id}`** here and updates the score/win-bar.
3. When the match ends, call **`POST /report`** with both innings' balls to render the report page.

Alternatively the frontend can accumulate the timeline itself and call `POST /predict` directly (no scorer round-trip from this service).

### One thing to confirm before live use
`app/scorer_client.py` → `ball_from_event()` and `_timeline_list()` map your
`/api/matches/:id` timeline into the four fields the model needs
(`runs_of_bat, extras, is_wicket, is_legal`). I wrote them against your `/score`
sample and common timeline shapes; check the field names match your actual
`/matches/:id` payload and adjust if needed. Everything else is derived. Set the
scorer URL via the `SCORER_BASE` env var (defaults to your Render URL).

## Deploy
- **Render:** push this folder; `render.yaml` is ready (or point a Web Service at the Dockerfile).
- **Docker:** `docker build -t cricket-predictor . && docker run -p 8000:8000 cricket-predictor`

## Model quality (honest, match-level cross-validation — see `models/metrics.json`)
- Projected score MAE ≈ 27 runs overall, ~21 at the death, ~33 in the powerplay.
- Win prob: accuracy 0.77, ROC-AUC 0.86, calibrated Brier 0.14. Probabilities are
  isotonic-calibrated, so "75%" really is ~75%.
- These are prototype-grade on 74 matches and will improve as your own scorer logs
  real games — just re-run `python train/train.py` on the refreshed tables.

## Notes / limitations
- Trained on **20-over** innings only; the feature builder reads match length live,
  but the model shouldn't be trusted on other formats without matching training data.
- Player identity is intentionally not a feature (too sparse for 74 matches).
- A handful of source matches have off-by-one/truncated final balls; the win rule
  (reach target, or finish with balls+wickets left) handles them consistently in
  both labels and reports.

## Stage 3 — the webapp (now served BY this service)
The dashboard is served at the service root `/` — the same service that runs the
model. Open `https://YOUR-SERVICE.onrender.com/` and it loads in demo mode. To use
real live matches, set `DEMO_MODE:false` in `webapp/index.html` (leave `API_BASE:""`
— it's same-origin) and redeploy. See DEPLOYMENT.md for the full guide.
