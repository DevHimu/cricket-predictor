# Deployment Guide — Single Service

Everything (frontend + model + API) is now ONE service. You upload one thing to
Render, and it serves both the webpage and the predictions on the same URL.
No separate static site, no CORS, no frontend URL to configure.

## What runs where
- **Cricket Scorer** — your existing Node app (already live). Unchanged.
- **Cricket Predictor (this repo)** — one Render **Web Service** that serves:
  - `GET /`            → the dashboard webpage
  - `GET /health`      → liveness check
  - `POST /predict`    → prediction from a timeline
  - `GET /live/:id`    → fetch scorer + predict current ball
  - `POST /report`     → post-match analysis
  - `GET /api/matches` → proxy to the scorer's match list (keeps the browser same-origin)

## Deploy (one service)
1. Push this `cricket-predictor` folder to a GitHub repo.
2. Render → **New → Web Service** → pick the repo. It uses the Dockerfile
   (or the `render.yaml` blueprint).
3. Set environment variable: `SCORER_BASE = https://cricket-scorer-pk3j.onrender.com`
4. Deploy. Open the service URL — the dashboard loads immediately in **demo mode**.

That's it. The page and the API are the same service, so it "just works".

### Verify
- Open `https://YOUR-SERVICE.onrender.com/`      → dashboard loads (demo data).
- Open `https://YOUR-SERVICE.onrender.com/health` → `{"status":"ok",...}`.
- Open `https://YOUR-SERVICE.onrender.com/docs`   → interactive API tester.

## Going live (real matches instead of demo)
The default is demo mode so the site always works out of the box. To use real
live matches from your scorer, edit ONE line in `webapp/index.html`:
```js
const CONFIG = { DEMO_MODE: false, API_BASE: "" };   // was DEMO_MODE: true
```
Leave `API_BASE` as `""` — the frontend talks to this same service. Redeploy.
Now the home page lists matches from your scorer (`/api/matches`), and opening a
live match polls `/live/:id` every 2 seconds to update the score, win bar,
projected score, worm, and over-by-over chart.

## Why it wasn't working before
The old setup had the webpage and the API on two different URLs. The webpage was
also demo-only and never actually called the API. Now they are one service on one
origin, and the frontend really calls `/live`, `/predict`, and `/api/matches`.

## If the service is asleep
Render's free tier spins down after ~15 min idle → the first request takes
30–60s (cold start). Pre-warm by opening `/health` before a demo, or ping it
every 10 min with a free cron (cron-job.org).

## Docker note
The image installs `libgomp1` (LightGBM needs it) and honours `$PORT`. Nothing to
change — `docker build . && docker run -p 8000:8000 -e SCORER_BASE=... image` works
locally too, then open http://localhost:8000/.

## Retraining
Run `python train/train.py` to regenerate the model files in `models/`, commit,
and redeploy. No frontend change needed.
