# Deployment Guide

Three pieces, deployed in this order:

1. **Scorer API** — already live (your Stage-1 app on Render).
2. **Prediction service** — the FastAPI app in `app/` (this repo).
3. **Webapp** — the static page in `webapp/index.html`.

Deploy the prediction service first (you need its URL for the webapp), then the webapp.

---

## 1. Prediction service (FastAPI)

Two ways — pick one.

### Option A — Render, native Python (simplest)
1. Push this folder to a GitHub repo.
2. Render → **New → Blueprint**, pick the repo. It reads `render.yaml` and creates the web service.
   (Or **New → Web Service** manually with:
   Build `pip install -r requirements.txt`,
   Start `uvicorn main:app --app-dir app --host 0.0.0.0 --port $PORT`.)
3. Env var: `SCORER_BASE = https://cricket-scorer-pk3j.onrender.com`.
4. Deploy → note the URL, e.g. `https://cricket-predictor.onrender.com`.

### Option B — Docker (any host: Render, Railway, Fly, a VPS)
```bash
docker build -t cricket-predictor .
docker run -p 8000:8000 -e SCORER_BASE=https://cricket-scorer-pk3j.onrender.com cricket-predictor
```
The image already installs `libgomp1` (LightGBM needs it) and honours `$PORT`.

### Verify
```bash
curl https://YOUR-PREDICTOR-URL/health
# {"status":"ok","models":["projected_score","win_prob"]}
```
Interactive docs: `https://YOUR-PREDICTOR-URL/docs`.

> **Render free tier** spins the service down when idle, so the first request after a
> pause takes ~30–60s (cold start). For a smoother showcase, use a paid instance or ping
> `/health` every ~10 min with a free cron (e.g. cron-job.org).

---

## 2. Webapp (static page)

`webapp/index.html` has no build step. Before deploying for live use, edit the `CONFIG`
block near the top:
```js
const CONFIG = {
  DEMO_MODE: false,                                  // was true
  SCORER_BASE: "https://cricket-scorer-pk3j.onrender.com",
  PREDICTOR_BASE: "https://YOUR-PREDICTOR-URL",       // from step 1
};
```
Leave `DEMO_MODE:true` if you just want the offline showcase.

Host it anywhere static:
- **Render → New → Static Site**: publish directory `webapp`, no build command.
- **Netlify / Vercel**: drag-and-drop the `webapp` folder, or connect the repo.
- **GitHub Pages**: put `index.html` on a `gh-pages` branch.

---

## 3. Wiring / CORS

- **Predictor → browser:** already allows all origins (CORS is open in `main.py`). Nothing to do.
- **Scorer → browser:** in live mode the webapp calls the scorer directly (`/api/matches`,
  the SSE `/stream`). Your scorer must send `Access-Control-Allow-Origin` for the webapp's
  domain. If the browser console shows CORS errors, enable CORS on the scorer for that origin.
- **Timeline mapping:** confirm `app/scorer_client.py` → `ball_from_event()` and
  `_timeline_list()` match your real `/api/matches/:id` payload. Send me one sample response
  and I'll lock this exactly.

---

## Going live — end-to-end check
1. `curl https://PREDICTOR/health` → ok.
2. Score a test match on the scorer so a `match_id` exists and has a few balls.
3. `curl https://PREDICTOR/live/THAT_MATCH_ID` → returns a prediction (not a 502).
4. Open the webapp (DEMO_MODE:false) → the live tile shows the match; opening it shows the
   score, win bar / projected score, and worm updating.

---

## Retraining (as your scorer logs real matches)
1. Rebuild the two training tables from a refreshed master, drop them in `train/data/`.
2. `python train/train.py` → overwrites the model files in `models/`.
3. Commit the new `models/` and redeploy the prediction service. No webapp change needed.
