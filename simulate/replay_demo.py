"""Offline end-to-end demo: replays a real match ball-by-ball through the exact
same feature builder + models the live service uses, printing predictions as
they evolve, then prints the post-match report. No live server needed.

Run:  python simulate/replay_demo.py [match_no]   (default 7)
"""
import os, sys, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from feature_builder import build_features
import predictor
from report import generate_report

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = list(csv.DictReader(open(os.path.join(HERE, "sample_matches.csv"))))
MATCH = sys.argv[1] if len(sys.argv) > 1 else "7"

def ball(r):
    return {"runs_of_bat": int(r["runs_of_bat"]), "extras": int(r["extras"]),
            "is_wicket": int(r["is_wicket"]), "is_legal": int(r["is_legal_delivery"])}

inn = {1: [], 2: []}
meta_rows = [r for r in ROWS if r["match_no"] == MATCH]
for r in meta_rows:
    if r["innings"] in ("1", "2"):
        inn[int(r["innings"])].append(r)

teamA = inn[1][0]["batting_team_full"]; teamB = inn[2][0]["batting_team_full"]
venue = inn[1][0]["venue"]; target = int(inn[2][-1]["target"])

print(f"\n=== REPLAY: Match {MATCH} — {teamA} vs {teamB} @ {venue} ===\n")

print(f"--- 1st innings ({teamA}) : live projected final score ---")
tl = []
for i, r in enumerate(inn[1]):
    tl.append(ball(r))
    f = build_features(tl, innings=1, batting_team=teamA, bowling_team=teamB, venue=venue)
    if f["balls_bowled"] % 18 == 0 and f["balls_bowled"] > 0:      # every 3 overs
        p = predictor.predict(f, 1)
        print(f"  {f['balls_bowled']//6:>2} ov  {p['current']:>7}  "
              f"CRR {p['crr']:>5}  ->  projected {p['projected_score']:.0f}")
actual1 = sum(b['runs_of_bat'] + b['extras'] for b in tl)
print(f"  ACTUAL final: {actual1}\n")

print(f"--- 2nd innings ({teamB}) : live win probability  (target {target}) ---")
tl = []
for r in inn[2]:
    tl.append(ball(r))
    f = build_features(tl, innings=2, batting_team=teamB, bowling_team=teamA,
                       venue=venue, target=target)
    if f["balls_bowled"] % 18 == 0 and f["balls_bowled"] > 0:
        p = predictor.predict(f, 2)
        wp = p["win_probability"][teamB]
        bar = "#" * int(wp * 20)
        print(f"  {f['balls_bowled']//6:>2} ov  {p['current']:>7}  need {p['runs_to_win']:>3} "
              f"in {p['balls_remaining']:>3}  RRR {p['rrr']:>5}  |{bar:<20}| {int(wp*100):>3}% {teamB}")

print("\n=== POST-MATCH REPORT ===")
match = {"meta": {"match_no": MATCH, "venue": venue, "batting_first": teamA, "second": teamB},
         "innings": {1: [ball(r) for r in inn[1]], 2: [ball(r) for r in inn[2]]},
         "target": target}
rep = generate_report(match)
print(rep["text_summary"])
print("\nPhase breakdown (1st inns):", rep["first_innings"]["phase_breakdown"])
print("Top partnerships (1st inns):", rep["first_innings"]["top_partnerships"])
print("Turning points (2nd inns):", rep["second_innings"]["turning_points"])
