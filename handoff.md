# Handoff — Xspace

Last updated: 2026-10-18. Read with [docs/PLAN.md](docs/PLAN.md) (the roadmap),
[docs/methodology.md](docs/methodology.md) (how everything is computed) and
[docs/validation.md](docs/validation.md) (Phase 4 results).

## Where things stand

| | |
|---|---|
| Branches | PRs #1–#6 merged (#6 on 2026-10-18). Phase 5 on `phase5-ratings` (local, not pushed yet). |
| Phase | 0–2 ✅ · 3 ✅ (video eye test *assumed* passed, not recorded) · 4 ✅ · **5 in progress** |
| Tests | 73 passing (`pytest`, ~8 s), ruff clean |
| Data | 71 matches; actions rebuilt with the likely cell, possessions / players with the new thresholds; ratings in `data/processed/ratings/` |

## What Phase 5 has so far (methodology.md section 12)

- **Likely best cell** on every action: xSpace × `exp(−d / 10 m)` (V1 prior). Edge-row best
  cells 27% → 0.7%. `likely_gap` and space held (`n_likely_space`, threshold 0.003) use it.
  The players' thresholds are now in the possession settings hash.
- **`metrics/ratings.py`, `scripts/build_ratings.py`** (~10 s): team-match ratings (attack /
  defence, 80% possession-bootstrap intervals, 5 grades, summary line), team profiles, leak
  maps, player ratings (carriers, runners; empirical-Bayes shrinkage towards position means).
- **Player directory** (`io.lineups.player_directory`): PFF rosters fix 153 `UNK` starters;
  cached in `ratings/directory.parquet`.
- **First checks**: Spain concede least, Costa Rica most; team xSpace created / conceded vs
  shots per possession r = 0.56 / 0.45 (32 teams); space created follows possession share
  (0.50). Runners: ST > AM > W > CM > FB > CB; within-position split-half 0.74.

## What Phase 4 changed

- **Physics fitted to 45k PFF pass outcomes** (`config.PhysicsParams`; the old values are
  `UNCALIBRATED_PARAMS`).
  - Two model changes came first. Each defender now gets one interception chance per lane, not
    one per lane sample (`lane_combine="max"`), and an `intercept_factor` (fitted 0.152) sets
    how often a defender who reaches the lane in time actually cuts the ball out.
  - Held-out completion log loss went from 0.633 to 0.351.
- **xSpace is ground-only** (`air_speed = 0`). The fitted lofted ball (20.8 m/s, 1.32 s hang
  time) is used only by the pass-completion model (`validation/pass_model.py`,
  `calibration.json`). Taking the better of the two balls per cell made xSpace worse at V1 and
  V3.
- **V1 now includes a distance prior**, P ∝ exp(β·s − γ·d). Distance alone beats every surface;
  the old V1 numbers were mostly the slow Phases 0–3 ball acting as that prior.
- **The joint fit** (`validation/joint.py`, `scripts/fit_joint.py`) fits the physics to
  destinations and completion together. It wasn't adopted: +0.01 nats on V1, −0.02 on V2.
- **Thresholds rescaled** to the new xSpace scale (about 5 per team-match, where it used to be
  about 1.15): `exploit_min` 0.01, `missed_best_min` 0.045.
- New modules: `validation/{passes,pass_model,calibrate,joint,ablations,report}.py`.
  New scripts: `build_pass_set`, `calibrate`, `fit_joint`, `build_ablations`, `validate`.

## Things learned the hard way

- **Fitting completion only on attempted passes** can't see lanes nobody tried. Unbounded, the
  fit runs away to an instant ball, so it needs bounds (`calibrate.BOUNDS`).
- **Score destinations against a distance baseline.** Without one, any physics that makes far
  cells worse looks good.
- **PFF `highPointType`** (peak height) separates ground and lofted passes. `ballHeightType` is
  the height at contact.
- **Long jobs get killed here.** Claude Code's background shells are reaped under memory
  pressure (Discord, League and Chrome use most of the 32 GB) and stopped after 2 h.
  - One rebuild hung its process pool after 34 matches.
  - Run long builds in your own terminal: `.\.venv\Scripts\python.exe scripts\...` in
    PowerShell. The timeline, actions and possession builds skip up-to-date files, so they
    resume. `fit_joint.py --resume` restarts from its checkpoint.
- **Notebooks:** run headless with `nbclient` (installed; nbconvert isn't). Set
  `PYTHONIOENCODING=utf-8` when printing their outputs.
- Earlier notes still apply:
  - `write_bytes` for scripted edits (CRLF);
  - commit before building, or outputs get stamped `-dirty`;
  - ≤ 16 workers;
  - `uv` isn't on PATH.

## Open issues

1. **V3: the Phases 0–3 physics predicts danger better** (PFF shot gain 1.24% against 0.75%,
   box entry 1.16% against 0.33%), through its heavier lane blocking. **Accepted for now**
   (2026-10-09, `validation.md` "Decision: keep the calibrated physics"): ratings are framed
   as space / progression, not danger. Revisit in Phase 6 (PLAN step 4 there):
   - score the joint fit's physics (`intercept_factor` 0.49, in `joint.json`) on V3;
   - fit reach to V3 directly.
2. **Calibrated physics is worse than the old on IDSSE V1** (0.71 against 0.80 nats, forward
   passes). The fit used PFF only.
3. **Touchline best cells** (30% of frames in the edge row, 17% on the far side): eye-checked
   2026-10-09 (`methodology.md` section 8). Real space, but a ~67 m ground pass almost nobody
   plays. Harmless for totals and `missed`; biases `decision_gap` and space held. **Done
   (Phase 5):** ratings use the likely cell (distance-weighted).
4. **Findings that changed with calibration:**
   - the transition "space behind" effect reversed;
   - early-possession xSpace predicts shots less strongly;
   - player split-half re-measured: space held per 90 r = 0.92 (32 players), 0.70 within
     position (`methodology.md` section 11).
5. **Exploitation rate is unreliable per team** (V4 reliability 0.37–0.54). **Done
   (Phase 5):** shrunk with an empirical-Bayes beta-binomial before grading.
6. Carried over:
   - the video eye-test checklist (`scripts/eye_test.py`): **Andrew is doing this by hand**;
   - PFF 3845 estimated players;
   - web skeleton;
   - re-download PFF 10510 / 10511;
   - set-piece / transition windows.

## Next steps

1. Phase 5, remaining:
   - leaderboard notebook with sanity checks (low-block teams and space behind; summaries
     read right);
   - PFF grade benchmark for players (event `grades`, `passerGrade`, ...);
   - V5 write-up (ratings vs results) in validation.md;
   - defender rating (experimental);
   - decide whether space held should be normalised by team possession (Spain's players lead).
2. Andrew: the video eye test (open issue 6).
3. Deferred: a heat map of all space conceded needs per-cell sums in the timeline (rebuild).
