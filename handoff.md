# Handoff — Xspace

Last updated: 2026-10-18. Read with [docs/PLAN.md](docs/PLAN.md) (the roadmap),
[docs/methodology.md](docs/methodology.md) (how everything is computed) and
[docs/validation.md](docs/validation.md) (Phase 4 and V5 results).

## Where things stand

| | |
|---|---|
| Branches | PRs #1–#6 merged (#6 on 2026-10-18). Phase 5 on `phase5-ratings`, PR #7. Phase 6 branches from `main` once it merges. |
| Phase | 0–2 ✅ · 3 ✅ (video eye test *assumed* passed, not recorded) · 4 ✅ · 5 ✅ pending Andrew's leaderboard eye test · **6 next** |
| Tests | 74 passing (`pytest`, ~5 s), ruff clean |
| Data | 71 matches; actions (likely cell, responsible defender), possessions, players rebuilt at `ba096a4`; ratings in `data/processed/ratings/` |

## What Phase 5 did (methodology.md section 12, validation.md V5)

- **Likely best cell** on every action: xSpace × `exp(−d / 10 m)` (the V1 prior). Edge-row
  best cells fall from 27% to 0.7%. `likely_gap` and space held (`n_likely_space`, ≥ 0.003)
  use it. `likely_defender_id` is the outfield defender quickest to that cell.
- **Players** get minutes in / out of possession. Runners are graded per 90 *in possession*,
  and defenders (experimental) per 90 out of possession.
- **`metrics/ratings.py`, `scripts/build_ratings.py`** (~10 s):
  - team-match ratings: attack / defence, 80% possession-bootstrap intervals, 5 grades and a
    summary line;
  - team profiles and leak maps;
  - player ratings: carriers, runners, defenders, with empirical-Bayes shrinkage towards
    position means.
- **Player directory** (`io.lineups.player_directory`): the PFF rosters fix 153 `UNK`
  starters. Cached in `ratings/directory.parquet`.
- **`validation/pff_grades.py`**: PFF analysts' event grades and "better option" flags.
- **`notebooks/ratings.ipynb`**: leaderboards, sanity checks, PFF benchmark, V5.
- **V5 results**:
  - team ratings track shots for and against (ρ 0.38–0.66 over 32 teams);
  - player ratings don't agree with PFF grades (|ρ| ≤ 0.16 within position);
  - the likely gap doesn't separate PFF's better-option flags (AUC 0.50);
  - deep blocks concede *more* value-weighted space behind (the low-block check fails as
    posed).

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

- **Up-to-date checks compare settings, not code.** A code change to a build (e.g. a new
  action column) needs `--force`. A threshold change in `config.py` / `players.py` triggers
  rebuilds by itself.

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

1. Merge PR #7 (Phase 5) after Andrew's eye test of `notebooks/ratings.ipynb` (PLAN's
   definition of done: leaderboards pass an expert eye test).
2. Andrew: the video eye test (open issue 6).
3. Phase 5 follow-ups, if wanted:
   - a relative decision gap: the absolute one penalises carriers in rich areas;
   - an area-only "behind", to test the low-block hypothesis (validation.md V5);
   - a heat map of all space conceded: needs per-cell sums in the timeline (rebuild); the
     leak map stands in.
4. Phase 6 (own value model). It also revisits V3 (danger) and may help the player ratings,
   which have no outside support yet.
