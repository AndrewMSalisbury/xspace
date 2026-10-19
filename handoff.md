# Handoff — Xspace

Last updated: 2026-10-09. Read with [docs/PLAN.md](docs/PLAN.md) (the roadmap),
[docs/methodology.md](docs/methodology.md) (how everything is computed) and
[docs/validation.md](docs/validation.md) (Phase 4 results).

## Where things stand

| | |
|---|---|
| Branches | PRs #1–#5 merged (Phase 4 = #5, 2026-10-09). Follow-up checks on `phase4-followups` (PR #6). Phase 5 branches from `main`. |
| Phase | 0–2 ✅ · 3 ✅ (video eye test *assumed* passed, not recorded) · 4 ✅ (V3 trade-off accepted) · **5 next** |
| Tests | 63 passing (`pytest`, ~5 s), ruff clean |
| Data | 71 matches; timelines, actions, possessions, players rebuilt with the calibrated, ground-only physics (hashes: timeline `1dd23ad89967`, actions `7faba5708c37`, possessions `51510fcd8820`); pass sets and ablations in `data/processed/validation/` |

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
   plays. Harmless for totals and `missed`; biases `decision_gap` and space held. **Phase 5:**
   weight the best cell by the V1 distance prior before using it in ratings.
4. **Findings that changed with calibration:**
   - the transition "space behind" effect reversed;
   - early-possession xSpace predicts shots less strongly;
   - player split-half re-measured: space held per 90 r = 0.92 (32 players), 0.70 within
     position (`methodology.md` section 11).
5. **Exploitation rate is unreliable per team** (V4 reliability 0.37–0.54). **For Phase 5:**
   pool it over matches or shrink it towards the mean (empirical Bayes); don't rank it raw.
6. Carried over:
   - the video eye-test checklist (`scripts/eye_test.py`): **Andrew is doing this by hand**;
   - PFF 3845 estimated players;
   - web skeleton;
   - re-download PFF 10510 / 10511;
   - set-piece / transition windows.

## Next steps

1. Merge PR #6 (follow-up checks; docs only).
2. Andrew: the video eye test (open issue 6).
3. Phase 5 (ratings), on a new branch from `main`:
   - lean on team xSpace (stable), not the exploitation rate, which needs pooling / shrinkage;
   - frame ratings as space / progression, not danger (V3 decision);
   - weight the best cell by the distance prior before it feeds `decision_gap` / space held.
