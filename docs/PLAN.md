# Xspace — Project Plan

**Expected Space (xSpace):** how much reachable, valuable space a defence leaves open in soccer, and
how well the attacking team exploits it — measured from tracking data, every frame, and rolled up
into match ratings, team profiles and player ratings, presented through a public web app.

| | |
|---|---|
| Owner | Andrew Salisbury |
| Repo | https://github.com/AndrewMSalisbury/xspace |
| Plan written | 2026-09-30 |
| Target timeline | ~16 weeks: 2026-10-01 → 2027-01-20 (incl. holiday buffer) |
| Goals | (1) portfolio piece that stands up to expert scrutiny, (2) public interactive tool |
| Status | Phases 0–2 complete (M1, M2 reached 2026-10-01); Phase 3 core done 2026-10-06 (event metrics + moments notebook); possession-level metrics and video check remain |

---

## Contents

1. [Goals and non-goals](#1-goals-and-non-goals)
2. [Positioning: what's new here](#2-positioning-whats-new-here)
3. [Current state (Phase 0)](#3-current-state-phase-0)
4. [Data](#4-data)
5. [Architecture](#5-architecture)
6. [Phase 1 — Data layer](#phase-1--data-layer-weeks-12)
7. [Phase 2 — Match timeline engine](#phase-2--match-timeline-engine-weeks-34)
8. [Phase 3 — Exploitation: linking space to actions](#phase-3--exploitation-linking-space-to-actions-weeks-56)
9. [Phase 4 — Validation and calibration](#phase-4--validation-and-calibration-weeks-78)
10. [Phase 5 — Ratings: matches, teams, players](#phase-5--ratings-matches-teams-players-weeks-910)
11. [Phase 6 — Own value model](#phase-6--own-value-model-weeks-911-parallel)
12. [Phase 7 — Web app](#phase-7--web-app-weeks-1114-skeleton-earlier)
13. [Phase 8 — Write-up and launch](#phase-8--write-up-and-launch-weeks-1516)
14. [Timeline and milestones](#timeline-and-milestones)
15. [Engineering practices](#engineering-practices)
16. [Risks and mitigations](#risks-and-mitigations)
17. [Open questions](#open-questions)
18. [Stretch ideas](#stretch-ideas)
19. [References](#references)

---

## 1. Goals and non-goals

### Goals

- **A defensible metric.** xSpace must be clearly defined, grounded in published methods, and
  *validated* — shown to predict what happens next better than simpler baselines.
- **Three levels of output.**
  - *Match ratings* for every team in every analysed match.
  - *Team profiles*: where a defence leaks space; how an attack finds it.
  - *Player ratings*: ball carriers, off-ball runners, and (later) defenders.
- **A public web app** with two modes: scrub through a match frame by frame with live space
  overlays, and browse precomputed match / team / player pages.
- **Open source and reproducible.** Anyone with the free data can regenerate every number.
- **Portfolio quality.** Clean code, tests, CI, a methodology write-up, and a polished front end.

### Non-goals (for this version)

- Real-time / live-match analysis.
- Commercial product or paid data.
- Full possession-value model from scratch on tracking data (we'll fit an event-based one; see
  Phase 6).
- Video integration.

---

## 2. Positioning: what's new here

Space in soccer is well studied; the novelty has to come from the angle, the synthesis and the
quality of execution.

| Prior work | What it measures | What Xspace adds |
|---|---|---|
| Pitch control (Spearman 2017/2018) | Who controls each location | Weighted by reachability and value; zoned by defensive lines |
| OBSO (Spearman 2018) | Off-ball scoring opportunity | Value of *progression* (xT gained), not only scoring; available vs. used |
| Wide Open Spaces / SOG (Fernández & Bornn 2018) | Space occupation and generation | Defensive-shape zoning; match/team/player ratings |
| Dangerous Accessible Space (Bischofberger & Baca) | Reachable dangerous space | Explicit link to decisions (available vs. exploited); public tool |
| Pressing Intensity (Bekkers 2025) | Defensive pressure on players | The mirror image, from the *same* time-to-intercept model |

**The core contribution:** a frame-level measure of *exploitable* space (reachable + valuable +
located relative to the defensive shape), linked to what the attack *actually did*, producing:

1. **Space conceded** — a defensive metric.
2. **Space exploited / exploitation rate** — an attacking metric.
3. **Decision gap** — a player decision-making metric.

…validated against outcomes, and explorable in a public app.

---

## 3. Current state (Phase 0)

Done as of 2026-09-30:

- [x] Repo, MIT licence, CI (ruff + pytest + web build), README, methodology notes.
- [x] Loader: any kloppy provider → `MatchTracking` dense arrays; orientation normalised; cache.
- [x] Savitzky–Golay velocities, clipped at 12 m/s.
- [x] Vectorised Spearman pitch control (per-player shares kept).
- [x] Pass-lane reachability using the Pressing Intensity `1 − ∏(1 − p)` combination.
- [x] xT gained (Karun Singh 12×8 grid), area-weighted totals.
- [x] Defensive line detection (≥2 players per line, medians) and zones.
- [x] Web: React + TS + D3 single-frame viewer with layer toggles.
- [x] Data: IDSSE (7 matches) loads; PFF 2022 World Cup (64 matches) fully downloaded and loads.
- [x] Fixed: GK selection (rosters include bench keepers).

Measured performance: ~0.2 s per frame at a 1 m grid (single core).

---

## 4. Data

### 4.1 Sources

| Source | Matches | Tracking | Events | Licence | Public display |
|---|---|---|---|---|---|
| IDSSE (Bassek et al. 2025) | 7 (Bundesliga / 2. BL 2022/23) | 25 Hz optical (TRACAB) | DFL events | CC BY 4.0 | ✅ Yes, with attribution |
| PFF FC / Gradient Sports 2022 WC | 64 | 29.97 fps broadcast | PFF events (rich) | Free on request; terms TBC | ⚠️ Pending terms review |
| StatsBomb Open Data (Phase 6) | 3,000+ across competitions | — (360 freeze frames for some) | StatsBomb events | Open, attribution required | Derived models only |

### 4.2 PFF event data — what's available

Discovered in `data/raw/pff/Event Data/*.json` (one list of events per match):

- **`gameEvents.setpieceType`**: `O` open play, `T` throw-in, `F` free kick, `G` goal kick,
  `C` corner, `D` drop ball, `K` kick-off, `P` penalty. → **set-piece filtering**.
- **`possessionEvents.possessionEventType`**: `PA` pass, `CR` cross, `SH` shot, `BC` ball carry,
  `CH` challenge, `CL` clearance, `RE` rebound, `IT` initial touch, `TC` touch.
- Pass fields: `passerPlayerId`, **`targetPlayerId`** (intended), `receiverPlayerId` (actual),
  `passOutcomeType`, `passType`, `ballHeightType`, **`linesBrokenType`**, `accuracyType`.
- `homePlayers` / `awayPlayers` / `ball`: freeze-frame positions at the event.
- `startTime` / `endTime` (video seconds), `gameEvents.period`, `startGameClock`.
- `grades`: PFF's play-by-play player grades — an external benchmark for player ratings.

The *intended target* is unusually valuable: it lets us validate "where did the passer *want*
to go" separately from "where did the ball end up".

### 4.3 Known data-quality issues

| Issue | Source | Handling |
|---|---|---|
| Off-camera players estimated | PFF | Flag frames where players are estimated (check spec v2.2 for visibility fields); report sensitivity |
| Ball position missing (~6% of frames) | PFF | Skip frame; never interpolate across > 0.5 s |
| Pitch sizes vary | Both | v0 assumes 105×68; Phase 1 uses per-stadium dimensions (PFF metadata has them) |
| Substitutions / GK changes | Both | Per-frame on-pitch mask; per-frame GK identity (Phase 1) |
| Noisy velocities | Both | SG smoothing; validate against provider speed fields where present |
| Event/tracking clock offsets | Both | Sync step with tolerance checks (Phase 1) |

---

## 5. Architecture

```
            ┌────────────────────┐
 raw data → │ io/ loaders        │  kloppy → MatchTracking (+ MatchEvents)
            └─────────┬──────────┘
                      ▼
            ┌────────────────────┐
            │ phases/            │  possessions, set pieces, transitions, quality flags
            └─────────┬──────────┘
                      ▼
            ┌────────────────────┐
            │ physics/ value/    │  time-to-intercept, control, reach, value surfaces
            └─────────┬──────────┘
                      ▼
            ┌────────────────────┐
            │ metrics/           │  frame xSpace → event exploitation → ratings
            └─────────┬──────────┘
                      ▼
  data/processed/  Parquet per match (frames, events, ratings)
                      ▼
            ┌────────────────────┐
            │ export/            │  compact JSON / binary for the web
            └─────────┬──────────┘
                      ▼
               web/ (static site, GitHub Pages)
```

### Package layout (target)

```
src/xspace/
  constants.py
  config.py           # PhysicsParams, grid sizes, sample rates, paths (one place)
  io/
    loaders.py        # tracking (exists)
    events.py         # PFF + IDSSE events → MatchEvents (Phase 1)
    sync.py           # event ↔ frame alignment (Phase 1)
  phases/
    possession.py     # possession sequences, open play vs set piece, transitions
    quality.py        # per-frame quality flags
  physics/            # exists
  value/
    xt.py             # exists (borrowed grid)
    xt_fit.py         # own xT (Phase 6)
  metrics/
    space.py          # frame xSpace (exists)
    timeline.py       # whole-match computation (Phase 2)
    exploitation.py   # event-level metrics (Phase 3)
    ratings.py        # match / team / player (Phase 5)
  validation/         # Phase 4
  export/
    web.py            # exists
    site.py           # full site build (Phase 7)
scripts/              # thin CLIs over the package
notebooks/            # exploration only; nothing the pipeline depends on
web/                  # React app
docs/                 # PLAN.md, methodology.md, figures
```

### Storage formats

| Artefact | Format | Location | Committed? |
|---|---|---|---|
| Raw data | provider native | `data/raw/` | No |
| Loaded match cache | pickle | `data/processed/*.pkl` | No |
| Frame timeline | Parquet (one row per frame) | `data/processed/timeline/{source}_{match}.parquet` | No |
| Event metrics | Parquet | `data/processed/events/…` | No |
| Ratings | Parquet + CSV | `data/processed/ratings/` | No (published via site) |
| Web data | JSON + binary | `web/public/data/` | Only small IDSSE samples |

---

## Phase 1 — Data layer (weeks 1–2)

**Goal:** a clean, trustworthy, synchronised view of every match: tracking + events + phases.

### Tasks

1. **Event loaders** (`io/events.py`) ✅
   - PFF: custom parser (kloppy has no PFF event loader) → `MatchEvents` table:
     `event_id, period, time_s, frame, team_side, player_id, type, setpiece_type, start_xy,
     end_xy, target_player_id, receiver_player_id, outcome, lines_broken, height`.
   - IDSSE: `kloppy.sportec.load_open_event_data` → same schema.
2. **Event ↔ tracking sync** (`io/sync.py`) ✅ — per-period offsets; all 7 IDSSE pass.
   Follow-up: per-event refinement for IDSSE (only 52–62% of on-ball events within 3 m).
   - Map event times to frame indices per period.
   - Sanity check: ball within ~3 m of the event's player at the event frame; report
     median/95th-percentile mismatch per match; fail loudly if offsets are systematic.
3. ~~**Per-stadium pitch dimensions**~~ — not needed: all 71 matches are 105 × 68 m (checked
   2026-10-01). Revisit only if a new data source adds other sizes.
4. **Per-frame GK identity and on-pitch masks** (handles GK substitutions and red cards). ✅
   On-pitch mask = non-NaN position; 3 PFF matches change keeper mid-game.
5. **Phases of play** (`phases/possession.py`) ✅ — rules in `methodology.md`
   - Possession sequences (from events, cross-checked with tracking `ball_owner`).
   - `open_play` vs `set_piece` label per frame: a set piece "phase" lasts from the restart until
     the defending team has had ≥ N seconds to reorganise (start: until the ball is cleared from
     the box or 8 s pass, whichever first; tune by inspection).
   - Transition labels: first 10 s after a turnover.
   - Pitch thirds / build-up vs progression vs final third by ball x.
6. **Quality flags** (`phases/quality.py`): missing ball, fewer than 10 outfield players
   tracked, estimated players (if exposed by PFF), velocity spikes. ✅ (except estimated
   players: kloppy drops PFF visibility). Also: sent-off players' ghost tracks removed
   (`io/lineups.py`).
7. **Data audit** ✅ — `scripts/data_audit.py` → [`docs/data_audit.md`](data_audit.md).
   Possession split vs. official stats not done (no offline source for official figures).
   Findings: PFF 10510 / 10511 have **no extra-time tracking** in the raw files (re-download to
   confirm it isn't a truncated download).

### Deliverables

- `MatchEvents` for all 71 matches; sync report; phase labels.
- Tests: schema tests, sync tolerance test on a fixture, set-piece window logic on synthetic data.

### Definition of done

Every match loads with tracking + events + phases; median event/frame sync error < 0.2 s; audit
table committed to `docs/data_audit.md` (summary stats only, no raw data).

---

## Phase 2 — Match timeline engine (weeks 3–4)

**Goal:** compute xSpace for every frame of every match, fast enough to iterate.

### Design

- **Sampling:** 5 Hz (every 5th frame at 25 Hz, every 6th at 29.97). Ball-in-play, open-play,
  quality-passing frames only (other frames stored with a reason code).
- **Grid:** 2 m for timelines (≈1,800 cells), 1 m for figures.
- **Per-frame outputs** (one Parquet row): match, frame, period, time, attacking side, phase
  labels, totals per zone, best cell (x, y, value), mean control, offside / back / mid line x,
  block width, compactness (back–mid line distance), ball xT, quality flags.
- **Parallelism:** `concurrent.futures.ProcessPoolExecutor` over frame chunks (24 cores).
- **GPU path (optional):** port `pitch_control` / `pass_reachability` to PyTorch or CuPy
  (batched over frames). Only if CPU throughput is the bottleneck.

### Performance budget

| Config | Est. per match | Est. all 71 |
|---|---|---|
| 1 m grid, 1 core | ~60 min | ~70 h |
| 2 m grid, 1 core | ~15 min | ~18 h |
| 2 m grid, 24 cores | ~1 min | ~1–1.5 h |
| GPU batched | < 30 s | < 30 min |

Target: full recompute of all matches in **≤ 1.5 h** on this machine, single match in ≤ 2 min.

### Optimisations to try (in order)

1. Restrict pitch-control integration to cells within reach of the ball in ≤ 4 s.
2. Early exit per cell is already in place; vectorise the remaining loop over frames.
3. `float32` throughout.
4. Numba-jitted kernel if NumPy overhead dominates.
5. GPU.

### Deliverables

- `scripts/build_timeline.py --source pff --match all` ✅
- `metrics/timeline.py` + tests (determinism; parallel == serial results). ✅
- First exploratory plots: xSpace over a match, by team, with goals marked. ✅
  (`notebooks/timeline_sanity.ipynb`)

### Outcome (2026-10-01)

- `config.py` holds physics params, timeline settings and paths; outputs carry the git SHA and
  a hash of every result-affecting setting.
- **Optimisation needed only step 3 of the list, plus one trick**: float32, the logistic
  computed by repeated multiplication instead of `exp` per step, and dropping converged cells.
  Pitch control 24 → 4 ms, reachability 10 → 2 ms per frame (2 m grid); results match the
  float64 reference to ~1e-6. Numba / GPU not needed.
- Parallel scaling on this laptop flattens out (~9× at 24 workers; 30 workers broke the Windows
  process pool), so per-frame speed mattered more than core count. Default: 16 workers.
- All 71 timelines build in **22 min** (14–20 s per match), 1.28 M sampled frames, ~91 MB.
- The sanity check found a Phase 1 bug: **PFF extra time was loaded a whole pitch off**
  (kloppy); fixed on load (`recentre_periods`).
- Turnovers: space *behind* the defence rises for ~8 s after the ball is won; total xSpace
  doesn't, because the new attackers are still in defensive shape (see `methodology.md`).
- Web skeleton (planned alongside Phase 2) **not started**.

### Definition of done

All 71 timelines built reproducibly; a notebook shows sensible patterns (e.g. xSpace conceded
spikes after turnovers; set pieces excluded).

---

## Phase 3 — Exploitation: linking space to actions (weeks 5–6)

**Goal:** go from "space existed" to "space was used (or wasted)".

### Event-level metrics (passes, crosses, carries)

At the frame of release (pass/cross) or carry start:

| Metric | Definition |
|---|---|
| `xspace_available` | Total xSpace in the frame (and per zone) |
| `xspace_best` | Max cell xSpace, with location and zone |
| `xspace_chosen` | xSpace at the target location (intended target's position, or end location) |
| `xt_gained_actual` | xT(reception or carry end) − xT(origin), if successful; 0 / negative if lost |
| `decision_gap` | `xspace_best − xspace_chosen` (≥ 0); also as a percentile within the frame |
| `exploited` | 1 if the action moved the ball into a top-k% xSpace cell and was completed |
| `zone_targeted` | behind / between / wide / in front |

Use PFF's `targetPlayerId` for intent; fall back to end location for IDSSE.

### Player-level ingredients

- **Ball carriers:** exploitation rate, mean decision gap, "missed big chances" (frames where
  `xspace_best` was top-5% but the choice was in the bottom half).
- **Off-ball runners:** per-player control share (`ControlSurface.attack_players`) of high-xSpace
  cells → *space occupied*. Credit the player who *owns* the space the ball goes into.
- **Space creation (v2):** counterfactual — recompute defenders' control if the runner had stood
  still (approximate: freeze the runner at t−1 s) and measure how much xSpace opened elsewhere.
  Inspired by Fernández & Bornn's SOG. Only if time allows.
- **Defenders (v2):** responsibility via defender Voronoi / nearest-defender assignment of
  conceded high-xSpace cells. Flag as experimental.

### Possession / sequence level

- xSpace conceded per possession, time-to-exploit after a turnover, max xSpace reached in a
  possession, and whether the possession ended in a box entry / shot.

### Deliverables

- [x] `metrics/exploitation.py`, event metrics Parquet for all matches, tests on hand-built frames.
- [x] A "moments" notebook: top 20 exploited and top 20 missed opportunities (`notebooks/moments.ipynb`;
  frame figures from IDSSE, World Cup moments as tables until PFF's terms are confirmed).
- [x] Prerequisite: per-event release frames for IDSSE (`io.sync.refine_release_frames`).
- [x] Space owner per action (`owner_id`, `best_owner_id`): the runner-credit ingredient.
- [ ] Possession-level metrics (xSpace conceded per possession, time-to-exploit, box entry / shot).
- [ ] Eye test against video (PFF `videoUrl`) for a sample of top / bottom moments.

### Outcome so far (2026-10-06)

All 71 matches: 68,464 open-play actions, 96–99% computed, 9.6 min build. The control model
assigns the chosen space to the actual receiver 86% (PFF) / 77% (IDSSE) of the time. Exploited
moments look right; missed moments lean on six-yard-box cells (high borrowed xT). Main finding:
long passes are under-rated by reachability (55% completed where the model says ~unreachable),
so rank-based ratings wait for Phase 4 calibration. Details in `methodology.md` section 9.

### Definition of done

Event metrics computed for all open-play passes / crosses / carries; top/bottom moments pass the
eye test (watch a sample against video via PFF's `videoUrl` where accessible).

---

## Phase 4 — Validation and calibration (weeks 7–8)

**Goal:** show xSpace means something, and tune the physics honestly.

### Validation tasks

| # | Question | Target | Metric |
|---|---|---|---|
| V1 | Does xSpace predict *where* the next pass goes? | Pass end location / intended target | Log-likelihood of the observed target under a softmax over cells; top-k accuracy |
| V2 | Does reachability predict pass *success*? | `passOutcomeType` | AUC, calibration (reliability curve), Brier |
| V3 | Does frame xSpace predict near-term danger? | Box entry / shot / xG in next 10 s | AUC, partial correlation controlling for ball position |
| V4 | Is the team signal stable? | Split-half (odd/even possessions) | Correlation of team xSpace conceded/exploited |
| V5 | Do ratings relate to results? | Goals, xG, PFF grades | Correlation; directionality, not causality |

### Baselines and ablations

Pitch control only → control × xT → control × reach × xT → + zones. Also compare against a pure
location model (xT of the ball) to show tracking adds information. Each step must earn its place.

### Calibration

- Fit `reaction_time`, `max_speed`, `tti_sigma`, `ball_speed` (and possibly a lofted-pass
  speed) by maximising V1/V2 likelihood (Spearman 2018 did this for pitch control).
- Train/test split **by match** (no leakage); report test metrics only.
- Compare IDSSE (optical) vs PFF (broadcast) on the same metrics to quantify broadcast error.

### Deliverables

- `src/xspace/validation/`, `scripts/validate.py`, results tables + plots in
  `docs/validation.md`.
- Updated `PhysicsParams` defaults with provenance.

### Definition of done

Full model beats every ablation on held-out matches for V1–V3; any task where it doesn't is
reported honestly with a hypothesis.

---

## Phase 5 — Ratings: matches, teams, players (weeks 9–10)

**Goal:** turn metrics into ratings people can read.

### Match ratings (every team, every match)

For team T in match M, open play only:

- **Attack**
  - `xspace_created` — mean xSpace per second of T's possession (and per possession).
  - `xspace_exploited` — total xT gained from actions classed as exploiting space.
  - `exploitation_rate` — exploited / available.
  - Zone split: % behind / between / wide.
- **Defence**
  - `xspace_conceded` — mean xSpace per second of the *opponent's* possession.
  - Zone split of space conceded; conceded in transition vs settled.
  - `compactness` — mean back-to-mid-line distance; block width.
- **Headline grades**: each component scaled to a 0–100 percentile against all team-matches in
  the dataset, with a one-line natural-language summary
  (e.g. "Left space between the lines; opponents rarely used it").

### Team profiles (aggregated over a team's matches)

- Space conceded heatmap (pitch-shaped), by phase (build-up, progression, transition).
- "Where they leak": top zones vs tournament average.
- Attack: preferred zones exploited, time-to-exploit after turnover, exploitation rate.
- Uncertainty: bootstrap over possessions → 80% intervals. Teams with 3 matches will be noisy;
  say so on the page.

### Player ratings

| Role | Metrics | Minimum sample |
|---|---|---|
| Ball carrier | Exploitation rate, mean decision gap, missed big chances / 90 | ≥ 50 open-play passes |
| Runner / receiver | Space occupied / 90, high-xSpace receptions / 90 | ≥ 180 open-play minutes |
| Defender (experimental) | Space conceded in zone of responsibility / 90 | ≥ 180 minutes |

- **Small-sample shrinkage:** empirical-Bayes shrink each rate toward the position mean
  (beta-binomial for rates, normal for means). Show raw and shrunk values.
- **Benchmark:** correlate with PFF grades (expect positive but imperfect correlation —
  discuss where and why they disagree).

### Deliverables

- `metrics/ratings.py`, `scripts/build_ratings.py`, Parquet + CSV outputs.
- Notebook: tournament-wide leaderboards and team comparisons with sanity checks
  (e.g. do teams known for low blocks concede little "behind" space?).

### Definition of done

Every team-match has a rating; leaderboards pass expert eye-test; uncertainty shown everywhere.

---

## Phase 6 — Own value model (weeks 9–11, parallel)

**Goal:** replace the borrowed xT grid with a model we fit and can explain.

### Approach

1. **xT v1 (own):** classic Markov xT (Singh 2018) fitted on **StatsBomb Open Data** events
   (thousands of matches) + PFF events, on a finer grid (e.g. 24×16), with smoothing.
   Event data is plentiful even though tracking isn't.
2. **Context-aware v2 (stretch):** gradient-boosted model of "probability of a shot/goal within
   the next N actions" from location + basic context (phase, pressure from Pressing Intensity,
   number of defenders goal-side). Evaluate against xT v1.
3. Plug in through the existing `value(points)` interface; re-run Phases 2–5; compare validation.

### Deliverables

- `value/xt_fit.py`, fitted grid in the package with provenance, comparison in
  `docs/validation.md`.

### Definition of done

Own model ≥ borrowed xT on validation V1–V3, or a documented reason to keep the borrowed one.

---

## Phase 7 — Web app (weeks 11–14; skeleton earlier)

**Goal:** a fast, polished, static site that makes the work explorable.

### Pages

| Route | Content |
|---|---|
| `/` | Hook + 30-second explainer: one animated example of space opening and being used |
| `/how-it-works` | Interactive explainer: toggle control → reach → value → xSpace on the same frame |
| `/matches` | Match list with ratings summary (sortable) |
| `/matches/:id` | **Match explorer**: pitch + scrubber + timeline + event markers + report |
| `/teams`, `/teams/:id` | Team profiles: heatmaps, zone bars, per-match ratings |
| `/players`, `/players/:id` | Leaderboards with filters; player page with moments |
| `/methodology` | Rendered methodology + validation results |

### Match explorer (the showpiece)

- Pitch with player dots, velocity vectors, ball, defensive lines, layer toggle.
- **Scrubber**: play/pause, 0.5×/1×/2×/4×, frame step, keyboard shortcuts.
- **Timeline chart** under the pitch: xSpace for each team over the match; goals, shots and big
  "exploited" / "missed" moments marked; click to jump.
- Side panel: current frame totals by zone, best option vs chosen action.

### Rendering strategy (key decision)

Shipping precomputed surfaces is too heavy: 2 m grid × 4 layers × 5 Hz ≈ 130 MB per match.
**Plan: compute surfaces in the browser.**

- Ship only **positions + velocities** at 10 Hz (≈ 23 objects × 4 floats, quantised to
  int16 → ~1–2 MB per match gzipped) and precomputed **scalar timelines**.
- Port time-to-intercept / control / reach to TypeScript in a **Web Worker**; upgrade to a
  **WebGL/WebGPU fragment shader** if needed (the math is per-cell and embarrassingly parallel).
- **Parity tests:** Python and TS implementations must agree within 1e-3 on fixture frames (run in
  CI).
- Fallback: precomputed low-res (3 m) xSpace-only surfaces for devices without WebGL.

### Data licensing on the site

- IDSSE: full explorer, with attribution.
- PFF: **ratings and aggregates** published; **frame-level scrubbing for PFF only after the
  terms are confirmed** to allow it. The build has a per-source switch.

### Tech

React + TypeScript + Vite, React Router, D3 (scales, axes, shapes), Web Worker (+ WebGL),
CSS variables for light/dark, GitHub Pages via Actions. Performance budget: first load < 300 KB
JS gzipped; 60 fps scrubbing on a mid-range laptop.

### Accessibility and quality

Keyboard-operable scrubber; colour-blind-safe palettes (avoid red/green; test with a simulator);
text alternatives for charts (summary sentences); mobile layout down to 360 px.

### Deliverables

- Site live at `https://andrewmsalisbury.github.io/xspace/` (or a custom domain).
- `scripts/build_site_data.py` produces everything under `web/public/data/`.

### Definition of done

All pages live; Lighthouse ≥ 90 performance / accessibility; scrubbing smooth; parity tests green.

---

## Phase 8 — Write-up and launch (weeks 15–16)

### Deliverables

1. **Long-form article** (blog post / Substack-style): the question, the method in plain
   language, 3–4 standout findings from the World Cup, links to the app.
2. **Technical report** (`docs/report.md` → PDF; arXiv-style optional): full methodology,
   validation, limitations, related work.
3. **README polish**: demo GIF of the scrubber, badges, quick start.
4. **Launch**: share with the analytics community (e.g. Bluesky, Friends of Tracking, LinkedIn);
   tag the authors whose work this builds on.

### Definition of done

Article published; repo tagged `v1.0.0`; site linked from README and article.

---

## Timeline and milestones

| Weeks | Dates (2026–27) | Phase | Milestone |
|---|---|---|---|
| 0 | Sep 30 | Phase 0 | ✅ Pipeline + viewer + data |
| 1–2 | Oct 1 – Oct 14 | 1. Data layer | ✅ **M1:** events synced, phases labelled, audit done (Oct 1) |
| 3–4 | Oct 15 – Oct 28 | 2. Timeline engine | ✅ **M2:** all 71 match timelines built (Oct 1) |
| 5–6 | Oct 29 – Nov 11 | 3. Exploitation | **M3:** event metrics + "moments" notebook |
| 7–8 | Nov 12 – Nov 25 | 4. Validation | **M4:** validation report; tuned parameters |
| 9–10 | Nov 26 – Dec 9 | 5. Ratings (+6 in parallel) | **M5:** match/team/player ratings |
| 11 | Dec 10 – Dec 16 | 6. Value model wrap-up | **M6:** own xT in pipeline |
| 11–14 | Dec 10 – Jan 6 | 7. Web app | **M7:** site live (holidays = buffer) |
| 15–16 | Jan 7 – Jan 20 | 8. Write-up | **M8:** v1.0 launch |

A web **skeleton** (routing, match explorer with IDSSE positions, client-side control) starts
alongside Phase 2 so front-end risk is retired early.

### Checkpoints

At each milestone: update this plan (what changed and why), tag a release (`v0.1`, `v0.2`, …),
and post a short progress note — useful portfolio material in itself.

---

## Engineering practices

- **One config source** (`config.py`): physics params, grid sizes, sample rates, paths.
- **Tests**: unit tests on synthetic scenarios for every metric (known answers); regression
  tests on 2–3 fixture frames from IDSSE (small, licensable); Python ↔ TS parity tests.
- **CI**: ruff, pytest, web lint + build, parity tests. Heavy data jobs run locally, not in CI.
- **Reproducibility**: every output regenerated by scripts; a `Makefile`/`justfile` with targets
  `data`, `timeline`, `events`, `validate`, `ratings`, `site`.
- **Versioning outputs**: write the git SHA + params hash into every Parquet's metadata.
- **Notebooks** for exploration only; anything reused moves into the package.
- **Branches + PRs** for each phase, so the history reads well to reviewers.
- **Docs as you go**: `methodology.md` updated in the same PR as the code it describes.

---

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PFF terms forbid public frame-level visuals | Medium | Medium | Per-source switch; IDSSE carries the explorer; publish PFF aggregates only |
| Broadcast tracking noise distorts far-side space | High | Medium | Quality flags; IDSSE comparison; report sensitivity |
| xSpace doesn't beat baselines | Medium | High | Ablations will show which parts help; reframe as descriptive tool if needed — still publishable |
| Small samples → noisy player ratings | High | Medium | Shrinkage, minimum minutes, intervals, honest copy |
| Compute too slow | Low | Medium | 24 cores; 2 m grid; GPU path |
| Browser compute too slow for scrubbing | Medium | Medium | WebGL shader; fallback to precomputed low-res surfaces |
| Scope creep (space creation, defenders, own EPV) | High | Medium | Marked v2/stretch; only after M5 |
| Event/tracking sync errors | Medium | High | Explicit sync checks with thresholds in Phase 1 |
| Over-tuning parameters to the data | Medium | Medium | Train/test by match; report test only |

---

## Open questions

1. **PFF terms of use** — what exactly is allowed for public display of derived frame-level
   visuals? (Read the terms that came with access; email Gradient Sports if unclear.)
2. **Hosting** — GitHub Pages under the repo, or a custom domain (e.g. `xspace.<something>`)?
3. **Branding** — keep "Xspace / Expected Space" as the public name?
4. **Write-up venue** — personal blog, Substack, Medium, or a submission (e.g. a sports
   analytics conference / workshop)?
5. **Player ratings scope** — include the experimental defender ratings publicly, or keep them
   in the report only?
6. **More data** — add SkillCorner open data (10 A-League matches, broadcast) as a third source?

---

## Stretch ideas

- **Lofted passes**: second trajectory model (slower, can't be intercepted mid-flight, but
  receivable only after landing) — important for "behind" space.
- **Offside-aware reach**: receivers beyond the offside line at pass time don't count.
- **Space creation credit** (counterfactual runs).
- **Pressing × Space**: join with Pressing Intensity (unravelsports) — press resistance =
  xSpace found under high pressure.
- **Per-player physical params** (speed, reaction) estimated from tracking.
- **"What if" mode** in the app: drag a player and watch space change.
- **Similarity search**: find moments across matches with similar space configurations.

---

## References

- Spearman, W. (2018). *Beyond Expected Goals.* MIT Sloan Sports Analytics Conference.
- Spearman, W. et al. (2017). *Physics-Based Modeling of Pass Probabilities in Soccer.*
- Shaw, L. *LaurieOnTracking* (Friends of Tracking) — pitch control implementation.
- Fernández, J. & Bornn, L. (2018). *Wide Open Spaces: A statistical technique for measuring
  space creation in professional soccer.* MIT SSAC.
- Fernández, J., Bornn, L. & Cervone, D. (2019/2021). *A framework for the fine-grained
  evaluation of the instantaneous expected value of soccer possessions.*
- Fernández, J. & Bornn, L. (2020). *SoccerMap.*
- Singh, K. (2018). *Introducing Expected Threat (xT).*
- Bischofberger, J. & Baca, A. *Dangerous Accessible Space.*
- Llana, S., Madrero, P. & Fernández, J. (2020). *The right place at the right time: Advanced
  off-ball metrics for exploiting an opponent's spatial weaknesses in soccer.* MIT SSAC.
- Bekkers, J. (2025). *Pressing Intensity: An Intuitive Measure for Pressing in Soccer.*
  arXiv:2501.04712.
- Bassek, M. et al. (2025). *An integrated dataset of spatiotemporal and event data in elite
  soccer.* Scientific Data 12, 195.
- PFF FC / Gradient Sports (2024–25). *2022 FIFA World Cup dataset* and specifications.
