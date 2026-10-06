# Handoff — Xspace

Last updated: 2026-10-01. Read with [docs/PLAN.md](docs/PLAN.md) (the roadmap) and
[docs/methodology.md](docs/methodology.md) (how everything is computed).

## Where things stand

| | |
|---|---|
| Branch | `phase1-data-layer` — **pushed to `origin`** on 2026-10-01 (5 commits ahead of `main`, incl. this handoff). **No PR opened yet.** |
| Phase | 0 ✅ · 1 ✅ (milestone M1 reached Oct 1, two weeks early) · **2 next** |
| Tests | 21 passing (`pytest`), ruff clean |
| Data | All 71 matches (7 IDSSE + 64 PFF) load, sync and are cached in `data/processed/` |

Commits on the branch:

| Commit | What |
|---|---|
| `1abea95` | Project plan, event loaders, event ↔ tracking sync |
| `5d75fea` | Phase-of-play labels, per-frame goalkeepers, restart snapping |
| `5cd5bff` | Quality flags, card events, sent-off player removal |
| `8d9a33b` | Data audit over all 71 matches |
| (latest) | This handoff file |

## What Phase 1 built

| Module | Does |
|---|---|
| `src/xspace/io/events.py` | PFF + IDSSE events → one `MatchEvents` table, same coordinates (home attacks +x) and clock as tracking. Includes cards and subs. |
| `src/xspace/io/sync.py` | Per-period clock offset, event → frame mapping, sync report + check (`synchronise`). |
| `src/xspace/io/lineups.py` | Removes sent-off players' tracks after their card (PFF keeps a ghost). |
| `src/xspace/io/loaders.py` | Now also assigns the goalkeeper **per frame** (`gk_at`); done on load, so caches stay valid. |
| `src/xspace/phases/possession.py` | Per-frame possession, possession id, set-piece windows, transitions, thirds (`label_phases`). |
| `src/xspace/phases/quality.py` | Per-frame bit flags: ball missing, too few players, ball/player jumps. |
| `scripts/sync_report.py` | Sync check over all matches of a source → `data/processed/sync_report_{source}.csv`. |
| `scripts/data_audit.py` | Whole data layer on all 71 matches → `docs/data_audit.md` + CSV. |

Typical use per match:

```python
from xspace.io.loaders import load_match
from xspace.io.events import load_events
from xspace.io.sync import synchronise
from xspace.io.lineups import remove_sent_off_players
from xspace.phases.possession import label_phases
from xspace.phases.quality import quality_flags

match = load_match("pff", "3828")
events, report = synchronise(load_events("pff", "3828"), match)
remove_sent_off_players(match, events)
phases = label_phases(match, events)
flags = quality_flags(match)
```

## Things learned the hard way

- **PFF event rows**: a touch-then-pass is two rows (`IT` + `PA`) of one game event. Use each
  row's `eventTime`, not the shared `startTime`. PFF gives no end locations; we use the ball at
  the next event.
- **PFF ball**: kloppy loads the *smoothed* ball, which PFF pulls onto the player at tagged
  events. So PFF actor–ball distance is ~0 m by construction. Event freeze frames use the *raw*
  ball (≈2.4 m different).
- **IDSSE clock offsets** range from −1.2 to +1.3 s and differ between halves of one match, so
  they are estimated per period. Even after that, only 54–65% of on-ball events have the player
  within 3 m of the ball.
- **Restart events** are often stamped up to ~1.5 s before the first live frame; sync snaps them
  forward (≤ 3 s).
- **Timestamps restart each period.** Any `searchsorted` over time must stay inside one period
  (this caused set-piece windows to run on for a whole half).
- **Possession** is built from events, not tracking `ball_owner` (IDSSE's flickers every ~3 s).
- **Extra-time PFF matches** have no period start times in metadata; the kick-off event is used.
- **PFF 10510 and 10511 have no extra-time tracking** in the raw files (periods 1–2 only).
- All 71 pitches are 105 × 68 m, so per-stadium dimensions were skipped.

## Environment notes

- `uv` is not on PATH in some shells. Use `.venv/Scripts/python.exe`, or the full path
  `C:/Users/andre/AppData/Local/Microsoft/WinGet/Packages/astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe/uv.exe`.
- First-time PFF tracking loads take minutes and ~3–4 GB RAM each. **Use `--workers 2`**; 4
  workers ran the machine out of memory. Everything is cached now, so reruns are fast.
- Caches are pickles in `data/processed/` (`idsse_*.pkl`, `pff_*.pkl`, ~110–170 MB each).
- New dev dependency: `tabulate` (for the audit's markdown tables).

## Open issues

1. **Open a PR** for Phase 1 from `phase1-data-layer` into `main` (the plan asks for one PR per
   phase). The branch is already pushed; everything is committed.
2. **Re-download PFF 10510 / 10511 tracking** to check whether the missing extra time is a
   truncated download or missing at source.
3. **IDSSE per-event sync refinement** (ETSY-style, Van Roy et al. 2021): search a small window
   around each event for the frame where the ball is with the actor. Needed before Phase 3,
   which depends on exact pass-release frames.
4. **Tune set-piece and transition windows** (currently 8 s / 4 s / 2 s and 10 s) by watching
   examples. Transitions cover ~28% of play, which may be generous.
5. **PFF "estimated" player visibility** isn't flagged: kloppy drops it. Would need reading the
   raw `homePlayersSmoothed` / `awayPlayersSmoothed` visibility fields ourselves.
6. Possession split vs. official stats wasn't checked (no offline source).
7. PFF card and sub rows have `team_side = -1`; derive it from rosters if it's ever needed.

## Next steps — Phase 2: match timeline engine

Goal (from the plan): compute xSpace for every frame of every match, fast enough to iterate.

1. **`src/xspace/config.py`**: one place for physics params, grid size (2 m for timelines),
   sample rate (5 Hz), paths.
2. **`src/xspace/metrics/timeline.py`**: for each match, sample every 5th frame (25 Hz) / 6th
   (29.97 fps), skip non-open-play and flagged frames (store a reason code instead), and compute
   one row per frame: totals per zone, best cell, mean control, line positions, block width,
   compactness, ball xT, phase labels, quality flags. Use the Phase 1 pipeline above for inputs.
3. **Parallelism**: `ProcessPoolExecutor` over frame chunks. Measure first; current cost is
   ~0.2 s/frame at 1 m on one core. Optimise in the plan's order (restrict to reachable cells,
   vectorise, float32, Numba, GPU last).
4. **Output**: Parquet per match in `data/processed/timeline/{source}_{match}.parquet`, with git
   SHA + params hash in the metadata.
5. **`scripts/build_timeline.py --source pff --match all`**.
6. **Tests**: determinism; parallel == serial.
7. **Sanity notebook**: xSpace over a match per team with goals marked; expect spikes after
   turnovers and no set-piece frames.

Done when all 71 timelines build reproducibly (target ≤ 1.5 h for everything, ≤ 2 min per
match) and the plots show sensible patterns.

Also planned alongside Phase 2: a **web skeleton** (routing + match explorer with IDSSE
positions + client-side pitch control), to retire front-end risk early.
