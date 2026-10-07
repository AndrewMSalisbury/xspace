# Handoff — Xspace

Last updated: 2026-10-06. Read with [docs/PLAN.md](docs/PLAN.md) (the roadmap) and
[docs/methodology.md](docs/methodology.md) (how everything is computed).

## Where things stand

| | |
|---|---|
| Branches | PR #1 (Phase 1) and PR #2 (Phase 2) merged. **Phase 3 PR** open from `phase3-exploitation` into `main`. |
| Phase | 0 ✅ · 1 ✅ (M1) · 2 ✅ (M2) · **3 core done** (event metrics + moments notebook); possession-level metrics and video check remain |
| Tests | 39 passing (`pytest`, ~5 s), ruff clean |
| Data | 71 matches cached; 71 timelines in `data/processed/timeline/`; 71 action files in `data/processed/actions/` |

## What Phase 3 built so far

| Module | Does |
|---|---|
| `io/sync.py` | `refine_release_frames`: per-event release frame (the kick) for IDSSE, stored as `release_frame`. `frame` and everything Phases 1–2 built on it is unchanged. `synchronise(..., refine=True)`; `prepare_match` turns it on for IDSSE. |
| `metrics/exploitation.py` | `build_actions`: one row per open-play pass / cross / carry, at the release frame on a 1 m grid: available / best / chosen xSpace, rank, decision gap, zone, xT gained, exploited / missed, space owner. |
| `config.py` | `ExploitationConfig` (grid, thresholds), `ACTIONS_DIR`. Its settings join the params hash for action files only Current hashes: timeline `89404c61b49c`, actions `d29ce1be84de`. |
| `metrics/timeline.py` | `output_metadata` shared by timeline and action files. |
| `scripts/build_actions.py` | Like `build_timeline.py`; all 71 matches in ~10 min. |
| `notebooks/moments.ipynb` | Coverage, sanity checks, zones, teams / players, top 20 exploited / missed (IDSSE figures, PFF tables). |

```python
from xspace.pipeline import prepare_match
from xspace.metrics.exploitation import build_actions

pm = prepare_match("idsse", "J03WMX")     # events have release_frame
acts = build_actions(pm.match, pm.events, pm.phases, pm.flags)  # pass executor= for parallel
```

## Things learned the hard way

- **Release frames**: picking the frame where the actor is closest to the ball is ambiguous
  (they're close the whole time they dribble). Score the *jump* in the ball's speed away from
  the actor, counting only outward movement, or a reception / the flight after the kick wins.
  Validated on PFF, whose tagged frames are exact: 87–91% within 2 frames. IDSSE passes heading
  toward their end location: 48–57% → 77–87%.
- "% heading toward end location" looked capped at 63% until I noticed a quarter of IDSSE
  passes (the incomplete ones) have no end location. Check the denominator.
- **Long passes are under-rated** by reachability (fixed 15 m/s ball, ground lanes): completed
  55% of the time where the model says ~unreachable. `chosen_rank` therefore favours short,
  safe progressions. First Phase 4 task: calibrate reach against observed completion.
- **Missed moments** lean on six-yard-box cells, where borrowed xT is very high.
- `Path.write_text` on Windows writes CRLF; the repo normalises to LF, but use `write_bytes`
  (or `newline=""`) for scripted edits to avoid noisy diffs.
- Build outputs while tracked files are being edited get stamped `-dirty`; commit first.
- Earlier notes still apply: 16 workers (30 broke the Windows pool), `--workers 2` for
  *uncached* PFF loads, `uv` not on PATH (use `.venv/Scripts/python.exe`).

## Open issues

1. **Review and merge the Phase 3 PR** (merge commit, not squash).
2. **Reachability for long passes** (above): Phase 4, before ratings.
3. **Thresholds** for `exploited` / `missed` are starting values (≈ 1–2% of actions each).
4. **Eye test against video**: sample top / bottom PFF moments via `videoUrl` (the notebook
   lists them). Needs a human.
5. **PFF 3845 (Qatar)** shows up again among the top missed moments: far side often empty
   (estimated players). Phase 4's broadcast-vs-optical comparison.
6. Carried over: web skeleton not started; re-download PFF 10510 / 10511; tune set-piece /
   transition windows; PFF "estimated" visibility flag; off-pitch quality flag; break plot
   lines at gaps for published figures.

## Next steps

1. **Possession-level metrics** (PLAN Phase 3): xSpace conceded per possession, time-to-exploit
   after a turnover, max xSpace reached, box entry / shot at the end. Join timeline rows to
   possessions via `possession_id`.
2. **Runner credit**: aggregate `owner_id` over high-xSpace targets → "space occupied" per player.
3. Then Phase 4 (validation and calibration), starting with long-pass reachability.
