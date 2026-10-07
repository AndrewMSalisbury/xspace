# Handoff — Xspace

Last updated: 2026-10-01. Read with [docs/PLAN.md](docs/PLAN.md) (the roadmap) and
[docs/methodology.md](docs/methodology.md) (how everything is computed).

## Where things stand

| | |
|---|---|
| Branches | `phase1-data-layer` → **PR #1** open into `main`. `phase2-timeline` is branched from it (merge #1 first, then open the Phase 2 PR; or retarget). |
| Phase | 0 ✅ · 1 ✅ (M1) · 2 ✅ (M2, Oct 1 — four weeks early) · **3 next** |
| Tests | 32 passing (`pytest`, ~3 s), ruff clean |
| Data | All 71 matches cached in `data/processed/`; all 71 timelines in `data/processed/timeline/` |

## What Phase 2 built

| Module | Does |
|---|---|
| `src/xspace/config.py` | One place for `PhysicsParams`, `TimelineConfig` (2 m grid, 5 Hz, quality mask), paths, `settings_dict` / `params_hash`, `git_sha`. Old import paths still work. |
| `src/xspace/pipeline.py` | `prepare_match(source, id)`: the whole Phase 1 data layer in one call. |
| `src/xspace/metrics/timeline.py` | `build_timeline`: one row per sampled frame with status, labels and metrics; Parquet I/O with metadata. |
| `src/xspace/metrics/space.py` | New `space_from_arrays` (array-level core); `frame_space(..., side=)` accepts event-based possession. |
| `src/xspace/physics/pitch_control.py` | ~6× faster pitch control and reachability (float32, multiplicative logistic, active-cell compaction). |
| `src/xspace/io/loaders.py` | `recentre_periods`: fixes PFF extra time, applied on load. |
| `scripts/build_timeline.py` | `--source idsse|pff|all --match ... --workers 16 --force`; skips up-to-date files. |
| `notebooks/timeline_sanity.ipynb` | Coverage, one match over time, turnovers, shape checks, team averages. |

```python
from xspace.pipeline import prepare_match
from xspace.metrics.timeline import build_timeline, read_timeline

pm = prepare_match("pff", "3828")
df = build_timeline(pm.match, pm.phases, pm.flags)          # serial; pass executor= for parallel
df, meta = read_timeline("data/processed/timeline/pff_3828.parquet")
```

## Things learned the hard way

- **Performance**: frame cost was 36 ms (2 m grid); now 6.6 ms. Parallel scaling on this
  laptop flattens out (~9× at 24 workers); **30 workers broke the Windows process pool**. Use 16.
- **PFF extra time** (10506, 10508, 10517) came out of kloppy 3.19 shifted by (±105, ±68) m.
  Phase 1's audit missed it (it checks timing, not positions). Fixed on load; a quality flag
  for off-pitch positions would catch the next one of these.
- **float32 vs float64** pitch control differs only where a cell stops at the convergence
  tolerance one step apart (≤ 1e-3; median 1e-6). Tests keep a float64 oracle.
- **Turnovers**: total xSpace is *not* higher in transitions; *behind* space is (it rises ~8 s
  after winning the ball). The team that just won the ball controls less space. Matters for
  how Phase 3 defines "space conceded" vs. "space usable".
- `nbformat` / `nbclient` are dev deps now; notebooks can be executed headless.
- Phase 1 notes still apply: per-period timestamps, events-based possession, `--workers 2` for
  *uncached* PFF loads, `uv` not on PATH (use `.venv/Scripts/python.exe`).

## Open issues

1. **Merge PR #1**, then open the Phase 2 PR from `phase2-timeline`.
2. **PFF 3845 (Qatar)**: second-half xSpace median 3.2 vs ~1.2 typical. Frames look plausible,
   but the far side is often empty, which is where PFF's *estimated* players would show.
   Ties into issue 5 below and Phase 4's broadcast-vs-optical comparison.
3. **Web skeleton** (planned alongside Phase 2): not started.
4. Carried over from Phase 1: re-download PFF 10510 / 10511 (no extra-time tracking); IDSSE
   per-event sync refinement (**needed before Phase 3**); tune set-piece / transition windows;
   PFF "estimated" visibility flag; possession vs official stats; card/sub `team_side = -1`.
5. Consider an **off-pitch quality flag** (positions > a few m outside the pitch).
6. Timeline frames in plots are connected across set-piece gaps; fine for sanity checks, but
   break lines at gaps for published figures.

## Next steps — Phase 3: exploitation

Goal: from "space existed" to "space was used (or wasted)". See PLAN.md Phase 3.

1. **First, IDSSE per-event sync refinement** (pass-release frames must be exact).
2. `metrics/exploitation.py`: at each open-play pass / cross / carry release frame compute
   `xspace_available`, `xspace_best`, `xspace_chosen` (PFF `targetPlayerId`; IDSSE end
   location), `xt_gained_actual`, `decision_gap`, `exploited`, `zone_targeted`. Reuse
   `space_from_arrays` on a 1–2 m grid; the timeline's 5 Hz rows are too coarse for release
   frames.
3. Off-ball runners: per-player control shares (`ControlSurface.attack_players`).
4. "Moments" notebook: top 20 exploited / missed opportunities (IDSSE for frame-level figures).
