"""Run the full data layer on every match and summarise it (Phase 1 audit).

    uv run python scripts/data_audit.py --workers 2

Per match: ball-in-play minutes, quality flags, event sync, phases, goalkeeper changes and
dismissals. Writes data/processed/data_audit.csv and docs/data_audit.md (summary stats only).
Run scripts/sync_report.py first so all tracking is cached; otherwise this is slow.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from xspace.io.events import load_events, pff_match_ids
from xspace.io.lineups import remove_sent_off_players
from xspace.io.loaders import CACHE_DIR, IDSSE_MATCHES, load_match
from xspace.io.sync import synchronise
from xspace.phases.possession import label_phases
from xspace.phases.quality import flag_summary, quality_flags

DOCS = Path(__file__).resolve().parents[1] / "docs"


def audit(source: str, match_id: str) -> dict:
    match = load_match(source, match_id)
    events, rep = synchronise(load_events(source, match_id), match, check=False)
    sent_off = remove_sent_off_players(match, events)
    phases = label_phases(match, events)
    flags = flag_summary(quality_flags(match))

    gk_changes = sum(int(np.sum(np.diff(gk[gk >= 0]) != 0))
                     for gk in (match.home_gk, match.away_gk))
    event_periods = set(events["period"].unique())
    missing_periods = sorted(int(p) for p in event_periods - set(np.unique(match.period)))
    side = phases.possession_side
    return {
        "source": source,
        "match_id": match_id,
        "home": match.home.name,
        "away": match.away.name,
        "in_play_min": match.n_frames / match.frame_rate / 60,
        "clean_pct": flags["clean"],
        "ball_missing_pct": flags["ball_missing"],
        "few_players_pct": flags["few_players"],
        "jump_pct": flags["ball_jump"] + flags["player_jump"],
        "events": len(events),
        "synced_pct": rep.pct_synced,
        "actor_ball_median_m": rep.median_actor_ball_m,
        "actor_within_3m_pct": rep.pct_actor_within_3m,
        "max_abs_offset_s": max((abs(v) for v in rep.offsets_s.values()), default=0.0),
        "residual_offset_s": rep.residual_offset_s,
        "open_play_pct": 100 * float(phases.open_play.mean()),
        "transition_pct": 100 * float(phases.transition.mean()),
        "home_possession_pct": 100 * float(np.mean(side[side >= 0] == 0)),
        "possessions": int(len(np.unique(phases.possession_id[phases.possession_id >= 0]))),
        "gk_changes": gk_changes,
        "sent_off": len(sent_off),
        "tracking_missing_periods": ",".join(map(str, missing_periods)),
    }


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    def summary(d: pd.DataFrame) -> str:
        cols = {
            "in_play_min": "Ball in play (min)", "clean_pct": "Clean frames %",
            "ball_missing_pct": "Ball missing %", "synced_pct": "On-ball events synced %",
            "actor_ball_median_m": "Actor–ball median (m)",
            "actor_within_3m_pct": "Actor within 3 m %",
            "max_abs_offset_s": "Max |clock offset| (s)", "open_play_pct": "Open play %",
            "transition_pct": "Transition %", "possessions": "Possessions",
        }
        q = d[list(cols)].describe(percentiles=[0.05, 0.5, 0.95]).T
        q = q[["min", "5%", "50%", "95%", "max"]].round(2)
        q.index = [cols[c] for c in q.index]
        return q.to_markdown()

    lines = [
        "# Data audit",
        "",
        f"Generated {date.today().isoformat()} by `scripts/data_audit.py` (summary statistics "
        "only; no raw data). Definitions are in [methodology.md](methodology.md).",
        "",
        "How to read this:",
        "",
        "- *Actor–ball* is the distance between the acting player and the tracking ball at the "
        "synced event frame. For PFF it is ~0 m by construction (the smoothed ball is pulled "
        "onto the player at tagged events), so it confirms timing, not ball accuracy.",
        "- *Clock offset* is the per-period shift applied to event times. PFF frames are "
        "tagged with event ids, so its offsets are ~0; IDSSE's vary by match and half.",
        "- IDSSE's lower *actor within 3 m* reflects per-event timing noise in DFL events "
        "(candidate fix: per-event refinement).",
        "- Matches with **no tracking for periods 3–4** lose all extra-time frames; their "
        "extra-time events are kept but unsynced.",
        "",
    ]
    for source, d in df.groupby("source"):
        lines += [f"## {source.upper()} ({len(d)} matches)", "", summary(d), ""]
        notes = []
        for _, r in d.iterrows():
            bits = []
            if r.gk_changes:
                bits.append(f"{r.gk_changes} GK change(s)")
            if r.sent_off:
                bits.append(f"{r.sent_off} sent off")
            if r.tracking_missing_periods:
                bits.append(f"**no tracking for period(s) {r.tracking_missing_periods}**")
            if bits:
                notes.append(f"- {r.match_id} ({r.home} v {r.away}): " + "; ".join(bits))
        if notes:
            lines += ["Match notes:", "", *notes, ""]
    lines += [
        "## Per match",
        "",
        df[["source", "match_id", "home", "away", "in_play_min", "clean_pct", "synced_pct",
            "actor_within_3m_pct", "max_abs_offset_s", "open_play_pct",
            "home_possession_pct"]].round(1).to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    jobs = [("idsse", m) for m in IDSSE_MATCHES] + [("pff", m) for m in pff_match_ids()]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(audit, s, m): (s, m) for s, m in jobs}
        for fut in as_completed(futures):
            s, m = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"{s} {m}: {type(e).__name__}: {e}")
            print(f"[{len(rows)}/{len(jobs)}] {s} {m}", flush=True)

    df = pd.DataFrame(rows).sort_values(["source", "match_id"]).reset_index(drop=True)
    df.to_csv(Path(CACHE_DIR) / "data_audit.csv", index=False)
    write_markdown(df, DOCS / "data_audit.md")
    print(f"wrote {DOCS / 'data_audit.md'}")


if __name__ == "__main__":
    main()
