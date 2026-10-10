"""Write a video eye-test checklist of PFF moments (Phase 3 definition of done).

    uv run python scripts/eye_test.py            # 8 exploited, 8 missed, 4 controls
    uv run python scripts/eye_test.py --seed 1   # a different set of controls

Picks the top exploited moments (by chosen xSpace) and the top missed ones (by decision gap),
at most one of each per match, plus random ordinary forward passes as controls. Each item has
the match's PFF film-room link, the video time to seek to, and what the model claims, in words.
Writes data/processed/eye_test/checklist.md (not committed: it quotes PFF data).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from xspace.config import ACTIONS_DIR, CACHE_DIR
from xspace.io.events import load_events
from xspace.io.loaders import PFF_DIR
from xspace.metrics.space import ZONES
from xspace.metrics.timeline import read_timeline

OUT = CACHE_DIR / "eye_test" / "checklist.md"
LEAD_S = 6.0  # start watching this long before the event
QUESTIONS = {
    "exploited": "Was there really open, reachable, valuable space where the ball went, and "
                 "did the player find it on purpose?",
    "missed": "Was the model's best option really on (open, reachable, worth more than what was "
              "chosen)? Would a coach say the player missed it?",
    "control": "An ordinary pass: is it right that this is neither a standout nor a miss?",
}
HEADER = """# Video eye test — Phase 3

For each moment: open the film-room link, seek to the video time, watch ~10 s, then tick
**right**, **wrong** or **unsure** and add a note. Directions are from the attacking team's
view (left / right as they attack). Zones: *behind* = past the defensive line, *between* =
between defence and midfield, *wide* = outside the block, *in front* = in front of midfield.
xSpace numbers are xT·m² in one 1 m cell; "better than N%" ranks the cell among the frame's
cells with any xSpace.

Pass bar: a clear majority of `right` in each group (say 6 of 8), and the controls
unremarkable. Bring the ticked file back and the results go into methodology.md.
"""


def load_actions() -> pd.DataFrame:
    parts = []
    for path in sorted(ACTIONS_DIR.glob("pff_*.parquet")):
        df, m = read_timeline(path)
        df["match_id"], df["home"], df["away"] = m["match_id"], m["home"], m["away"]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    return df[df["status"] == "ok"].copy()


def video_info(match_id: str) -> tuple[str, dict[str, tuple[float, str]], pd.DataFrame]:
    """Film-room URL; event id → (video time s, formatted game clock); events by id."""
    meta = json.loads((PFF_DIR / "Metadata" / f"{match_id}.json").read_text(encoding="utf-8"))
    meta = meta[0] if isinstance(meta, list) else meta
    raw = json.loads((PFF_DIR / "Event Data" / f"{match_id}.json").read_text(encoding="utf-8"))
    times = {}
    for ev in raw:
        pe = ev.get("possessionEvents") or {}
        if ev.get("possessionEventId") is not None:
            times[str(ev["possessionEventId"])] = (float(ev["eventTime"]),
                                                   pe.get("formattedGameClock") or "?")
    events = load_events("pff", match_id).drop_duplicates("event_id").set_index("event_id")
    return meta.get("videoUrl") or "", times, events


def where(x: float, y: float, bx: float, by: float, zone: float) -> str:
    """A point relative to the ball, in the attacking frame (+x toward goal, +y to the left)."""
    dx, dy = x - bx, y - by
    side = "centre" if abs(y) < 9 else ("left" if y > 0 else "right")
    return (f"{abs(dx):.0f} m {'ahead' if dx >= 0 else 'back'}, {abs(dy):.0f} m "
            f"{'left' if dy > 0 else 'right'} of the ball ({side}, "
            f"{ZONES[int(zone)].replace('_', ' ')})")


def clock(seconds: float) -> str:
    s = max(seconds, 0.0)
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"


def pick(df: pd.DataFrame, by: str, n: int) -> pd.DataFrame:
    return df.sort_values(by, ascending=False).drop_duplicates("match_id").head(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="exploited and missed moments each")
    ap.add_argument("--controls", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acts = load_actions()
    names = pd.read_csv(PFF_DIR / "players.csv", dtype={"id": str}).drop_duplicates("id")
    names = names.set_index("id")["nickname"]

    def name(pid) -> str:
        return "?" if pd.isna(pid) else names.get(str(pid), f"player {pid}")

    exploited = pick(acts[acts["exploited"]], "chosen", args.n)
    missed = pick(acts[acts["missed"]], "decision_gap", args.n)
    ordinary = acts[acts["success"].fillna(False).astype(bool) & ~acts["exploited"]
                    & ~acts["missed"] & acts["chosen_rank"].between(0.4, 0.8)
                    & (acts["type"] == "pass")]
    controls = ordinary.sample(args.controls, random_state=args.seed)
    items = ([("exploited", r) for _, r in exploited.iterrows()]
             + [("missed", r) for _, r in missed.iterrows()]
             + [("control", r) for _, r in controls.iterrows()])

    cache: dict[str, tuple[str, dict, pd.DataFrame]] = {}
    lines = [HEADER]
    for i, (kind, r) in enumerate(items, 1):
        mid = r["match_id"]
        if mid not in cache:
            cache[mid] = video_info(mid)
        url, times, events = cache[mid]
        vt, game_clock = times.get(str(r["event_id"]), (np.nan, "?"))
        ev = events.loc[r["event_id"]]
        flip = -1.0 if r["team_side"] == 1 else 1.0  # events: home attacks +x
        bx, by = flip * ev["start_x"], flip * ev["start_y"]
        team = r["home"] if r["team_side"] == 0 else r["away"]
        done = "completed" if r["success"] is True else "not completed"
        went = where(r["chosen_x"], r["chosen_y"], bx, by, r["chosen_zone"])
        best = where(r["best_x"], r["best_y"], bx, by, r["best_zone"])
        seek = "?" if np.isnan(vt) else f"**{clock(vt - LEAD_S)}** (event at {clock(vt)})"
        lines += [
            f"## {i}. {kind.upper()} — {r['home']} v {r['away']}, {game_clock}", "",
            f"- **Video:** {url} — seek to {seek}",
            f"- **On the ball:** {name(r['player_id'])} ({team}), {r['type']}, {done}",
            f"- **Ball went:** {went}, space owned by {name(r['owner_id'])}; xSpace "
            f"{r['chosen']:.4f}, better than {r['chosen_rank']:.0%} of the frame's open cells",
            f"- **Model's best option:** {best}, owned by {name(r['best_owner_id'])}; xSpace "
            f"{r['best']:.4f}",
            f"- **Question:** {QUESTIONS[kind]}",
            "- [ ] right  - [ ] wrong  - [ ] unsure",
            "- Notes:", "",
        ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(items)} moments from {len(cache)} matches)")


if __name__ == "__main__":
    main()
