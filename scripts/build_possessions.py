"""Build Phase 3 roll-ups for whole matches: possessions (one row per possession) and players
(one row per player: minutes, own actions, credit for the space they held).

    uv run python scripts/build_possessions.py --source all
    uv run python scripts/build_possessions.py --source idsse --match J03WMX --force

Needs the match's timeline and action files, built with the current settings (run
build_timeline.py and build_actions.py first). Writes data/processed/possessions/ and
data/processed/players/{source}_{match}.parquet, stamped like them. No physics: each match
takes about as long as loading it, so whole matches run in parallel (each worker holds one
match in memory).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from xspace.config import (
    ACTIONS_DIR,
    DEFAULT_EXPLOITATION,
    DEFAULT_PARAMS,
    PLAYERS_DIR,
    POSSESSIONS_DIR,
    TIMELINE_DIR,
    params_hash,
    settings_dict,
)
from xspace.metrics.players import build_players
from xspace.metrics.possessions import (
    build_possessions,
    possessions_metadata,
    possessions_settings,
)
from xspace.metrics.timeline import read_metadata, read_timeline, write_timeline
from xspace.pipeline import match_ids, prepare_match


def build_one(source: str, match_id: str) -> str:
    """Top-level so worker processes can import it."""
    pm = prepare_match(source, match_id)
    timeline, _ = read_timeline(TIMELINE_DIR / f"{source}_{match_id}.parquet")
    actions, _ = read_timeline(ACTIONS_DIR / f"{source}_{match_id}.parquet")
    df = build_possessions(pm.match, pm.events, pm.phases, timeline, actions)
    write_timeline(df, POSSESSIONS_DIR / f"{source}_{match_id}.parquet",
                   possessions_metadata(source, pm.match))
    players = build_players(pm.match, actions)
    write_timeline(players, PLAYERS_DIR / f"{source}_{match_id}.parquet",
                   possessions_metadata(source, pm.match))
    return f"{len(df)} possessions, {int(df['shot'].sum())} with a shot; {len(players)} players"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="idsse", choices=["idsse", "pff", "all"])
    ap.add_argument("--match", nargs="+", default=["all"])
    ap.add_argument("--workers", type=int, default=3,
                    help="matches loaded at once (each needs ~1-2 GB)")
    ap.add_argument("--force", action="store_true", help="rebuild up-to-date files")
    args = ap.parse_args()

    sources = ["idsse", "pff"] if args.source == "all" else [args.source]
    jobs = [(s, m) for s in sources
            for m in (match_ids(s) if args.match == ["all"] else args.match)]
    inputs = {
        TIMELINE_DIR: params_hash(settings_dict(DEFAULT_PARAMS)),
        ACTIONS_DIR: params_hash(settings_dict(DEFAULT_PARAMS,
                                               exploitation=DEFAULT_EXPLOITATION)),
    }
    current = params_hash(possessions_settings())

    todo = []
    for source, match_id in jobs:
        name = f"{source}_{match_id}.parquet"
        stale = [d.name for d, h in inputs.items()
                 if not (d / name).exists() or read_metadata(d / name).get("params_hash") != h]
        if stale:
            print(f"{source} {match_id}: SKIPPED, {' and '.join(stale)} missing or out of date")
        elif not args.force and all(
                (d / name).exists() and read_metadata(d / name).get("params_hash") == current
                for d in (POSSESSIONS_DIR, PLAYERS_DIR)):
            print(f"{source} {match_id}: up to date")
        else:
            todo.append((source, match_id))

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_one, s, m): (s, m) for s, m in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            source, match_id = futures[fut]
            try:
                msg = fut.result()
            except Exception as e:  # keep going; report the failure
                msg = f"FAILED {type(e).__name__}: {e}"
            print(f"[{i}/{len(todo)}] {source} {match_id}: {msg}", flush=True)
    print(f"done in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
