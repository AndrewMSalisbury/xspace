"""Build the Expected Space timeline (one row per sampled frame) for whole matches.

    uv run python scripts/build_timeline.py --source pff --match all
    uv run python scripts/build_timeline.py --source all --workers 16
    uv run python scripts/build_timeline.py --source idsse --match J03WMX --force

Writes data/processed/timeline/{source}_{match}.parquet, stamped with the git SHA and a hash
of every setting (config.settings_dict). Up-to-date files (same params hash) are skipped unless
--force. Matches are prepared one at a time in this process; their frames are spread over the
worker pool in chunks, so memory stays at about one match.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from xspace.config import DEFAULT_PARAMS, DEFAULT_TIMELINE, TIMELINE_DIR, params_hash, settings_dict
from xspace.metrics.timeline import (
    build_timeline,
    read_metadata,
    timeline_metadata,
    write_timeline,
)
from xspace.pipeline import match_ids, prepare_match


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="idsse", choices=["idsse", "pff", "all"])
    ap.add_argument("--match", nargs="+", default=["all"])
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                    help="processes computing frames (each needs only ~100 MB)")
    ap.add_argument("--force", action="store_true", help="rebuild up-to-date timelines")
    args = ap.parse_args()

    sources = ["idsse", "pff"] if args.source == "all" else [args.source]
    jobs = [(s, m) for s in sources
            for m in (match_ids(s) if args.match == ["all"] else args.match)]
    current = params_hash(settings_dict(DEFAULT_PARAMS, DEFAULT_TIMELINE))

    start = time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=args.workers)
    try:
        for i, (source, match_id) in enumerate(jobs, 1):
            path = TIMELINE_DIR / f"{source}_{match_id}.parquet"
            if (not args.force and path.exists()
                    and read_metadata(path).get("params_hash") == current):
                print(f"[{i}/{len(jobs)}] {source} {match_id}: up to date")
                continue
            t0 = time.perf_counter()
            try:
                pm = prepare_match(source, match_id)
                t1 = time.perf_counter()
                try:
                    df = build_timeline(pm.match, pm.phases, pm.flags, executor=pool)
                except BrokenProcessPool:
                    # A worker died (on Windows, e.g. WinError 6 at start-up or low memory): a
                    # broken pool fails every later task, so replace it and retry once.
                    print(f"[{i}/{len(jobs)}] {source} {match_id}: worker pool broke; retrying",
                          flush=True)
                    pool.shutdown(wait=False, cancel_futures=True)
                    pool = ProcessPoolExecutor(max_workers=args.workers)
                    df = build_timeline(pm.match, pm.phases, pm.flags, executor=pool)
            except Exception as e:  # keep going; report the failure
                print(f"[{i}/{len(jobs)}] {source} {match_id}: FAILED {type(e).__name__}: {e}")
                continue
            meta = timeline_metadata(source, pm.match, sync_pct=round(pm.sync.pct_synced, 2))
            write_timeline(df, path, meta)
            ok = int((df["status"] == "ok").sum())
            print(f"[{i}/{len(jobs)}] {source} {match_id}: {ok}/{len(df)} frames computed, "
                  f"prepare {t1 - t0:.0f}s, compute {time.perf_counter() - t1:.0f}s", flush=True)
    finally:
        pool.shutdown()
    print(f"done in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
