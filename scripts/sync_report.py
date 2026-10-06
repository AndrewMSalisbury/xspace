"""Load events + tracking, synchronise them, and report alignment quality per match.

    uv run python scripts/sync_report.py --source idsse --match all
    uv run python scripts/sync_report.py --source pff --match 3812 10517 --workers 4

Writes data/processed/sync_report_{source}.csv. Loading PFF tracking for the first time is slow
(minutes per match); later runs use the cache in data/processed/.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from xspace.io.events import load_events, pff_match_ids
from xspace.io.loaders import CACHE_DIR, IDSSE_MATCHES, load_match
from xspace.io.sync import check_sync, synchronise


def one(source: str, match_id: str) -> dict:
    match = load_match(source, match_id)
    _, report = synchronise(load_events(source, match_id), match, check=False)
    row = report.as_dict()
    offsets = row.pop("offsets_s")
    row.update({f"offset_p{p}_s": v for p, v in offsets.items()})
    try:
        check_sync(report)
        row["ok"] = True
    except ValueError as e:
        row["ok"] = False
        print(e)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="idsse", choices=["idsse", "pff"])
    ap.add_argument("--match", nargs="+", default=["all"])
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel matches; uncached PFF loads need ~3-4 GB RAM each")
    args = ap.parse_args()

    ids = args.match
    if ids == ["all"]:
        ids = list(IDSSE_MATCHES) if args.source == "idsse" else pff_match_ids()

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, args.source, m): m for m in ids}
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:  # keep going; report the failure
                print(f"{futures[fut]}: {type(e).__name__}: {e}")
                rows.append({"match_id": futures[fut], "ok": False, "error": str(e)})
            print(f"[{len(rows)}/{len(ids)}] {futures[fut]}", flush=True)

    df = pd.DataFrame(rows).sort_values("match_id")
    out = Path(CACHE_DIR) / f"sync_report_{args.source}.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.round(2).to_string(index=False))
    print(f"\nwrote {out}; {int(df['ok'].sum())}/{len(df)} matches pass")


if __name__ == "__main__":
    main()
