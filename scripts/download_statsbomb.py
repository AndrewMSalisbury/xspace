"""Download StatsBomb Open Data men's events and reduce them for the value model (Phase 6).

    uv run python scripts/download_statsbomb.py            # every comp-season in SEASONS
    uv run python scripts/download_statsbomb.py --workers 8

StatsBomb Open Data is free with attribution ("Data provided by StatsBomb"). Each match's raw
JSON (~3 MB) is reduced in memory (`io.statsbomb.reduce_events`) and only the reduced table is
kept: data/raw/statsbomb/{competition_id}_{season_id}.parquet (gitignored). Comp-seasons
already on disk are skipped, so an interrupted run resumes.

Men's competitions from 2003/04 on. The 2022 World Cup is left out on purpose: its matches are
the PFF tracking matches the validation tasks score on.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from xspace.config import RAW_DIR
from xspace.io.statsbomb import BASE_URL, reduce_events

OUT = RAW_DIR / "statsbomb"
EXCLUDED = {(43, 106)}  # FIFA World Cup 2022: the PFF validation matches
MIN_YEAR = 2003


def fetch(url: str, tries: int = 4) -> object:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def seasons() -> list[dict]:
    comps = fetch(f"{BASE_URL}/competitions.json")
    return [c for c in comps
            if c["competition_gender"] == "male"
            and int(c["season_name"][:4]) >= MIN_YEAR
            and (c["competition_id"], c["season_id"]) not in EXCLUDED]


def one_match(match_id: int) -> pd.DataFrame:
    return reduce_events(fetch(f"{BASE_URL}/events/{match_id}.json"), match_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = seasons()
    catalogue = []
    start = time.perf_counter()
    for i, c in enumerate(todo, 1):
        cid, sid = c["competition_id"], c["season_id"]
        matches = fetch(f"{BASE_URL}/matches/{cid}/{sid}.json")
        catalogue += [{"competition_id": cid, "season_id": sid,
                       "competition": c["competition_name"], "season": c["season_name"],
                       "match_id": m["match_id"], "match_date": m["match_date"]}
                      for m in matches]
        path = OUT / f"{cid}_{sid}.parquet"
        label = f"[{i}/{len(todo)}] {c['competition_name']} {c['season_name']}"
        if path.exists():
            print(f"{label}: on disk", flush=True)
            continue
        with ThreadPoolExecutor(args.workers) as pool:
            frames = list(pool.map(one_match, [m["match_id"] for m in matches]))
        df = pd.concat(frames, ignore_index=True).assign(competition_id=cid, season_id=sid)
        df.to_parquet(path, index=False)
        print(f"{label}: {len(matches)} matches, {len(df):,} rows "
              f"({(time.perf_counter() - start) / 60:.1f} min)", flush=True)
    pd.DataFrame(catalogue).to_parquet(OUT / "matches.parquet", index=False)
    print("Data provided by StatsBomb.")


if __name__ == "__main__":
    main()
