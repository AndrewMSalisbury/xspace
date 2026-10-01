"""Download the PFF FC (Gradient Sports) 2022 World Cup dataset from its Drive folder.

The data is free but gated behind a request form at
https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset — request access there, then
put the Drive folder link you receive in the environment variable PFF_FOLDER_URL.
Data lands in data/raw/pff/ (gitignored).

    uv run python scripts/download_pff.py                 # metadata, rosters, events, docs
    uv run python scripts/download_pff.py --tracking 3812 # + tracking for one match
    uv run python scripts/download_pff.py --tracking all  # + tracking for all 64 matches
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import gdown

FOLDER_URL_ENV = "PFF_FOLDER_URL"  # the Drive link Gradient Sports sends after their access form
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "pff"


def _latest_event_folder(paths: list[str]) -> str | None:
    """Event Data has dated update subfolders (e.g. 'May 1, 2025'); pick the newest."""
    from datetime import datetime

    dated = set()
    for p in paths:
        parts = p.split("/")
        if parts[0] == "Event Data" and len(parts) == 3:
            dated.add(parts[1])
    if not dated:
        return None
    return max(dated, key=lambda d: datetime.strptime(d, "%B %d, %Y"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking", nargs="*", default=[],
                    help="match ids to download tracking for, or 'all' (~large)")
    args = ap.parse_args()

    folder_url = os.environ.get(FOLDER_URL_ENV)
    if not folder_url:
        raise SystemExit(f"Set {FOLDER_URL_ENV} to the Drive folder link from Gradient Sports.")
    files = gdown.download_folder(folder_url, skip_download=True, quiet=True)
    by_path = {f.path.replace("\\", "/"): f for f in files}
    latest_events = _latest_event_folder(list(by_path))
    want_all = "all" in args.tracking

    selected = []
    for path, f in by_path.items():
        parts = path.split("/")
        top = parts[0]
        if top in ("Metadata", "Rosters") or path.endswith((".csv", ".pdf")):
            selected.append((path, f))
        elif top == "Event Data" and len(parts) == 3 and parts[1] == latest_events:
            selected.append((f"Event Data/{parts[2]}", f))
        elif top == "Tracking Data" and path.endswith(".jsonl.bz2"):
            match_id = re.sub(r"\.jsonl\.bz2$", "", parts[-1])
            if want_all or match_id in args.tracking:
                selected.append((path, f))

    print(f"{len(selected)} files selected (events version: {latest_events})")
    for i, (rel, f) in enumerate(sorted(selected), 1):
        dest = OUT / rel
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(selected)}] {rel}")
        gdown.download(id=f.id, output=str(dest), quiet=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
