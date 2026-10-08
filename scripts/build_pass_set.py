"""Build the Phase 4 pass dataset (one frame snapshot per open-play pass / cross).

    uv run python scripts/build_pass_set.py --source all

Writes data/processed/validation/passes_{source}_{match}.npz (see xspace.validation.passes).
"""

from __future__ import annotations

import argparse
import time

from xspace.config import VALIDATION_DIR
from xspace.pipeline import match_ids, prepare_match
from xspace.validation.passes import build_pass_set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all", choices=["idsse", "pff", "all"])
    ap.add_argument("--match", nargs="+", default=["all"])
    args = ap.parse_args()

    sources = ["idsse", "pff"] if args.source == "all" else [args.source]
    jobs = [(s, m) for s in sources
            for m in (match_ids(s) if args.match == ["all"] else args.match)]
    start = time.perf_counter()
    for i, (source, match_id) in enumerate(jobs, 1):
        try:
            pm = prepare_match(source, match_id)
            ps = build_pass_set(source, pm.match, pm.events, pm.flags)
        except Exception as e:  # keep going; report the failure
            print(f"[{i}/{len(jobs)}] {source} {match_id}: FAILED {type(e).__name__}: {e}")
            continue
        ps.save(VALIDATION_DIR / f"passes_{source}_{match_id}.npz")
        print(f"[{i}/{len(jobs)}] {source} {match_id}: {len(ps)} passes", flush=True)
    print(f"done in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
