"""Ablation surfaces for validation tasks V1 and V3 (see xspace.validation.ablations).

    uv run python scripts/build_ablations.py            # the physics in config.py (calibrated)
    uv run python scripts/build_ablations.py --default  # UNCALIBRATED_PARAMS (Phases 0-3)

The calibrated run switches on the fitted lofted ball (calibration.json) so that the
`xspace_lofted` surface can be compared with ground-only `xspace`; the Phases 0-3 physics had no
lofted ball, so there the two are the same.

Per match, writes data/processed/validation/ablations[_default]/
{source}_{match}_{frames,passes}.parquet:
V3 frames sampled at --hz in open play, with danger labels; V1 passes from the pass dataset
(scripts/build_pass_set.py), in the same order.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from xspace.config import DEFAULT_PARAMS, DEFAULT_TIMELINE, UNCALIBRATED_PARAMS, VALIDATION_DIR
from xspace.metrics.space import orient
from xspace.metrics.timeline import frame_status, sample_frames
from xspace.pipeline import match_ids, prepare_match
from xspace.validation import ablations as ab
from xspace.validation.passes import PassSet

CHUNK = 64


def run(pool, match, frames, side, end, cell_size, params) -> dict[str, np.ndarray]:
    jobs = [pool.submit(ab.compute, ab.snapshots(match, frames[s:s + CHUNK], side[s:s + CHUNK],
                                                 None if end is None else end[s:s + CHUNK]),
                        cell_size, params)
            for s in range(0, len(frames), CHUNK)]
    parts = [j.result() for j in jobs]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all", choices=["idsse", "pff", "all"])
    ap.add_argument("--match", nargs="+", default=["all"])
    ap.add_argument("--default", action="store_true",
                    help="use the uncalibrated Phases 0-3 physics")
    ap.add_argument("--hz", type=float, default=1.0, help="V3 frame sample rate")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()

    if args.default:
        params = UNCALIBRATED_PARAMS
    else:
        fit = json.loads((VALIDATION_DIR / "calibration.json").read_text())["fitted"]
        params = replace(DEFAULT_PARAMS, **{k: fit[k] for k in
                                            ("air_speed", "air_time", "air_lambda_factor")})
    print(params)
    out_dir = VALIDATION_DIR / ("ablations_default" if args.default else "ablations")
    out_dir.mkdir(parents=True, exist_ok=True)
    cell = DEFAULT_TIMELINE.cell_size

    sources = ["idsse", "pff"] if args.source == "all" else [args.source]
    jobs = [(s, m) for s in sources
            for m in (match_ids(s) if args.match == ["all"] else args.match)]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (source, match_id) in enumerate(jobs, 1):
            t0 = time.perf_counter()
            pm = prepare_match(source, match_id)
            m = pm.match

            frames = sample_frames(m, args.hz)
            status = frame_status(pm.phases, pm.flags, frames, DEFAULT_TIMELINE.skip_flags)
            frames = frames[status == 0]
            side = pm.phases.possession_side[frames].astype(int)
            res = run(pool, m, frames, side, None, cell, params)
            df = ab.frame_rows(m, frames, side, res)
            labels = ab.danger_labels(m, pm.events, pm.phases, frames)
            df = pd.concat([df, labels], axis=1)
            df.insert(0, "possession_id", pm.phases.possession_id[frames])
            df.insert(0, "match_id", match_id)
            df.insert(0, "source", source)
            df.to_parquet(out_dir / f"{source}_{match_id}_frames.parquet")

            ps = PassSet.load(VALIDATION_DIR / f"passes_{source}_{match_id}.npz")
            end = np.array([orient(e, s) for e, s in zip(ps.end, ps.team_side, strict=True)])
            res = run(pool, m, ps.frame, ps.team_side.astype(int), end, cell, params)
            pdf = ab.pass_rows(res)
            pdf.insert(0, "event_id", ps.event_id)
            pdf.insert(0, "match_id", match_id)
            pdf.insert(0, "source", source)
            pdf.to_parquet(out_dir / f"{source}_{match_id}_passes.parquet")
            print(f"[{i}/{len(jobs)}] {source} {match_id}: {len(df)} frames, {len(pdf)} passes,"
                  f" {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"done in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
