"""Fit the physics to pass outcomes and report held-out scores (Phase 4, V2).

    uv run python scripts/calibrate.py            # fit, evaluate, write calibration.json
    uv run python scripts/calibrate.py --no-fit   # evaluate the saved fit only
    uv run python scripts/calibrate.py --unbounded  # no bounds: calibration_unbounded.json

Needs the pass dataset (scripts/build_pass_set.py). Writes
data/processed/validation/calibration.json: fitted parameters, the train / test match split
and every score printed here.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from dataclasses import replace

import numpy as np

from xspace.config import DEFAULT_PARAMS, VALIDATION_DIR, git_sha
from xspace.validation import calibrate as cal
from xspace.validation import pass_model as pm
from xspace.validation.passes import PassSet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--maxiter", type=int, default=400)
    ap.add_argument("--unbounded", action="store_true", help="fit without parameter bounds")
    ap.add_argument("--warm", action="store_true",
                    help="start from the saved fit (as far as it goes) instead of the defaults")
    args = ap.parse_args()
    out = VALIDATION_DIR / ("calibration_unbounded.json" if args.unbounded
                            else "calibration.json")
    bounds = None if args.unbounded else cal.BOUNDS

    ps = PassSet.concat([PassSet.load(f)
                         for f in sorted(glob.glob(str(VALIDATION_DIR / "passes_*.npz")))])
    train, test = cal.split_matches(ps)
    print(f"{len(ps)} passes; train {train.sum()} ({len(set(ps.match_id[train]))} matches), "
          f"test {test.sum()} ({len(set(ps.match_id[test]))} matches)")

    default = cal.PassModel(DEFAULT_PARAMS, pm.Trajectory("air", 15.0, 0.5, 1.0))
    lane_max = replace(default, params=replace(DEFAULT_PARAMS, lane_combine="max"))
    if args.no_fit:
        fitted = cal.model_from_dict(json.loads(out.read_text())["fitted"])
        history = []
    else:
        t0 = time.perf_counter()

        def report(i, loss, model):
            if i % 20 == 0:
                print(f"  iter {i}: train log loss {loss:.4f} ({time.perf_counter() - t0:.0f}s)",
                      flush=True)

        start = lane_max
        if args.warm:
            start = cal.model_from_dict(json.loads(out.read_text())["fitted"])
        fitted, history = cal.fit(ps.subset(train & (ps.target >= 0)), start,
                                  maxiter=args.maxiter, callback=report, bounds=bounds)
    models = {"default": default, "lane_max": lane_max, "fitted": fitted}

    held = ps.subset(test)
    results = {
        "test_intent": cal.evaluate(held, models),
        "test_intent_best_of": cal.evaluate(held, models, use_trajectory=False),
        "test_end": cal.evaluate(held, models, target="end"),
        "idsse_end": cal.evaluate(ps.subset(ps.source == "idsse"), models, target="end"),
    }
    for name, df in results.items():
        print(f"\n== {name}\n" + df.round(3).to_string(index=False))
    dist = cal.by_distance(held, cal.predict(held, fitted))
    print("\n== fitted, test, by distance\n" + dist.round(3).to_string(index=False))
    rel = pm.reliability(held.success, cal.predict(held, fitted))
    print("\n== fitted, test, reliability (mean p, observed, n)\n" + np.array2string(
        rel, precision=3, suppress_small=True))
    print("\nfitted:", json.dumps(fitted.as_dict(), indent=1))
    if bounds:
        print("at bounds:", cal.at_bounds(fitted))

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "git_sha": git_sha(),
        "fitted": fitted.as_dict(), "start": lane_max.as_dict(),
        "bounds": bounds, "at_bounds": cal.at_bounds(fitted) if bounds else {},
        "train_matches": sorted(set(ps.match_id[train]), key=int),
        "test_matches": sorted(set(ps.match_id[test]), key=int),
        "history": history,
        "scores": {k: v.to_dict(orient="records") for k, v in results.items()},
        "by_distance": dist.to_dict(orient="records"),
        "reliability": rel.tolist(),
    }, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
