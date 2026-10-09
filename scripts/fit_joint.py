"""Fit the physics to where passes go and whether they arrive (see xspace.validation.joint).

    uv run python scripts/fit_joint.py

Same match split as scripts/calibrate.py. Writes data/processed/validation/joint.json: the
fitted parameters and β, the loss history, and held-out scores for the Phases 0-3 physics,
the completion-only fit (calibration.json) and the joint fit.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from xspace.config import UNCALIBRATED_PARAMS, VALIDATION_DIR, git_sha
from xspace.validation import calibrate as cal
from xspace.validation import joint
from xspace.validation import pass_model as pm
from xspace.validation.passes import PassSet

OUT = VALIDATION_DIR / "joint.json"


def forward(ps: PassSet) -> np.ndarray:
    return ps.end[:, 0] - ps.ball[:, 0] > 5


def held_out_destination(pool, ps: PassSet, model: cal.PassModel, beta: float,
                         cell: float) -> float:
    """Mean log P(end cell) minus the uniform guess's, in nats (V1's gain), on `ps`."""
    parts = np.array_split(np.arange(len(ps)), 32)
    subs = [ps.subset(p) for p in parts]
    lls = pool.map(joint.destination_ll, [np.arange(len(s)) for s in subs],
                   [model.physics()] * len(subs), [beta] * len(subs), [cell] * len(subs), subs)
    n_cells = len(joint._grid(cell))
    return float(np.concatenate(list(lls)).mean() + np.log(n_cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dest", type=int, default=8000, help="destination passes (subsample)")
    ap.add_argument("--n-test", type=int, default=3000, help="held-out destination passes")
    ap.add_argument("--cell", type=float, default=3.0)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 1))
    args = ap.parse_args()

    ps = PassSet.concat([PassSet.load(f)
                         for f in sorted(glob.glob(str(VALIDATION_DIR / "passes_*.npz")))])
    train, test = cal.split_matches(ps)
    rng = np.random.default_rng(0)
    success = ps.subset(train & (ps.target >= 0))
    dest_rows = rng.choice(np.flatnonzero(train & forward(ps)), args.n_dest, replace=False)
    dest = ps.subset(np.sort(dest_rows))
    test_rows = rng.choice(np.flatnonzero(test & forward(ps)), args.n_test, replace=False)
    dest_test = ps.subset(np.sort(test_rows))
    print(f"completion: {len(success)} passes; destination: {len(dest)} forward passes "
          f"(held out: {len(dest_test)})", flush=True)

    start = cal.PassModel(replace(UNCALIBRATED_PARAMS, lane_combine="max",
                                  intercept_factor=0.9),
                          pm.Trajectory("air", 20.8, 1.32, 1.0))
    t0 = time.perf_counter()

    def report(i, loss, parts, model, beta):
        if i % 10 == 0:
            print(f"  iter {i}: loss {loss:.4f} (completion {parts[0]:.4f}, destination "
                  f"{parts[1]:.4f}), beta {beta:.2f} ({time.perf_counter() - t0:.0f}s)",
                  flush=True)

    with ProcessPoolExecutor(args.workers, initializer=joint.init_worker,
                             initargs=(dest,)) as pool:
        model, beta, history = joint.fit_joint(success, len(dest), start, 3.0, pool,
                                               args.cell, maxiter=args.maxiter,
                                               callback=report)
        print("\nfitted:", json.dumps({**model.as_dict(), "beta": beta}, indent=1))
        print("at bounds:", cal.at_bounds(model))

        completion_only = cal.model_from_dict(
            json.loads((VALIDATION_DIR / "calibration.json").read_text())["fitted"])
        # (completion model, xSpace model): the Phases 0-3 xSpace had no lofted ball, but
        # scoring PFF's lofted passes needs one (the same baseline as scripts/calibrate.py).
        old = cal.PassModel(UNCALIBRATED_PARAMS, pm.Trajectory("air", 15.0, 0.5, 1.0))
        models = {
            "phases_0_3": (old, replace(old, air=replace(old.air, air_speed=0.0))),
            "completion_only": (completion_only, completion_only),
            "joint": (model, model),
        }
        held = ps.subset(test)
        scores = {}
        for name, (m_v2, m_v1) in models.items():
            # The joint β is fitted; the others use V1's best for forward passes (3).
            b = beta if name == "joint" else 3.0
            p = cal.predict(held, m_v2)
            ok = ~np.isnan(p)
            scores[name] = {
                "v2_log_loss": pm.log_loss(held.success[ok], p[ok]),
                "v2_auc": pm.auc(held.success[ok], p[ok]),
                "v1_gain_nats": held_out_destination(pool, dest_test, m_v1, b, args.cell),
            }
            print(name, {k: round(v, 4) for k, v in scores[name].items()}, flush=True)

    OUT.write_text(json.dumps({
        "git_sha": git_sha(), "fitted": model.as_dict(), "beta": beta,
        "at_bounds": cal.at_bounds(model), "bounds": cal.BOUNDS, "cell_size": args.cell,
        "n_dest": len(dest), "n_test": len(dest_test), "history": history, "scores": scores,
    }, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
