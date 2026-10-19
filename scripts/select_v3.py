"""Pick physics for V3 (danger) on the training matches only (Phase 6, PLAN step 4).

    uv run python scripts/select_v3.py if015 if030 if050 if075 if100

For each ablation folder (scripts/build_ablations.py --tag), the PFF training matches
(calibration.json) are split in two by match; a logistic model of shot / box entry within 10 s
on ball position, with and without total xSpace, is fitted on one half and scored on the other,
and the other way round. Prints the mean log-loss gain of xSpace over ball position; the test
matches are left alone for the final comparison (scripts/validate.py --tag).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from xspace.config import VALIDATION_DIR
from xspace.validation.report import _design, fit_logistic, log_loss, predict_logistic


def inner_gain(frames: pd.DataFrame, feature: str, target: str) -> float:
    ids = np.array(sorted(frames["match_id"].unique()))
    half = frames["match_id"].isin(ids[::2]).to_numpy()
    y = frames[target].to_numpy(float)
    gains = []
    for fit_on in (half, ~half):
        score_on = ~fit_on
        base = fit_logistic(_design(frames[fit_on], None), y[fit_on])
        full = fit_logistic(_design(frames[fit_on], feature), y[fit_on])
        lb = log_loss(y[score_on], predict_logistic(base, _design(frames[score_on], None)))
        lf = log_loss(y[score_on], predict_logistic(full, _design(frames[score_on], feature)))
        gains.append(100 * (1 - lf / lb))
    return float(np.mean(gains))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="ablation folders: ablations_<tag>")
    args = ap.parse_args()
    train = set(json.loads((VALIDATION_DIR / "calibration.json").read_text())["train_matches"])
    rows = []
    for tag in args.tags:
        folder = VALIDATION_DIR / ("ablations" if tag == "calibrated" else f"ablations_{tag}")
        files = sorted(folder.glob("pff_*_frames.parquet"))
        frames = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        frames = frames[frames["match_id"].isin(train)]
        row = {"tag": tag, "frames": len(frames)}
        for target in ("shot10", "box10"):
            for feature in ("total_xspace", "best_xspace"):
                row[f"{target}_{feature}"] = inner_gain(frames, feature, target)
        rows.append(row)
    print(pd.DataFrame(rows).round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
