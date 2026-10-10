"""Validation tasks V1 and V3 from the ablation files (scripts/build_ablations.py), and V4
(split-half stability of team metrics) from the possession files (scripts/build_possessions.py:
whatever physics those were built with).

    uv run python scripts/validate.py            # calibrated physics
    uv run python scripts/validate.py --default  # config.py physics

Uses the same match split as the calibration (calibration.json): models are fitted on the PFF
training matches and scored on the PFF test matches and on all IDSSE matches (another
provider and league: an out-of-source check). Prints markdown tables and writes
data/processed/validation/validation.json.
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from xspace.config import POSSESSIONS_DIR, VALIDATION_DIR
from xspace.metrics.timeline import read_metadata
from xspace.validation.passes import TRAJECTORIES, PassSet
from xspace.validation.report import v1_scores, v3_scores, v4_split_half


def load(ab_dir, kind: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(ab_dir / f"*_{kind}.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--default", action="store_true", help="ablations built with --default")
    args = ap.parse_args()
    tag = "_default" if args.default else ""
    ab_dir = VALIDATION_DIR / f"ablations{tag}"
    cal = json.loads((VALIDATION_DIR / "calibration.json").read_text())
    train_ids, test_ids = set(cal["train_matches"]), set(cal["test_matches"])
    out = {}

    passes = load(ab_dir, "passes")
    ps = PassSet.concat([PassSet.load(VALIDATION_DIR / f"passes_{s}_{m}.npz")
                         for s, m in passes[["source", "match_id"]].drop_duplicates()
                         .itertuples(index=False)])
    assert (ps.event_id == passes["event_id"].to_numpy()).all(), "pass order mismatch"
    subsets = {
        "all": np.ones(len(ps), dtype=bool),
        "forward (> 5 m)": ps.end[:, 0] - ps.ball[:, 0] > 5,
        "completed": ps.success == 1,
        "lofted (PFF)": ps.trajectory == TRAJECTORIES.index("air"),
    }
    pff = (passes["source"] == "pff").to_numpy()
    train = pff & passes["match_id"].isin(train_ids).to_numpy()
    for name, test in (("pff_test", pff & passes["match_id"].isin(test_ids).to_numpy()),
                       ("idsse", ~pff)):
        df = v1_scores(passes, train, test, subsets)
        out[f"v1_{name}"] = df.to_dict(orient="records")
        print(f"\n## V1, {name}\n\n" + df.round(3).to_markdown(index=False))

    frames = load(ab_dir, "frames")
    pff = (frames["source"] == "pff").to_numpy()
    train = pff & frames["match_id"].isin(train_ids).to_numpy()
    print(f"\nV3 frames: {len(frames)}; shot within 10 s {frames['shot10'].mean():.3f}, "
          f"box entry {frames['box10'].mean():.3f}")
    for name, test in (("pff_test", pff & frames["match_id"].isin(test_ids).to_numpy()),
                       ("idsse", ~pff)):
        for target in ("shot10", "box10"):
            df = v3_scores(frames, train, test, target)
            out[f"v3_{name}_{target}"] = df.to_dict(orient="records")
            print(f"\n## V3 {target}, {name}\n\n" + df.round(4).to_markdown(index=False))
    parts = []
    for f in sorted(POSSESSIONS_DIR.glob("pff_*.parquet")):
        meta, d = read_metadata(f), pd.read_parquet(f)
        names = np.array([meta["home"], meta["away"]])
        d["match_id"] = meta["match_id"]
        d["team"], d["opponent"] = names[d["team_side"]], names[1 - d["team_side"]]
        parts.append(d)
    df = v4_split_half(pd.concat(parts, ignore_index=True))
    out["v4_pff"] = df.to_dict(orient="records")
    print("\n## V4, PFF possessions\n\n" + df.round(3).to_markdown(index=False))

    (VALIDATION_DIR / f"validation{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
