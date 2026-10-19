"""Fit our own xT grid on StatsBomb Open Data and compare it with Singh's (Phase 6).

    uv run python scripts/fit_xt.py

Needs data/raw/statsbomb/ (scripts/download_statsbomb.py). Fits on every comp-season except
the held-out ones (HELD_OUT: Euro 2024, Copa América 2024, Bundesliga 2023/24), then scores
grids on the held-out matches: at every open-play action, how well does xT at its location
predict the xG the team creates from it to the end of the possession (`possession_targets`)?

Writes src/xspace/value/xt_own_24x16.json (the grid, with provenance) and
data/processed/validation/xt_fit.json (the comparison). Takes about a minute.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from xspace.config import RAW_DIR, VALIDATION_DIR, git_sha
from xspace.value import xt as xt_module
from xspace.value.xt_fit import DEFAULT_XT, fit_xt, possession_targets

HELD_OUT = {(55, 282), (223, 282), (9, 281)}  # Euro 2024, Copa América 2024, Bundesliga 23/24


def score(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """MSE, R² against the mean, calibration slope (target on prediction) and mean levels."""
    mse = float(np.mean((pred - target) ** 2))
    base = float(np.mean((target - target.mean()) ** 2))
    slope = float(np.cov(pred, target)[0, 1] / np.var(pred))
    return {"mse": mse, "r2": 1 - mse / base, "slope": slope,
            "mean_pred": float(pred.mean()), "mean_target": float(target.mean()),
            "spearman": float(pd.Series(pred).corr(pd.Series(target), method="spearman"))}


def deciles(pred: np.ndarray, target: np.ndarray) -> list[dict[str, float]]:
    q = pd.qcut(pd.Series(pred).rank(method="first"), 10, labels=False)
    df = pd.DataFrame({"pred": pred, "target": target, "q": q}).groupby("q").mean()
    return df.round(5).to_dict("records")


def main() -> None:
    sb = RAW_DIR / "statsbomb"
    events = pd.concat([pd.read_parquet(p) for p in sorted(sb.glob("*_*.parquet"))],
                       ignore_index=True)
    held = pd.Series(list(zip(events["competition_id"], events["season_id"], strict=True)))
    test_mask = held.isin(HELD_OUT).to_numpy()
    train, test = events[~test_mask], events[test_mask]
    print(f"train: {train['match_id'].nunique()} matches, {len(train):,} events; "
          f"test: {test['match_id'].nunique()} matches, {len(test):,} events")

    target = possession_targets(test)
    pts = test.loc[target.index, ["x", "y"]].to_numpy(float)
    y = target.to_numpy()

    models = {"own_24x16": DEFAULT_XT, "own_12x8": replace(DEFAULT_XT, nx=12, ny=8),
              "singh_chain_24x16": replace(DEFAULT_XT, chain="singh")}
    fits = {name: fit_xt(train, cfg) for name, cfg in models.items()}
    report = {"n_train_matches": int(train["match_id"].nunique()),
              "n_test_matches": int(test["match_id"].nunique()), "n_test_actions": int(len(y)),
              "held_out": sorted(map(list, HELD_OUT)), "models": {}}
    preds = {"singh_12x8": xt_module.xt_value(pts, "singh_12x8")}
    for name, model in fits.items():
        ny, nx = model.grid.shape
        ix = np.clip(((pts[:, 0] + 52.5) / 105 * nx).astype(int), 0, nx - 1)
        iy = np.clip(((pts[:, 1] + 34) / 68 * ny).astype(int), 0, ny - 1)
        preds[f"{name}_cells"] = model.grid[iy, ix]  # the grid as fitted (no interpolation)
    for name, p in preds.items():
        report["models"][name] = {**score(p, y), "deciles": deciles(p, y)}
        s = report["models"][name]
        print(f"{name:18s} MSE {s['mse']:.6f}  R² {s['r2']:.4f}  slope {s['slope']:.2f}  "
              f"mean {s['mean_pred']:.4f} (target {s['mean_target']:.4f})  "
              f"ρ {s['spearman']:.3f}")

    own = fits["own_24x16"]
    print(f"own 24x16: {own.iterations} iterations; max xT {own.grid.max():.3f}, "
          f"own-half mean {own.grid[:, :12].mean():.4f}")
    competitions = (train[["competition_id", "season_id"]].drop_duplicates()
                    .sort_values(["competition_id", "season_id"]).to_numpy().tolist())
    provenance = {
        "method": "Markov xT (Singh 2018), value/xt_fit.py",
        "data": "StatsBomb Open Data (Data provided by StatsBomb), men's, 2003/04 on, "
                "open play; FIFA World Cup 2022 and the held-out seasons excluded",
        "train_matches": report["n_train_matches"],
        "train_competition_seasons": competitions,
        "git_sha": git_sha(),
    }
    out = xt_module.resources.files("xspace.value").joinpath(xt_module.VALUE_MODELS["own_24x16"])
    with xt_module.resources.as_file(out) as path:
        path.write_text(own.to_json(provenance), encoding="utf-8")
    # Interpolated, as the pipeline will use it.
    xt_module._interpolator.cache_clear()
    p = xt_module.xt_value(pts, "own_24x16")
    report["models"]["own_24x16"] = {**score(p, y), "deciles": deciles(p, y)}
    s = report["models"]["own_24x16"]
    print(f"{'own_24x16 (interp)':18s} MSE {s['mse']:.6f}  R² {s['r2']:.4f}  "
          f"slope {s['slope']:.2f}  ρ {s['spearman']:.3f}")
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    (VALIDATION_DIR / "xt_fit.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path.name}; Data provided by StatsBomb.")


if __name__ == "__main__":
    main()
