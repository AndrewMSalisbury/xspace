"""Fit the xSpace physics to pass outcomes (Phase 4, validation task V2).

The pass-success model (`pass_model`) is `control x reach` at the target, scored per pass
with that pass's trajectory: PFF tags every pass with its peak height (`PassSet.trajectory`),
so ground passes are scored with the ground model and lofted ones with the air model. Where
the trajectory is unknown (IDSSE, or a grid cell that hasn't been passed to yet) the passer is
assumed to pick the better of the two: `max(P_ground, P_air)`, which is what xSpace uses.

Fitting maximises the likelihood of completed / failed on training matches (Nelder-Mead in
log-parameter space, since everything fitted is positive). Matches, not passes, are split, so
the test set shares no frames with training.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from xspace.config import DEFAULT_PARAMS, PhysicsParams
from xspace.validation import pass_model as pm
from xspace.validation.passes import TRAJECTORIES, PassSet

PHYSICS_FIELDS = ("reaction_time", "max_speed", "tti_sigma", "lambda_att", "kappa_def",
                  "ball_speed", "intercept_factor")
AIR_FIELDS = ("air_speed", "air_time", "lambda_factor")
AIR = TRAJECTORIES.index("air")
# Physically plausible ranges. Unbounded, the fit runs away to an instant ball (ball_speed in
# the thousands, reaction ~0) under which nothing is cut out and control is "is a defender
# nearer the receiver than the receiver?": a good marking statistic, but not physics.
BOUNDS = {
    "reaction_time": (0.3, 1.0),  # s
    "max_speed": (4.0, 7.0),  # m/s, an average top speed (sprints peak at 9-10)
    "tti_sigma": (0.2, 1.0),  # s
    "lambda_att": (1.0, 10.0),  # 1/s
    "kappa_def": (0.5, 2.0),
    "ball_speed": (10.0, 30.0),  # m/s, ground passes
    "intercept_factor": (0.1, 1.0),
    "air_speed": (8.0, 30.0),  # m/s, horizontal
    "air_time": (0.2, 2.0),  # s
    "lambda_factor": (0.3, 1.5),
}


@dataclass(frozen=True)
class PassModel:
    params: PhysicsParams = DEFAULT_PARAMS
    air: pm.Trajectory = pm.Trajectory("air")

    def vector(self) -> np.ndarray:
        return np.array([getattr(self.params, f) for f in PHYSICS_FIELDS]
                        + [getattr(self.air, f) for f in AIR_FIELDS], dtype=float)

    def with_vector(self, v: np.ndarray) -> PassModel:
        n = len(PHYSICS_FIELDS)
        physics = dict(zip(PHYSICS_FIELDS, map(float, v[:n]), strict=True))
        return PassModel(replace(self.params, **physics),
                         replace(self.air, **dict(zip(AIR_FIELDS, map(float, v[n:]), strict=True))))

    def physics(self) -> PhysicsParams:
        """The physics xSpace uses: these parameters with lofted passes switched on."""
        return replace(self.params, air_speed=self.air.air_speed, air_time=self.air.air_time,
                       air_lambda_factor=self.air.lambda_factor)

    def as_dict(self) -> dict[str, float]:
        return {**{f: getattr(self.params, f) for f in PHYSICS_FIELDS},
                **{f"air_{f}" if not f.startswith("air") else f: getattr(self.air, f)
                   for f in AIR_FIELDS},
                "lane_combine": self.params.lane_combine}


def predict(ps: PassSet, model: PassModel, target: str = "intent",
            use_trajectory: bool = True) -> np.ndarray:
    """P(success) per pass. With `use_trajectory`, known trajectories pick the model; unknown
    ones (and all passes, without it) get max(ground, air)."""
    need_ground = np.ones(len(ps), dtype=bool)
    need_air = np.ones(len(ps), dtype=bool)
    if use_trajectory:
        need_ground = ps.trajectory != AIR
        need_air = ps.trajectory != TRAJECTORIES.index("ground")
    ground = np.full(len(ps), np.nan)
    air = np.full(len(ps), np.nan)
    if need_ground.any():
        ground[need_ground] = pm.predict(ps.subset(need_ground), model.params, pm.GROUND,
                                         target).p
    if need_air.any():
        air[need_air] = pm.predict(ps.subset(need_air), model.params, model.air, target).p
    # np.fmax ignores the NaN of the trajectory that wasn't needed; NaN only if both are.
    return np.fmax(ground, air)


def split_matches(ps: PassSet, test_every: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) masks over PFF passes: every `test_every`-th PFF match (sorted by id)
    is held out. IDSSE passes are in neither (they're an out-of-source check)."""
    pff = ps.source == "pff"
    ids = sorted(set(ps.match_id[pff]), key=int)
    test_ids = set(ids[test_every - 1::test_every])
    test = pff & np.isin(ps.match_id, list(test_ids))
    return pff & ~test, test


def fit(ps: PassSet, start: PassModel, target: str = "intent", maxiter: int = 400,
        callback=None, bounds: dict[str, tuple[float, float]] | None = BOUNDS,
        ) -> tuple[PassModel, list[float]]:
    """Maximum-likelihood fit of PHYSICS_FIELDS and AIR_FIELDS on `ps` (passes with a usable
    target only), within `bounds` (None: unbounded). Returns the fitted model and the loss
    after each iteration."""
    y = ps.success
    history: list[float] = []

    def loss(logv: np.ndarray) -> float:
        p = predict(ps, start.with_vector(np.exp(logv)), target)
        ok = ~np.isnan(p)
        return pm.log_loss(y[ok], p[ok])

    def step(logv: np.ndarray) -> None:
        history.append(loss(logv))
        if callback is not None:
            callback(len(history), history[-1], start.with_vector(np.exp(logv)))

    x0 = np.log(start.vector())
    log_bounds = None
    if bounds is not None:
        log_bounds = [tuple(np.log(bounds[f])) for f in (*PHYSICS_FIELDS, *AIR_FIELDS)]
        lo, hi = np.array(log_bounds).T
        # Start a little inside the box: a simplex started on a bound collapses there.
        margin = 0.05 * (hi - lo)
        x0 = np.clip(x0, lo + margin, hi - margin)
    res = minimize(loss, x0, method="Nelder-Mead", callback=step, bounds=log_bounds,
                   options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-5, "adaptive": True})
    return start.with_vector(np.exp(res.x)), history


def evaluate(ps: PassSet, models: dict[str, PassModel], target: str = "intent",
             use_trajectory: bool = True) -> pd.DataFrame:
    """Scores per model on all passes and per trajectory."""
    rows = []
    for name, model in models.items():
        p = predict(ps, model, target, use_trajectory)
        for subset in ("all", *TRAJECTORIES):
            m = ~np.isnan(p) & (np.ones(len(ps), bool) if subset == "all"
                                else ps.trajectory == TRAJECTORIES.index(subset))
            if m.sum() < 50:
                continue
            rows.append({"model": name, "subset": subset, **pm.scores(ps.success[m], p[m])})
    return pd.DataFrame(rows)


def by_distance(ps: PassSet, p: np.ndarray,
                bins: tuple[float, ...] = (0, 10, 20, 30, 40, 60, 120)) -> pd.DataFrame:
    """Observed and predicted completion per distance band."""
    d = ps.distance
    rows = []
    for a, b in zip(bins[:-1], bins[1:], strict=True):
        m = (d >= a) & (d < b) & ~np.isnan(p)
        if m.any():
            rows.append({"distance": f"{a:g}-{b:g} m", "n": int(m.sum()),
                         "observed": float(ps.success[m].mean()), "predicted": float(p[m].mean())})
    return pd.DataFrame(rows)


def at_bounds(model: PassModel, bounds: dict[str, tuple[float, float]] = BOUNDS,
              rtol: float = 0.01) -> dict[str, str]:
    """Fitted values pinned (within rtol) to a bound: 'lower' or 'upper'."""
    v = dict(zip((*PHYSICS_FIELDS, *AIR_FIELDS), model.vector(), strict=True))
    out = {}
    for f, (lo, hi) in bounds.items():
        if v[f] <= lo * (1 + rtol):
            out[f] = "lower"
        elif v[f] >= hi * (1 - rtol):
            out[f] = "upper"
    return out


def model_from_dict(d: dict) -> PassModel:
    params = replace(DEFAULT_PARAMS, **{f: d[f] for f in PHYSICS_FIELDS if f in d},
                     lane_combine=d.get("lane_combine", DEFAULT_PARAMS.lane_combine))
    air = pm.Trajectory("air", d["air_speed"], d["air_time"], d["air_lambda_factor"])
    return PassModel(params, air)

