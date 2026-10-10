"""Joint fit of the physics to where passes go *and* whether they arrive (Phase 4).

Fitting the physics to pass completion alone (`calibrate.fit`, task V2) learns from passes
players chose to attempt. They rarely attempt lanes that are really shut, so that fit decides
lanes barely matter (`intercept_factor` ≈ 0.15), and xSpace loses most of what made it
predict where passes go (V1) and danger (V3). Adding the destination makes a blocked lane cost
something: it's a lane the passer didn't choose.

    loss = mean log loss of completion (V2, intended receivers, every training pass)
         + mean -log P(the cell the pass went to)   (V1, a subsample of forward passes)

    P(cell) ∝ exp(β · xSpace(cell) / max xSpace)   (β fitted with the physics)

The destination term needs a whole xSpace grid per pass, so it runs on a subsample, on a
coarser grid, in worker processes that load the subsample once (`init_worker`).
"""

from __future__ import annotations

from concurrent.futures import Executor

import numpy as np
from scipy.optimize import minimize

from xspace.config import PhysicsParams
from xspace.metrics.space import space_from_arrays
from xspace.metrics.timeline import _grid
from xspace.validation import calibrate as cal
from xspace.validation import pass_model as pm
from xspace.validation.passes import PassSet

BETA_BOUNDS = (0.5, 64.0)
FIELDS = (*cal.PHYSICS_FIELDS, *cal.AIR_FIELDS, "beta")

_DEST: PassSet | None = None  # each worker's copy of the destination passes


def init_worker(ps: PassSet) -> None:
    global _DEST
    _DEST = ps


def destination_ll(rows: np.ndarray, params: PhysicsParams, beta: float,
                   cell_size: float, ps: PassSet | None = None) -> np.ndarray:
    """log P(end cell) under the xSpace softmax for passes `rows` of the destination set
    (the worker's copy unless `ps` is given). Top-level so worker processes can import it."""
    ps = _DEST if ps is None else ps
    grid = _grid(cell_size)
    out = np.empty(len(rows))
    for k, i in enumerate(rows):
        gk = int(ps.def_gk[i])
        # PassSet arrays are already in the passing team's frame: side 0 leaves them be.
        fs = space_from_arrays(ps.att_pos[i], ps.att_vel[i], ps.def_pos[i], ps.def_vel[i],
                               ps.ball[i], gk if gk >= 0 else None, 0, grid, params)
        x = fs.xspace
        top = x.max()
        z = beta * (x / top if top > 0 else np.zeros_like(x))
        cell = int(np.argmin(((grid - ps.end[i]) ** 2).sum(axis=1)))
        zmax = z.max()
        out[k] = z[cell] - zmax - np.log(np.exp(z - zmax).sum())
    return out


def fit_joint(success: PassSet, n_dest: int, start: cal.PassModel, beta: float,
              executor: Executor, cell_size: float = 3.0, chunks: int = 64,
              maxiter: int = 300, callback=None,
              bounds: dict[str, tuple[float, float]] = cal.BOUNDS,
              ) -> tuple[cal.PassModel, float, list[float]]:
    """Fit the physics and β jointly. `success`: passes for the completion term (with
    intended receivers). The destination passes live in the executor's workers (`n_dest` of
    them, see `init_worker`). Returns (model, β, loss after each iteration)."""
    y = success.success
    parts = np.array_split(np.arange(n_dest), chunks)
    history: list[float] = []
    last = {"loss": np.nan}

    def unpack(logv: np.ndarray) -> tuple[cal.PassModel, float]:
        v = np.exp(logv)
        return start.with_vector(v[:-1]), float(v[-1])

    def loss(logv: np.ndarray) -> float:
        model, b = unpack(logv)
        p = cal.predict(success, model)
        ok = ~np.isnan(p)
        physics = model.physics()
        lls = executor.map(destination_ll, parts, [physics] * len(parts), [b] * len(parts),
                           [cell_size] * len(parts))
        dest = -np.concatenate(list(lls)).mean()
        last["loss"] = pm.log_loss(y[ok], p[ok]) + dest
        last["parts"] = (pm.log_loss(y[ok], p[ok]), dest)
        return last["loss"]

    def step(logv: np.ndarray) -> None:
        history.append(last["loss"])
        if callback is not None:
            model, b = unpack(logv)
            callback(len(history), last["loss"], last["parts"], model, b)

    log_bounds = np.log([bounds[f] for f in FIELDS[:-1]] + [BETA_BOUNDS])
    lo, hi = log_bounds.T
    x0 = np.log(np.r_[start.vector(), beta])
    x0 = np.clip(x0, lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo))
    res = minimize(loss, x0, method="Nelder-Mead", callback=step, bounds=list(map(tuple,
                   log_bounds)), options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4,
                                          "adaptive": True})
    model, b = unpack(res.x)
    return model, b, history

