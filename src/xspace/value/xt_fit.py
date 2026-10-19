"""Fit our own Expected Threat (xT) grid (Phase 6): Markov models on open-play events.

xT(z) estimates the xG a team goes on to create in the possession from an action at z (zones
of an nx × ny grid, the team in possession attacking +x). Two chains (`XTConfig.chain`):

**singh** (Singh 2018):

    xT(z) = s(z) · g(z) + m(z) · Σ_z' T(z → z') · xT(z')

- s(z), m(z): shares of actions in z that are shots / moves (passes, carries, and ball losses:
  failed dribbles, dispossessions, miscontrols), s + m = 1.
- g(z): goal probability of a shot from z, the mean StatsBomb xG of shots there (less noisy
  than goals), shrunk towards a logistic shot model on distance and angle with weight
  `shot_prior` shots, so zones with almost no shots get a sensible value.
- T(z → z'): of all moves from z, the share that reached z' and were kept. Failed moves and
  losses go nowhere, so they lower xT(z).

**possession** (the default): the next state is where the same team's *next action in the same
possession* happened, whatever came between, and the chain ends when the possession does:

    xT(z) = s(z) · g(z) + Σ_z' P(z → z') · xT(z')

P(z → z') is the share of all actions at z (shots included: a rebound can follow) whose next
team action was at z'. Singh's chain ends a possession at every failed pass, but in event data
a possession often survives an opponent's touch (a block that bounces back, a rebound), so
Singh's chain under-counts what teams go on to create.

The fixed point is found by iteration from xT = 0 (each step adds one more action of look-ahead).

`possession_targets` gives the check used to compare grids on held-out matches: at every
open-play action, xT should equal the xG the team creates from it to the end of the possession.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from xspace.constants import PITCH_LENGTH, PITCH_WIDTH

GOAL = np.array([PITCH_LENGTH / 2, 0.0])
GOAL_HALF_WIDTH = 3.66


@dataclass(frozen=True)
class XTConfig:
    nx: int = 24  # zones along the length
    ny: int = 16  # zones across the width
    shot_prior: float = 20.0  # pseudo-shots pulling g(z) towards the distance / angle model
    chain: str = "possession"  # "possession" or "singh" (see the module docstring)
    max_iter: int = 200
    tol: float = 1e-7


def zones(x: np.ndarray, y: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Flat zone index (row-major: iy * nx + ix) for pitch points; -1 if NaN."""
    ix = np.floor((np.asarray(x, float) + PITCH_LENGTH / 2) / PITCH_LENGTH * nx)
    iy = np.floor((np.asarray(y, float) + PITCH_WIDTH / 2) / PITCH_WIDTH * ny)
    ok = ~(np.isnan(ix) | np.isnan(iy))
    z = np.clip(np.nan_to_num(iy), 0, ny - 1) * nx + np.clip(np.nan_to_num(ix), 0, nx - 1)
    return np.where(ok, z, -1).astype(np.int64)


def zone_centres(nx: int, ny: int) -> np.ndarray:
    """(ny * nx, 2) centres, in zone order."""
    xs = -PITCH_LENGTH / 2 + (np.arange(nx) + 0.5) * PITCH_LENGTH / nx
    ys = -PITCH_WIDTH / 2 + (np.arange(ny) + 0.5) * PITCH_WIDTH / ny
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def shot_features(points: np.ndarray) -> np.ndarray:
    """[1, distance to goal (m), goal mouth angle (rad)] per point."""
    d = points - GOAL
    dist = np.hypot(d[:, 0], d[:, 1])
    left = np.arctan2(GOAL_HALF_WIDTH - points[:, 1], GOAL[0] - points[:, 0])
    right = np.arctan2(-GOAL_HALF_WIDTH - points[:, 1], GOAL[0] - points[:, 0])
    angle = np.abs(left - right)
    return np.stack([np.ones(len(points)), dist, angle], axis=1)


def fit_shot_model(points: np.ndarray, xg: np.ndarray, iters: int = 50) -> np.ndarray:
    """Logistic regression of xG (a probability, used as a soft label) on distance and angle,
    by Newton's method. Returns the coefficients."""
    X = shot_features(points)
    w = np.zeros(X.shape[1])
    w[0] = np.log(xg.mean() / (1 - xg.mean()))
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (xg - p)
        hess = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def shot_model(points: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-shot_features(points) @ w))


DEFAULT_XT = XTConfig()


@dataclass
class XTModel:
    config: XTConfig
    grid: np.ndarray  # (ny, nx) xT
    shot_share: np.ndarray  # (ny, nx) s
    move_share: np.ndarray  # (ny, nx) m
    goal_prob: np.ndarray  # (ny, nx) g
    n_actions: np.ndarray  # (ny, nx)
    n_shots: np.ndarray  # (ny, nx)
    iterations: int

    def to_json(self, provenance: dict) -> str:
        return json.dumps({
            "provenance": provenance, "config": asdict(self.config),
            "grid": np.round(self.grid, 6).tolist(),
        }, indent=1)


def fit_xt(events: pd.DataFrame, config: XTConfig = DEFAULT_XT) -> XTModel:
    """Fit from reduced events (`io.statsbomb.reduce_events` columns: kind, x, y, end_x, end_y,
    success, xg)."""
    nx, ny = config.nx, config.ny
    n = nx * ny
    z = zones(events["x"].to_numpy(), events["y"].to_numpy(), nx, ny)
    kind = events["kind"].astype(str).to_numpy()
    ok = z >= 0
    shots = ok & (kind == "shot")
    moves = ok & (kind != "shot")  # moves and losses

    n_shots = np.bincount(z[shots], minlength=n).astype(float)
    n_moves = np.bincount(z[moves], minlength=n).astype(float)
    total = n_shots + n_moves
    s = np.divide(n_shots, total, out=np.zeros(n), where=total > 0)
    m = 1 - s

    xg = events["xg"].to_numpy(float)
    pts = events[["x", "y"]].to_numpy(float)
    w = fit_shot_model(pts[shots], xg[shots])
    prior = shot_model(zone_centres(nx, ny), w)
    xg_sum = np.bincount(z[shots], weights=xg[shots], minlength=n)
    denom = n_shots + config.shot_prior
    g = np.divide(xg_sum + config.shot_prior * prior, denom, out=prior.copy(), where=denom > 0)

    counts = np.zeros((n, n))
    if config.chain == "singh":
        kept = moves & (kind == "move") & events["success"].to_numpy(bool)
        z_end = zones(events["end_x"].to_numpy(), events["end_y"].to_numpy(), nx, ny)
        kept &= z_end >= 0
        np.add.at(counts, (z[kept], z_end[kept]), 1.0)
        T = m[:, None] * np.divide(counts, n_moves[:, None], out=np.zeros((n, n)),
                                   where=n_moves[:, None] > 0)
    elif config.chain == "possession":
        z_next = next_zones(events, z)
        kept = ok & (z_next >= 0)
        np.add.at(counts, (z[kept], z_next[kept]), 1.0)
        T = np.divide(counts, total[:, None], out=np.zeros((n, n)), where=total[:, None] > 0)
    else:
        raise ValueError(f"unknown chain {config.chain!r}")

    xt = np.zeros(n)
    iterations = 0
    while iterations < config.max_iter:
        iterations += 1
        new = s * g + T @ xt
        done = np.abs(new - xt).max() < config.tol
        xt = new
        if done:
            break
    shape = (ny, nx)
    return XTModel(config, xt.reshape(shape), s.reshape(shape), m.reshape(shape),
                   g.reshape(shape), total.reshape(shape), n_shots.reshape(shape), iterations)


def _possession_key(df: pd.DataFrame) -> list[pd.Series]:
    return [df["match_id"], df["possession"], df["team_id"]]


def next_zones(events: pd.DataFrame, z: np.ndarray) -> np.ndarray:
    """Zone of the same team's next action in the same possession (event order), -1 if none."""
    order = np.lexsort((events["index"].to_numpy(), events["match_id"].to_numpy()))
    df = events.iloc[order]
    nxt = pd.Series(z[order], index=df.index).groupby(_possession_key(df)).shift(-1)
    out = np.empty(len(events), dtype=np.int64)
    out[order] = nxt.fillna(-1).to_numpy(np.int64)
    return out


def possession_targets(events: pd.DataFrame) -> pd.Series:
    """For every action: the xG of the open-play shots the same team takes from this action
    (included, if it is a shot) to the end of the possession (StatsBomb `possession` ids).
    What xT at the action's location estimates."""
    df = events.sort_values(["match_id", "index"])
    key = _possession_key(df)
    shot_xg = np.where(df["kind"].astype(str) == "shot", df["xg"].fillna(0.0), 0.0)
    # xG from here on = total for the (match, possession, team) − what came before this event.
    s = pd.Series(shot_xg, index=df.index)
    total = s.groupby(key).transform("sum")
    before = s.groupby(key).cumsum() - s
    return (total - before).rename("xg_to_come")
