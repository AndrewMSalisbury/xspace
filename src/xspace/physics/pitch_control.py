"""Vectorised physics-based pitch control and pass reachability.

Pitch control follows Spearman (2018), "Beyond Expected Goals", as implemented by
Shaw (LaurieOnTracking) but vectorised over every grid cell at once. Time-to-intercept is the
same quantity used by Bekkers (2025), "Pressing Intensity", so defensive pressure and
offensive space are computed from one consistent model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xspace.config import DEFAULT_PARAMS, PhysicsParams  # noqa: F401  (re-exported)
from xspace.constants import PITCH_LENGTH, PITCH_WIDTH


def make_grid(cell_size: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell-centre coordinates. Returns (xs, ys, grid) with grid shaped (n_y * n_x, 2)."""
    n_x = int(round(PITCH_LENGTH / cell_size))
    n_y = int(round(PITCH_WIDTH / cell_size))
    xs = -PITCH_LENGTH / 2 + (np.arange(n_x) + 0.5) * PITCH_LENGTH / n_x
    ys = -PITCH_WIDTH / 2 + (np.arange(n_y) + 0.5) * PITCH_WIDTH / n_y
    gx, gy = np.meshgrid(xs, ys)
    return xs, ys, np.stack([gx.ravel(), gy.ravel()], axis=1)


def time_to_intercept(pos: np.ndarray, vel: np.ndarray, targets: np.ndarray,
                      params: PhysicsParams = DEFAULT_PARAMS) -> np.ndarray:
    """Time for each player to reach each target point (float32).

    Players keep moving along their current velocity for `reaction_time`, then run in a
    straight line at `max_speed`. pos, vel: (N, 2). targets: (..., 2). Returns (N, ...).
    """
    r_react = (pos + vel * params.reaction_time).astype(np.float32)
    r_react = r_react.reshape((len(pos),) + (1,) * (targets.ndim - 1) + (2,))
    t = targets.astype(np.float32)
    dx = t[None, ..., 0] - r_react[..., 0]
    dy = t[None, ..., 1] - r_react[..., 1]
    return np.float32(params.reaction_time) + np.sqrt(dx * dx + dy * dy) / np.float32(
        params.max_speed)


def _logistic_rate(sigma: float) -> float:
    """Slope of the arrival logistic: P(arrived by t) = 1 / (1 + exp(-k (t - tti)))."""
    return np.pi / np.sqrt(3.0) / sigma


# Cap on logistic exponents: e^60 can't fall to O(1) within max_int_time (k * 10 s ≈ 40), and
# it keeps float32 from overflowing.
_MAX_EXPONENT = 60.0
# Drop converged cells from the working arrays once fewer than this share is still active.
_COMPACT_BELOW = 0.7


def _valid(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = ~(np.isnan(pos).any(axis=1) | np.isnan(vel).any(axis=1))
    return pos[mask], vel[mask], mask


@dataclass
class ControlSurface:
    attack: np.ndarray  # (G,) P(attacking team controls cell)
    defence: np.ndarray  # (G,)
    attack_players: np.ndarray  # (N_att, G) per-player share of attack control
    attack_mask: np.ndarray  # (N_att_roster,) which roster slots were on the pitch


def pitch_control(att_pos: np.ndarray, att_vel: np.ndarray, def_pos: np.ndarray,
                  def_vel: np.ndarray, ball: np.ndarray, grid: np.ndarray,
                  def_gk: int | None = None,
                  params: PhysicsParams = DEFAULT_PARAMS) -> ControlSurface:
    """Spearman pitch control for every grid cell simultaneously.

    Roster-shaped arrays may contain NaN rows for players not on the pitch; they are dropped.
    `def_gk` is the defending goalkeeper's roster index (gets a higher control rate).
    """
    a_pos, a_vel, a_mask = _valid(att_pos, att_vel)
    d_pos, d_vel, d_mask = _valid(def_pos, def_vel)
    n_att, n_grid = len(a_pos), len(grid)
    f32 = np.float32

    # Both teams in one (N, G) block: attackers first, then defenders.
    tti = time_to_intercept(np.concatenate([a_pos, d_pos]), np.concatenate([a_vel, d_vel]),
                            grid, params)
    ball_t = (np.sqrt(((grid - ball) ** 2).sum(axis=1)) / params.ball_speed).astype(f32)
    lam = np.r_[np.full(n_att, params.lambda_att),
                np.full(len(d_pos), params.lambda_att * params.kappa_def)]
    if def_gk is not None and d_mask[def_gk]:
        lam[n_att + np.flatnonzero(d_mask).tolist().index(def_gk)] *= params.lambda_gk_factor
    lam_dt = (lam * params.int_dt).astype(f32)[:, None]

    # Integrate from the ball's arrival in steps of int_dt. P(arrived) = 1 / (1 + E) with
    # E = exp(-k (t - tti)), so each step just multiplies E by exp(-k dt): no exp in the loop.
    k = _logistic_rate(params.tti_sigma)
    E = np.exp(np.minimum(k * (tti - ball_t[None]), _MAX_EXPONENT)).astype(f32)
    decay = f32(np.exp(-k * params.int_dt))
    tol = f32(1.0 - params.convergence_tol)

    ppcf_all = np.zeros((len(lam), n_grid), f32)
    total_all = np.zeros(n_grid, f32)
    cells = np.arange(n_grid)  # grid index of each working column
    ppcf = np.zeros_like(E)
    total = np.zeros(n_grid, f32)
    active = np.ones(n_grid, dtype=bool)
    for _ in range(int(params.max_int_time / params.int_dt)):
        remaining = np.where(active, np.maximum(1 - total, 0), 0).astype(f32, copy=False)
        step = lam_dt / (1 + E) * remaining
        ppcf += step
        total += step.sum(axis=0)
        active &= total < tol
        n_active = int(active.sum())
        if n_active == 0:
            break
        if n_active < _COMPACT_BELOW * len(cells):
            done = ~active
            ppcf_all[:, cells[done]] = ppcf[:, done]
            total_all[cells[done]] = total[done]
            cells, E, ppcf = cells[active], E[:, active], ppcf[:, active]
            total, active = total[active], active[active]
        E *= decay
    ppcf_all[:, cells] = ppcf
    total_all[cells] = total

    # Normalise away the small residual left by the convergence tolerance.
    ppcf_all /= np.maximum(total_all, 1e-9)
    ppcf_att, ppcf_def = ppcf_all[:n_att], ppcf_all[n_att:]
    return ControlSurface(
        attack=ppcf_att.sum(axis=0),
        defence=ppcf_def.sum(axis=0),
        attack_players=ppcf_att,
        attack_mask=a_mask,
    )


def pass_reachability(ball: np.ndarray, def_pos: np.ndarray, def_vel: np.ndarray,
                      grid: np.ndarray, params: PhysicsParams = DEFAULT_PARAMS) -> np.ndarray:
    """P(a ground pass from the ball to each cell is NOT intercepted en route).

    Points are sampled along each straight passing lane; for each, a defender intercepts with
    logistic probability in (ball arrival time - defender time-to-intercept). Combined with
    the Pressing Intensity rule P = 1 - prod(1 - p_i), then reach = 1 - P.
    """
    d_pos, d_vel, _ = _valid(def_pos, def_vel)
    if len(d_pos) == 0:
        return np.ones(len(grid))
    s = np.linspace(0.0, 1.0, params.lane_samples + 2)[1:-1]  # exclude passer and target
    lane = ball[None, None, :] + s[None, :, None] * (grid - ball)[:, None, :]  # (G, S, 2)
    dist = np.sqrt(((grid - ball) ** 2).sum(axis=1))
    ball_t = (s[None, :] * dist[:, None] / params.ball_speed).astype(np.float32)  # (G, S)
    tti = time_to_intercept(d_pos, d_vel, lane, params)  # (Nd, G, S)
    # 1 - p_intercept = 1 - logistic(k (ball_t - tti)) = 1 / (1 + exp(k (ball_t - tti)))
    k = np.float32(_logistic_rate(params.tti_sigma))
    p_free = 1.0 / (1.0 + np.exp(np.minimum(k * (ball_t[None] - tti), _MAX_EXPONENT)))
    return np.prod(p_free, axis=(0, 2)).astype(np.float64)
