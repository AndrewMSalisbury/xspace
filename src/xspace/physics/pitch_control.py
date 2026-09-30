"""Vectorised physics-based pitch control and pass reachability.

Pitch control follows Spearman (2018), "Beyond Expected Goals", as implemented by
Shaw (LaurieOnTracking) but vectorised over every grid cell at once. Time-to-intercept is the
same quantity used by Bekkers (2025), "Pressing Intensity", so defensive pressure and
offensive space are computed from one consistent model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xspace.constants import PITCH_LENGTH, PITCH_WIDTH


@dataclass(frozen=True)
class PhysicsParams:
    reaction_time: float = 0.7  # s before a player can change course
    max_speed: float = 5.0  # m/s, average max running speed
    tti_sigma: float = 0.45  # s, uncertainty in arrival time (same sigma as Pressing Intensity)
    lambda_att: float = 4.3  # 1/s, rate of gaining control once at the ball
    kappa_def: float = 1.0  # defender advantage multiplier on lambda
    lambda_gk_factor: float = 3.0  # goalkeepers can handle the ball
    ball_speed: float = 15.0  # m/s, average ground-pass speed
    int_dt: float = 0.04  # s, integration step
    max_int_time: float = 10.0  # s
    convergence_tol: float = 0.01
    lane_samples: int = 12  # points sampled along a pass to test interception


DEFAULT_PARAMS = PhysicsParams()


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
    """Time for each player to reach each target point.

    Players keep moving along their current velocity for `reaction_time`, then run in a
    straight line at `max_speed`. pos, vel: (N, 2). targets: (..., 2). Returns (N, ...).
    """
    r_react = pos + vel * params.reaction_time  # (N, 2)
    shape = (len(pos),) + (1,) * (targets.ndim - 1) + (2,)
    dist = np.linalg.norm(targets[None] - r_react.reshape(shape), axis=-1)
    return params.reaction_time + dist / params.max_speed


def _arrival_prob(t: np.ndarray, tti: np.ndarray, sigma: float) -> np.ndarray:
    """P(player has arrived by time t), logistic in (t - tti)."""
    return 1.0 / (1.0 + np.exp(-np.pi / np.sqrt(3.0) / sigma * (t - tti)))


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
    n_grid = len(grid)

    tti_att = time_to_intercept(a_pos, a_vel, grid, params)  # (Na, G)
    tti_def = time_to_intercept(d_pos, d_vel, grid, params)  # (Nd, G)
    ball_t = np.linalg.norm(grid - ball, axis=1) / params.ball_speed  # (G,)

    lam_att = np.full(len(a_pos), params.lambda_att)
    lam_def = np.full(len(d_pos), params.lambda_att * params.kappa_def)
    if def_gk is not None and d_mask[def_gk]:
        lam_def[np.flatnonzero(d_mask).tolist().index(def_gk)] *= params.lambda_gk_factor

    ppcf_att = np.zeros((len(a_pos), n_grid))
    ppcf_def = np.zeros((len(d_pos), n_grid))
    total = np.zeros(n_grid)
    active = np.ones(n_grid, dtype=bool)
    n_steps = int(params.max_int_time / params.int_dt)

    for k in range(1, n_steps + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        t = ball_t[idx] - params.int_dt + k * params.int_dt
        remaining = 1.0 - total[idx]
        d_att = remaining * _arrival_prob(t, tti_att[:, idx], params.tti_sigma) * lam_att[:, None]
        d_def = remaining * _arrival_prob(t, tti_def[:, idx], params.tti_sigma) * lam_def[:, None]
        ppcf_att[:, idx] += np.maximum(d_att * params.int_dt, 0.0)
        ppcf_def[:, idx] += np.maximum(d_def * params.int_dt, 0.0)
        total[idx] = ppcf_att[:, idx].sum(axis=0) + ppcf_def[:, idx].sum(axis=0)
        active[idx] = total[idx] < 1.0 - params.convergence_tol

    # Normalise away the small residual left by the convergence tolerance.
    norm = np.maximum(total, 1e-9)
    ppcf_att /= norm
    ppcf_def /= norm
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
    ball_t = np.linalg.norm(lane - ball, axis=-1) / params.ball_speed  # (G, S)
    tti = time_to_intercept(d_pos, d_vel, lane, params)  # (Nd, G, S)
    p_int = _arrival_prob(ball_t[None], tti, params.tti_sigma)
    p_blocked = 1.0 - np.prod(1.0 - p_int, axis=(0, 2))
    return 1.0 - p_blocked
