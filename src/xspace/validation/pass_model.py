"""Pass success under the xSpace physics, evaluated at one target point per pass (batched).

    P(success) = control_att(target) * reach(target)

the same two factors that weight every cell of xSpace (`metrics.space`): can an attacker get
to the target first once the ball arrives, and does the ball get there without being cut out.
`control_at` and `ground_reach` reproduce `physics.pitch_control.pitch_control` and
`pass_reachability` exactly, but for N passes x one cell instead of one frame x every cell, so
a whole tournament's passes take a second or two and the physics can be fitted to outcomes.

Trajectories (`Trajectory`):

- ground: the ball rolls at `ball_speed`; any defender who can reach a point on the lane before
  the ball can cut it out (the current xSpace reach model).
- air: a lofted ball. Flight time `air_time + distance / air_speed`; it can't be intercepted
  in flight, only contested where it lands (by pitch control, with the longer flight time).

Targets: `end` is where the ball went (`PassSet.end`); `intent` is the intended receiver (PFF
`targetPlayerId`) projected along their velocity for the flight time, for every pass,
completed or not, so both outcomes are scored at the same kind of point.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import rankdata

from xspace.config import DEFAULT_PARAMS, PhysicsParams
from xspace.physics.pitch_control import _MAX_EXPONENT, _logistic_rate
from xspace.validation.passes import PassSet

P_CLIP = 1e-4  # predictions are clipped to [P_CLIP, 1 - P_CLIP] for log loss
_FAR = 1e4  # time-to-intercept for empty (NaN) player slots: never arrives


@dataclass(frozen=True)
class Trajectory:
    kind: str = "ground"  # "ground" or "air"
    air_speed: float = 15.0  # m/s horizontal, lofted balls
    air_time: float = 0.0  # s added to every lofted flight (hang time)
    # Multiplies every player's control rate where a lofted ball lands: a dropping ball is
    # harder to bring under control than a rolling one.
    lambda_factor: float = 1.0

    def speed(self, params: PhysicsParams) -> float:
        return params.ball_speed if self.kind == "ground" else self.air_speed

    @property
    def delay(self) -> float:
        return 0.0 if self.kind == "ground" else self.air_time

    def flight_time(self, dist: np.ndarray, params: PhysicsParams) -> np.ndarray:
        return self.delay + dist / self.speed(params)

    def meet_time(self, offset: np.ndarray, vel: np.ndarray, params: PhysicsParams) -> np.ndarray:
        """Flight time T at which the ball meets a receiver at `offset` (N, 2) from the ball
        running at constant `vel` (N, 2): |offset + vel T| = speed (T - delay). Receivers
        running at 90% of the ball's speed or more are capped there, so a solution exists."""
        speed = self.speed(params)
        v_norm = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = vel * np.minimum(1.0, 0.9 * speed / np.maximum(v_norm, 1e-9))
        d = offset + vel * self.delay  # where the receiver is when the delay is over
        a = (vel * vel).sum(axis=1) - speed**2  # < 0
        b = 2 * (d * vel).sum(axis=1)
        c = (d * d).sum(axis=1)
        u = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)  # the root ≥ 0 (a < 0 ≤ c)
        return self.delay + u


GROUND = Trajectory()


def _tti(pos: np.ndarray, vel: np.ndarray, target: np.ndarray,
         params: PhysicsParams) -> np.ndarray:
    """Time to intercept: pos, vel (N, P, 2), target (N, [S,] 2) → (N, P[, S])."""
    r = pos + vel * params.reaction_time
    if target.ndim == 3:
        r = r[:, :, None, :]
        target = target[:, None, :, :]
    else:
        target = target[:, None, :]
    t = params.reaction_time + np.linalg.norm(target - r, axis=-1) / params.max_speed
    return np.where(np.isnan(t), _FAR, t)


def control_at(ps: PassSet, target: np.ndarray, ball_t: np.ndarray,
               params: PhysicsParams = DEFAULT_PARAMS,
               lambda_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Attacking pitch control at one target per pass, with the ball arriving after `ball_t`.

    Offside attackers are left out. Returns (attack control (N,), per-attacker shares (N, P)).
    """
    att_pos = np.where(ps.offside[..., None], np.nan, ps.att_pos)
    pos = np.concatenate([att_pos, ps.def_pos], axis=1)
    vel = np.concatenate([np.nan_to_num(ps.att_vel), np.nan_to_num(ps.def_vel)], axis=1)
    n, n_att = len(ps), ps.att_pos.shape[1]
    tti = _tti(pos, vel, target, params)  # (N, 2P)

    lam = np.r_[np.full(n_att, params.lambda_att),
                np.full(ps.def_pos.shape[1], params.lambda_att * params.kappa_def)]
    lam = np.tile(lam, (n, 1))
    has_gk = ps.def_gk >= 0
    lam[np.flatnonzero(has_gk), n_att + ps.def_gk[has_gk]] *= params.lambda_gk_factor
    lam_dt = lam * (params.int_dt * lambda_scale)

    k = _logistic_rate(params.tti_sigma)
    E = np.exp(np.minimum(k * (tti - ball_t[:, None]), _MAX_EXPONENT))
    decay = np.exp(-k * params.int_dt)
    tol = 1.0 - params.convergence_tol
    ppcf = np.zeros_like(E)
    total = np.zeros(n)
    active = np.ones(n, dtype=bool)
    for _ in range(int(params.max_int_time / params.int_dt)):
        remaining = np.where(active, np.maximum(1 - total, 0), 0)
        step = lam_dt / (1 + E) * remaining[:, None]
        ppcf += step
        total += step.sum(axis=1)
        active &= total < tol
        if not active.any():
            break
        E *= decay
    ppcf /= np.maximum(total, 1e-9)[:, None]
    return ppcf[:, :n_att].sum(axis=1), ppcf[:, :n_att]


def ground_reach(ps: PassSet, target: np.ndarray,
                 params: PhysicsParams = DEFAULT_PARAMS) -> np.ndarray:
    """P(a ground pass from the ball to `target` (N, 2) isn't cut out), as pass_reachability."""
    s = np.linspace(0.0, 1.0, params.lane_samples + 2)[1:-1]
    lane = ps.ball[:, None, :] + s[None, :, None] * (target - ps.ball)[:, None, :]  # (N, S, 2)
    dist = np.linalg.norm(target - ps.ball, axis=1)
    ball_t = s[None, :] * dist[:, None] / params.ball_speed  # (N, S)
    tti = _tti(ps.def_pos, np.nan_to_num(ps.def_vel), lane, params)  # (N, P, S)
    k = _logistic_rate(params.tti_sigma)
    p_free = 1.0 / (1.0 + np.exp(np.minimum(k * (ball_t[:, None, :] - tti), _MAX_EXPONENT)))
    if params.intercept_factor != 1.0:
        p_free = 1.0 - params.intercept_factor * (1.0 - p_free)
    if params.lane_combine == "max":
        return p_free.min(axis=2).prod(axis=1)
    return p_free.prod(axis=(1, 2))


def target_points(ps: PassSet, kind: str, flight: np.ndarray | None = None) -> np.ndarray:
    """(N, 2) target per pass: `end`, or `intent` (NaN where there is no intended target)."""
    if kind == "end":
        return ps.end
    if kind != "intent":
        raise ValueError(kind)
    out = np.full((len(ps), 2), np.nan)
    has = ps.target >= 0
    rows = np.flatnonzero(has)
    p = ps.att_pos[rows, ps.target[has]]
    v = np.nan_to_num(ps.att_vel[rows, ps.target[has]])
    out[rows] = p if flight is None else p + v * flight[rows][:, None]
    return out


def intent_points(ps: PassSet, trajectory: Trajectory,
                  params: PhysicsParams) -> tuple[np.ndarray, np.ndarray]:
    """Where the ball meets the intended receiver running on at their current velocity, and
    the flight time, per pass (NaN without a target)."""
    flight = np.full(len(ps), np.nan)
    has = ps.target >= 0
    rows = np.flatnonzero(has)
    p = ps.att_pos[rows, ps.target[has]]
    v = np.nan_to_num(ps.att_vel[rows, ps.target[has]])
    flight[rows] = trajectory.meet_time(p - ps.ball[rows], v, params)
    return target_points(ps, "intent", flight), flight


@dataclass
class Prediction:
    control: np.ndarray
    reach: np.ndarray
    flight: np.ndarray
    target: np.ndarray

    @property
    def p(self) -> np.ndarray:
        return self.control * self.reach


def predict(ps: PassSet, params: PhysicsParams = DEFAULT_PARAMS,
            trajectory: Trajectory = GROUND, target: str = "end") -> Prediction:
    """Model pass success for every pass, all with the same trajectory."""
    if target == "intent":
        tgt, _ = intent_points(ps, trajectory, params)
    else:
        tgt = target_points(ps, target)
    dist = np.linalg.norm(tgt - ps.ball, axis=1)
    flight = trajectory.flight_time(dist, params)
    ok = ~np.isnan(tgt).any(axis=1)
    control = np.full(len(ps), np.nan)
    reach = np.full(len(ps), np.nan)
    sub = ps.subset(ok)
    scale = trajectory.lambda_factor if trajectory.kind == "air" else 1.0
    control[ok], _ = control_at(sub, tgt[ok], flight[ok], params, scale)
    reach[ok] = ground_reach(sub, tgt[ok], params) if trajectory.kind == "ground" else 1.0
    return Prediction(control, reach, flight, tgt)


def with_params(params: PhysicsParams, **changes) -> PhysicsParams:
    return replace(params, **changes)


# --- scores ---------------------------------------------------------------------------------

def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U; ties count half)."""
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"n": int(len(y)), "log_loss": log_loss(y, p), "brier": brier(y, p),
            "auc": auc(y, p), "mean_p": float(np.mean(p)), "rate": float(np.mean(y))}


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> np.ndarray:
    """(bins, 3): mean prediction, observed rate and count per equal-width bin of p."""
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = np.full((bins, 3), np.nan)
    for b in range(bins):
        m = idx == b
        if m.any():
            out[b] = p[m].mean(), y[m].mean(), m.sum()
    return out
