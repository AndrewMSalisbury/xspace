"""Ablation surfaces for validation tasks V1 (where does the next pass go?) and V3 (does space
now predict danger soon?).

Each rung of the ablation ladder is a per-cell surface built from one `FrameSpace`:

| Surface | Per cell |
|---|---|
| value | xT gained by moving the ball there (location only, no tracking) |
| control | ground-pass pitch control (Spearman) |
| control_value | control x value |
| xspace | control x ground reach x value: xSpace (ground passes, the default) |
| xspace_lofted | receive x reach x value: ground or lofted ball, whichever is better |
| pass_prob | control x ground reach: P(a pass there arrives and is received), no value |

`compute` should get physics with the lofted ball on (`air_speed > 0`) so that `xspace_lofted`
differs from `xspace`; the other surfaces don't use it.

V3 frames (`frame_rows`) store each surface's area-weighted total and best cell, plus the ball's
xT, with labels: did the team shoot / enter the box within `HORIZON_S` in the same possession.
V1 passes (`pass_rows`) store, for each surface, where the actual end cell ranks in the frame
(`rank_*`: share of cells with a lower value) and the softmax log-likelihood of the end cell for
each temperature in `BETAS` and distance weight in `GAMMAS` (`ll_{s}_{β}_{γ}`):

    P(cell) ∝ exp(β · s / max s − γ · d / 10 m)       d: distance from the ball

Passes are mostly short whatever the space looks like, so distance alone predicts the end cell
well. γ > 0 gives every surface that prior, and asks what the surface adds on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xspace.config import PhysicsParams
from xspace.constants import PITCH_LENGTH, PITCH_WIDTH
from xspace.io.loaders import MatchTracking
from xspace.metrics.possessions import BOX_DEPTH, BOX_HALF_WIDTH
from xspace.metrics.space import FrameSpace, orient, space_from_arrays
from xspace.metrics.timeline import _grid
from xspace.phases.possession import PhaseLabels
from xspace.value.xt import xt_value

SURFACES = ("value", "control", "control_value", "xspace", "xspace_lofted", "pass_prob")
BETAS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)  # per 10 m
HORIZON_S = 10.0


def surfaces(fs: FrameSpace) -> dict[str, np.ndarray]:
    ground = fs.control.attack
    return {
        "value": fs.value,
        "control": ground,
        "control_value": ground * fs.value,
        "xspace": ground * fs.ground_reach * fs.value,
        "xspace_lofted": fs.xspace,
        "pass_prob": ground * fs.ground_reach,
    }


@dataclass
class Snapshots:
    """Frames to evaluate, in pitch coordinates. `end` is NaN for V3 frames."""

    side: np.ndarray  # (N,)
    ball: np.ndarray  # (N, 2)
    end: np.ndarray  # (N, 2)
    home_pos: np.ndarray
    home_vel: np.ndarray
    away_pos: np.ndarray
    away_vel: np.ndarray
    home_gk: np.ndarray
    away_gk: np.ndarray


def snapshots(match: MatchTracking, frames: np.ndarray, side: np.ndarray,
              end: np.ndarray | None = None) -> Snapshots:
    return Snapshots(side=side, ball=match.ball[frames],
                     end=np.full((len(frames), 2), np.nan) if end is None else end,
                     home_pos=match.home_pos[frames], home_vel=match.home_vel[frames],
                     away_pos=match.away_pos[frames], away_vel=match.away_vel[frames],
                     home_gk=match.home_gk[frames], away_gk=match.away_gk[frames])


def compute(snap: Snapshots, cell_size: float, params: PhysicsParams) -> dict[str, np.ndarray]:
    """Per-snapshot totals, best cells, end-cell ranks and log-likelihoods for every surface.
    Top-level so worker processes can import it."""
    grid = _grid(cell_size)
    area = PITCH_LENGTH * PITCH_WIDTH / len(grid)
    n, nb, ng = len(snap.side), len(BETAS), len(GAMMAS)
    out = {}
    for s in SURFACES:
        out[f"total_{s}"] = np.full(n, np.nan)
        out[f"best_{s}"] = np.full(n, np.nan)
        out[f"rank_{s}"] = np.full(n, np.nan)
        out[f"ll_{s}"] = np.full((n, nb, ng), np.nan, dtype=np.float32)
    out["air_share"] = np.full(n, np.nan)
    betas = np.asarray(BETAS)[:, None, None]
    gammas = np.asarray(GAMMAS)[None, :, None]
    for i in range(n):
        side = int(snap.side[i])
        teams = ((snap.home_pos, snap.home_vel, snap.home_gk),
                 (snap.away_pos, snap.away_vel, snap.away_gk))
        att_pos, att_vel, _ = teams[side]
        def_pos, def_vel, def_gk = teams[1 - side]
        gk = int(def_gk[i])
        fs = space_from_arrays(att_pos[i], att_vel[i], def_pos[i], def_vel[i], snap.ball[i],
                               gk if gk >= 0 else None, side, grid, params)
        out["air_share"][i] = fs.air.mean()
        has_end = not np.isnan(snap.end[i]).any()
        if has_end:
            e = orient(snap.end[i], side)
            cell = int(np.argmin(((grid - e) ** 2).sum(axis=1)))
            b = orient(snap.ball[i], side)
            dist = gammas * (np.sqrt(((grid - b) ** 2).sum(axis=1)) / 10.0)[None, None]
        for name, surf in surfaces(fs).items():
            out[f"total_{name}"][i] = surf.sum() * area
            top = surf.max()
            out[f"best_{name}"][i] = top
            if has_end:
                out[f"rank_{name}"][i] = (surf < surf[cell]).mean()
                z = betas * (surf / top if top > 0 else np.zeros_like(surf)) - dist
                zmax = z.max(axis=2, keepdims=True)
                lse = zmax[..., 0] + np.log(np.exp(z - zmax).sum(axis=2))
                out[f"ll_{name}"][i] = z[..., cell] - lse
    return out


def danger_labels(match: MatchTracking, events: pd.DataFrame, phases: PhaseLabels,
                  frames: np.ndarray, horizon_s: float = HORIZON_S) -> pd.DataFrame:
    """For each frame: does the team in possession shoot, or get the ball into the opponent's
    box, within `horizon_s` and without losing the possession?"""
    pid = phases.possession_id
    side = phases.possession_side
    horizon = int(round(horizon_s * match.frame_rate))
    n = match.n_frames

    shots = events[(events["type"] == "shot") & (events["frame"] >= 0)]
    shot_frame = np.zeros(n, dtype=bool)
    shot_frame[shots["frame"].to_numpy().astype(int)] = True
    bx = np.where(side == 1, -match.ball[:, 0], match.ball[:, 0])
    by = match.ball[:, 1]
    in_box = ((bx > PITCH_LENGTH / 2 - BOX_DEPTH) & (np.abs(by) < BOX_HALF_WIDTH)
              & (side >= 0))

    shot10 = np.zeros(len(frames), dtype=bool)
    box10 = np.zeros(len(frames), dtype=bool)
    for i, f in enumerate(frames):
        stop = min(f + horizon + 1, n)
        same = (pid[f + 1:stop] == pid[f]) & (match.period[f + 1:stop] == match.period[f])
        shot10[i] = (shot_frame[f + 1:stop] & same).any()
        box10[i] = (in_box[f + 1:stop] & same).any()
    return pd.DataFrame({"shot10": shot10, "box10": box10})


def frame_rows(match: MatchTracking, frames: np.ndarray, side: np.ndarray,
               results: dict[str, np.ndarray]) -> pd.DataFrame:
    ball = np.where(side[:, None] == 1, -match.ball[frames], match.ball[frames])
    df = pd.DataFrame({"frame": frames, "team_side": side, "ball_x": ball[:, 0],
                       "ball_y": ball[:, 1], "ball_xt": xt_value(ball),
                       "air_share": results["air_share"]})
    for s in SURFACES:
        df[f"total_{s}"] = results[f"total_{s}"]
        df[f"best_{s}"] = results[f"best_{s}"]
    return df


def pass_rows(results: dict[str, np.ndarray]) -> pd.DataFrame:
    cols = {}
    for s in SURFACES:
        cols[f"rank_{s}"] = results[f"rank_{s}"]
        for j, b in enumerate(BETAS):
            for k, g in enumerate(GAMMAS):
                cols[f"ll_{s}_{b:g}_{g:g}"] = results[f"ll_{s}"][:, j, k]
    return pd.DataFrame(cols)
