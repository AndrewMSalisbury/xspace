"""Phase 4 pass dataset: one frame snapshot per open-play pass or cross, for calibration.

Calibrating the physics against pass outcomes only needs the model evaluated at one point
per pass, so instead of whole grids this keeps, for each pass at its release frame:

- the on-pitch players of both teams, in the passing team's frame (it attacks +x), packed
  into `MAX_PLAYERS` slots (NaN padding), with velocities;
- which attackers were offside (left out of pitch control, as in `metrics.space`);
- the defending goalkeeper's slot, the intended target's and the actual receiver's slots;
- where the ball went (`end`), whether the pass was completed, and PFF's ball-height codes
  (at contact, and the peak in flight, which sets the trajectory).

`PassSet` holds the arrays for many passes; `build_pass_set` makes one for a prepared match.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from xspace.config import DEFAULT_EXPLOITATION, DEFAULT_PARAMS, ExploitationConfig, PhysicsParams
from xspace.io.loaders import MatchTracking
from xspace.metrics.exploitation import CHOSEN_SOURCES, chosen_points, select_actions
from xspace.metrics.space import defensive_shape, offside_attackers, orient

MAX_PLAYERS = 12  # on-pitch players per team (11, plus one spare for a stray ghost track)
# PFF highPointType (the ball's peak height in flight) grouped by trajectory. G = ground and
# L = low stay interceptable along the lane; A = above head, M and H go over players. (The other
# code, ballHeightType, is the height at contact: a header is "A" there even if it's 5 m.)
GROUND_CODES = ("G", "L")
AIR_CODES = ("A", "M", "H")
TRAJECTORIES = ("ground", "air", "unknown")


@dataclass
class PassSet:
    """Arrays for N passes (all positions in the passing team's frame, attacking +x)."""

    source: np.ndarray  # (N,) str
    match_id: np.ndarray  # (N,) str
    event_id: np.ndarray  # (N,) str
    type: np.ndarray  # (N,) str: pass / cross
    frame: np.ndarray  # (N,) release frame
    team_side: np.ndarray  # (N,) passing team: 0 home, 1 away
    success: np.ndarray  # (N,) float: 1, 0
    trajectory: np.ndarray  # (N,) int8 index into TRAJECTORIES
    height: np.ndarray  # (N,) str: raw ball height at contact ('' if none)
    high_point: np.ndarray  # (N,) str: raw peak height in flight ('' if none)
    ball: np.ndarray  # (N, 2)
    end: np.ndarray  # (N, 2) chosen point (see metrics.exploitation.chosen_points)
    end_source: np.ndarray  # (N,) str
    att_pos: np.ndarray  # (N, MAX_PLAYERS, 2); offside attackers included
    att_vel: np.ndarray
    def_pos: np.ndarray
    def_vel: np.ndarray
    offside: np.ndarray  # (N, MAX_PLAYERS) bool
    def_gk: np.ndarray  # (N,) slot, -1 if none
    passer: np.ndarray  # (N,) attacker slot, -1 if not found
    target: np.ndarray  # (N,) attacker slot of the intended receiver (PFF), -1 if none
    receiver: np.ndarray  # (N,) attacker slot of the actual receiver, -1 if none

    def __len__(self) -> int:
        return len(self.frame)

    def subset(self, mask: np.ndarray) -> PassSet:
        return PassSet(**{f.name: getattr(self, f.name)[mask] for f in fields(self)})

    @property
    def distance(self) -> np.ndarray:
        return np.linalg.norm(self.end - self.ball, axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f.name: getattr(self, f.name) for f in fields(self)}
        arrays = {k: v.astype(str) if v.dtype == object else v for k, v in arrays.items()}
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> PassSet:
        with np.load(path, allow_pickle=False) as z:
            return cls(**{f.name: z[f.name] for f in fields(cls)})

    @classmethod
    def concat(cls, sets: list[PassSet]) -> PassSet:
        return cls(**{f.name: np.concatenate([getattr(s, f.name) for s in sets])
                      for f in fields(cls)})


def trajectory_code(height: str | None) -> int:
    if height in GROUND_CODES:
        return TRAJECTORIES.index("ground")
    if height in AIR_CODES:
        return TRAJECTORIES.index("air")
    return TRAJECTORIES.index("unknown")


def _pack(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """On-pitch rows of a roster-shaped (P, 2) array packed into MAX_PLAYERS slots. Returns the
    packed positions, velocities and the roster slot of each packed slot (-1 for padding)."""
    on = np.flatnonzero(~(np.isnan(pos).any(axis=1) | np.isnan(vel).any(axis=1)))
    on = on[:MAX_PLAYERS]
    p = np.full((MAX_PLAYERS, 2), np.nan)
    v = np.full((MAX_PLAYERS, 2), np.nan)
    slots = np.full(MAX_PLAYERS, -1)
    p[:len(on)], v[:len(on)], slots[:len(on)] = pos[on], vel[on], on
    return p, v, slots


def build_pass_set(source: str, match: MatchTracking, events: pd.DataFrame, flags: np.ndarray,
                   config: ExploitationConfig = DEFAULT_EXPLOITATION,
                   params: PhysicsParams = DEFAULT_PARAMS) -> PassSet:
    """Open-play passes and crosses of a synced match (events need `release_frame`) with a
    usable release frame, a known outcome and a chosen point."""
    actions = select_actions(events)
    actions = actions[actions["type"].isin(["pass", "cross"]).to_numpy()]
    end, end_source = chosen_points(actions, events, match, params)
    frames = actions["release_frame"].to_numpy().astype(np.int64)
    success = actions["success"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    safe = np.maximum(frames, 0)
    keep = ((frames >= 0) & ~np.isnan(match.ball[safe]).any(axis=1)
            & ((flags[safe] & config.skip_flags) == 0)
            & ~np.isnan(end).any(axis=1) & ~np.isnan(success))
    actions, end, end_source = actions[keep], end[keep], end_source[keep]
    frames, success = frames[keep], success[keep]

    n, P = len(actions), MAX_PLAYERS
    out = {k: np.full((n, P, 2), np.nan) for k in ("att_pos", "att_vel", "def_pos", "def_vel")}
    offside = np.zeros((n, P), dtype=bool)
    ball = np.full((n, 2), np.nan)
    end_o = np.full((n, 2), np.nan)
    slot_cols = {k: np.full(n, -1, dtype=np.int16)
                 for k in ("def_gk", "passer", "target", "receiver")}
    sides = actions["team_side"].to_numpy().astype(int)
    teams = (match.home, match.away)
    for i, (f, side) in enumerate(zip(frames, sides, strict=True)):
        att_pos, att_vel = match.team_arrays(side)
        def_pos, def_vel = match.team_arrays(1 - side)
        ap, av, a_slots = _pack(orient(att_pos[f], side), orient(att_vel[f], side))
        dp, dv, d_slots = _pack(orient(def_pos[f], side), orient(def_vel[f], side))
        b = orient(match.ball[f], side)
        gk = match.gk_at(1 - side, f)
        gk_slot = int(np.flatnonzero(d_slots == gk)[0]) if gk is not None and gk in d_slots else -1
        shape = defensive_shape(dp, gk_slot if gk_slot >= 0 else None, float(b[0]))
        offside[i] = offside_attackers(ap, b, shape.offside_line, params.offside_margin)
        out["att_pos"][i], out["att_vel"][i], out["def_pos"][i], out["def_vel"][i] = ap, av, dp, dv
        ball[i], end_o[i] = b, orient(end[i], side)
        slot_cols["def_gk"][i] = gk_slot
        ids = teams[side].player_ids
        for col, pid in (("passer", actions["player_id"].iat[i]),
                         ("target", actions["target_player_id"].iat[i]),
                         ("receiver", actions["receiver_player_id"].iat[i])):
            if isinstance(pid, str) and pid in ids:
                hit = np.flatnonzero(a_slots == ids.index(pid))
                if len(hit):
                    slot_cols[col][i] = hit[0]

    heights = actions["height"].fillna("").astype(str).to_numpy()
    peaks = actions["high_point"].fillna("").astype(str).to_numpy()
    return PassSet(
        source=np.full(n, source), match_id=np.full(n, str(match.match_id)),
        event_id=actions["event_id"].astype(str).to_numpy(),
        type=actions["type"].astype(str).to_numpy(), frame=frames, team_side=sides,
        success=success,
        trajectory=np.array([trajectory_code(h or None) for h in peaks], dtype=np.int8),
        height=heights, high_point=peaks, ball=ball, end=end_o,
        end_source=np.array([CHOSEN_SOURCES[s] for s in end_source]),
        offside=offside, **out, **slot_cols,
    )
