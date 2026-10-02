"""Expected Space (xSpace): reachable, valuable space the defence leaves open.

For every grid cell g in a frame, with the attacking team oriented towards +x:

    xspace(g) = control_att(g) * reach(g) * value(g)

- control_att: Spearman pitch control (can an attacker get there first?)
- reach:       P(a pass from the ball to g is not intercepted en route)
- value:       xT gained by moving the ball there, max(xT(g) - xT(ball), 0). Raw xT would let
               the huge, safe, low-value area in a team's own half dominate every total.

Totals are area-weighted (sum * cell area, units xT·m²) so they don't depend on grid size.

Cells are labelled by zone relative to the defensive shape: behind the last line, between the
lines, wide of the block, or in front of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from xspace.constants import PITCH_LENGTH, PITCH_WIDTH
from xspace.io.loaders import MatchTracking
from xspace.physics.pitch_control import (
    DEFAULT_PARAMS,
    ControlSurface,
    PhysicsParams,
    pass_reachability,
    pitch_control,
)
from xspace.value.xt import xt_value

ZONES = ("behind", "between", "wide", "in_front")


@dataclass
class DefensiveShape:
    offside_line: float  # x of the second-deepest defender (incl. GK), not behind halfway
    back_line: float  # median x of the defensive back line
    mid_line: float  # median x of the defending midfield line
    block_y: tuple[float, float]  # lateral extent of the outfield defenders


@dataclass
class FrameSpace:
    frame: int
    attacking_side: int  # 0 home, 1 away
    control: ControlSurface
    reach: np.ndarray  # (G,)
    value: np.ndarray  # (G,)
    xspace: np.ndarray  # (G,)
    zone: np.ndarray  # (G,) int index into ZONES
    shape: DefensiveShape
    totals: dict[str, float] = field(default_factory=dict)


def orient(points: np.ndarray, attacking_side: int) -> np.ndarray:
    """Rotate so the attacking team always attacks +x (home already does)."""
    return points if attacking_side == 0 else -points


MIN_LINE_SIZE = 2


def _split_lines(xs: np.ndarray) -> list[np.ndarray]:
    """Split x-positions (sorted deepest first) into up to 3 lines at the largest gaps.

    Every line needs MIN_LINE_SIZE players, so a lone defender tracking a runner deep doesn't
    get mistaken for the whole back line.
    """
    n = len(xs)
    gaps = xs[:-1] - xs[1:]  # gaps[i] separates xs[i] and xs[i + 1]
    best, best_cuts = -1.0, None
    for i in range(MIN_LINE_SIZE - 1, n - 2 * MIN_LINE_SIZE):
        for j in range(i + MIN_LINE_SIZE, n - MIN_LINE_SIZE):
            if gaps[i] + gaps[j] > best:
                best, best_cuts = gaps[i] + gaps[j], (i, j)
    if best_cuts is None:  # too few players for three lines: split once
        cut = int(np.argmax(gaps[: max(n - 1, 1)])) if n > 1 else 0
        return [xs[: cut + 1], xs[cut + 1:]]
    i, j = best_cuts
    return [xs[: i + 1], xs[i + 1: j + 1], xs[j + 1:]]


def defensive_shape(def_pos: np.ndarray, gk_index: int | None, ball_x: float) -> DefensiveShape:
    """Offside line plus back and midfield lines of the defending team (deepest = largest x)."""
    on = ~np.isnan(def_pos).any(axis=1)
    xs_all = np.sort(def_pos[on, 0])[::-1]
    offside = float(max(xs_all[1] if len(xs_all) > 1 else 0.0, ball_x, 0.0))

    outfield = on.copy()
    if gk_index is not None:
        outfield[gk_index] = False
    pts = def_pos[outfield]
    if len(pts) < 2:
        return DefensiveShape(offside, offside, offside, (-34.0, 34.0))

    lines = _split_lines(np.sort(pts[:, 0])[::-1])
    back = float(np.median(lines[0]))
    mid = float(np.median(lines[1])) if len(lines) > 1 and len(lines[1]) else back
    return DefensiveShape(
        offside_line=offside,
        back_line=back,
        mid_line=min(mid, back),
        block_y=(float(pts[:, 1].min()), float(pts[:, 1].max())),
    )


def zone_cells(grid: np.ndarray, shape: DefensiveShape) -> np.ndarray:
    x, y = grid[:, 0], grid[:, 1]
    inside = (y >= shape.block_y[0]) & (y <= shape.block_y[1])
    zone = np.full(len(grid), ZONES.index("in_front"))
    beyond_mid = x > shape.mid_line
    zone[beyond_mid & inside] = ZONES.index("between")
    zone[beyond_mid & ~inside] = ZONES.index("wide")
    zone[x > shape.back_line] = ZONES.index("behind")
    return zone


def frame_space(match: MatchTracking, frame: int, grid: np.ndarray,
                params: PhysicsParams = DEFAULT_PARAMS) -> FrameSpace | None:
    """Expected Space for one frame, or None if possession or the ball is unknown."""
    side = int(match.ball_owner[frame])
    ball = match.ball[frame]
    if side < 0 or np.isnan(ball).any():
        return None

    att_pos, att_vel = match.team_arrays(side)
    def_pos, def_vel = match.team_arrays(1 - side)
    def_gk = match.gk_at(1 - side, frame)
    ap, av = orient(att_pos[frame], side), orient(att_vel[frame], side)
    dp, dv = orient(def_pos[frame], side), orient(def_vel[frame], side)
    b = orient(ball, side)

    control = pitch_control(ap, av, dp, dv, b, grid, def_gk=def_gk, params=params)
    reach = pass_reachability(b, dp, dv, grid, params)
    value = np.maximum(xt_value(grid) - xt_value(b[None])[0], 0.0)
    xspace = control.attack * reach * value
    shape = defensive_shape(dp, def_gk, float(b[0]))
    zone = zone_cells(grid, shape)

    cell_area = PITCH_LENGTH * PITCH_WIDTH / len(grid)
    totals = {"total": float(xspace.sum() * cell_area), "best": float(xspace.max())}
    for i, name in enumerate(ZONES):
        totals[name] = float(xspace[zone == i].sum() * cell_area)
    return FrameSpace(frame, side, control, reach, value, xspace, zone, shape, totals)
