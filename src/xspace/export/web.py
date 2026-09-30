"""Export computed surfaces as compact JSON for the web front end.

Surfaces are quantised to 0-255 per layer (with the layer max stored), which keeps a frame at a
2 m grid around 10 KB. Coordinates are in the attacking-team-oriented frame (attack goes +x).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xspace.io.loaders import MatchTracking
from xspace.metrics.space import ZONES, FrameSpace, orient


def _quantise(surface: np.ndarray, shape: tuple[int, int]) -> dict:
    vmax = float(surface.max()) if surface.size else 0.0
    q = np.zeros_like(surface) if vmax <= 0 else np.round(surface / vmax * 255)
    return {"max": vmax, "values": q.astype(np.uint8).reshape(shape).tolist()}


def _players(pos: np.ndarray, vel: np.ndarray, team, side: int) -> list[dict]:
    out = []
    for j, pid in enumerate(team.player_ids):
        if np.isnan(pos[j]).any():
            continue
        p, v = orient(pos[j], side), orient(vel[j], side)
        out.append({
            "id": pid,
            "number": team.jersey_numbers[j],
            "position": team.positions[j],
            "x": round(float(p[0]), 2),
            "y": round(float(p[1]), 2),
            "vx": round(float(v[0]), 2),
            "vy": round(float(v[1]), 2),
        })
    return out


def frame_payload(match: MatchTracking, fs: FrameSpace, xs: np.ndarray, ys: np.ndarray) -> dict:
    f, side = fs.frame, fs.attacking_side
    att, dfn = (match.home, match.away) if side == 0 else (match.away, match.home)
    att_pos, att_vel = match.team_arrays(side)
    def_pos, def_vel = match.team_arrays(1 - side)
    shape = (len(ys), len(xs))
    ball = orient(match.ball[f], side)
    return {
        "match_id": match.match_id,
        "frame": f,
        "period": int(match.period[f]),
        "time_s": round(float(match.timestamp[f]), 2),
        "attacking_team": att.name,
        "defending_team": dfn.name,
        "ball": {"x": round(float(ball[0]), 2), "y": round(float(ball[1]), 2)},
        "attackers": _players(att_pos[f], att_vel[f], att, side),
        "defenders": _players(def_pos[f], def_vel[f], dfn, side),
        "grid": {"xs": np.round(xs, 2).tolist(), "ys": np.round(ys, 2).tolist()},
        "layers": {
            "control": _quantise(fs.control.attack, shape),
            "reach": _quantise(fs.reach, shape),
            "value": _quantise(fs.value, shape),
            "xspace": _quantise(fs.xspace, shape),
        },
        "shape": {
            "offside_line": fs.shape.offside_line,
            "back_line": fs.shape.back_line,
            "mid_line": fs.shape.mid_line,
            "block_y": list(fs.shape.block_y),
        },
        "zones": list(ZONES),
        "totals": fs.totals,
    }


def write_json(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return path
