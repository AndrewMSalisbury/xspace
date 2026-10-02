"""Per-frame data-quality flags, as bit masks so several can be set at once.

| Flag | Meaning |
|---|---|
| BALL_MISSING | no ball position |
| FEW_PLAYERS | a team has fewer than `MIN_PLAYERS` tracked (one red card is still fine) |
| BALL_JUMP | ball moved faster than `MAX_BALL_SPEED` since the previous frame |
| PLAYER_JUMP | a player moved faster than `MAX_PLAYER_SPEED` since the previous frame |

Jumps are measured on raw positions within a period and ignore dead-ball gaps (the time step is
taken from the timestamps). PFF's per-player "estimated" visibility isn't exposed by kloppy, so
it isn't flagged yet.
"""

from __future__ import annotations

import numpy as np

from xspace.io.loaders import MatchTracking

BALL_MISSING = 1
FEW_PLAYERS = 2
BALL_JUMP = 4
PLAYER_JUMP = 8
FLAG_NAMES = {BALL_MISSING: "ball_missing", FEW_PLAYERS: "few_players", BALL_JUMP: "ball_jump",
              PLAYER_JUMP: "player_jump"}

MIN_PLAYERS = 10
MAX_BALL_SPEED = 45.0  # m/s; hardest shots are ~35 m/s
MAX_PLAYER_SPEED = 13.0  # m/s; sprint records are ~12.5 m/s


def _speeds(xy: np.ndarray, match: MatchTracking) -> np.ndarray:
    """Speed from the previous frame (T, ...); NaN at period starts and after gaps."""
    dt = np.diff(match.timestamp)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    dt = dt.reshape(dt.shape + (1,) * (step.ndim - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = step / dt
    contiguous = (match.period[1:] == match.period[:-1]) & (np.diff(match.timestamp) > 0) & (
        np.diff(match.timestamp) < 1.5 / match.frame_rate)
    speed[~contiguous] = np.nan
    return np.concatenate([np.full((1,) + speed.shape[1:], np.nan), speed])


def quality_flags(match: MatchTracking) -> np.ndarray:
    """(T,) int bit mask; 0 means the frame passed every check."""
    flags = np.zeros(match.n_frames, dtype=np.int64)
    flags[np.isnan(match.ball).any(axis=1)] |= BALL_MISSING

    n_home = (~np.isnan(match.home_pos[:, :, 0])).sum(axis=1)
    n_away = (~np.isnan(match.away_pos[:, :, 0])).sum(axis=1)
    flags[(n_home < MIN_PLAYERS) | (n_away < MIN_PLAYERS)] |= FEW_PLAYERS

    with np.errstate(invalid="ignore"):
        flags[_speeds(match.ball, match) > MAX_BALL_SPEED] |= BALL_JUMP
        players = np.concatenate([match.home_pos, match.away_pos], axis=1)
        flags[np.nanmax(np.nan_to_num(_speeds(players, match), nan=0.0), axis=1)
              > MAX_PLAYER_SPEED] |= PLAYER_JUMP
    return flags


def flag_summary(flags: np.ndarray) -> dict[str, float]:
    """Percentage of frames with each flag, plus `clean` (no flags)."""
    out = {name: 100.0 * float(np.mean(flags & bit > 0)) for bit, name in FLAG_NAMES.items()}
    out["clean"] = 100.0 * float(np.mean(flags == 0))
    return out
