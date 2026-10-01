import numpy as np
from test_events import line_match

from xspace.phases.quality import (
    BALL_JUMP,
    BALL_MISSING,
    FEW_PLAYERS,
    PLAYER_JUMP,
    flag_summary,
    quality_flags,
)


def test_quality_flags():
    match = line_match(4.0)
    assert (quality_flags(match) == 0).all()

    match.ball[10] = np.nan
    match.home_pos[20:30, :2] = np.nan  # home down to 9 players
    match.ball[40, 0] += 5.0  # 5 m in one 0.04 s frame = 125 m/s
    match.away_pos[60, 3, 1] += 2.0  # 50 m/s
    flags = quality_flags(match)

    assert flags[10] & BALL_MISSING
    assert (flags[20:30] & FEW_PLAYERS).all() and not (flags[30] & FEW_PLAYERS)
    assert flags[40] & BALL_JUMP and flags[60] & PLAYER_JUMP
    assert flag_summary(flags)["clean"] < 100
