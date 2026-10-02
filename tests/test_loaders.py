import numpy as np
from test_events import line_match

from xspace.io.loaders import recentre_periods


def test_recentre_periods_undoes_whole_pitch_offsets():
    match = line_match(10.0)
    match.period[len(match.period) // 2:] = 3
    before = (match.ball.copy(), match.home_pos.copy(), match.away_pos.copy())
    et = match.period == 3
    for arr in (match.ball, match.home_pos, match.away_pos):
        arr[et] += np.array([105.0, -68.0])  # what kloppy 3.19 leaves in PFF extra time

    recentre_periods(match)

    for fixed, original in zip((match.ball, match.home_pos, match.away_pos), before, strict=True):
        np.testing.assert_allclose(fixed, original)


def test_recentre_periods_leaves_normal_periods_alone():
    match = line_match(10.0)
    before = match.home_pos.copy()
    recentre_periods(match)
    np.testing.assert_array_equal(match.home_pos, before)
