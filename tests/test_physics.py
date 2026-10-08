import numpy as np
import pytest

from xspace.metrics.space import ZONES, defensive_shape, zone_cells
from xspace.physics.kinematics import smooth_velocities
from xspace.physics.pitch_control import make_grid, pass_reachability, pitch_control
from xspace.value.xt import xt_value

STILL = np.zeros((1, 2))


def at(x, y):
    return np.array([[x, y]], dtype=float)


def test_grid_covers_pitch():
    xs, ys, grid = make_grid(1.0)
    assert grid.shape == (105 * 68, 2)
    assert xs.min() == pytest.approx(-52.0) and ys.max() == pytest.approx(33.5)


def test_control_sums_to_one_and_favours_nearest_player():
    _, _, grid = make_grid(5.0)
    surf = pitch_control(at(-20, 0), STILL, at(20, 0), STILL, ball=np.array([0.0, 0.0]),
                         grid=grid)
    np.testing.assert_allclose(surf.attack + surf.defence, 1.0, atol=1e-6)
    left = grid[:, 0] < -30
    right = grid[:, 0] > 30
    assert surf.attack[left].mean() > 0.9
    assert surf.attack[right].mean() < 0.1


def test_running_player_controls_space_ahead():
    target = np.array([[10.0, 0.0]])
    still = pitch_control(at(0, 0), STILL, at(4, 8), STILL, np.zeros(2), target)
    running = pitch_control(at(0, 0), np.array([[6.0, 0.0]]), at(4, 8), STILL, np.zeros(2),
                            target)
    assert running.attack[0] > still.attack[0]


def test_nan_players_are_ignored():
    _, _, grid = make_grid(10.0)
    att = np.array([[-10.0, 0.0], [np.nan, np.nan]])
    surf = pitch_control(att, np.zeros((2, 2)), at(10, 0), STILL, np.zeros(2), grid)
    assert surf.attack_players.shape == (1, len(grid))
    assert list(surf.attack_mask) == [True, False]


def test_defender_in_lane_blocks_pass():
    ball = np.array([0.0, 0.0])
    targets = np.array([[30.0, 0.0], [0.0, 30.0]])
    reach = pass_reachability(ball, at(15, 0), STILL, targets)
    assert reach[0] < 0.2  # straight through the defender
    assert reach[1] > 0.8  # open lane


def test_xt_increases_towards_goal():
    pts = np.array([[-40.0, 0.0], [0.0, 0.0], [45.0, 0.0]])
    v = xt_value(pts)
    assert v[0] < v[1] < v[2]


def test_constant_velocity_recovered():
    fps, t = 25.0, 100
    x = np.arange(t) / fps * 4.0  # 4 m/s along x
    pos = np.stack([x, np.zeros(t)], axis=1)[:, None, :]
    vel = smooth_velocities(pos, fps, np.ones(t, dtype=int))
    np.testing.assert_allclose(vel[10:-10, 0, 0], 4.0, atol=1e-6)


def test_zones_follow_defensive_lines():
    # GK, back four at x=30, midfield four at x=15, two forwards at x=0 (attack goes +x)
    defs = np.array(
        [[50, 0]] + [[30, y] for y in (-20, -7, 7, 20)] + [[15, y] for y in (-15, -5, 5, 15)]
        + [[0, -5], [0, 5]], dtype=float,
    )
    shape = defensive_shape(defs, gk_index=0, ball_x=-10.0)
    assert shape.offside_line == pytest.approx(30)
    assert shape.mid_line == pytest.approx(15)
    cells = np.array([[40, 0], [22, 0], [22, 30], [-10, 0]], dtype=float)
    assert [ZONES[z] for z in zone_cells(cells, shape)] == ["behind", "between", "wide",
                                                            "in_front"]


def test_lone_deep_defender_is_not_the_back_line():
    # A centre-back tracks a runner to x=48 while the rest of the back line holds at x=34
    defs = np.array(
        [[50, 0], [48, 23]] + [[34, y] for y in (-3, 7, 30)] + [[33, 20]]
        + [[25, 14], [20, 19], [19, 14], [13, -14], [12, 18]], dtype=float,
    )
    shape = defensive_shape(defs, gk_index=0, ball_x=10.0)
    assert shape.offside_line == pytest.approx(48)
    assert 33 <= shape.back_line <= 35
    assert shape.mid_line < 26


def test_offside_attackers_get_no_control():
    from dataclasses import replace

    from xspace.config import DEFAULT_PARAMS
    from xspace.metrics.space import offside_attackers, space_from_arrays

    _, _, grid = make_grid(2.0)
    # Defenders: back line at x = 20, keeper at 50 → offside line 20 (ball at 0).
    dp = np.array([[50.0, 0.0], [20.0, -10.0], [20.0, 10.0], [10.0, 0.0]])
    # Attackers: carrier at the ball, one level (within the margin), one clearly offside.
    ap = np.array([[0.0, 0.0], [20.3, 20.0], [32.0, 0.0]])
    zeros_a, zeros_d = np.zeros_like(ap), np.zeros_like(dp)
    ball = np.array([0.0, 0.0])

    fs = space_from_arrays(ap, zeros_a, dp, zeros_d, ball, 0, 0, grid)
    assert fs.offside.tolist() == [False, False, True]
    off = replace(DEFAULT_PARAMS, offside_margin=float("inf"))
    no_rule = space_from_arrays(ap, zeros_a, dp, zeros_d, ball, 0, 0, grid, off)
    assert not no_rule.offside.any()

    near = np.argmin(np.linalg.norm(grid - [32.0, 0.0], axis=1))
    assert no_rule.control.attack[near] > 0.8  # he'd own the space around him ...
    assert fs.control.attack[near] < 0.2  # ... but he's offside, so the defence does
    assert fs.totals["behind"] < no_rule.totals["behind"]
    # Offside players never own space: their rows are gone from the per-player shares.
    assert fs.control.attack_players.shape[0] == 2

    # The player on the ball is never offside, even a step ahead of it at the line.
    carrier = np.array([[21.0, 0.0], [5.0, 5.0]])
    assert not offside_attackers(carrier, np.array([20.0, 0.0]), 20.0, 0.5).any()
