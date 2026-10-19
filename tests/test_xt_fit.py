import numpy as np
import pandas as pd

from xspace.io.statsbomb import reduce_events, to_metres
from xspace.value.xt_fit import XTConfig, fit_xt, possession_targets, zone_centres, zones


def test_zones_and_centres():
    assert list(zones(np.array([-52.5, 52.5, 0.1, np.nan]), np.array([-34, 34, 0.1, 0]),
                      4, 2)) == [0, 7, 6, -1]
    c = zone_centres(4, 2)
    assert list(zones(c[:, 0], c[:, 1], 4, 2)) == list(range(8))


def ev(kind, x, y, end_x=np.nan, end_y=np.nan, success=True, xg=np.nan, possession=1,
       index=0):
    return dict(match_id=1, index=index, possession=possession, team_id=1, kind=kind, x=x, y=y,
                end_x=end_x, end_y=end_y, success=success, xg=xg)


def test_fit_on_a_two_zone_pitch():
    """Own half always passes into the opponent's half; there, half the actions are shots
    (xG 0.4) and half are lost. xT(attack) = 0.5 · 0.4 = 0.2; xT(own) = 1 · xT(attack)."""
    rows = [ev("move", -20, 0, 20, 0) for _ in range(10)]
    rows += [ev("shot", 40, 0, xg=0.4, success=False) for _ in range(5)]
    rows += [ev("loss", 30, 0, success=False) for _ in range(5)]
    model = fit_xt(pd.DataFrame(rows), XTConfig(nx=2, ny=1, shot_prior=0.0, chain="singh"))
    np.testing.assert_allclose(model.grid, [[0.2, 0.2]], atol=1e-6)
    np.testing.assert_allclose(model.shot_share, [[0.0, 0.5]])
    # A failed pass from the own half halves its value.
    rows += [ev("move", -20, 0, 20, 0, success=False) for _ in range(10)]
    model = fit_xt(pd.DataFrame(rows), XTConfig(nx=2, ny=1, shot_prior=0.0, chain="singh"))
    np.testing.assert_allclose(model.grid, [[0.1, 0.2]], atol=1e-6)


def test_possession_chain():
    """Two possessions: own half -> attack -> shot (xG 0.4) -> rebound in attack -> lost; and
    own half -> a failed pass the same team gets back in attack -> lost. In the attacking zone
    1 of 3 actions is a shot and 1 of 3 is followed by another action there."""
    rows = [
        ev("move", -20, 0, 20, 0, possession=1, index=0),
        ev("shot", 40, 0, xg=0.4, success=False, possession=1, index=1),
        ev("loss", 30, 0, success=False, possession=1, index=2),
        ev("move", -20, 0, 20, 0, success=False, possession=2, index=3),
        ev("loss", 30, 0, success=False, possession=2, index=4),
    ]
    model = fit_xt(pd.DataFrame(rows), XTConfig(nx=2, ny=1, shot_prior=0.0))
    # attack: xT = 0.4 / 3 + xT / 3, so 0.2; the own half always leads to the attack: 0.2.
    np.testing.assert_allclose(model.grid, [[0.2, 0.2]], atol=1e-6)


def test_possession_targets():
    df = pd.DataFrame([
        ev("move", 0, 0, index=0), ev("shot", 40, 0, xg=0.1, index=1),
        ev("move", 30, 0, index=2), ev("shot", 45, 0, xg=0.3, index=3),
        ev("move", 0, 0, index=4, possession=2),
    ])
    np.testing.assert_allclose(possession_targets(df).to_numpy(), [0.4, 0.4, 0.3, 0.3, 0.0])


def test_statsbomb_reduce():
    assert to_metres([60, 40]) == (0.0, 0.0)
    assert to_metres([120, 0]) == (52.5, 34.0)  # StatsBomb y points down
    base = {"period": 1, "minute": 1, "second": 0, "possession": 3, "team": {"id": 9}}
    raw = [
        {**base, "index": 1, "type": {"name": "Pass"}, "location": [60, 40],
         "pass": {"end_location": [90, 40]}},
        {**base, "index": 2, "type": {"name": "Pass"}, "location": [120, 0],
         "pass": {"end_location": [110, 40], "type": {"name": "Corner"}}},
        {**base, "index": 3, "type": {"name": "Pass"}, "location": [90, 40],
         "pass": {"end_location": [100, 40], "outcome": {"name": "Incomplete"}}},
        {**base, "index": 4, "type": {"name": "Shot"}, "location": [108, 40],
         "shot": {"type": {"name": "Open Play"}, "outcome": {"name": "Goal"},
                  "statsbomb_xg": 0.3}},
        {**base, "index": 5, "type": {"name": "Shot"}, "location": [108, 40],
         "shot": {"type": {"name": "Penalty"}, "outcome": {"name": "Goal"}}},
        {**base, "index": 6, "type": {"name": "Dispossessed"}, "location": [80, 30]},
        {**base, "index": 7, "type": {"name": "Pressure"}, "location": [80, 30]},
    ]
    df = reduce_events(raw, 5)
    assert list(df["index"]) == [1, 3, 4, 6]
    assert list(df["kind"].astype(str)) == ["move", "move", "shot", "loss"]
    assert list(df["success"]) == [True, False, True, False]
    np.testing.assert_allclose(df["xg"].iloc[2], 0.3)
