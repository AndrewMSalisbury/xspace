from dataclasses import replace

import numpy as np
import pytest

from xspace.config import DEFAULT_PARAMS
from xspace.physics.pitch_control import pass_reachability, pitch_control
from xspace.validation import calibrate as cal
from xspace.validation import pass_model as pm
from xspace.validation.passes import MAX_PLAYERS, TRAJECTORIES, PassSet

P = MAX_PLAYERS


def random_passes(n: int, seed: int = 0, n_players: int = 11) -> PassSet:
    """Random but plausible frames: 11 v 11 on the pitch, ball near an attacker."""
    rng = np.random.default_rng(seed)

    def team():
        pos = np.full((n, P, 2), np.nan)
        vel = np.full((n, P, 2), np.nan)
        pos[:, :n_players] = rng.uniform([-50, -33], [50, 33], (n, n_players, 2))
        vel[:, :n_players] = rng.normal(0, 2.5, (n, n_players, 2))
        return pos, vel

    ap, av = team()
    dp, dv = team()
    ball = ap[:, 0] + rng.normal(0, 0.5, (n, 2))
    end = rng.uniform([-50, -33], [50, 33], (n, 2))
    offside = np.zeros((n, P), dtype=bool)
    offside[::5, 3] = True
    return PassSet(
        source=np.full(n, "pff"), match_id=np.array([str(i % 8) for i in range(n)]),
        event_id=np.array([str(i) for i in range(n)]), type=np.full(n, "pass"),
        frame=np.arange(n), success=(rng.random(n) < 0.8).astype(float),
        trajectory=(np.arange(n) % 3).astype(np.int8), height=np.full(n, ""),
        high_point=np.full(n, ""), ball=ball, end=end, end_source=np.full(n, "end"),
        att_pos=ap, att_vel=av, def_pos=dp, def_vel=dv, offside=offside,
        def_gk=np.where(np.arange(n) % 2 == 0, 0, -1).astype(np.int16),
        passer=np.zeros(n, np.int16), target=np.full(n, 1, np.int16),
        receiver=np.full(n, 1, np.int16),
    )


@pytest.mark.parametrize("lane_combine", ["product", "max"])
def test_batched_model_matches_reference_physics(lane_combine):
    params = replace(DEFAULT_PARAMS, lane_combine=lane_combine)
    ps = random_passes(30)
    pred = pm.predict(ps, params)
    for i in range(len(ps)):
        att = np.where(ps.offside[i][:, None], np.nan, ps.att_pos[i])
        gk = int(ps.def_gk[i]) if ps.def_gk[i] >= 0 else None
        surf = pitch_control(att, ps.att_vel[i], ps.def_pos[i], ps.def_vel[i], ps.ball[i],
                             ps.end[i][None], def_gk=gk, params=params)
        reach = pass_reachability(ps.ball[i], ps.def_pos[i], ps.def_vel[i], ps.end[i][None],
                                  params)
        assert pred.control[i] == pytest.approx(surf.attack[0], abs=1e-5)
        assert pred.reach[i] == pytest.approx(reach[0], abs=1e-5)


def test_lane_max_never_lowers_reach():
    ps = random_passes(50, seed=1)
    prod = pm.predict(ps, DEFAULT_PARAMS).reach
    mx = pm.predict(ps, replace(DEFAULT_PARAMS, lane_combine="max")).reach
    assert np.all(mx >= prod - 1e-12)
    assert (mx > prod + 1e-3).any()


def test_one_defender_beside_lane_counts_once():
    """A defender who can reach the middle of the lane: with "max" their chance is taken once,
    so reach is 1 - p at their best point, not the product over every lane sample."""
    ball = np.array([0.0, 0.0])
    target = np.array([[30.0, 0.0]])
    defender = np.array([[15.0, 6.0]])
    params = replace(DEFAULT_PARAMS, lane_combine="max")
    r_max = pass_reachability(ball, defender, np.zeros((1, 2)), target, params)[0]
    r_prod = pass_reachability(ball, defender, np.zeros((1, 2)), target)[0]
    assert r_prod < r_max < 1.0


def test_air_trajectory_is_not_intercepted_but_takes_longer():
    ps = random_passes(20, seed=2)
    air = pm.Trajectory("air", air_speed=15.0, air_time=1.0)
    pred = pm.predict(ps, DEFAULT_PARAMS, air)
    ground = pm.predict(ps, DEFAULT_PARAMS)
    np.testing.assert_array_equal(pred.reach, 1.0)
    assert np.all(pred.flight > ground.flight)


def test_intent_target_follows_receiver():
    ps = random_passes(5, seed=3)
    pred = pm.predict(ps, DEFAULT_PARAMS, target="intent")
    # Receivers faster than 90% of the ball are capped; the fixture's are all slower.
    assert (np.linalg.norm(ps.att_vel[:, 1], axis=1) < 0.9 * DEFAULT_PARAMS.ball_speed).all()
    expected = ps.att_pos[:, 1] + ps.att_vel[:, 1] * pred.flight[:, None]
    np.testing.assert_allclose(pred.target, expected, atol=1e-6)
    dist = np.linalg.norm(pred.target - ps.ball, axis=1)
    np.testing.assert_allclose(pred.flight, dist / DEFAULT_PARAMS.ball_speed, atol=1e-6)


def test_scores():
    y = np.array([0, 0, 1, 1])
    assert pm.auc(y, np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert pm.auc(y, np.array([0.4, 0.3, 0.2, 0.1])) == 0.0
    assert pm.auc(y, np.full(4, 0.5)) == 0.5
    assert pm.brier(y, y.astype(float)) == 0.0
    assert pm.log_loss(y, np.full(4, 0.5)) == pytest.approx(np.log(2))


def test_best_of_is_max_of_trajectories():
    ps = random_passes(30, seed=4)
    model = cal.PassModel()
    best = cal.predict(ps, model, use_trajectory=False)
    g = pm.predict(ps, model.params, pm.GROUND, "intent").p
    a = pm.predict(ps, model.params, model.air, "intent").p
    np.testing.assert_allclose(best, np.maximum(g, a))
    known = cal.predict(ps, model)
    is_air = ps.trajectory == TRAJECTORIES.index("air")
    np.testing.assert_allclose(known[is_air], a[is_air])


def test_split_is_by_match():
    ps = random_passes(80)
    train, test = cal.split_matches(ps, test_every=4)
    assert not (train & test).any() and (train | test).all()
    assert not set(ps.match_id[train]) & set(ps.match_id[test])
    assert set(ps.match_id[test]) == {"3", "7"}


def test_model_vector_round_trip():
    model = cal.PassModel()
    again = model.with_vector(model.vector() * 1.5)
    assert again.params.max_speed == pytest.approx(DEFAULT_PARAMS.max_speed * 1.5)
    assert cal.model_from_dict(again.as_dict()) == again


def test_pass_set_round_trip(tmp_path):
    ps = random_passes(10)
    ps.save(tmp_path / "p.npz")
    back = PassSet.load(tmp_path / "p.npz")
    np.testing.assert_array_equal(back.att_pos, ps.att_pos)
    assert list(back.event_id) == list(ps.event_id)
