import numpy as np
import pandas as pd
from test_exploitation import events_for
from test_timeline import FPS, random_match

from xspace.metrics.possessions import build_possessions, possession_spans
from xspace.phases.possession import label_phases


def scripted():
    """Home has the ball until an away recovery at frame 92; away then shoots and scores with
    the ball in the home box (away attacks -x)."""
    match = random_match()
    match.ball[160:, 0] = -45.0
    ev = events_for([
        dict(time_s=0.4, frame=10, player_id="h1", success=True),
        dict(time_s=2.0, frame=50, player_id="h2", success=False),
        dict(time_s=3.68, frame=92, player_id="a4", team_side=1, type="recovery"),
        dict(time_s=4.4, frame=110, player_id="a2", team_side=1, success=True),
        dict(time_s=6.8, frame=170, player_id="a9", team_side=1, type="shot", success=True),
    ])
    return match, ev, label_phases(match, ev)


def timeline_for(match, phases):
    frames = np.arange(0, match.n_frames, 5)
    total = np.linspace(0.5, 2.0, len(frames))
    total[frames == 40] = 9.0  # home's peak, 1.6 s in
    return pd.DataFrame({
        "frame": frames, "time_s": frames / FPS,
        "status": np.where(frames == 60, "set_piece", "ok"),
        "possession_id": phases.possession_id[frames],
        "total": total, "behind": total / 4,
    })


def actions_for():
    return pd.DataFrame({
        "time_s": [0.4, 2.0, 4.4, 5.2], "possession_id": [0, 0, 1, 1],
        "status": ["ok", "ok", "ok", "no_frame"],
        "exploited": [False, False, True, True], "missed": [False, True, False, False],
        "xt_gained": [0.01, -0.02, 0.05, np.nan],
    })


def test_spans_and_types():
    match, _, phases = scripted()
    spans = possession_spans(match, phases)
    assert spans["possession_id"].tolist() == [0, 1]
    assert spans["team_side"].tolist() == [0, 1]
    assert spans["start_frame"].tolist() == [0, 92]
    assert spans["end_frame"].tolist() == [91, match.n_frames - 1]
    assert spans["start_type"].tolist() == ["restart", "regain"]
    assert spans["end_type"].tolist() == ["turnover", "dead_ball"]
    np.testing.assert_allclose(spans["duration_s"], [91 / FPS, (match.n_frames - 93) / FPS])


def test_aggregates_and_outcomes():
    match, ev, phases = scripted()
    tl = timeline_for(match, phases)
    df = build_possessions(match, ev, phases, tl, actions_for())
    home, away = df.iloc[0], df.iloc[1]

    ok = tl[(tl["status"] == "ok") & (tl["possession_id"] == 0)]
    assert home["n_ok"] == len(ok)
    assert home["xspace_max"] == 9.0
    np.testing.assert_allclose(home["xspace_max_s"], 1.6)
    np.testing.assert_allclose(home["xspace_mean"], ok["total"].mean())
    np.testing.assert_allclose(home["xspace_integral"], ok["total"].sum() / 5.0)
    np.testing.assert_allclose(home["behind_max"], 9.0 / 4)

    assert (home["n_actions"], home["n_exploited"], home["n_missed"]) == (2, 0, 1)
    assert np.isnan(home["first_exploit_s"])
    np.testing.assert_allclose(home["xt_gained"], -0.01)
    assert away["n_actions"] == 1  # the no_frame row doesn't count
    np.testing.assert_allclose(away["first_exploit_s"], 4.4 - 92 / FPS)

    assert not home["shot"] and not home["box_entry"] and not home["final_third"]
    assert away["shot"] and away["goal"] and away["box_entry"] and away["final_third"]


def test_possession_without_timeline_rows():
    match, ev, phases = scripted()
    tl = timeline_for(match, phases)
    df = build_possessions(match, ev, phases, tl[tl["possession_id"] == 0],
                           actions_for().iloc[:2])
    away = df.iloc[1]
    assert away["n_ok"] == 0 and np.isnan(away["xspace_max"]) and away["n_actions"] == 0
