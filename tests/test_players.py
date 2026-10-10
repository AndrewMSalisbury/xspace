import numpy as np
import pandas as pd
from test_timeline import FPS, random_match

from xspace.metrics.players import build_players, minutes_played


def actions_for():
    """h1 plays twice, h2 once. h3 owns the big best cell twice: found once, ignored once."""
    return pd.DataFrame({
        "status": ["ok", "ok", "ok", "no_frame"],
        "team_side": np.array([0, 0, 0, 0], dtype=np.int8),
        "player_id": pd.array(["h1", "h1", "h2", "h1"], dtype="string"),
        "success": pd.array([True, False, True, True], dtype="boolean"),
        "exploited": [True, False, False, True], "missed": [False, True, False, False],
        "xt_gained": [0.05, -0.02, 0.01, 0.3], "decision_gap": [0.0, 0.04, 0.001, 0.0],
        "chosen": [0.03, 0.0, 0.002, 0.5], "best": [0.03, 0.05, 0.003, 0.5],
        "owner_id": pd.array(["h3", "h4", "h1", "h3"], dtype="string"),
        "best_owner_id": pd.array(["h3", "h3", "h3", "h3"], dtype="string"),
    })


def test_minutes():
    match = random_match()
    match.home_pos[100:, 5] = np.nan  # subbed off halfway
    m = minutes_played(match).set_index("player_id")["minutes"]
    np.testing.assert_allclose(m["h0"], match.n_frames / FPS / 60)
    np.testing.assert_allclose(m["h5"], 100 / FPS / 60)
    assert len(m) == 22


def test_credit():
    match = random_match()
    match.away_pos[:, 7] = np.nan  # never played
    df = build_players(match, actions_for()).set_index("player_id")
    assert len(df) == 21 and "a7" not in df.index

    h1, h2, h3 = df.loc["h1"], df.loc["h2"], df.loc["h3"]
    assert (h1["n_actions"], h1["n_exploited"], h1["n_missed"]) == (2, 1, 1)
    np.testing.assert_allclose(h1["xt_gained"], 0.03)
    np.testing.assert_allclose(h1["decision_gap"], 0.02)
    # Completed into space h1 owned (h2's pass); the failed pass gives h4 nothing.
    assert h1["n_received_space"] == 1 and df.loc["h4", "n_received_space"] == 0
    # h3: the best cell twice above MIN_BEST (h2's best is below it); found once, ignored once.
    assert (h3["n_best_space"], h3["n_best_found"], h3["n_best_ignored"]) == (2, 1, 1)
    np.testing.assert_allclose(h3["xspace_received"], 0.03)
    assert h2["n_actions"] == 1 and h2["n_best_space"] == 0
