from dataclasses import replace

import numpy as np
import pandas as pd
from test_events import line_match, passes_at

from xspace.io.events import SETPIECE_TYPES
from xspace.io.sync import attach_frames
from xspace.phases.possession import THIRDS, label_phases


def stopped_match():
    """40 s of play with the ball dead (frames dropped) from 20 s to 25 s."""
    match = line_match(40.0)
    keep = (match.timestamp < 20.0) | (match.timestamp >= 25.0)
    arrays = {k: v[keep] for k, v in vars(match).items()
              if isinstance(v, np.ndarray) and len(v) == len(keep)}
    return replace(match, **arrays)


def scripted_events():
    ev = passes_at(np.array([0.0, 5.0, 12.0, 25.5, 35.0]), [1, 1, 2, 3, 4])
    ev["team_side"] = pd.array([0, 0, 1, 1, 0], dtype="int64")
    ev["setpiece"] = pd.array(["kick_off", "open_play", "open_play", "throw_in", "open_play"],
                              dtype="string")
    return ev


def at(match, labels, t, name):
    frame = int(np.argmin(np.abs(match.timestamp - t)))
    return getattr(labels, name)[frame]


def test_possessions_restarts_setpieces_and_transitions():
    match = stopped_match()
    events = attach_frames(scripted_events(), match)
    assert (events["frame"] >= 0).all()  # the 25.5 s throw-in is live; nothing to snap
    ph = label_phases(match, events)

    assert [at(match, ph, t, "possession_side") for t in (3, 15, 25.0, 30, 36)] == [0, 1, 1, 1, 0]
    # Four possessions: kick-off, turnover at 12 s, restart after the stoppage, turnover at 35 s.
    ids = ph.possession_id
    assert len(np.unique(ids)) == 4 and (np.diff(ids) >= 0).all()

    sp = lambda t: SETPIECE_TYPES[at(match, ph, t, "setpiece")]  # noqa: E731
    assert (sp(1), sp(5), sp(27), sp(31)) == ("kick_off", "open_play", "throw_in", "open_play")

    tr = lambda t: bool(at(match, ph, t, "transition"))  # noqa: E731
    assert (tr(3), tr(15), tr(19.9), tr(27), tr(36)) == (False, True, True, False, True)

    th = lambda t: THIRDS[at(match, ph, t, "third")]  # noqa: E731
    # Ball x = t; home attacks +x, away attacks -x.
    assert (th(3), th(15), th(30), th(36)) == ("middle", "middle", "defensive", "final")


def test_restart_event_before_live_ball_is_snapped_forward():
    match = stopped_match()
    ev = scripted_events()
    ev.loc[3, "time_s"] = 23.8  # stamped 1.2 s before the first live frame (25.0 s)
    events = attach_frames(ev, match)
    assert match.timestamp[events.loc[3, "frame"]] == 25.0
    open_play = ev.copy()
    open_play.loc[3, "setpiece"] = "open_play"
    assert attach_frames(open_play, match).loc[3, "frame"] == -1  # only restarts are snapped
