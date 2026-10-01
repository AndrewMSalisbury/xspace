"""Per-frame phase-of-play labels: possession, set-piece windows, transitions, pitch thirds.

Possession comes from synchronised events, not the tracking `ball_owner` field: PFF's is derived
from events anyway, and IDSSE's (DFL) flickers during duels. Rules:

- The team in possession is the team of the most recent *controlling* event (pass, cross, shot,
  carry, reception, recovery). Challenges, clearances, rebounds and touches don't change it.
- A new possession starts when the team changes or play restarts after a stoppage (a gap in the
  ball-in-play frames).
- A set-piece window runs from a restart event for `SETPIECE_WINDOW_S[type]` seconds, ending
  early if possession changes. Inside it the defence is not in its open-play shape.
- A transition is the first `TRANSITION_S` seconds of a possession won in open play.
- Thirds use ball x in the attacking direction of the team in possession.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xspace.constants import PITCH_LENGTH
from xspace.io.events import SETPIECE_TYPES
from xspace.io.loaders import MatchTracking

CONTROL_TYPES = ("pass", "cross", "shot", "carry", "reception", "recovery")
# Seconds after a restart during which the defence is treated as not settled. Starting values;
# to be tuned by inspection (see docs/PLAN.md Phase 1).
SETPIECE_WINDOW_S = {
    "kick_off": 4.0, "throw_in": 4.0, "goal_kick": 4.0, "drop_ball": 2.0,
    "free_kick": 8.0, "corner": 8.0, "penalty": 8.0,
}
TRANSITION_S = 10.0
STOPPAGE_GAP_S = 0.5  # a jump in tracking time this large means the ball was dead
RESTART_FILL_S = 3.0  # max delay from the first live frame to the restart event
THIRDS = ("defensive", "middle", "final")


@dataclass
class PhaseLabels:
    """Per-frame labels, aligned with `MatchTracking` rows."""

    possession_side: np.ndarray  # (T,) 0 home, 1 away, -1 unknown (before first event)
    possession_id: np.ndarray  # (T,) increasing integer, -1 unknown
    setpiece: np.ndarray  # (T,) index into SETPIECE_TYPES; 0 = open play
    transition: np.ndarray  # (T,) bool
    third: np.ndarray  # (T,) index into THIRDS, -1 unknown

    @property
    def open_play(self) -> np.ndarray:
        return self.setpiece == SETPIECE_TYPES.index("open_play")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "possession_side": self.possession_side,
            "possession_id": self.possession_id,
            "setpiece": pd.Categorical.from_codes(self.setpiece, SETPIECE_TYPES),
            "transition": self.transition,
            "third": pd.Categorical.from_codes(self.third, THIRDS),
        })


def _stoppage_starts(match: MatchTracking) -> np.ndarray:
    """(T,) bool: first frame of each period and first frame after a dead-ball gap."""
    start = np.zeros(match.n_frames, dtype=bool)
    if match.n_frames == 0:
        return start
    start[0] = True
    start[1:] = (match.period[1:] != match.period[:-1]) | (
        np.diff(match.timestamp) > STOPPAGE_GAP_S)
    return start


def _window_end(match: MatchTracking, start: int, seconds: float) -> int:
    """First frame index at or after `start` + `seconds`, never past the end of its period
    (timestamps restart each period, so the search must stay inside it)."""
    period_end = start + np.searchsorted(match.period[start:] != match.period[start], True)
    t = match.timestamp[start:period_end]
    return start + int(np.searchsorted(t, t[0] + seconds))


def label_phases(match: MatchTracking, events: pd.DataFrame) -> PhaseLabels:
    """Label every tracking frame. `events` must come from `io.sync.attach_frames`."""
    n = match.n_frames
    ev = events[events["frame"] >= 0].sort_values("frame", kind="stable")

    # Possession side: forward-fill the team of controlling events.
    side = np.full(n, -1, dtype=np.int64)
    ctrl = ev[ev["type"].isin(CONTROL_TYPES) & (ev["team_side"] >= 0)]
    marks = np.full(n, -1, dtype=np.int64)
    marks[ctrl["frame"].to_numpy()] = ctrl["team_side"].to_numpy()
    has = marks >= 0
    last = np.maximum.accumulate(np.where(has, np.arange(n), -1))
    side[last >= 0] = marks[last[last >= 0]]
    # Frames before the first event of a period inherit nothing from the previous period.
    period_start = np.r_[True, match.period[1:] != match.period[:-1]] if n else np.zeros(0, bool)
    last_start = np.maximum.accumulate(np.where(period_start, np.arange(n), 0))
    side[(last >= 0) & (last < last_start)] = -1

    # After a stoppage the restarting team has the ball from the first live frame, not only
    # from its restart event: back-fill from the first controlling event within RESTART_FILL_S.
    restarts = _stoppage_starts(match)
    t = match.timestamp
    ctrl_frames = np.flatnonzero(has)
    for r in np.flatnonzero(restarts):
        k = np.searchsorted(ctrl_frames, r)
        if k == len(ctrl_frames):
            continue
        f = ctrl_frames[k]
        if match.period[f] == match.period[r] and t[f] - t[r] <= RESTART_FILL_S:
            side[r:f] = marks[f]

    # Possession ids: new id on a change of team or a restart.
    new_poss = restarts.copy()
    new_poss[1:] |= side[1:] != side[:-1]
    possession_id = np.cumsum(new_poss) - 1
    possession_id[side < 0] = -1

    # Set-piece windows.
    setpiece = np.zeros(n, dtype=np.int64)
    sp_events = ev[ev["setpiece"] != "open_play"]
    for frame, sp in zip(sp_events["frame"].to_numpy(), sp_events["setpiece"], strict=True):
        window = np.arange(frame, _window_end(match, frame, SETPIECE_WINDOW_S[sp]))
        poss = possession_id[frame]
        if poss >= 0:
            window = window[possession_id[window] == poss]
        setpiece[window] = SETPIECE_TYPES.index(sp)

    # Transitions: start of a possession won in open play (not at a restart / set piece).
    transition = np.zeros(n, dtype=bool)
    starts = np.flatnonzero(new_poss & (side >= 0))
    for s in starts:
        if restarts[s] or setpiece[s] != 0:
            continue
        window = np.arange(s, _window_end(match, s, TRANSITION_S))
        transition[window[possession_id[window] == possession_id[s]]] = True

    # Thirds, in the attacking direction of the team in possession.
    x_att = np.where(side == 1, -match.ball[:, 0], match.ball[:, 0])
    edges = np.array([-PITCH_LENGTH / 6, PITCH_LENGTH / 6])
    third = np.searchsorted(edges, x_att).astype(np.int64)
    third[(side < 0) | np.isnan(x_att)] = -1

    return PhaseLabels(side, possession_id, setpiece, transition, third)
