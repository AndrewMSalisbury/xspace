"""Possession level: what space a possession had, whether it was used, and how it ended.

One row per possession (`PhaseLabels.possession_id`), from the match's timeline and action
files plus its events. No new physics: everything here is an aggregate.

| Column | Meaning |
|---|---|
| team_side, period | team in possession |
| start_frame, end_frame, start_time_s, duration_s | span, on the tracking clock |
| start_type | `regain` (won in open play: a transition), a set-piece type, or `restart` |
| end_type | `turnover` (the other team took over in play), `dead_ball` (stoppage / period end) |
| start_third, start_x | where it began (ball x in the attacking frame) |
| n_ok | timeline rows with xSpace computed (5 Hz, open play) |
| xspace_mean, xspace_max, behind_max | timeline total / behind xSpace, xT·m² |
| xspace_max_s | seconds from the start to the frame of `xspace_max` |
| xspace_integral | Σ total / sample rate: xSpace held over time, xT·m²·s |
| n_actions, n_exploited, n_missed, xt_gained | open-play passes / crosses / carries computed |
| first_exploit_s | seconds from the start to the first exploited action (NaN if none) |
| final_third, box_entry | the ball reached the final third / the opponent's penalty area |
| shot, goal | the team shot / scored in the possession |

From the defending team's side the same rows are xSpace *conceded*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xspace.config import (
    DEFAULT_EXPLOITATION,
    DEFAULT_PARAMS,
    DEFAULT_TIMELINE,
    ExploitationConfig,
    PhysicsParams,
    TimelineConfig,
    settings_dict,
)
from xspace.constants import PITCH_LENGTH
from xspace.io.events import SETPIECE_TYPES
from xspace.io.loaders import MatchTracking
from xspace.metrics.timeline import output_metadata
from xspace.phases.possession import THIRDS, PhaseLabels

BOX_DEPTH = 16.5  # penalty area, m from the goal line
BOX_HALF_WIDTH = 20.16  # m either side of the centre line
START_TYPES = ("regain", "restart", *SETPIECE_TYPES[1:])
END_TYPES = ("turnover", "dead_ball")


def possession_spans(match: MatchTracking, phases: PhaseLabels) -> pd.DataFrame:
    """First / last frame, team, start and end type of every possession."""
    pid = phases.possession_id
    valid = np.flatnonzero(pid >= 0)
    ids, first = np.unique(pid[valid], return_index=True)
    starts = valid[first]
    # Ids are contiguous in frame order, so a possession ends where the next one starts
    # (or at its last labelled frame, if unlabelled frames follow).
    last = np.r_[valid[first[1:] - 1], valid[-1]] if len(valid) else np.zeros(0, int)

    sp = phases.setpiece[starts]
    start_type = np.where(phases.transition[starts], START_TYPES.index("regain"),
                          START_TYPES.index("restart"))
    start_type = np.where(sp > 0, START_TYPES.index("restart") + sp, start_type)

    # Turnover: the next frame is in play (no time gap, same period), labelled for the other
    # team, and its possession isn't a restart.
    nxt = np.minimum(last + 1, match.n_frames - 1)
    side = phases.possession_side
    gap = match.timestamp[nxt] - match.timestamp[last]
    in_play = ((nxt > last) & (match.period[nxt] == match.period[last])
               & (gap < 2.0 / match.frame_rate) & (side[nxt] >= 0) & (side[nxt] != side[last]))
    end_type = np.where(in_play, END_TYPES.index("turnover"), END_TYPES.index("dead_ball"))

    ball_x = np.where(side[starts] == 1, -match.ball[starts, 0], match.ball[starts, 0])
    return pd.DataFrame({
        "possession_id": ids.astype(np.int32),
        "team_side": side[starts].astype(np.int8),
        "period": match.period[starts].astype(np.int8),
        "start_frame": starts.astype(np.int32),
        "end_frame": last.astype(np.int32),
        "start_time_s": match.timestamp[starts],
        "duration_s": match.timestamp[last] - match.timestamp[starts],
        "start_type": pd.Categorical.from_codes(start_type, START_TYPES),
        "end_type": pd.Categorical.from_codes(end_type, END_TYPES),
        "start_third": pd.Categorical.from_codes(phases.third[starts], THIRDS),
        "start_x": ball_x.astype(np.float32),
    })


def ball_outcomes(match: MatchTracking, phases: PhaseLabels) -> pd.DataFrame:
    """Per possession: did the ball reach the final third / the opponent's box."""
    pid = phases.possession_id
    side = phases.possession_side
    flip = np.where(side == 1, -1.0, 1.0)[:, None]
    ball = match.ball * flip  # attacking frame
    final = ball[:, 0] > PITCH_LENGTH / 6
    box = (ball[:, 0] > PITCH_LENGTH / 2 - BOX_DEPTH) & (np.abs(ball[:, 1]) < BOX_HALF_WIDTH)
    df = pd.DataFrame({"possession_id": pid, "final_third": final, "box_entry": box})
    return df[pid >= 0].groupby("possession_id").any()


def build_possessions(match: MatchTracking, events: pd.DataFrame, phases: PhaseLabels,
                      timeline: pd.DataFrame, actions: pd.DataFrame,
                      sample_hz: float = DEFAULT_TIMELINE.sample_hz) -> pd.DataFrame:
    """One row per possession. `timeline` and `actions` are the match's Phase 2 / Phase 3
    outputs (`build_timeline`, `build_actions`); `events` must be synced (have `frame`)."""
    df = possession_spans(match, phases).set_index("possession_id")
    df = df.join(ball_outcomes(match, phases))

    tl = timeline[timeline["status"] == "ok"]
    start = df["start_time_s"].reindex(tl["possession_id"]).to_numpy()
    tl = tl.assign(since=tl["time_s"].to_numpy() - start)
    g = tl.groupby("possession_id")
    peak = tl.loc[g["total"].idxmax().dropna()].set_index("possession_id")
    df = df.join(pd.DataFrame({
        "n_ok": g.size(),
        "xspace_mean": g["total"].mean(),
        "xspace_max": g["total"].max(),
        "behind_max": g["behind"].max(),
        "xspace_max_s": peak["since"],
        "xspace_integral": g["total"].sum() / sample_hz,
    }))
    df["n_ok"] = df["n_ok"].fillna(0).astype(np.int32)

    act = actions[actions["status"] == "ok"]
    act = act.assign(since=act["time_s"].to_numpy()
                     - df["start_time_s"].reindex(act["possession_id"]).to_numpy())
    g = act.groupby("possession_id")
    df = df.join(pd.DataFrame({
        "n_actions": g.size(),
        "n_exploited": g["exploited"].sum(),
        "n_missed": g["missed"].sum(),
        "xt_gained": g["xt_gained"].sum(),
        "first_exploit_s": act[act["exploited"]].groupby("possession_id")["since"].min(),
    }))
    for col in ("n_actions", "n_exploited", "n_missed"):
        df[col] = df[col].fillna(0).astype(np.int32)

    # Shots by the team in possession, mapped to the possession at the shot's frame.
    shots = events[(events["type"] == "shot") & (events["frame"] >= 0)]
    pid = phases.possession_id[shots["frame"].to_numpy()]
    mine = phases.possession_side[shots["frame"].to_numpy()] == shots["team_side"].to_numpy()
    shots = shots[mine & (pid >= 0)].assign(possession_id=pid[mine & (pid >= 0)])
    scored = shots["success"].fillna(False).astype(bool)
    df["shot"] = df.index.isin(shots["possession_id"])
    df["goal"] = df.index.isin(shots.loc[scored.to_numpy(), "possession_id"])
    for col in ("final_third", "box_entry"):
        df[col] = df[col].fillna(False).astype(bool)
    return df.reset_index()


def possessions_settings(params: PhysicsParams = DEFAULT_PARAMS,
                         timeline: TimelineConfig = DEFAULT_TIMELINE,
                         exploitation: ExploitationConfig = DEFAULT_EXPLOITATION) -> dict:
    """Everything a possession file depends on: the timeline and action settings, plus the box."""
    return {**settings_dict(params, timeline, exploitation),
            "possessions": {"box_depth": BOX_DEPTH, "box_half_width": BOX_HALF_WIDTH}}


def possessions_metadata(source: str, match: MatchTracking, **extra) -> dict[str, str]:
    return output_metadata(source, match, possessions_settings(), **extra)
