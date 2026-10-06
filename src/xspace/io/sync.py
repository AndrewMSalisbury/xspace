"""Align events with tracking frames and check that the alignment is trustworthy.

`estimate_offsets` finds a per-period clock offset (the time shift that brings the tracking ball
closest to the acting player at on-ball events); `attach_frames` maps each event to the nearest
tracking frame after applying it; `sync_report` measures the remaining alignment error.
`synchronise` runs all three.

PFF frames are tagged with their event ids, so PFF offsets come out as 0. PFF's smoothed ball
(what kloppy loads) is pulled onto the player at tagged events, which makes the actor-ball
distance near zero for PFF by construction. IDSSE (DFL) events lag tracking by ~1 s.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from xspace.io.loaders import MatchTracking

# Event types where the acting player is on the ball at the event instant.
ON_BALL_TYPES = ("pass", "cross", "shot", "carry", "reception", "touch", "clearance")


def _nearest_frames(match: MatchTracking, periods: np.ndarray, times: np.ndarray,
                    tolerance_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest tracking frame (row index) in the same period; -1 if further than tolerance."""
    frame = np.full(len(times), -1, dtype=np.int64)
    dt = np.full(len(times), np.nan)
    for p in np.unique(periods):
        idx = np.flatnonzero(match.period == p)
        if len(idx) == 0:
            continue
        ts = match.timestamp[idx]
        rows = np.flatnonzero(periods == p)
        t = times[rows]
        right = np.clip(np.searchsorted(ts, t), 0, len(ts) - 1)
        left = np.clip(right - 1, 0, len(ts) - 1)
        k = np.where(np.abs(ts[left] - t) < np.abs(ts[right] - t), left, right)
        d = ts[k] - t
        ok = np.abs(d) <= tolerance_s
        frame[rows[ok]] = idx[k[ok]]
        dt[rows] = d
    return frame, dt


def attach_frames(events: pd.DataFrame, match: MatchTracking,
                  offsets: dict[int, float] | None = None,
                  tolerance_s: float = 0.1, restart_snap_s: float = 3.0) -> pd.DataFrame:
    """Return a copy of `events` with `frame` (row index into `match` arrays), `sync_offset_s`
    (per-period offset added to `time_s`) and `sync_dt_s` (frame time − corrected event time).

    `frame` is -1 when no tracking frame lies within `tolerance_s` (e.g. the event happened
    while the ball was dead and those frames were dropped). Set-piece restarts up to
    `restart_snap_s` before the next live frame are snapped forward to it.
    """
    out = events.copy()
    periods = out["period"].to_numpy()
    offset = np.array([(offsets or {}).get(int(p), 0.0) for p in periods], dtype=float)
    times = out["time_s"].to_numpy(float) + offset
    frame, dt = _nearest_frames(match, periods, times, tolerance_s)

    # Restarts are often stamped just before the ball is live (dead frames are dropped):
    # snap them forward to the first live frame of the same period.
    restart = (out["setpiece"] != "open_play").to_numpy(dtype=bool) & (frame < 0)
    for i in np.flatnonzero(restart):
        idx = np.flatnonzero(match.period == periods[i])
        k = np.searchsorted(match.timestamp[idx], times[i])
        if k < len(idx) and match.timestamp[idx[k]] - times[i] <= restart_snap_s:
            frame[i] = idx[k]
            dt[i] = match.timestamp[idx[k]] - times[i]
    out["frame"] = frame
    out["sync_offset_s"] = offset
    out["sync_dt_s"] = dt
    return out


@dataclass
class _OnBall:
    periods: np.ndarray
    times: np.ndarray
    sides: np.ndarray
    slots: np.ndarray
    start_xy: np.ndarray
    frames: np.ndarray  # from `attach_frames`; -1 if not attached


def _on_ball(events: pd.DataFrame, match: MatchTracking) -> _OnBall:
    """On-ball events whose actor is on a roster, with the actor's (side, slot)."""
    index = {}
    for side, team in ((0, match.home), (1, match.away)):
        for j, pid in enumerate(team.player_ids):
            index[str(pid)] = (side, j)
    ev = events[events["type"].isin(ON_BALL_TYPES) & events["player_id"].notna()]
    ev = ev[ev["player_id"].map(lambda pid: pid in index).to_numpy(dtype=bool)]
    side_slot = np.array([index[pid] for pid in ev["player_id"]], dtype=np.int64).reshape(-1, 2)
    return _OnBall(
        periods=ev["period"].to_numpy(),
        times=ev["time_s"].to_numpy(float),
        sides=side_slot[:, 0],
        slots=side_slot[:, 1],
        start_xy=ev[["start_x", "start_y"]].to_numpy(float),
        frames=ev["frame"].to_numpy() if "frame" in ev else np.full(len(ev), -1),
    )


def _actor_ball_distance(match: MatchTracking, frames: np.ndarray, sides: np.ndarray,
                         slots: np.ndarray) -> np.ndarray:
    pos = np.where(
        (sides == 0)[:, None],
        match.home_pos[frames, np.where(sides == 0, slots, 0)],
        match.away_pos[frames, np.where(sides == 1, slots, 0)],
    )
    return np.linalg.norm(pos - match.ball[frames], axis=1)


def _median_distance_at(match: MatchTracking, ob: _OnBall, mask: np.ndarray, shift_s: float,
                        tolerance_s: float) -> tuple[float, int]:
    frame, _ = _nearest_frames(match, ob.periods[mask], ob.times[mask] + shift_s, tolerance_s)
    ok = frame >= 0
    if not ok.any():
        return np.inf, 0
    d = _actor_ball_distance(match, frame[ok], ob.sides[mask][ok], ob.slots[mask][ok])
    return float(np.nanmedian(d)), int(ok.sum())


def estimate_offsets(events: pd.DataFrame, match: MatchTracking, max_shift_s: float = 2.0,
                     step_s: float = 0.04, tolerance_s: float = 0.1) -> dict[int, float]:
    """Per-period time shift (s, added to event times) minimising median actor-ball distance.

    Shifts that keep fewer than half the events matched to a frame are ignored.
    """
    ob = _on_ball(events, match)
    shifts = np.arange(-max_shift_s, max_shift_s + step_s / 2, step_s)
    offsets: dict[int, float] = {}
    for p in np.unique(ob.periods):
        mask = ob.periods == p
        _, n0 = _median_distance_at(match, ob, mask, 0.0, tolerance_s)
        best, best_med = 0.0, np.inf
        for s in shifts:
            med, n = _median_distance_at(match, ob, mask, s, tolerance_s)
            if n >= 0.5 * max(n0, 1) and med < best_med - 1e-9:
                best, best_med = float(s), med
        offsets[int(p)] = round(best, 3)
    return offsets


# Event types where the ball leaves the actor's foot: these get a refined release frame.
RELEASE_TYPES = ("pass", "cross", "shot", "clearance")


def refine_release_frames(events: pd.DataFrame, match: MatchTracking, window_s: float = 2.0,
                          after_s: float = 0.32, near_m: float = 2.0,
                          penalty_m_per_s: float = 1.0) -> np.ndarray:
    """Per-event release frame for events returned by `attach_frames`.

    A clock offset fixes the average lag, but DFL event times are noisy per event (± 1 s), and
    Phase 3 needs the exact frame the ball was played. Around each release event's frame we
    pick the frame where the ball is at the actor's feet (within `near_m`, or the closest
    approach + 0.5 m if never that close) and its speed away from the actor jumps the most:
    separation speed over the next `after_s` minus that over the previous `after_s`, both
    counting only movement away (so a kick wins over the flight after it or a reception),
    minus a small penalty per second of shift. Frames stay in event order: a release
    is never placed at or before the previous one. Other events keep their frame.

    Only for providers whose events aren't tagged to frames (IDSSE). On PFF, whose tags are
    exact, it agrees with the tagged frame (±2 frames) for 87-91% of passes.
    """
    if "frame" not in events:
        raise ValueError("events have no `frame`; run attach_frames first")
    frames = events["frame"].to_numpy().copy()
    fps = match.frame_rate
    w, ahead = int(round(window_s * fps)), max(1, int(round(after_s * fps)))
    index = {str(pid): (side, j) for side, team in ((0, match.home), (1, match.away))
             for j, pid in enumerate(team.player_ids)}
    types = events["type"].to_numpy()
    players = events["player_id"].to_numpy()

    last = -1  # previous release frame
    for i in np.flatnonzero(frames >= 0):
        f = frames[i]
        if types[i] not in RELEASE_TYPES or players[i] not in index:
            continue
        side, slot = index[players[i]]
        cand = np.arange(max(f - w, last + 1, 0), min(f + w + 1, match.n_frames - ahead))
        cand = cand[(match.period[cand] == match.period[f])
                    & (match.period[cand + ahead] == match.period[f])]
        # At a period's start there is no "before": compare with the frame itself.
        prev = np.maximum(cand - ahead, 0)
        prev = np.where(match.period[prev] == match.period[f], prev, cand)
        actor = match.team_arrays(side)[0][:, slot]
        d, d_before, d_after = (np.linalg.norm(actor[c] - match.ball[c], axis=1)
                                for c in (cand, prev, cand + ahead))
        if len(cand) == 0 or np.all(np.isnan(d)):
            continue
        # Only movement away counts, so a reception (ball arriving, then still) scores 0.
        jump = np.maximum(d_after - d, 0) - np.maximum(d - np.fmin(d_before, d), 0)
        score = jump * fps / ahead - penalty_m_per_s * np.abs(cand - f) / fps
        score[~(d <= max(near_m, np.nanmin(d) + 0.5)) | np.isnan(score)] = -np.inf
        if np.isfinite(score).any():
            frames[i] = last = int(cand[np.argmax(score)])
    return frames


@dataclass
class SyncReport:
    match_id: str
    n_events: int
    n_on_ball: int
    pct_synced: float  # on-ball events with a tracking frame within tolerance
    median_actor_ball_m: float  # actor ↔ tracking ball at the event frame
    p95_actor_ball_m: float
    pct_actor_within_3m: float
    median_event_ball_m: float  # event coordinates ↔ tracking ball (checks orientation)
    offsets_s: dict[int, float] = field(default_factory=dict)  # applied per-period offsets
    residual_offset_s: float = 0.0  # best further shift after applying offsets (≈ 0 if good)

    def as_dict(self) -> dict:
        return asdict(self)


def sync_report(events: pd.DataFrame, match: MatchTracking, max_shift_s: float = 2.0
                ) -> SyncReport:
    """Measure event ↔ tracking alignment for events returned by `attach_frames`."""
    if "frame" not in events:
        events = attach_frames(events, match)
    offsets = (events.groupby("period")["sync_offset_s"].first().to_dict()
               if len(events) else {})
    corrected = events.assign(time_s=events["time_s"] + events["sync_offset_s"])
    ob = _on_ball(corrected, match)
    frames = ob.frames
    ok = frames >= 0

    nan = float("nan")
    if not ok.any():
        return SyncReport(match.match_id, len(events), len(ob.times), 0.0, nan, nan, nan, nan,
                          offsets, nan)

    d_actor = _actor_ball_distance(match, frames[ok], ob.sides[ok], ob.slots[ok])
    d_event = np.linalg.norm(ob.start_xy[ok] - match.ball[frames[ok]], axis=1)
    residual = estimate_offsets(corrected, match, max_shift_s=max_shift_s)
    worst = max(residual.values(), key=abs) if residual else 0.0

    return SyncReport(
        match_id=match.match_id,
        n_events=len(events),
        n_on_ball=len(ob.times),
        pct_synced=100.0 * float(ok.sum()) / max(len(ob.times), 1),
        median_actor_ball_m=float(np.nanmedian(d_actor)),
        p95_actor_ball_m=float(np.nanpercentile(d_actor, 95)),
        pct_actor_within_3m=100.0 * float(np.nanmean(d_actor <= 3.0)),
        median_event_ball_m=float(np.nanmedian(d_event)),
        offsets_s={int(k): float(v) for k, v in offsets.items()},
        residual_offset_s=float(worst),
    )


def check_sync(report: SyncReport, max_offset_s: float = 0.2,
               max_median_actor_ball_m: float = 3.0) -> None:
    """Fail loudly if events and tracking are still systematically misaligned."""
    problems = []
    if not abs(report.residual_offset_s) <= max_offset_s:
        problems.append(f"residual clock offset {report.residual_offset_s:+.2f} s")
    if not report.median_actor_ball_m <= max_median_actor_ball_m:
        problems.append(f"median actor-ball distance {report.median_actor_ball_m:.1f} m")
    if problems:
        raise ValueError(f"match {report.match_id}: events out of sync: " + "; ".join(problems))


def synchronise(events: pd.DataFrame, match: MatchTracking, check: bool = True,
                refine: bool = False) -> tuple[pd.DataFrame, SyncReport]:
    """Estimate per-period offsets, attach frames, report, and (optionally) check.

    Adds `release_frame`: the refined frame for release events if `refine` (use it when the
    provider's events aren't tagged to frames), otherwise a copy of `frame`. `frame` itself,
    and so everything Phases 1-2 build on it, is unchanged.
    """
    synced = attach_frames(events, match, offsets=estimate_offsets(events, match))
    synced["release_frame"] = (refine_release_frames(synced, match) if refine
                               else synced["frame"].to_numpy())
    report = sync_report(synced, match)
    if check:
        check_sync(report)
    return synced, report
