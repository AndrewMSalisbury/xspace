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
                  tolerance_s: float = 0.1) -> pd.DataFrame:
    """Return a copy of `events` with `frame` (row index into `match` arrays), `sync_offset_s`
    (per-period offset added to `time_s`) and `sync_dt_s` (frame time − corrected event time).

    `frame` is -1 when no tracking frame lies within `tolerance_s` (e.g. the event happened
    while the ball was dead and those frames were dropped).
    """
    out = events.copy()
    periods = out["period"].to_numpy()
    offset = np.array([(offsets or {}).get(int(p), 0.0) for p in periods], dtype=float)
    frame, dt = _nearest_frames(match, periods, out["time_s"].to_numpy(float) + offset,
                                tolerance_s)
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


def synchronise(events: pd.DataFrame, match: MatchTracking, check: bool = True
                ) -> tuple[pd.DataFrame, SyncReport]:
    """Estimate per-period offsets, attach frames, report, and (optionally) check."""
    synced = attach_frames(events, match, offsets=estimate_offsets(events, match))
    report = sync_report(synced, match)
    if check:
        check_sync(report)
    return synced, report
