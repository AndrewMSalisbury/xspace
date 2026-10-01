"""Load event data into one provider-independent table (`MatchEvents`, a pandas DataFrame).

Coordinates use the same frame as `MatchTracking`: metres, centre spot at (0, 0), home team
attacking +x for the whole match. Times are seconds since the start of the period, on the same
clock as `MatchTracking.timestamp`, so `io.sync` can map events to tracking frames.

One row per on-ball action (pass, cross, shot, carry, reception, ...) or game event (ball out,
substitution, ...). Columns are listed in `EVENT_COLUMNS`; provider-specific codes that don't map
cleanly (outcome, height, lines broken) are kept as raw strings.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xspace.io.loaders import PFF_DIR

EVENT_COLUMNS: dict[str, str] = {
    "event_id": "string",
    "period": "int64",
    "time_s": "float64",  # seconds since period start (tracking clock)
    "team_side": "int64",  # 0 = home, 1 = away, -1 = none
    "player_id": "string",
    "type": "string",  # see EVENT_TYPES
    "setpiece": "string",  # see SETPIECE_TYPES
    "start_x": "float64",
    "start_y": "float64",
    "end_x": "float64",  # NaN when unknown / not meaningful
    "end_y": "float64",
    "target_player_id": "string",  # intended receiver (PFF only)
    "receiver_player_id": "string",  # actual receiver
    "success": "boolean",  # pass/cross completed, shot scored, carry retained; NA otherwise
    "outcome": "string",  # raw provider outcome code
    "height": "string",  # raw provider ball-height code
    "lines_broken": "string",  # raw provider code (PFF only)
}

EVENT_TYPES = (
    "pass", "cross", "shot", "carry", "reception", "touch", "challenge", "clearance", "rebound",
    "recovery", "out", "foul", "card", "sub", "other",
)
SETPIECE_TYPES = (
    "open_play", "kick_off", "throw_in", "free_kick", "goal_kick", "corner", "penalty",
    "drop_ball",
)
# Types whose end location is where the ball went next.
BALL_MOVING_TYPES = ("pass", "cross", "shot", "carry", "clearance")


def empty_events() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in EVENT_COLUMNS.items()})


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return empty_events()
    df = pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
    df = df.astype(EVENT_COLUMNS)
    return df.sort_values(["period", "time_s"], kind="stable").reset_index(drop=True)


def validate_events(df: pd.DataFrame) -> None:
    """Raise if `df` doesn't match the MatchEvents schema."""
    if list(df.columns[: len(EVENT_COLUMNS)]) != list(EVENT_COLUMNS):
        raise ValueError(f"columns {list(df.columns)} != {list(EVENT_COLUMNS)}")
    for col, dtype in EVENT_COLUMNS.items():
        if str(df[col].dtype) != dtype:
            raise ValueError(f"column {col!r} has dtype {df[col].dtype}, expected {dtype}")
    bad = set(df["type"].dropna()) - set(EVENT_TYPES)
    if bad:
        raise ValueError(f"unknown event types {bad}")
    bad = set(df["setpiece"].dropna()) - set(SETPIECE_TYPES)
    if bad:
        raise ValueError(f"unknown set-piece types {bad}")


# --- PFF FC 2022 World Cup ----------------------------------------------------------------------

PFF_TYPES = {
    "PA": "pass", "CR": "cross", "SH": "shot", "BC": "carry", "IT": "reception", "TC": "touch",
    "CH": "challenge", "CL": "clearance", "RE": "rebound",
}
PFF_GAME_TYPES = {"OUT": "out", "SUB": "sub"}
PFF_SETPIECES = {
    "O": "open_play", "K": "kick_off", "T": "throw_in", "F": "free_kick", "G": "goal_kick",
    "C": "corner", "P": "penalty", "D": "drop_ball",
}
# Next-ball lookahead for end locations; longer gaps mean the next event is a different phase.
PFF_END_LOOKAHEAD_S = 10.0


def _pff_id(value) -> str | None:
    return None if value is None else str(value)


def _pff_period_starts(meta: dict, raw: list[dict]) -> dict[int, float]:
    """Video time (s) at which each period starts.

    Metadata has `startPeriod1/2` for most matches but is empty for extra-time matches; there we
    use the period's first event (the kick-off), which matches `startPeriodN` where both exist.
    """
    starts: dict[int, float] = {}
    for ev in raw:
        p = ev["gameEvents"]["period"]
        if p not in starts:
            starts[p] = ev["eventTime"]
    for p in list(starts):
        if meta.get(f"startPeriod{p}") is not None:
            starts[p] = float(meta[f"startPeriod{p}"])
    return starts


def _pff_home_attacks_left(meta: dict, period: int) -> bool:
    """True if the home team attacks -x (raw PFF coords) in this period."""
    if period <= 2:
        start_left = bool(meta["homeTeamStartLeft"])
    else:
        et = meta.get("homeTeamStartLeftExtraTime")
        start_left = bool(meta["homeTeamStartLeft"] if et is None else et)
    # Starting on the left means attacking +x in the first period of each half-pair.
    first_of_pair = period % 2 == 1
    return start_left != first_of_pair


def _pff_outcome(ptype: str, pe: dict) -> tuple[str | None, bool | None]:
    if ptype == "PA":
        code = pe.get("passOutcomeType")
        return code, None if code is None else code == "C"
    if ptype == "CR":
        code = pe.get("crossOutcomeType")
        return code, None if code is None else code == "C"
    if ptype == "SH":
        code = pe.get("shotOutcomeType")
        return code, None if code is None else code == "G"
    if ptype == "BC":
        code = pe.get("ballCarryOutcome")
        return code, None if code is None else code == "R"
    if ptype == "CL":
        return pe.get("clearanceOutcomeType"), None
    if ptype == "CH":
        return pe.get("challengeOutcomeType"), None
    return None, None


def parse_pff_events(raw: list[dict], meta: dict) -> pd.DataFrame:
    """Convert PFF event JSON (a list of event dicts) + match metadata into MatchEvents."""
    home_team = str(meta["homeTeam"]["id"])
    starts = _pff_period_starts(meta, raw)

    rows: list[dict] = []
    ball_at: list[tuple[int, float, float, float]] = []  # (period, video time, x, y)
    for ev in raw:
        ge = ev["gameEvents"]
        pe = ev.get("possessionEvents") or {}
        period = int(ge["period"])
        if period not in starts:
            continue
        flip = -1.0 if _pff_home_attacks_left(meta, period) else 1.0

        ball = ev.get("ball") or []
        bx = by = np.nan
        if ball and ball[0].get("x") is not None:
            bx, by = flip * ball[0]["x"], flip * ball[0]["y"]
            ball_at.append((period, ev["eventTime"], bx, by))

        ptype = pe.get("possessionEventType")
        if ptype is not None:
            if pe.get("nonEvent") or (ptype == "IT" and ge.get("initialNonEvent")):
                continue  # annotated as "didn't happen"
            etype = PFF_TYPES.get(ptype, "other")
        else:
            etype = PFF_GAME_TYPES.get(ge["gameEventType"])
            if etype is None:
                continue  # kick-off markers, period ends, ...

        team_id = ge.get("teamId")
        team_side = -1 if team_id is None else (0 if str(team_id) == home_team else 1)
        outcome, success = _pff_outcome(ptype, pe) if ptype else (None, None)
        if ptype == "PA":
            height = pe.get("ballHeightType")
        elif ptype == "CR":
            height = pe.get("ballHeightType") or pe.get("crossType")
        else:
            height = None
        rows.append({
            "event_id": str(ev["possessionEventId"] or f"g{ev['gameEventId']}-{ptype or etype}"),
            "period": period,
            "time_s": ev["eventTime"] - starts[period],
            "team_side": team_side,
            "player_id": _pff_id(ge.get("playerId")),
            "type": etype,
            "setpiece": PFF_SETPIECES.get(ge.get("setpieceType") or "O", "open_play"),
            "start_x": bx,
            "start_y": by,
            "end_x": np.nan,
            "end_y": np.nan,
            "target_player_id": _pff_id(pe.get("targetPlayerId")),
            "receiver_player_id": _pff_id(pe.get("receiverPlayerId")),
            "success": success,
            "outcome": outcome,
            "height": height,
            "lines_broken": pe.get("linesBrokenType"),
            "_video_t": ev["eventTime"],
        })

    _pff_end_locations(rows, ball_at)
    for r in rows:
        del r["_video_t"]
    return _to_frame(rows)


def _pff_end_locations(rows: list[dict], ball_at: list[tuple[int, float, float, float]]) -> None:
    """PFF has no end coordinates: use the ball at the next event (usually the reception)."""
    if not ball_at:
        return
    periods = np.array([b[0] for b in ball_at])
    times = np.array([b[1] for b in ball_at])
    order = np.lexsort((times, periods))
    periods, times = periods[order], times[order]
    xy = np.array([(b[2], b[3]) for b in ball_at])[order]
    for r in rows:
        if r["type"] not in BALL_MOVING_TYPES:
            continue
        in_p = np.flatnonzero(periods == r["period"])
        k = np.searchsorted(times[in_p], r["_video_t"], side="right")
        if k < len(in_p) and times[in_p[k]] - r["_video_t"] <= PFF_END_LOOKAHEAD_S:
            r["end_x"], r["end_y"] = xy[in_p[k]]


def load_pff_events(match_id: str | int) -> pd.DataFrame:
    """Load a PFF FC 2022 World Cup match's events (download with scripts/download_pff.py)."""
    match_id = str(match_id)
    path = PFF_DIR / "Event Data" / f"{match_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run: uv run python scripts/download_pff.py")
    raw = json.loads(path.read_text(encoding="utf-8"))
    meta = json.loads((PFF_DIR / "Metadata" / f"{match_id}.json").read_text(encoding="utf-8"))
    return parse_pff_events(raw, meta[0] if isinstance(meta, list) else meta)


def pff_match_ids() -> list[str]:
    return sorted(p.stem for p in Path(PFF_DIR / "Event Data").glob("*.json"))


# --- IDSSE (DFL events via kloppy) --------------------------------------------------------------

KLOPPY_TYPES = {
    "PASS": "pass", "SHOT": "shot", "CARRY": "carry", "RECOVERY": "recovery",
    "BALL_OUT": "out", "FOUL_COMMITTED": "foul", "CARD": "card", "SUBSTITUTION": "sub",
    "CLEARANCE": "clearance", "DUEL": "challenge", "TAKE_ON": "challenge",
    "INTERCEPTION": "recovery", "MISCONTROL": "touch", "BALL_RECEIPT": "reception",
}
KLOPPY_SETPIECES = {
    "KICK_OFF": "kick_off", "THROW_IN": "throw_in", "FREE_KICK": "free_kick",
    "GOAL_KICK": "goal_kick", "CORNER_KICK": "corner", "PENALTY": "penalty",
}


def from_kloppy_events(dataset) -> pd.DataFrame:
    """Convert a kloppy EventDataset (any provider) into MatchEvents."""
    from kloppy.domain import Orientation, PassQualifier, PassType, SetPieceQualifier

    dataset = dataset.transform(
        to_orientation=Orientation.STATIC_HOME_AWAY,
        to_coordinate_system="secondspectrum",
    )
    home_id = dataset.metadata.teams[0].team_id

    rows: list[dict] = []
    for e in dataset.events:
        etype = KLOPPY_TYPES.get(e.event_type.name, "other")
        qualifiers = e.qualifiers or []
        setpiece = next((KLOPPY_SETPIECES.get(q.value.name, "open_play")
                         for q in qualifiers if isinstance(q, SetPieceQualifier)), "open_play")
        pass_types = {q.value for q in qualifiers if isinstance(q, PassQualifier)}
        if etype == "pass" and PassType.CROSS in pass_types:
            etype = "cross"

        end = None
        receiver = None
        if etype in ("pass", "cross"):
            end = e.receiver_coordinates
            receiver = e.receiver_player.player_id if e.receiver_player else None
        elif etype == "shot":
            end = getattr(e, "result_coordinates", None)
        elif etype == "carry":
            end = getattr(e, "end_coordinates", None)

        result = e.result.name if getattr(e, "result", None) is not None else None
        if etype in ("pass", "cross"):
            success = None if result is None else result == "COMPLETE"
        elif etype == "shot":
            success = None if result is None else result == "GOAL"
        elif etype == "carry":
            success = None if result is None else result == "COMPLETE"
        else:
            success = None

        rows.append({
            "event_id": str(e.event_id),
            "period": int(e.period.id),
            "time_s": e.timestamp.total_seconds(),
            "team_side": -1 if e.team is None else (0 if e.team.team_id == home_id else 1),
            "player_id": e.player.player_id if e.player else None,
            "type": etype,
            "setpiece": setpiece,
            "start_x": e.coordinates.x if e.coordinates else np.nan,
            "start_y": e.coordinates.y if e.coordinates else np.nan,
            "end_x": end.x if end else np.nan,
            "end_y": end.y if end else np.nan,
            "target_player_id": None,
            "receiver_player_id": receiver,
            "success": success,
            "outcome": result,
            "height": "high" if PassType.HIGH_PASS in pass_types else None,
            "lines_broken": None,
        })
    return _to_frame(rows)


def load_idsse_events(match_id: str = "J03WMX") -> pd.DataFrame:
    """Load one of the 7 open IDSSE matches' DFL events (CC BY 4.0, Bassek et al. 2025)."""
    from kloppy import sportec

    return from_kloppy_events(sportec.load_open_event_data(match_id=match_id))


def load_events(source: str, match_id: str) -> pd.DataFrame:
    """Dispatch to an event loader by source name: 'idsse' or 'pff'."""
    loaders = {"idsse": load_idsse_events, "pff": load_pff_events}
    if source not in loaders:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(loaders)}")
    return loaders[source](match_id)
