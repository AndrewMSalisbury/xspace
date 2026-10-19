"""StatsBomb Open Data events, reduced to what a value model needs (Phase 6).

StatsBomb Open Data (https://github.com/statsbomb/open-data) is free under StatsBomb's user
agreement: attribution required ("Data provided by StatsBomb"). Only derived models (the fitted
xT grid) leave this machine; the events stay in data/raw/statsbomb/ (gitignored).

One row per on-ball event that moves the ball, loses it or shoots, in **open play**:

| kind | Events | end_x / end_y | success |
|---|---|---|---|
| move | open-play passes, carries | where the ball went | pass completed; carries always |
| loss | failed dribbles, dispossessions, miscontrols | NaN | False |
| shot | open-play shots | NaN | goal |

Coordinates are metres, centre spot (0, 0), the acting team attacking +x, y up (StatsBomb
uses 120 × 80 yards with y down and always orients events towards the acting team's attack).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xspace.constants import PITCH_LENGTH, PITCH_WIDTH

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
SB_LENGTH, SB_WIDTH = 120.0, 80.0
SET_PIECE_PASSES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
LOSSES = {"Dispossessed", "Miscontrol"}
COLUMNS = ["match_id", "index", "period", "time_s", "possession", "team_id", "kind", "x", "y",
           "end_x", "end_y", "success", "xg"]


def to_metres(loc: list[float] | None) -> tuple[float, float]:
    """StatsBomb [x, y] (yards, y down) → metres, centre origin, y up."""
    if not loc:
        return np.nan, np.nan
    return (loc[0] / SB_LENGTH * PITCH_LENGTH - PITCH_LENGTH / 2,
            (SB_WIDTH / 2 - loc[1]) / SB_WIDTH * PITCH_WIDTH)


def reduce_events(events: list[dict], match_id: int) -> pd.DataFrame:
    """One match's raw StatsBomb events → the rows above."""
    rows = []
    for e in events:
        kind_name = e["type"]["name"]
        x, y = to_metres(e.get("location"))
        if np.isnan(x):
            continue
        end = (np.nan, np.nan)
        success, xg = None, np.nan
        if kind_name == "Pass":
            p = e["pass"]
            if p.get("type", {}).get("name") in SET_PIECE_PASSES:
                continue
            kind, end = "move", to_metres(p.get("end_location"))
            success = "outcome" not in p
        elif kind_name == "Carry":
            kind, end, success = "move", to_metres(e["carry"].get("end_location")), True
        elif kind_name == "Dribble":
            if e["dribble"].get("outcome", {}).get("name") == "Complete":
                continue  # the carry that follows moves the ball
            kind, success = "loss", False
        elif kind_name in LOSSES:
            kind, success = "loss", False
        elif kind_name == "Shot":
            s = e["shot"]
            if s.get("type", {}).get("name") != "Open Play":
                continue
            kind = "shot"
            success = s.get("outcome", {}).get("name") == "Goal"
            xg = s.get("statsbomb_xg", np.nan)
        else:
            continue
        rows.append((match_id, e["index"], e["period"], e["minute"] * 60 + e["second"],
                     e["possession"], e["team"]["id"], kind, x, y, end[0], end[1],
                     bool(success), xg))
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.astype({"kind": "category", "x": "float32", "y": "float32", "end_x": "float32",
                      "end_y": "float32", "xg": "float32"})
