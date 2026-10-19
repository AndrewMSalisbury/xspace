"""Lineups: who is on the pitch, and who each player is.

PFF's smoothed tracking keeps a sent-off player's track after the card (3828: Hennessey stays
"on the pitch" after his 84th-minute red, alongside the replacement keeper). Substituted players
do disappear correctly, so only dismissals are fixed here.

`player_directory` gives names and positions. For PFF it reads the match roster, which has a
position for every player; kloppy's tracking metadata leaves some starters (wing-backs, a
middle centre-back) as "UNK".
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from xspace.config import PFF_DIR
from xspace.io.events import SENT_OFF
from xspace.io.loaders import MatchTracking, assign_goalkeepers


def remove_sent_off_players(match: MatchTracking, events: pd.DataFrame) -> list[str]:
    """Blank (NaN) sent-off players' positions and velocities from their card onwards, in place,
    then re-assign per-frame goalkeepers. Returns the affected player ids.

    Uses `events.time_s` (+ `sync_offset_s` if present), so it works whether or not the card
    happened while the ball was in play.
    """
    cards = events[(events["type"] == "card") & events["outcome"].isin(SENT_OFF)
                   & events["player_id"].notna()]
    removed = []
    for _, card in cards.iterrows():
        t = card["time_s"] + (card["sync_offset_s"] if "sync_offset_s" in card else 0.0)
        after = (match.period > card["period"]) | (
            (match.period == card["period"]) & (match.timestamp >= t))
        for team, pos, vel in ((match.home, match.home_pos, match.home_vel),
                               (match.away, match.away_pos, match.away_vel)):
            if card["player_id"] in team.player_ids:
                j = team.player_ids.index(card["player_id"])
                pos[after, j] = np.nan
                vel[after, j] = np.nan
                removed.append(str(card["player_id"]))
    if removed:
        assign_goalkeepers(match)
    return removed


def player_directory(source: str, match_id: str) -> pd.DataFrame:
    """player_id, name, position code (as the source spells it; "UNK" if none) per player."""
    if source == "pff":
        roster = json.loads((PFF_DIR / "Rosters" / f"{match_id}.json").read_text(encoding="utf-8"))
        rows = [(str(r["player"]["id"]), r["player"]["nickname"], r.get("positionGroupType"))
                for r in roster]
    elif source == "idsse":
        from kloppy import sportec

        meta = sportec.load_open_event_data(match_id=match_id).metadata
        rows = [(str(p.player_id), p.full_name or p.name,
                 p.starting_position.code if p.starting_position else None)
                for team in meta.teams for p in team.players]
    else:
        raise ValueError(f"unknown source {source!r}")
    df = pd.DataFrame(rows, columns=["player_id", "name", "position"])
    df["position"] = df["position"].fillna("UNK")
    return df.drop_duplicates("player_id").astype({"player_id": "string", "name": "string"})
