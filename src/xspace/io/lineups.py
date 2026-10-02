"""Use events to correct who is on the pitch.

PFF's smoothed tracking keeps a sent-off player's track after the card (3828: Hennessey stays
"on the pitch" after his 84th-minute red, alongside the replacement keeper). Substituted players
do disappear correctly, so only dismissals are fixed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
