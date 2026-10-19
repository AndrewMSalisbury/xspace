"""Player level, per match: minutes, on-ball choices, and credit for the space a player holds.

Built from the action file (`build_actions`), whose `owner_id` / `best_owner_id` name the
attacker with the largest pitch-control share at the chosen and at the best cell.

- `team_side`, `player_id`, `position`, `minutes`: from tracking; minutes on the pitch while
  the ball was live. `minutes_in_possession` / `minutes_out_of_possession`: the part of those
  with the player's own team / the opponent in possession (needs the phase labels).
- `n_actions`, `n_exploited`, `n_missed`, `xt_gained`, `decision_gap`, `likely_gap`: the
  player's own passes, crosses and carries.
- `n_received_space`, `xspace_received`: completed actions into a cell this player owned
  (receiver or not), and the sum of their xSpace.
- `n_best_space`: team-mates' actions whose best cell (best ≥ `min_best`) this player owned:
  space occupied. `n_best_found`: of those, the ball went into this player's space.
  `n_best_ignored`: of those, the action was `missed`.
- `n_likely_space`, `n_likely_found`: the same for the *likely* best cell (weighted by the
  distance prior, likely ≥ `min_likely`), which doesn't credit far-touchline space nobody
  passes to. Phase 5 rates space held on these.
- `n_likely_conceded`, `likely_conceded`, `n_likely_conceded_found`: opponents' actions whose
  likely cell (≥ `min_likely`) this player, as the outfield defender quickest to it, was
  responsible for; the sum of its xSpace; how often the ball went there. Experimental.

Credit is a count of moments, not a model of the run that made the space; the counterfactual
version ("freeze the runner, recompute") is a v2 idea in docs/PLAN.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xspace.config import DEFAULT_EXPLOITATION
from xspace.io.loaders import MatchTracking
from xspace.phases.possession import PhaseLabels

MIN_BEST = DEFAULT_EXPLOITATION.exploit_min  # a best cell worth crediting (≈ median frame's best)
# The likely cell is worth less (it trades value for distance): its median over actions is
# 0.0027, where the best cell's is 0.011. The same meaning, "at least a median option":
MIN_LIKELY = 0.003


def minutes_played(match: MatchTracking, phases: PhaseLabels | None = None) -> pd.DataFrame:
    """Minutes each rostered player had a position, counting only live-ball frames; with
    `phases`, also split by which team had possession."""
    rows = []
    for side, team, pos in ((0, match.home, match.home_pos), (1, match.away, match.away_pos)):
        present = ~np.isnan(pos[:, :, 0])
        row = {
            "team_side": np.int8(side), "player_id": pd.array(team.player_ids, dtype="string"),
            "position": team.positions,
            "minutes": present.sum(axis=0) / match.frame_rate / 60,
        }
        if phases is not None:
            for name, who in (("in", side), ("out_of", 1 - side)):
                frames = present & (phases.possession_side == who)[:, None]
                row[f"minutes_{name}_possession"] = frames.sum(axis=0) / match.frame_rate / 60
        rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True)


def build_players(match: MatchTracking, actions: pd.DataFrame,
                  phases: PhaseLabels | None = None, min_best: float = MIN_BEST,
                  min_likely: float = MIN_LIKELY) -> pd.DataFrame:
    """One row per player who was on the pitch."""
    df = minutes_played(match, phases).set_index(["team_side", "player_id"])
    df = df[df["minutes"] > 0]
    act = actions[actions["status"] == "ok"]
    completed = act["success"].fillna(False).astype(bool)

    def by(frame: pd.DataFrame, id_col: str) -> pd.core.groupby.DataFrameGroupBy:
        keys = [frame["team_side"], frame[id_col].rename("player_id")]
        return frame.groupby(keys)

    g = by(act, "player_id")
    own = pd.DataFrame({
        "n_actions": g.size(), "n_exploited": g["exploited"].sum(),
        "n_missed": g["missed"].sum(), "xt_gained": g["xt_gained"].sum(),
        "decision_gap": g["decision_gap"].mean(), "likely_gap": g["likely_gap"].mean(),
    })
    g = by(act[completed], "owner_id")
    received = pd.DataFrame({"n_received_space": g.size(), "xspace_received": g["chosen"].sum()})
    # The best cell's owner, when it isn't the player on the ball.
    big = act[(act["best"] >= min_best) & (act["best_owner_id"] != act["player_id"])]
    g = by(big.assign(found=big["owner_id"] == big["best_owner_id"]), "best_owner_id")
    space = pd.DataFrame({"n_best_space": g.size(), "n_best_found": g["found"].sum(),
                          "n_best_ignored": g["missed"].sum()})
    big = act[(act["likely"] >= min_likely) & (act["likely_owner_id"] != act["player_id"])]
    g = by(big.assign(found=big["owner_id"] == big["likely_owner_id"]), "likely_owner_id")
    likely = pd.DataFrame({"n_likely_space": g.size(), "n_likely_found": g["found"].sum()})
    # The same moments from the defending side: the defender quickest to the likely cell.
    big = act[(act["likely"] >= min_likely) & act["likely_defender_id"].notna()]
    big = big.assign(team_side=(1 - big["team_side"]).astype(np.int8),
                     found=big["owner_id"] == big["likely_owner_id"])
    g = by(big, "likely_defender_id")
    conceded = pd.DataFrame({"n_likely_conceded": g.size(), "likely_conceded": g["likely"].sum(),
                             "n_likely_conceded_found": g["found"].sum()})

    df = df.join(own).join(received).join(space).join(likely).join(conceded)
    counts = [c for c in df.columns if c.startswith("n_")]
    df[counts] = df[counts].fillna(0).astype(np.int32)
    sums = ["xt_gained", "xspace_received", "likely_conceded"]
    df[sums] = df[sums].fillna(0.0)
    return df.reset_index()
