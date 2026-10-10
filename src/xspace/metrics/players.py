"""Player level, per match: minutes, on-ball choices, and credit for the space a player holds.

Built from the action file (`build_actions`), whose `owner_id` / `best_owner_id` name the
attacker with the largest pitch-control share at the chosen and at the best cell.

- `team_side`, `player_id`, `position`, `minutes`: from tracking; minutes on the pitch while
  the ball was live.
- `n_actions`, `n_exploited`, `n_missed`, `xt_gained`, `decision_gap`: the player's own passes,
  crosses and carries.
- `n_received_space`, `xspace_received`: completed actions into a cell this player owned
  (receiver or not), and the sum of their xSpace.
- `n_best_space`: team-mates' actions whose best cell (best ≥ `min_best`) this player owned:
  space occupied. `n_best_found`: of those, the ball went into this player's space.
  `n_best_ignored`: of those, the action was `missed`.

Credit is a count of moments, not a model of the run that made the space; the counterfactual
version ("freeze the runner, recompute") is a v2 idea in docs/PLAN.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xspace.config import DEFAULT_EXPLOITATION
from xspace.io.loaders import MatchTracking

MIN_BEST = DEFAULT_EXPLOITATION.exploit_min  # a best cell worth crediting (≈ median frame's best)


def minutes_played(match: MatchTracking) -> pd.DataFrame:
    """Minutes each rostered player had a position, counting only live-ball frames."""
    rows = []
    for side, team, pos in ((0, match.home, match.home_pos), (1, match.away, match.away_pos)):
        frames = (~np.isnan(pos[:, :, 0])).sum(axis=0)
        rows.append(pd.DataFrame({
            "team_side": np.int8(side), "player_id": pd.array(team.player_ids, dtype="string"),
            "position": team.positions, "minutes": frames / match.frame_rate / 60,
        }))
    return pd.concat(rows, ignore_index=True)


def build_players(match: MatchTracking, actions: pd.DataFrame,
                  min_best: float = MIN_BEST) -> pd.DataFrame:
    """One row per player who was on the pitch."""
    df = minutes_played(match).set_index(["team_side", "player_id"])
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
        "decision_gap": g["decision_gap"].mean(),
    })
    g = by(act[completed], "owner_id")
    received = pd.DataFrame({"n_received_space": g.size(), "xspace_received": g["chosen"].sum()})
    # The best cell's owner, when it isn't the player on the ball.
    big = act[(act["best"] >= min_best) & (act["best_owner_id"] != act["player_id"])]
    g = by(big.assign(found=big["owner_id"] == big["best_owner_id"]), "best_owner_id")
    space = pd.DataFrame({"n_best_space": g.size(), "n_best_found": g["found"].sum(),
                          "n_best_ignored": g["missed"].sum()})

    df = df.join(own).join(received).join(space)
    counts = [c for c in df.columns if c.startswith("n_")]
    df[counts] = df[counts].fillna(0).astype(np.int32)
    df[["xt_gained", "xspace_received"]] = df[["xt_gained", "xspace_received"]].fillna(0.0)
    return df.reset_index()
