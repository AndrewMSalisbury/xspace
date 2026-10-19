"""PFF's own judgements, as a benchmark for the ratings (Phase 5).

PFF analysts grade players event by event: each possession event has a `grades` dict
(`passerGrade`, `receiverGrade`, `defenderGrade`, ...; steps of 0.5, mostly −2 to +2, 0 =
expected), and the player in each role is `possessionEvents.<role>PlayerId`. Analysts also flag
when a **better option** was on (`betterOptionType`: P pass, C cross, H hold / carry, L, B, O),
sometimes naming the player (`betterOptionPlayerId`): a direct check on the decision gap.

These are human judgements, not ground truth; they are graded relative to what the analyst
expected, so they reward execution as much as the choice.
"""

from __future__ import annotations

import json

import pandas as pd

from xspace.config import PFF_DIR

# Roles grouped into what a rating measures. Unlisted roles (keeper, shooter, touch, ...) are
# kept under "other".
ROLE_GROUPS = {
    "passing": ("passer", "crosser"),
    "receiving": ("receiver", "target"),
    "off_ball": ("movement", "position", "position2", "position3"),
    "carrying": ("carrier", "ballCarrier", "dribbler"),
    "defending": ("defender", "challenger", "blocker", "failedBlocker", "failedBlocker2",
                  "closingDown", "carryDefender", "clearer", "deflector"),
}
ROLE_TO_GROUP = {role: group for group, roles in ROLE_GROUPS.items() for role in roles}


def _events(match_id: str) -> list[dict]:
    path = PFF_DIR / "Event Data" / f"{match_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def event_grades(match_id: str) -> pd.DataFrame:
    """One row per graded (event, role): event_id, role, group, player_id, grade."""
    rows = []
    for ev in _events(match_id):
        grades, pe = ev.get("grades") or {}, ev.get("possessionEvents") or {}
        for key, grade in grades.items():
            if grade is None:
                continue
            role = key.removesuffix("Grade")
            player = pe.get(f"{role}PlayerId")
            if player is None:
                continue
            rows.append((str(ev.get("possessionEventId") or ""), role,
                         ROLE_TO_GROUP.get(role, "other"), str(player), float(grade)))
    return pd.DataFrame(rows, columns=["event_id", "role", "group", "player_id", "grade"])


def player_grades(match_id: str) -> pd.DataFrame:
    """Per player in a match: sum and count of grades per role group (`<group>_sum`,
    `<group>_n`)."""
    g = event_grades(match_id).groupby(["player_id", "group"])["grade"].agg(["sum", "count"])
    wide = g.unstack("group")
    wide.columns = [f"{grp}_{'sum' if stat == 'sum' else 'n'}" for stat, grp in wide.columns]
    return wide.fillna(0.0).reset_index().astype({"player_id": "string"})


def better_options(match_id: str) -> pd.DataFrame:
    """Every possession event with its better-option flag: event_id, better_option (bool),
    better_option_type, better_option_player_id."""
    rows = []
    for ev in _events(match_id):
        pe = ev.get("possessionEvents") or {}
        if ev.get("possessionEventId") is None:
            continue
        kind = pe.get("betterOptionType")
        player = pe.get("betterOptionPlayerId")
        rows.append((str(ev["possessionEventId"]), kind is not None, kind,
                     None if player is None else str(player)))
    return pd.DataFrame(rows, columns=["event_id", "better_option", "better_option_type",
                                       "better_option_player_id"]).astype(
        {"event_id": "string", "better_option_player_id": "string"})
