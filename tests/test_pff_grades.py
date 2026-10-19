from xspace.validation import pff_grades

EVENTS = [
    {"possessionEventId": 1, "grades": {"passerGrade": 0.5, "receiverGrade": -1.0,
                                        "keeperGrade": None},
     "possessionEvents": {"passerPlayerId": 10, "receiverPlayerId": 11,
                          "betterOptionType": "P", "betterOptionPlayerId": 12}},
    {"possessionEventId": 2, "grades": {"passerGrade": -0.5, "defenderGrade": 0.5},
     "possessionEvents": {"passerPlayerId": 10, "defenderPlayerId": 20}},
    {"possessionEventId": None, "grades": None, "possessionEvents": None},
]


def test_grades_and_better_options(monkeypatch):
    monkeypatch.setattr(pff_grades, "_events", lambda match_id: EVENTS)
    e = pff_grades.event_grades("x")
    assert len(e) == 4 and set(e["group"]) == {"passing", "receiving", "defending"}
    pg = pff_grades.player_grades("x").set_index("player_id")
    assert (pg.loc["10", "passing_sum"], pg.loc["10", "passing_n"]) == (0.0, 2)
    assert pg.loc["11", "receiving_sum"] == -1.0 and pg.loc["20", "defending_sum"] == 0.5
    b = pff_grades.better_options("x").set_index("event_id")
    assert len(b) == 2 and b.loc["1", "better_option"] and not b.loc["2", "better_option"]
    assert b.loc["1", "better_option_player_id"] == "12"
