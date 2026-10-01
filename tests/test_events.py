import numpy as np
import pandas as pd
import pytest

from xspace.io.events import EVENT_COLUMNS, parse_pff_events, validate_events
from xspace.io.loaders import MatchTracking, Team
from xspace.io.sync import attach_frames, check_sync, estimate_offsets, sync_report, synchronise

META = {
    "homeTeam": {"id": "1"}, "awayTeam": {"id": "2"},
    "homeTeamStartLeft": True, "homeTeamStartLeftExtraTime": None,
    "startPeriod1": 100.0, "startPeriod2": 4000.0,
}


def pff_event(t, period, ptype=None, game_type="OTB", team=1, player=10, ball=(0.0, 0.0),
              setpiece="O", event_id=None, **pe):
    possession = None
    if ptype is not None:
        possession = {"possessionEventType": ptype, "nonEvent": False, **pe}
    return {
        "gameEventId": 1, "possessionEventId": event_id, "eventTime": t,
        "gameEvents": {"gameEventType": game_type, "period": period, "teamId": team,
                       "playerId": player, "setpieceType": setpiece,
                       "initialNonEvent": False},
        "possessionEvents": possession,
        "ball": [] if ball is None else [{"x": ball[0], "y": ball[1], "z": 0.0}],
    }


def test_pff_schema_times_and_end_locations():
    raw = [
        pff_event(100.0, 1, "PA", setpiece="K", event_id=1, ball=(0, 0), passOutcomeType="C",
                  targetPlayerId=11, receiverPlayerId=11),
        pff_event(101.5, 1, "IT", player=11, ball=(10, 5)),
        pff_event(102.0, 1, "SH", player=11, team=1, event_id=2, ball=(12, 6),
                  shotOutcomeType="G"),
        pff_event(103.0, 1, game_type="OUT", ptype=None, ball=(52, 0)),
    ]
    ev = parse_pff_events(raw, META)
    validate_events(ev)
    assert list(ev.columns) == list(EVENT_COLUMNS)
    assert list(ev["type"]) == ["pass", "reception", "shot", "out"]
    np.testing.assert_allclose(ev["time_s"], [0.0, 1.5, 2.0, 3.0])
    p = ev.iloc[0]
    assert (p.end_x, p.end_y) == (10, 5)  # ball at the reception
    assert p.setpiece == "kick_off" and bool(p.success) and p.receiver_player_id == "11"
    assert (ev.iloc[2].end_x, ev.iloc[2].end_y) == (52, 0)
    assert bool(ev.iloc[2].success)
    assert np.isnan(ev.iloc[1].end_x)  # receptions don't move the ball


def test_pff_second_half_is_flipped_so_home_attacks_positive_x():
    raw = [pff_event(4000.0 + 5, 2, "PA", event_id=1, ball=(30.0, -10.0))]
    ev = parse_pff_events(raw, META)
    assert (ev.iloc[0].start_x, ev.iloc[0].start_y) == (-30.0, 10.0)
    assert ev.iloc[0].period == 2 and ev.iloc[0].time_s == pytest.approx(5.0)


def test_pff_team_side_and_non_events():
    raw = [
        pff_event(100.0, 1, "PA", team=2, event_id=1),
        pff_event(101.0, 1, "PA", event_id=2, nonEvent=True),
    ]
    ev = parse_pff_events(raw, META)
    assert len(ev) == 1 and ev.iloc[0].team_side == 1


def test_pff_extra_time_falls_back_to_first_event_of_period():
    meta = {**META, "startPeriod1": None, "homeTeamStartLeftExtraTime": False}
    raw = [pff_event(200.0, 1, "PA", event_id=1), pff_event(210.0, 1, "PA", event_id=2),
           pff_event(9000.0, 3, "PA", event_id=3, ball=(1.0, 2.0))]
    ev = parse_pff_events(raw, meta)
    np.testing.assert_allclose(ev["time_s"], [0.0, 10.0, 0.0])
    # Home starts extra time on the right → attacks -x in period 3 → flipped.
    assert (ev.iloc[2].start_x, ev.iloc[2].start_y) == (-1.0, -2.0)


# --- sync --------------------------------------------------------------------------------------

FPS = 25.0


def line_match(seconds: float = 60.0) -> MatchTracking:
    """Ball rolls along y=0 at 1 m/s; home players stand every 5 m along its path."""
    t = np.arange(0, seconds, 1 / FPS)
    n_players = 11
    home_pos = np.zeros((len(t), n_players, 2))
    home_pos[:, :, 0] = 5.0 * np.arange(n_players)
    away_pos = np.full((len(t), n_players, 2), 40.0)

    def team(prefix: str) -> Team:
        return Team(prefix, prefix, [f"{prefix}{i}" for i in range(n_players)],
                    list(range(n_players)), ["UNK"] * n_players, 0)

    zeros = np.zeros_like(home_pos)
    return MatchTracking(
        match_id="synthetic", frame_rate=FPS, period=np.ones(len(t), dtype=int), timestamp=t,
        ball=np.stack([t, np.zeros_like(t)], axis=1), ball_owner=np.zeros(len(t), dtype=int),
        home=team("h"), away=team("a"), home_pos=home_pos, away_pos=away_pos,
        home_vel=zeros, away_vel=zeros,
    )


def passes_at(times: np.ndarray, players: list[int]) -> pd.DataFrame:
    n = len(times)
    df = pd.DataFrame({
        "event_id": [str(i) for i in range(n)], "period": 1, "time_s": times, "team_side": 0,
        "player_id": [f"h{p}" for p in players], "type": "pass", "setpiece": "open_play",
        "start_x": 5.0 * np.array(players), "start_y": 0.0, "end_x": np.nan, "end_y": np.nan,
        "target_player_id": None, "receiver_player_id": None, "success": None,
        "outcome": None, "height": None, "lines_broken": None,
    })
    return df.astype(EVENT_COLUMNS)


def test_attach_frames_nearest_and_tolerance():
    match = line_match(10.0)
    ev = passes_at(np.array([0.0, 2.01, 50.0]), [0, 0, 0])
    out = attach_frames(ev, match)
    assert list(out["frame"]) == [0, 50, -1]  # 2.01 s → frame 50; 50 s is past the data


def test_estimate_offsets_recovers_event_lag():
    match = line_match()
    players = list(range(1, 11))
    lag = 0.8  # events recorded 0.8 s after the ball reached the player
    ev = passes_at(5.0 * np.array(players) + lag, players)
    offsets = estimate_offsets(ev, match)
    assert offsets[1] == pytest.approx(-lag, abs=1 / FPS)

    raw_report = sync_report(attach_frames(ev, match), match)
    assert raw_report.median_actor_ball_m == pytest.approx(lag, abs=0.05)
    with pytest.raises(ValueError, match="out of sync"):
        check_sync(raw_report)

    synced, report = synchronise(ev, match)  # check=True must not raise
    assert report.median_actor_ball_m < 0.05
    assert report.residual_offset_s == pytest.approx(0.0, abs=1 / FPS)
    assert (synced["frame"] >= 0).all()
