from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from xspace.config import TimelineConfig, params_hash, settings_dict
from xspace.io.loaders import MatchTracking, Team, assign_goalkeepers
from xspace.metrics.timeline import (
    METRICS,
    build_timeline,
    read_timeline,
    sample_frames,
    timeline_metadata,
    write_timeline,
)
from xspace.phases.possession import PhaseLabels
from xspace.phases.quality import BALL_MISSING, PLAYER_JUMP
from xspace.physics.pitch_control import make_grid, pass_reachability, pitch_control

FPS = 25.0
CONFIG = TimelineConfig(cell_size=5.0, chunk_size=7)


def random_match(seconds: float = 8.0, seed: int = 0) -> MatchTracking:
    """Two teams of 11 drifting around their own halves; home GK in slot 0, away in slot 0."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)

    def positions(x_centre: float) -> tuple[np.ndarray, np.ndarray]:
        start = np.stack([rng.uniform(-20, 20, 11) + x_centre, rng.uniform(-30, 30, 11)], 1)
        vel = rng.normal(0, 2, (11, 2))
        t = np.arange(n)[:, None, None] / FPS
        return start[None] + vel[None] * t, np.broadcast_to(vel, (n, 11, 2)).copy()

    home_pos, home_vel = positions(-15.0)
    away_pos, away_vel = positions(15.0)
    home_pos[:, 0] = [-50.0, 0.0]
    away_pos[:, 0] = [50.0, 0.0]
    home_vel[:, 0] = away_vel[:, 0] = 0.0

    def team(prefix: str) -> Team:
        return Team(prefix, prefix, [f"{prefix}{i}" for i in range(11)], list(range(11)),
                    ["GK"] + ["UNK"] * 10, 0)

    ball = np.stack([np.linspace(-10, 10, n), np.zeros(n)], 1)
    return assign_goalkeepers(MatchTracking(
        match_id="synthetic", frame_rate=FPS, period=np.ones(n, dtype=int),
        timestamp=np.arange(n) / FPS, ball=ball, ball_owner=np.zeros(n, dtype=int),
        home=team("h"), away=team("a"), home_pos=home_pos, away_pos=away_pos,
        home_vel=home_vel, away_vel=away_vel,
    ))


def labels(n: int) -> PhaseLabels:
    """Home in possession for the first half, away for the second; no side for 5 frames."""
    side = np.where(np.arange(n) < n // 2, 0, 1)
    side[:5] = -1
    setpiece = np.zeros(n, dtype=np.int64)
    setpiece[20:30] = 3  # free kick
    return PhaseLabels(possession_side=side, possession_id=np.where(side < 0, -1, side),
                       setpiece=setpiece, transition=np.zeros(n, dtype=bool),
                       third=np.ones(n, dtype=np.int64))


def flags(n: int) -> np.ndarray:
    f = np.zeros(n, dtype=np.int64)
    f[40:45] = BALL_MISSING
    f[60] = PLAYER_JUMP
    return f


def test_sampling_follows_frame_rate():
    match = random_match()
    assert np.diff(sample_frames(match, 5.0)).tolist() == [5] * 39
    pff_like = replace(match, frame_rate=29.97)
    assert set(np.diff(sample_frames(pff_like, 5.0))) == {6}


def test_status_codes_and_metrics():
    match = random_match()
    df = build_timeline(match, labels(match.n_frames), flags(match.n_frames), CONFIG)
    status = dict(zip(df["frame"], df["status"].astype(str), strict=True))
    assert status[0] == "no_possession"
    assert status[20] == status[25] == "set_piece"
    assert status[40] == status[60] == "quality"
    assert status[10] == status[100] == "ok"

    ok = df[df["status"] == "ok"]
    assert ok[list(METRICS)].notna().all().all()
    assert df.loc[df["status"] != "ok", "total"].isna().all()
    zones = ok[["behind", "between", "wide", "in_front"]].sum(axis=1)
    np.testing.assert_allclose(zones, ok["total"], rtol=1e-5)
    assert (ok["n_att"] == 11).all() and (ok["n_def"] == 11).all()
    # Attacking frame: the away team attacks -x on the pitch, so its ball x flips sign.
    away = ok[ok["possession_side"] == 1]
    np.testing.assert_allclose(away["ball_x"], -match.ball[away["frame"], 0], rtol=1e-6)


def test_quality_mask_is_configurable():
    match = random_match()
    lenient = replace(CONFIG, skip_flags=BALL_MISSING)
    df = build_timeline(match, labels(match.n_frames), flags(match.n_frames), lenient)
    assert df.set_index("frame").loc[60, "status"] == "ok"


def test_deterministic_and_parallel_equals_serial():
    match = random_match()
    lab, fl = labels(match.n_frames), flags(match.n_frames)
    serial = build_timeline(match, lab, fl, CONFIG)
    again = build_timeline(match, lab, fl, CONFIG)
    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel = build_timeline(match, lab, fl, CONFIG, executor=pool)
    assert serial.equals(again)
    assert serial.equals(parallel)


def test_parquet_round_trip_keeps_metadata(tmp_path):
    match = random_match()
    df = build_timeline(match, labels(match.n_frames), flags(match.n_frames), CONFIG)
    meta = timeline_metadata("synthetic", match, CONFIG, sync_pct=99.5)
    back, meta_back = read_timeline(write_timeline(df, tmp_path / "t.parquet", meta))
    assert back.equals(df)
    assert meta_back["params_hash"] == params_hash(settings_dict(timeline=CONFIG))
    assert meta_back["match_id"] == "synthetic" and meta_back["sync_pct"] == "99.5"


def test_params_hash_tracks_settings():
    assert params_hash(settings_dict()) == params_hash(settings_dict())
    assert params_hash(settings_dict()) != params_hash(settings_dict(timeline=CONFIG))
    # chunk_size changes how work is split, not the results
    assert params_hash(settings_dict(timeline=replace(CONFIG, chunk_size=99))) == params_hash(
        settings_dict(timeline=CONFIG))


# --- The fast kernels against a direct float64 implementation of the same model ---

def _reference_control(att_pos, att_vel, def_pos, def_vel, ball, grid, def_gk, p):
    """Shaw-style integration, one exp per step, float64, no compaction."""
    k = np.pi / np.sqrt(3) / p.tti_sigma

    def tti(pos, vel):
        r = pos + vel * p.reaction_time
        return p.reaction_time + np.linalg.norm(grid[None] - r[:, None], axis=-1) / p.max_speed

    ta, td = tti(att_pos, att_vel), tti(def_pos, def_vel)
    lam_a = np.full(len(att_pos), p.lambda_att)
    lam_d = np.full(len(def_pos), p.lambda_att * p.kappa_def)
    lam_d[def_gk] *= p.lambda_gk_factor
    ball_t = np.linalg.norm(grid - ball, axis=1) / p.ball_speed
    pa, pd_ = np.zeros_like(ta), np.zeros_like(td)
    total, active = np.zeros(len(grid)), np.ones(len(grid), bool)
    for step in range(int(p.max_int_time / p.int_dt)):
        t = ball_t + step * p.int_dt
        rem = np.where(active, np.maximum(1 - total, 0), 0)
        pa += rem * lam_a[:, None] * p.int_dt / (1 + np.exp(-k * (t - ta)))
        pd_ += rem * lam_d[:, None] * p.int_dt / (1 + np.exp(-k * (t - td)))
        total = pa.sum(0) + pd_.sum(0)
        active &= total < 1 - p.convergence_tol
        if not active.any():
            break
    return pa.sum(0) / total


@pytest.mark.parametrize("frame", [10, 90, 170])
def test_fast_kernels_match_reference(frame):
    from xspace.config import DEFAULT_PARAMS as p

    match = random_match(seed=frame)
    _, _, grid = make_grid(3.0)
    ap, av = match.home_pos[frame], match.home_vel[frame]
    dp, dv = match.away_pos[frame], match.away_vel[frame]
    ball = match.ball[frame]

    fast = pitch_control(ap, av, dp, dv, ball, grid, def_gk=0)
    ref = _reference_control(ap, av, dp, dv, ball, grid, 0, p)
    # float32 can stop a cell one integration step apart from float64 at the tolerance.
    np.testing.assert_allclose(fast.attack, ref, atol=2e-3)
    assert np.median(np.abs(fast.attack - ref)) < 1e-5

    s = np.linspace(0, 1, p.lane_samples + 2)[1:-1]
    lane = ball + s[None, :, None] * (grid - ball)[:, None, :]
    ball_t = np.linalg.norm(lane - ball, axis=-1) / p.ball_speed
    r = dp + dv * p.reaction_time
    tti = p.reaction_time + np.linalg.norm(lane[None] - r[:, None, None], axis=-1) / p.max_speed
    p_int = 1 / (1 + np.exp(-np.pi / np.sqrt(3) / p.tti_sigma * (ball_t[None] - tti)))
    np.testing.assert_allclose(pass_reachability(ball, dp, dv, grid),
                               np.prod(1 - p_int, axis=(0, 2)), atol=1e-5)
