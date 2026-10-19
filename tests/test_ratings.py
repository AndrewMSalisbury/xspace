import numpy as np
import pandas as pd

from xspace.metrics.ratings import (
    RatingsConfig,
    action_table,
    bootstrap_ratio,
    leak_map,
    percentile,
    player_ratings,
    position_group,
    possession_table,
    shrink_count_rate,
    shrink_mean,
    shrink_rate,
    team_match_ratings,
    team_profiles,
)

CONFIG = RatingsConfig(n_boot=200)


def test_percentile_mid_rank():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
    np.testing.assert_allclose(percentile(s)[:4], [12.5, 37.5, 62.5, 87.5])
    assert np.isnan(percentile(s)[4])
    np.testing.assert_allclose(percentile(pd.Series([0.0, 10.0]), s), [0, 100])


def test_shrink_rate():
    # True rates identical: the spread is all noise, so everything goes to the pooled rate.
    rng = np.random.default_rng(1)
    n = np.full(200, 100)
    k = rng.binomial(n, 0.1)
    shrunk, mu, kappa = shrink_rate(k, n)
    assert np.isinf(kappa) or kappa > 100  # nearly all of the spread is noise
    assert np.ptp(shrunk) < np.ptp(k / n) / 2
    # Real differences survive, and small samples move further than large ones.
    p = np.where(np.arange(200) < 100, 0.05, 0.3)
    n = np.where(np.arange(200) % 2 == 0, 20, 2000)
    shrunk, mu, _ = shrink_rate(rng.binomial(n, p), n)
    big, small = n == 2000, n == 20
    assert abs(shrunk[big & (p == 0.3)].mean() - 0.3) < 0.01
    raw = rng.binomial(n, p) / n
    assert np.abs(shrunk[small] - mu).mean() < np.abs(raw[small] - mu).mean()


def test_shrink_count_rate_and_mean():
    rng = np.random.default_rng(2)
    exposure = np.r_[np.full(50, 0.2), np.full(50, 20.0)]
    rate = np.tile([1.0, 5.0], 50)
    count = rng.poisson(rate * exposure)
    shrunk, mu = shrink_count_rate(count, exposure)
    # Long exposures keep their rate; short ones are pulled towards the mean.
    np.testing.assert_allclose(shrunk[50:][rate[50:] == 5.0].mean(), 5.0, rtol=0.05)
    short = count[:50] / exposure[:50]
    assert np.abs(shrunk[:50] - mu).mean() < np.abs(short - mu).mean()
    assert np.isnan(shrink_count_rate(np.array([1, 0]), np.array([1.0, 0.0]))[0][1])

    means, n = np.array([0.0, 1.0, 10.0]), np.array([1000, 1000, 1])
    shrunk, mu = shrink_mean(means, n, within_var=25.0)
    assert abs(shrunk[2] - mu) < abs(10.0 - mu) / 2  # one noisy observation barely counts
    assert abs(shrunk[0] - 0.0) < 0.1


def test_bootstrap_ratio_covers_the_estimate():
    rng = np.random.default_rng(3)
    groups = np.repeat([0, 1], 300)
    den = rng.integers(5, 50, 600).astype(float)
    num = den * np.where(groups == 0, 2.0, 5.0) + rng.normal(0, 5, 600)
    lo, hi = bootstrap_ratio(num, den, groups, 2, CONFIG)
    for g, true in ((0, 2.0), (1, 5.0)):
        assert lo[g] < true < hi[g] and hi[g] - lo[g] < 0.5


def test_position_groups():
    assert [position_group(c) for c in ("RCB", "LWB", "CF", "LM", "UNK", None, "XX")] == [
        "CB", "FB", "ST", "W", "UNK", "UNK", "UNK"]


# --- A small synthetic tournament: three teams, a round robin, one match each pairing ---------

TEAMS = ("Alpha", "Beta", "Gamma")


def fake_match(home: str, away: str, seed: int):
    """Possessions alternate; `Alpha` finds twice the space and exploits more."""
    rng = np.random.default_rng(seed)
    n_poss, frames_per = 40, 25
    side = np.arange(n_poss) % 2
    timeline = []
    for pid in range(n_poss):
        team = (home, away)[side[pid]]
        scale = 2.0 if team == "Alpha" else 1.0
        total = rng.gamma(4, scale, frames_per).astype(np.float32)
        zones = rng.dirichlet(np.ones(4), frames_per) * total[:, None]
        timeline.append(pd.DataFrame({
            "status": "ok", "possession_id": pid, "possession_side": side[pid],
            "transition": np.arange(frames_per) < 5,
            "third": pd.Categorical(rng.choice(["defensive", "middle", "final"], frames_per),
                                    categories=["defensive", "middle", "final"]),
            "total": total, "behind": zones[:, 0], "between": zones[:, 1],
            "wide": zones[:, 2], "in_front": zones[:, 3],
            "compactness": rng.normal(15, 2, frames_per),
            "block_width": rng.normal(40, 3, frames_per),
        }))
    timeline = pd.concat(timeline, ignore_index=True)
    n_actions = rng.integers(2, 8, n_poss)
    rate = np.where(np.array([(home, away)[s] for s in side]) == "Alpha", 0.3, 0.05)
    possessions = pd.DataFrame({
        "possession_id": np.arange(n_poss), "team_side": side.astype(np.int8),
        "start_type": pd.Categorical(np.where(np.arange(n_poss) % 3 == 0, "regain", "restart")),
        "xspace_integral": timeline.groupby("possession_id")["total"].sum().to_numpy() / 5,
        "n_actions": n_actions, "n_exploited": rng.binomial(n_actions, rate),
        "xt_gained": rng.normal(0, 0.02, n_poss), "first_exploit_s": rng.uniform(0, 10, n_poss),
        "shot": rng.random(n_poss) < 0.1, "goal": rng.random(n_poss) < 0.02,
        "box_entry": rng.random(n_poss) < 0.2, "final_third": rng.random(n_poss) < 0.4,
    })
    n_act = 120
    a_side = rng.integers(0, 2, n_act).astype(np.int8)
    players = [f"{t}{i}" for t in (home, away) for i in range(4)]
    actor = [players[s * 4 + rng.integers(0, 4)] for s in a_side]
    owner = [players[s * 4 + rng.integers(0, 4)] for s in a_side]
    actions = pd.DataFrame({
        "status": "ok", "team_side": a_side, "player_id": pd.array(actor, dtype="string"),
        "type": rng.choice(["pass", "cross", "carry"], n_act),
        "success": pd.array(rng.random(n_act) < 0.8, dtype="boolean"),
        "exploited": rng.random(n_act) < 0.1, "missed": rng.random(n_act) < 0.05,
        "xt_gained": rng.normal(0, 0.02, n_act), "chosen": rng.exponential(0.01, n_act),
        "likely": rng.exponential(0.02, n_act), "likely_gap": rng.exponential(0.01, n_act),
        "likely_x": rng.uniform(-52, 52, n_act), "likely_y": rng.uniform(-34, 34, n_act),
        "owner_id": pd.array(owner, dtype="string"),
    })
    pl = pd.DataFrame({
        "team_side": np.repeat([0, 1], 4).astype(np.int8),
        "player_id": pd.array(players, dtype="string"), "position": "UNK",
        "minutes": 60.0, "n_actions": 10, "n_exploited": 1, "n_missed": 0, "xt_gained": 0.1,
        "n_likely_space": rng.integers(0, 30, 8), "n_likely_found": 1, "n_best_space": 5,
    })
    return possessions, timeline, actions, pl


def tournament():
    poss, acts, players, directory = [], [], [], []
    for i, (home, away) in enumerate([("Alpha", "Beta"), ("Beta", "Gamma"),
                                      ("Gamma", "Alpha"), ("Beta", "Alpha")]):
        p, tl, a, pl = fake_match(home, away, i)
        ids = {"source": "test", "match_id": str(i)}
        poss.append(possession_table(p, tl, (home, away)).assign(**ids))
        acts.append(action_table(a, (home, away)).assign(**ids))
        pl["team"] = pl["team_side"].map({0: home, 1: away})
        players.append(pl.assign(**ids))
        directory.append(pd.DataFrame({
            "player_id": pl["player_id"], "name": pl["player_id"],
            "position": ["GK", "CB", "CM", "CF"] * 2}).assign(**ids))
    return (pd.concat(poss, ignore_index=True), pd.concat(acts, ignore_index=True),
            pd.concat(players, ignore_index=True), pd.concat(directory, ignore_index=True))


def test_possession_table_sums():
    p, tl, _, _ = fake_match("Alpha", "Beta", 0)
    df = possession_table(p, tl, ("Alpha", "Beta"))
    assert list(df["team"][:2]) == ["Alpha", "Beta"] and list(df["opponent"][:2]) == [
        "Beta", "Alpha"]
    np.testing.assert_allclose(df["sum_total"].sum(), tl["total"].sum(), rtol=1e-5)
    zones = df[["sum_behind", "sum_between", "sum_wide", "sum_in_front"]].sum(axis=1)
    np.testing.assert_allclose(zones, df["sum_total"], rtol=1e-4)
    assert (df["n_transition"] == 5).all() and (df["n_frames"] == 25).all()
    thirds = df[["n_defensive", "n_middle", "n_final"]].sum(axis=1)
    assert (thirds == df["n_frames"]).all()


def test_team_match_ratings():
    poss, acts, _, _ = tournament()
    df = team_match_ratings(poss, acts, CONFIG).set_index(["match_id", "team"])
    assert len(df) == 8
    # Attack of one team is the defence of the other.
    np.testing.assert_allclose(df.loc[("0", "Alpha"), "xspace_created"],
                               df.loc[("0", "Beta"), "xspace_conceded"])
    assert df.loc[("0", "Alpha"), "opponent"] == "Beta"
    # Alpha creates about twice the space (gamma(4, 2) against gamma(4, 1) per frame).
    alpha = df.xs("Alpha", level="team")["xspace_created"]
    assert (alpha > 6).all() and (df.xs("Gamma", level="team")["xspace_created"] < 6).all()
    assert (df["xspace_created_lo"] < df["xspace_created"]).all()
    assert (df["xspace_created"] < df["xspace_created_hi"]).all()
    for g in ("attack_space", "attack_use", "defence_space", "defence_use"):
        assert df[f"grade_{g}"].between(0, 100).all()
    assert (df.xs("Alpha", level="team")["grade_attack_space"] > 50).all()
    assert (df["grade_attack_space_lo"] <= df["grade_attack_space"]).all()
    shares = df[[f"share_{z}" for z in ("behind", "between", "wide", "in_front")]].sum(axis=1)
    np.testing.assert_allclose(shares, 1, rtol=1e-4)
    assert df["summary"].str.startswith("Created").all()
    # Shrunk exploitation rates sit between each raw rate and the pooled rate.
    mu = df["n_exploited"].sum() / df["n_actions"].sum()
    raw, shrunk = df["exploit_rate"], df["exploit_rate_shrunk"]
    assert (((shrunk - mu) * (raw - mu)) >= -1e-12).all()
    assert ((shrunk - mu).abs() <= (raw - mu).abs() + 1e-12).all()


def test_team_profiles_and_leak_map():
    poss, acts, _, _ = tournament()
    tm = team_match_ratings(poss, acts, CONFIG)
    prof = team_profiles(poss, acts, tm, CONFIG).set_index("team")
    assert prof.loc["Alpha", "matches"] == 3 and prof.loc["Gamma", "matches"] == 2
    assert prof["small_sample"].all()
    assert prof.loc["Alpha", "xspace_created"] > prof.loc["Beta", "xspace_created"]
    assert prof["leaks_most"].isin(["behind", "between", "wide", "in_front"]).all()
    lm = leak_map(acts, CONFIG)
    assert len(lm) == 3 * 12 * 8
    np.testing.assert_allclose(lm.groupby("team")["share"].sum(), 1.0)


def test_player_ratings():
    poss, acts, players, directory = tournament()
    cfg = RatingsConfig(n_boot=10, min_carrier_passes=5, min_runner_minutes=100,
                        min_group_size=2, min_graded=2)
    df = player_ratings(players, acts, directory, cfg).set_index("player_id")
    a0 = df.loc["Alpha0"]  # GK, played all three of Alpha's matches
    assert a0["matches"] == 3 and a0["position_group"] == "GK" and a0["minutes"] == 180
    assert not a0["runner"] and np.isnan(a0["grade_runner_space"])
    np.testing.assert_allclose(df["space_held_per90"],
                               df["n_likely_space"] / (df["minutes"] / 90))
    st = df[df["position_group"] == "ST"]
    assert st["grade_runner_space"].notna().all()
    assert df.loc[df["carrier"], "grade_carrier_use"].between(0, 100).all()
    # Too few qualified players in a position: no grades.
    few = player_ratings(players, acts, directory, RatingsConfig(
        n_boot=10, min_carrier_passes=5, min_runner_minutes=100, min_graded=10))
    assert few["grade_runner_space"].isna().all()
    # A reception counts only for team-mates' completed actions into the player's cell.
    rec = acts[acts["success"].fillna(False).astype(bool) & (acts["chosen"] >= cfg.high_xspace)
               & (acts["owner_id"] != acts["player_id"])]
    assert df["n_high_receptions"].sum() == len(rec)
