"""Phase 5 ratings: every team in every match, team profiles and player ratings.

Everything is an aggregate of the per-match outputs (timeline, actions, possessions, players);
no new physics. Ratings describe **space and progression, not danger**: the calibrated physics
predicts where passes go and whether they arrive, but the Phases 0-3 physics predicted danger
better (V3; docs/validation.md "Decision: keep the calibrated physics").

Building blocks, all concatenated over matches with `source`, `match_id`, `team`, `opponent`:

- `possession_table`: one row per possession, the possession file plus sums of the 5 Hz
  timeline (xSpace per zone, in transition, per third, defensive shape). Seen from `team` it is
  space created; from `opponent`, space conceded. Ratios of sums (Σ xSpace / Σ frames) are
  time averages, and resampling possessions gives their uncertainty.
- `action_table`: the action file with team names.

Outputs:

| Function | One row per | Notes |
|---|---|---|
| `team_match_ratings` | team-match | attack, defence, 80% intervals, 0-100 grades, a summary |
| `team_profiles` | team | the same pooled over a team's matches, phase split, regains |
| `leak_map` | team × pitch cell | where opponents' likely best option was, against the average |
| `player_ratings` | player | ball carrier and runner metrics, raw and shrunk, grades in position |

Small samples: the exploitation rate is unreliable per team-match (V4 reliability 0.37-0.54),
so it is shrunk towards the mean with an empirical-Bayes beta-binomial before it is graded.
Player rates are shrunk towards their position's mean (beta-binomial for rates, gamma-Poisson
for per-90 counts, normal for mean decision gaps). Raw values are kept beside shrunk ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xspace.config import DEFAULT_EXPLOITATION, DEFAULT_TIMELINE
from xspace.constants import PITCH_LENGTH, PITCH_WIDTH
from xspace.metrics.space import ZONES
from xspace.phases.possession import THIRDS


@dataclass(frozen=True)
class RatingsConfig:
    n_boot: int = 1000  # bootstrap resamples of possessions
    interval: float = 0.8  # width of the intervals
    seed: int = 0
    min_carrier_passes: int = 50  # open-play passes + crosses to rate a ball carrier
    # Live minutes to rate a runner: ≈ 180 match minutes (a match has about 55-60 live).
    min_runner_minutes: float = 110.0
    min_group_size: int = 10  # players with exposure needed for a position's own prior
    min_graded: int = 5  # qualified players a position needs before it gets grades
    high_xspace: float = DEFAULT_EXPLOITATION.exploit_min  # a reception worth counting
    leak_cells: tuple[int, int] = (12, 8)  # leak map grid (x, y), the xT grid's shape


DEFAULT_RATINGS = RatingsConfig()

# Position codes (PFF rosters, kloppy) → groups used for priors and grades.
POSITION_GROUPS = {
    "GK": "GK",
    "CB": "CB", "LCB": "CB", "RCB": "CB", "MCB": "CB",
    "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
    "DM": "DM", "LDM": "DM", "RDM": "DM", "CDM": "DM",
    "CM": "CM", "LCM": "CM", "RCM": "CM",
    "AM": "AM", "CAM": "AM",
    "LW": "W", "RW": "W", "LM": "W", "RM": "W", "LF": "W", "RF": "W",
    "ST": "ST", "CF": "ST",
}
GRADE_BANDS = ((80, "a lot of"), (60, "more than average"), (40, "average"),
               (20, "less than average"), (0, "little"))
ZONE_WORDS = {"behind": "behind the defence", "between": "between the lines",
              "wide": "wide of the block", "in_front": "in front of the block"}


def position_group(code: str | None) -> str:
    return POSITION_GROUPS.get(code or "UNK", "UNK")


# --- Building blocks ---------------------------------------------------------------------------

def possession_table(possessions: pd.DataFrame, timeline: pd.DataFrame,
                     teams: tuple[str, str]) -> pd.DataFrame:
    """One match's possessions with timeline sums. `teams` = (home, away) names.

    Added columns: `team`, `opponent`, and per possession `n_frames` (computed frames),
    `sum_total`, `sum_<zone>`, `n_transition` / `sum_transition`, `n_<third>` / `sum_<third>`
    (by the ball's third), `n_shape` / `sum_compactness` / `sum_block_width` (the defending
    team's shape).
    """
    tl = timeline[timeline["status"] == "ok"]
    g = tl.groupby("possession_id")
    sums = pd.DataFrame({"n_frames": g.size(), "sum_total": g["total"].sum(),
                         **{f"sum_{z}": g[z].sum() for z in ZONES}})
    tr = tl[tl["transition"].astype(bool)].groupby("possession_id")["total"]
    sums["n_transition"], sums["sum_transition"] = tr.size(), tr.sum()
    for third in THIRDS:
        t = tl[tl["third"] == third].groupby("possession_id")["total"]
        sums[f"n_{third}"], sums[f"sum_{third}"] = t.size(), t.sum()
    shape = tl.dropna(subset=["compactness", "block_width"]).groupby("possession_id")
    sums["n_shape"] = shape.size()
    sums["sum_compactness"] = shape["compactness"].sum()
    sums["sum_block_width"] = shape["block_width"].sum()

    df = possessions.set_index("possession_id").join(sums)
    df[sums.columns] = df[sums.columns].fillna(0.0)
    side = df["team_side"].to_numpy()
    df.insert(0, "team", np.where(side == 0, teams[0], teams[1]))
    df.insert(1, "opponent", np.where(side == 0, teams[1], teams[0]))
    return df.reset_index()


def action_table(actions: pd.DataFrame, teams: tuple[str, str]) -> pd.DataFrame:
    """One match's computed actions with `team` / `opponent` names."""
    df = actions[actions["status"] == "ok"].copy()
    side = df["team_side"].to_numpy()
    df.insert(0, "team", np.where(side == 0, teams[0], teams[1]))
    df.insert(1, "opponent", np.where(side == 0, teams[1], teams[0]))
    return df


# --- Statistics --------------------------------------------------------------------------------

def percentile(values: pd.Series, reference: pd.Series | None = None) -> pd.Series:
    """0-100 mid-rank percentile of each value among `reference` (default: the values)."""
    ref = np.sort((values if reference is None else reference).dropna().to_numpy())
    v = values.to_numpy(dtype=float)
    below = np.searchsorted(ref, v, side="left")
    at_or_below = np.searchsorted(ref, v, side="right")
    out = (below + at_or_below) / 2 / max(len(ref), 1) * 100
    return pd.Series(np.where(np.isnan(v), np.nan, out), index=values.index)


def bootstrap_ratio(num: np.ndarray, den: np.ndarray, groups: np.ndarray, n_groups: int,
                    config: RatingsConfig = DEFAULT_RATINGS) -> tuple[np.ndarray, np.ndarray]:
    """Interval for Σ num / Σ den per group, resampling rows (possessions) within each group.

    Poisson bootstrap: each row gets a Poisson(1) weight per resample, which approximates
    resampling n rows with replacement and lets every group share one pass over the rows.
    Returns (lo, hi) per group, `config.interval` wide.
    """
    rng = np.random.default_rng(config.seed)
    est = np.full((config.n_boot, n_groups), np.nan)
    for b in range(config.n_boot):
        w = rng.poisson(1.0, len(num))
        top = np.bincount(groups, weights=w * num, minlength=n_groups)
        bottom = np.bincount(groups, weights=w * den, minlength=n_groups)
        with np.errstate(invalid="ignore", divide="ignore"):
            est[b] = top / bottom
    tail = (1 - config.interval) / 2 * 100
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanpercentile(est, [tail, 100 - tail], axis=0)
    return lo, hi


def shrink_rate(k: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Empirical-Bayes beta-binomial: shrink k / n towards the pooled rate.

    Method of moments: with μ the pooled rate, the n-weighted variance of observed rates is
    τ² + m·μ(1−μ)/Σn for m units, where τ² is the variance of the true rates. The prior is a
    beta with mean μ and variance τ², i.e. κ = μ(1−μ)/τ² − 1 pseudo-trials. τ² ≤ 0 means the
    spread is all noise: everything goes to μ. Returns (shrunk, μ, κ).
    """
    k, n = np.asarray(k, float), np.asarray(n, float)
    use = n > 0
    if not use.any():
        return np.full(len(k), np.nan), np.nan, np.nan
    mu = k[use].sum() / n[use].sum()
    p = k[use] / n[use]
    s2 = (n[use] * (p - mu) ** 2).sum() / n[use].sum()
    tau2 = s2 - use.sum() * mu * (1 - mu) / n[use].sum()
    kappa = mu * (1 - mu) / tau2 - 1 if tau2 > 0 else np.inf
    if np.isinf(kappa):
        return np.full(len(k), mu), mu, kappa
    return (k + mu * kappa) / (n + kappa), mu, kappa


def shrink_count_rate(count: np.ndarray, exposure: np.ndarray) -> tuple[np.ndarray, float]:
    """Empirical-Bayes gamma-Poisson: shrink count / exposure towards the pooled rate.

    Same method of moments as `shrink_rate`, with Poisson noise (variance μ / exposure). The
    gamma prior has mean μ and variance τ²: shape μ²/τ², rate μ/τ². Returns (shrunk, μ).
    """
    c, e = np.asarray(count, float), np.asarray(exposure, float)
    use = e > 0
    if not use.any():
        return np.full(len(c), np.nan), np.nan
    mu = c[use].sum() / e[use].sum()
    r = c[use] / e[use]
    s2 = (e[use] * (r - mu) ** 2).sum() / e[use].sum()
    tau2 = s2 - use.sum() * mu / e[use].sum()
    if tau2 <= 0 or mu <= 0:
        return np.where(use, mu, np.nan), mu
    shape, rate = mu ** 2 / tau2, mu / tau2
    return np.where(use, (c + shape) / (e + rate), np.nan), mu


def shrink_mean(mean: np.ndarray, n: np.ndarray, within_var: float) -> tuple[np.ndarray, float]:
    """Empirical-Bayes normal-normal: shrink per-unit means of n observations towards the
    n-weighted grand mean. `within_var` is the variance of one observation around its unit's
    mean. Returns (shrunk, grand mean)."""
    x, n = np.asarray(mean, float), np.asarray(n, float)
    use = (n > 0) & ~np.isnan(x)
    if not use.any():
        return np.full(len(x), np.nan), np.nan
    mu = (n[use] * x[use]).sum() / n[use].sum()
    s2 = (n[use] * (x[use] - mu) ** 2).sum() / n[use].sum()
    tau2 = s2 - use.sum() * within_var / n[use].sum()
    if tau2 <= 0:
        return np.where(use, mu, np.nan), mu
    weight = tau2 / (tau2 + within_var / np.where(use, n, 1))
    return np.where(use, mu + weight * (x - mu), np.nan), mu


# --- Team ratings ------------------------------------------------------------------------------

def _sums(poss: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Possession sums per group: the numerators and denominators of every team metric."""
    cols = ["n_frames", "sum_total", *[f"sum_{z}" for z in ZONES], "n_transition",
            "sum_transition", *[c for t in THIRDS for c in (f"n_{t}", f"sum_{t}")],
            "n_shape", "sum_compactness", "sum_block_width", "xspace_integral",
            "n_actions", "n_exploited", "xt_gained"]
    g = poss.groupby(by, observed=True)
    out = g[cols].sum()
    out["possessions"] = poss[poss["n_frames"] > 0].groupby(by, observed=True).size()
    out["possessions"] = out["possessions"].fillna(0).astype(int)
    for flag in ("shot", "goal", "box_entry", "final_third"):
        out[f"n_{flag}"] = g[flag].sum().astype(int)
    out["regains"] = g["start_type"].apply(lambda s: int((s == "regain").sum()))
    return out


def _space_metrics(s: pd.DataFrame) -> pd.DataFrame:
    """Rates from `_sums`: xSpace per computed frame (= per second of possession), zone shares,
    transition / settled, and the phase split by the ball's third."""
    def ratio(a, b):
        return (s[a] / s[b].where(s[b] > 0)).astype(float)

    out = pd.DataFrame(index=s.index)
    out["xspace"] = ratio("sum_total", "n_frames")
    out["xspace_per_possession"] = ratio("xspace_integral", "possessions")
    for z in ZONES:
        out[f"xspace_{z}"] = ratio(f"sum_{z}", "n_frames")
        out[f"share_{z}"] = ratio(f"sum_{z}", "sum_total")
    out["xspace_transition"] = ratio("sum_transition", "n_transition")
    settled_n = s["n_frames"] - s["n_transition"]
    out["xspace_settled"] = ((s["sum_total"] - s["sum_transition"])
                             / settled_n.where(settled_n > 0)).astype(float)
    for t in THIRDS:
        out[f"xspace_{t}_third"] = ratio(f"sum_{t}", f"n_{t}")
    out["compactness"] = ratio("sum_compactness", "n_shape")
    out["block_width"] = ratio("sum_block_width", "n_shape")
    out["exploit_rate"] = ratio("n_exploited", "n_actions")
    return out


def _action_metrics(act: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = act.groupby(by, observed=True)
    exploited = act[act["exploited"]].groupby(by, observed=True)["xt_gained"]
    return pd.DataFrame({"xt_exploited": exploited.sum(), "n_missed": g["missed"].sum(),
                         "likely_gap": g["likely_gap"].mean()})


def _attack_defence(poss: pd.DataFrame, act: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Attack (team in possession) and defence (opponent's possessions) side by side, one row
    per `keys` + team."""
    att_by, def_by = [*keys, "team"], [*keys, "opponent"]
    att_s, def_s = _sums(poss, att_by), _sums(poss, def_by)
    att = _space_metrics(att_s).join(_action_metrics(act, att_by))
    dfn = _space_metrics(def_s).join(_action_metrics(act, def_by))
    def_s.index = def_s.index.rename(att_by)
    dfn.index = dfn.index.rename(att_by)

    out = pd.DataFrame(index=att.index)
    out["possessions"] = att_s["possessions"]
    out["live_minutes"] = att_s["n_frames"] / DEFAULT_TIMELINE.sample_hz / 60
    for c in ["xspace", "xspace_per_possession", *[f"xspace_{z}" for z in ZONES],
              *[f"share_{z}" for z in ZONES], "xspace_transition", "xspace_settled",
              *[f"xspace_{t}_third" for t in THIRDS], "exploit_rate", "xt_exploited",
              "likely_gap"]:
        name = c.replace("xspace", "xspace_created", 1) if c.startswith("xspace") else c
        out[name] = att[c]
    out["n_actions"], out["n_exploited"] = att_s["n_actions"], att_s["n_exploited"]
    out["n_missed"] = att["n_missed"].fillna(0).astype(int)
    out["xt_gained"] = att_s["xt_gained"]
    for flag in ("shot", "goal", "box_entry"):
        out[f"n_{flag}"] = att_s[f"n_{flag}"]
    out["regains"] = att_s["regains"]

    out["opp_possessions"] = def_s["possessions"]
    for c in ["xspace", "xspace_per_possession", *[f"xspace_{z}" for z in ZONES],
              *[f"share_{z}" for z in ZONES], "xspace_transition", "xspace_settled",
              *[f"xspace_{t}_third" for t in THIRDS]]:
        prefix = "xspace" if c.startswith("xspace") else "share"
        out[c.replace(prefix, f"{prefix}_conceded", 1)] = dfn[c]
    out["compactness"], out["block_width"] = dfn["compactness"], dfn["block_width"]
    out["opp_exploit_rate"] = dfn["exploit_rate"]
    out["opp_n_actions"], out["opp_n_exploited"] = def_s["n_actions"], def_s["n_exploited"]
    out["opp_xt_exploited"] = dfn["xt_exploited"]
    for flag in ("shot", "goal", "box_entry"):
        out[f"n_{flag}_against"] = def_s[f"n_{flag}"]
    return out


def _intervals(poss: pd.DataFrame, index: pd.MultiIndex, keys: list[str],
               config: RatingsConfig) -> pd.DataFrame:
    """80% intervals (by default) for the headline ratios, resampling possessions."""
    out = pd.DataFrame(index=index)
    specs = {  # name: (numerator, denominator, attack side?)
        "xspace_created": ("sum_total", "n_frames", True),
        "exploit_rate": ("n_exploited", "n_actions", True),
        "xspace_conceded": ("sum_total", "n_frames", False),
        "xspace_conceded_behind": ("sum_behind", "n_frames", False),
        "opp_exploit_rate": ("n_exploited", "n_actions", False),
    }
    for name, (num, den, attack) in specs.items():
        cols = [*keys, "team" if attack else "opponent"]
        codes = index.get_indexer(pd.MultiIndex.from_frame(poss[cols]))
        ok = codes >= 0
        lo, hi = bootstrap_ratio(poss[num].to_numpy(float)[ok], poss[den].to_numpy(float)[ok],
                                 codes[ok], len(index), config)
        out[f"{name}_lo"], out[f"{name}_hi"] = lo, hi
    return out


def _band(grade: float) -> str:
    return next(word for cut, word in GRADE_BANDS if grade >= cut)


def _lead_zone(row: pd.Series, prefix: str, average: dict[str, float]) -> str | None:
    """The zone whose share most exceeds the dataset average, if by at least 15%."""
    lifts = {z: row[f"{prefix}{z}"] / average[z] for z in ZONES if average[z] > 0}
    zone, lift = max(lifts.items(), key=lambda kv: kv[1])
    return zone if lift >= 1.15 else None


def summarise(row: pd.Series, avg_att: dict[str, float], avg_def: dict[str, float]) -> str:
    """One line, e.g. "Created a lot of space, mostly between the lines, and used it rarely.
    Conceded little; opponents rarely used it."."""
    def use(grade: float) -> str:
        return ("often" if grade >= 70 else "rarely" if grade < 30 else "about as often as most")

    zone = _lead_zone(row, "share_", avg_att)
    attack = f"Created {_band(row['grade_attack_space'])} space"
    if zone:
        attack += f", mostly {ZONE_WORDS[zone]}"
    attack += f", and used it {use(row['grade_attack_use'])}."
    # Defence grades are "higher is better", so conceding a lot is a low grade.
    conceded = _band(100 - row["grade_defence_space"])
    zone = _lead_zone(row, "share_conceded_", avg_def)
    defence = f"Conceded {conceded} space"
    if zone:
        defence += f", mostly {ZONE_WORDS[zone]}"
    defence += f"; opponents used it {use(100 - row['grade_defence_use'])}."
    return f"{attack} {defence}"


def _grades(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """0-100 grades (higher = better for the team) with intervals mapped through the same
    reference distribution: the interval's ends graded like any other value. Shrunk rates get
    no grade interval (their raw interval isn't on the shrunk scale); the rate keeps its own."""
    specs = {  # grade: (value column, higher is better)
        "attack_space": ("xspace_created", True),
        "attack_use": ("exploit_rate_shrunk", True),
        "defence_space": ("xspace_conceded", False),
        "defence_behind": ("xspace_conceded_behind", False),
        "defence_use": ("opp_exploit_rate_shrunk", False),
    }
    out = pd.DataFrame(index=df.index)
    for grade, (col, higher) in specs.items():
        ref = reference[col]
        g = percentile(df[col], ref)
        out[f"grade_{grade}"] = g if higher else 100 - g
        if f"{col}_lo" in df:
            lo, hi = percentile(df[f"{col}_lo"], ref), percentile(df[f"{col}_hi"], ref)
            out[f"grade_{grade}_lo"] = lo if higher else 100 - hi
            out[f"grade_{grade}_hi"] = hi if higher else 100 - lo
    return out


def team_match_ratings(poss: pd.DataFrame, act: pd.DataFrame,
                       config: RatingsConfig = DEFAULT_RATINGS) -> pd.DataFrame:
    """One row per team per match (open play): attack, defence, intervals, grades, summary.

    `poss` / `act`: `possession_table` / `action_table` outputs over all matches, with `source`
    and `match_id` columns. Grades are percentiles against every team-match in `poss`.
    """
    keys = ["source", "match_id"]
    df = _attack_defence(poss, act, keys)
    df = df.join(_intervals(poss, df.index, keys, config))
    df["exploit_rate_shrunk"] = shrink_rate(df["n_exploited"], df["n_actions"])[0]
    df["opp_exploit_rate_shrunk"] = shrink_rate(df["opp_n_exploited"], df["opp_n_actions"])[0]
    df = df.join(_grades(df, df))
    avg_att = {z: df[f"share_{z}"].mean() for z in ZONES}
    avg_def = {z: df[f"share_conceded_{z}"].mean() for z in ZONES}
    df["summary"] = df.apply(summarise, axis=1, args=(avg_att, avg_def))
    df = df.reset_index()
    opp = df.set_index([*keys, "team"])
    df.insert(3, "opponent", [
        next(t for t in opp.loc[(s, m)].index if t != team)
        for s, m, team in df[[*keys, "team"]].itertuples(index=False)])
    df["goals_for"], df["goals_against"] = df["n_goal"], df["n_goal_against"]
    return df


def team_profiles(poss: pd.DataFrame, act: pd.DataFrame, team_matches: pd.DataFrame,
                  config: RatingsConfig = DEFAULT_RATINGS) -> pd.DataFrame:
    """One row per team, pooled over its matches: the team-match metrics as time averages over
    all its possessions, with intervals, grades against the team-match distribution, the phase
    split, and what happens after a regain."""
    keys = ["source"]
    df = _attack_defence(poss, act, keys)
    df = df.join(_intervals(poss, df.index, keys, config))
    df["exploit_rate_shrunk"] = shrink_rate(df["n_exploited"], df["n_actions"])[0]
    df["opp_exploit_rate_shrunk"] = shrink_rate(df["opp_n_exploited"], df["opp_n_actions"])[0]
    # Grades against teams, so a team's average is compared with other teams' averages.
    df = df.join(_grades(df, df))
    n = team_matches.groupby(["source", "team"]).size().rename("matches")
    df = df.join(n)

    # Regains: how often an open-play regain leads to an exploited action, and how fast.
    reg = poss[poss["start_type"] == "regain"]
    g = reg.groupby(["source", "team"])
    df["regain_exploit_share"] = g["n_exploited"].apply(lambda s: float((s > 0).mean()))
    df["regain_first_exploit_s"] = g["first_exploit_s"].median()

    # Where they leak: conceded zone shares against the average team.
    for z in ZONES:
        df[f"leak_{z}"] = df[f"share_conceded_{z}"] / df[f"share_conceded_{z}"].mean()
    df["leaks_most"] = df[[f"leak_{z}" for z in ZONES]].idxmax(axis=1).str.removeprefix("leak_")
    df["small_sample"] = df["matches"] < 4
    return df.reset_index()


def leak_map(act: pd.DataFrame, config: RatingsConfig = DEFAULT_RATINGS) -> pd.DataFrame:
    """Per team and pitch cell (the defending team's view; opponents attack +x): the share of
    the opponents' likely best options (`likely` cell, weighted by its xSpace) that fell in the
    cell, and its lift over the average team. A heat map of where a defence leaves its most
    usable space, from action moments only (the timeline keeps no per-cell surfaces)."""
    nx, ny = config.leak_cells
    a = act[act["likely"] > 0]
    ix = np.clip(((a["likely_x"] + PITCH_LENGTH / 2) / PITCH_LENGTH * nx).astype(int), 0, nx - 1)
    iy = np.clip(((a["likely_y"] + PITCH_WIDTH / 2) / PITCH_WIDTH * ny).astype(int), 0, ny - 1)
    df = pd.DataFrame({"source": a["source"], "team": a["opponent"], "cell_x": ix,
                       "cell_y": iy, "value": a["likely"]})
    cells = df.groupby(["source", "team", "cell_x", "cell_y"])["value"].sum()
    teams = cells.index.droplevel(["cell_x", "cell_y"]).unique()
    full = pd.MultiIndex.from_tuples(
        [(s, t, x, y) for s, t in teams for x in range(nx) for y in range(ny)],
        names=cells.index.names)
    cells = cells.reindex(full, fill_value=0.0).rename("value").reset_index()
    cells["share"] = cells["value"] / cells.groupby(["source", "team"])["value"].transform("sum")
    avg = cells.groupby(["source", "cell_x", "cell_y"])["share"].transform("mean")
    cells["lift"] = cells["share"] / avg.where(avg > 0)
    cells["x_min"] = cells["cell_x"] * PITCH_LENGTH / nx - PITCH_LENGTH / 2
    cells["y_min"] = cells["cell_y"] * PITCH_WIDTH / ny - PITCH_WIDTH / 2
    return cells


# --- Player ratings ----------------------------------------------------------------------------

def _shrink_by_group(df: pd.DataFrame, fn, cols: list[str], config: RatingsConfig,
                     **kwargs) -> np.ndarray:
    """Apply a shrinkage function per position group; groups with too few players (and UNK)
    use the prior fitted on everyone."""
    out = np.full(len(df), np.nan)
    exposure = df[cols[1]].to_numpy(float)
    everyone = fn(*(df[c].to_numpy(float) for c in cols), **kwargs)[0]
    for group, idx in df.groupby("position_group").indices.items():
        if group == "UNK" or (exposure[idx] > 0).sum() < config.min_group_size:
            out[idx] = everyone[idx]
        else:
            sub = df.iloc[idx]
            out[idx] = fn(*(sub[c].to_numpy(float) for c in cols), **kwargs)[0]
    return out


def player_ratings(players: pd.DataFrame, act: pd.DataFrame, directory: pd.DataFrame,
                   config: RatingsConfig = DEFAULT_RATINGS) -> pd.DataFrame:
    """One row per player over all their matches.

    `players`: per-match player files (with `source`, `match_id`, `team`); `act`: the
    `action_table` over all matches; `directory`: `player_directory` rows with `source`,
    `match_id`. Per 90 means per 90 *live* minutes (a match has about 55-60).

    Ball carriers (≥ `min_carrier_passes` open-play passes + crosses): exploitation rate,
    mean likely gap (lower is better), missed chances per 90. Runners (≥ `min_runner_minutes`):
    space held per 90 (team-mates' actions whose likely best cell they owned) and high-xSpace
    receptions per 90 (team-mates' completed actions into their cell worth ≥ `high_xspace`).
    Each has a raw and a `_shrunk` value; grades are percentiles among qualified players of
    the same position group, on shrunk values (none if fewer than `min_graded` qualify).
    """
    keys = ["source", "team", "player_id"]
    pm = players.merge(directory, on=["source", "match_id", "player_id"], how="left",
                       suffixes=("_tracking", ""))
    pm["position"] = pm["position"].fillna(pm["position_tracking"])
    pm["position_group"] = pm["position"].map(position_group)

    # The player's group is where they played most minutes (UNK only if nothing else).
    known = pm[pm["position_group"] != "UNK"]
    main = (known.groupby([*keys, "position_group"])["minutes"].sum().reset_index()
            .sort_values("minutes").groupby(keys).tail(1).set_index(keys)["position_group"])
    names = pm.dropna(subset=["name"]).groupby(keys)["name"].first()
    sums = pm.groupby(keys)[["minutes", "n_actions", "n_exploited", "n_missed", "xt_gained",
                             "n_likely_space", "n_likely_found", "n_best_space"]].sum()
    df = sums.join(pm.groupby(keys)["match_id"].nunique().rename("matches"))
    df = df.join(names).join(main)
    df["position_group"] = df["position_group"].fillna("UNK")

    passes = act[act["type"].isin(["pass", "cross"])].groupby(
        ["source", "team", "player_id"]).size().rename("n_passes")
    g = act.groupby(["source", "team", "player_id"])["likely_gap"]
    gap = pd.DataFrame({"likely_gap": g.mean(), "n_gap": g.count()})
    completed = act["success"].fillna(False).astype(bool)
    rec = act[completed & (act["chosen"] >= config.high_xspace)
              & (act["owner_id"] != act["player_id"])]
    rec = rec.groupby(["source", "team", rec["owner_id"].rename("player_id")]).size()
    df = df.join(passes).join(gap).join(rec.rename("n_high_receptions"))
    for c in ("n_passes", "n_gap", "n_high_receptions"):
        df[c] = df[c].fillna(0).astype(int)
    df = df.reset_index()

    p90 = df["minutes"] / 90
    df["exploit_rate"] = df["n_exploited"] / df["n_actions"].where(df["n_actions"] > 0)
    df["missed_per90"] = df["n_missed"] / p90
    df["space_held_per90"] = df["n_likely_space"] / p90
    df["receptions_per90"] = df["n_high_receptions"] / p90
    df["exploit_rate_shrunk"] = _shrink_by_group(df, shrink_rate, ["n_exploited", "n_actions"],
                                                 config)
    df["p90"] = p90
    for col, count in (("missed_per90", "n_missed"), ("space_held_per90", "n_likely_space"),
                       ("receptions_per90", "n_high_receptions")):
        df[f"{col}_shrunk"] = _shrink_by_group(df, shrink_count_rate, [count, "p90"], config)
    within = act.groupby(["source", "team", "player_id"])["likely_gap"].var().mean()
    df["likely_gap_shrunk"] = _shrink_by_group(df, shrink_mean, ["likely_gap", "n_gap"], config,
                                               within_var=float(within))
    df = df.drop(columns="p90")

    df["carrier"] = df["n_passes"] >= config.min_carrier_passes
    df["runner"] = (df["minutes"] >= config.min_runner_minutes) & (df["position_group"] != "GK")
    grades = {  # grade: (column, role, higher is better)
        "carrier_use": ("exploit_rate_shrunk", "carrier", True),
        "carrier_choice": ("likely_gap_shrunk", "carrier", False),
        "runner_space": ("space_held_per90_shrunk", "runner", True),
        "runner_receptions": ("receptions_per90_shrunk", "runner", True),
    }
    for grade, (col, role, higher) in grades.items():
        out = pd.Series(np.nan, index=df.index)
        for _, idx in df[df[role]].groupby("position_group").groups.items():
            if len(idx) < config.min_graded:
                continue
            g = percentile(df.loc[idx, col])
            out[idx] = g if higher else 100 - g
        df[f"grade_{grade}"] = out
    return df
