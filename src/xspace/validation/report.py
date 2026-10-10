"""Scores for validation tasks V1 and V3 from the ablation files (`validation.ablations`).

V1 (where does the pass go?): each surface becomes a softmax over cells,
P(cell) ∝ exp(β · s / max s − γ · d / 10 m), d the distance from the ball. β (and γ) are
picked on training matches; the test score is the mean log-likelihood of the actual end cell
minus that of a uniform guess (nats per pass: higher is better, 0 = no better than uniform),
plus the end cell's mean rank within its frame. Each surface is scored alone (γ = 0) and on top
of the distance prior; a `distance` row is the prior alone (β = 0).

V3 (does space now predict danger soon?): for each surface's frame total and best cell, the
AUC for a shot / box entry within 10 s in the same possession, and — the real test — what it
adds to ball position: logistic regressions on training matches with ball features only
(xT, x, |y|) and with ball features + the surface, compared on test log loss and AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from xspace.validation.ablations import BETAS, GAMMAS, SURFACES
from xspace.validation.pass_model import auc, log_loss

BALL_FEATURES = ("ball_xt", "ball_x", "abs_ball_y")


def _best(passes: pd.DataFrame, cols: list[str], params: list, tr: np.ndarray,
          te: np.ndarray, uniform: str) -> tuple:
    """The column with the best mean on train: (its parameters, test gain over uniform)."""
    j = int(np.nanargmax(passes.loc[tr, cols].mean().to_numpy()))
    return params[j], float((passes.loc[te, cols[j]] - passes.loc[te, uniform]).mean())


def v1_scores(passes: pd.DataFrame, train: np.ndarray, test: np.ndarray,
              subsets: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """One row per (subset, surface): β picked on train and the test gain over uniform, alone
    (`beta`, `gain_nats`) and with the distance prior (`beta_d`, `gamma`, `gain_with_distance`),
    and the end cell's mean rank."""
    subsets = subsets or {"all": np.ones(len(passes), dtype=bool)}
    ll = lambda s, b, g: f"ll_{s}_{b:g}_{g:g}"  # noqa: E731
    uniform = ll(SURFACES[0], 0, 0)  # β = γ = 0: the same for every surface
    ok = passes[uniform].notna().to_numpy()
    grid = [(b, g) for b in BETAS for g in GAMMAS]
    rows = []
    for name, mask in subsets.items():
        tr, te = train & mask & ok, test & mask & ok
        g0, gain0 = _best(passes, [ll(SURFACES[0], 0, g) for g in GAMMAS], list(GAMMAS), tr,
                          te, uniform)
        rows.append({"subset": name, "surface": "distance", "n_test": int(te.sum()),
                     "beta": 0.0, "gain_nats": 0.0, "beta_d": 0.0, "gamma": g0,
                     "gain_with_distance": gain0, "mean_rank": np.nan})
        for s in SURFACES:
            b, gain = _best(passes, [ll(s, b, 0) for b in BETAS], list(BETAS), tr, te, uniform)
            (bd, g), gain_d = _best(passes, [ll(s, *bg) for bg in grid], grid, tr, te, uniform)
            rows.append({
                "subset": name, "surface": s, "n_test": int(te.sum()), "beta": b,
                "gain_nats": gain, "beta_d": bd, "gamma": g, "gain_with_distance": gain_d,
                "mean_rank": float(passes.loc[te, f"rank_{s}"].mean()),
            })
    return pd.DataFrame(rows)


def _design(frames: pd.DataFrame, extra: str | None) -> np.ndarray:
    cols = [frames["ball_xt"], frames["ball_x"], frames["ball_y"].abs()]
    if extra is not None:
        cols.append(np.log1p(frames[extra].clip(lower=0) * 100))
    return np.column_stack(cols).astype(float)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> tuple[np.ndarray, ...]:
    """Logistic regression on standardised features. Returns (mean, sd, weights)."""
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])

    def nll(w):
        z = Z @ w
        loss = np.mean(np.logaddexp(0, z) - y * z) + l2 * (w[1:] ** 2).sum()
        grad = Z.T @ (1 / (1 + np.exp(-z)) - y) / len(y)
        grad[1:] += 2 * l2 * w[1:]
        return loss, grad

    w = minimize(nll, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
    return mu, sd, w


def predict_logistic(model: tuple[np.ndarray, ...], X: np.ndarray) -> np.ndarray:
    mu, sd, w = model
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    return 1 / (1 + np.exp(-(Z @ w)))


def v3_scores(frames: pd.DataFrame, train: np.ndarray, test: np.ndarray,
              target: str = "shot10") -> pd.DataFrame:
    """One row per feature: test AUC alone, and test log loss / AUC on top of ball position."""
    y = frames[target].to_numpy(dtype=float)
    base = fit_logistic(_design(frames[train], None), y[train])
    p_base = predict_logistic(base, _design(frames[test], None))
    rows = [{"feature": "ball only", "auc_alone": np.nan,
             "test_log_loss": log_loss(y[test], p_base), "test_auc": auc(y[test], p_base)}]
    features = ["ball_xt"] + [f"{k}_{s}" for s in SURFACES for k in ("total", "best")]
    for f in features:
        alone = auc(y[test], frames.loc[test, f].to_numpy())
        if f == "ball_xt":
            rows[0]["auc_alone"] = alone
            continue
        model = fit_logistic(_design(frames[train], f), y[train])
        p = predict_logistic(model, _design(frames[test], f))
        rows.append({"feature": f, "auc_alone": alone, "test_log_loss": log_loss(y[test], p),
                     "test_auc": auc(y[test], p)})
    df = pd.DataFrame(rows)
    df["log_loss_gain_pct"] = 100 * (1 - df["test_log_loss"] / df["test_log_loss"].iloc[0])
    return df


# --- V4: is the team signal stable? ---------------------------------------------------------

V4_METRICS = {
    # name: (column, weight); see _weighted
    "xspace": ("xspace_mean", "n_ok"),  # time-weighted mean xSpace in possession
    "peak": ("xspace_max", None),  # mean peak xSpace per possession
    "exploit_rate": ("n_exploited", "n_actions"),  # exploited actions / actions
}


def _weighted(df: pd.DataFrame, col: str, weight: str | None) -> float:
    ok = df[col].notna()
    if weight is None:
        return float(df.loc[ok, col].mean())
    if weight == "n_actions":  # a rate: Σ exploited / Σ actions
        return float(df[col].sum() / max(df[weight].sum(), 1))
    w = df.loc[ok, weight]
    return float((df.loc[ok, col] * w).sum() / max(w.sum(), 1))


def v4_split_half(poss: pd.DataFrame, min_matches: int = 3) -> pd.DataFrame:
    """Odd vs even possessions (per team per match, in time order): the correlation of each
    team metric between the halves, with the Spearman-Brown reliability 2r / (1 + r).

    At team-match level a team's "created" is its opponent's "conceded", so only created is
    reported. Per team over the tournament (teams with at least `min_matches`), both are: what
    a team created in possession, and what it conceded when its opponents had the ball.
    `poss` needs `match_id`, `team` (in possession), `opponent` and the possession columns.
    """
    poss = poss.sort_values(["match_id", "team", "start_frame"]).copy()
    poss["half"] = poss.groupby(["match_id", "team"]).cumcount() % 2
    n_matches = poss.groupby("team")["match_id"].nunique()
    rows = []
    for level, keys, sides in (("team-match", ["match_id"], (("created", "team"),)),
                               ("team", [], (("created", "team"), ("conceded", "opponent")))):
        for side, who in sides:
            for name, (col, w) in V4_METRICS.items():
                g = (poss.groupby([*keys, who, "half"])
                     .apply(lambda d, c=col, w=w: _weighted(d, c, w), include_groups=False)
                     .unstack("half"))
                if level == "team":
                    g = g[n_matches.reindex(g.index).to_numpy() >= min_matches]
                g = g.dropna()
                with np.errstate(invalid="ignore", divide="ignore"):  # constant halves: NaN
                    r = float(np.corrcoef(g[0], g[1])[0, 1]) if len(g) > 2 else np.nan
                rows.append({"level": level, "metric": name, "side": side, "n": len(g),
                             "r": r, "reliability": 2 * r / (1 + r)})
    return pd.DataFrame(rows)
