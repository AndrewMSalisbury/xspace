"""Scores for validation tasks V1 and V3 from the ablation files (`validation.ablations`).

V1 (where does the pass go?): each surface becomes a softmax over cells,
P(cell) ∝ exp(β · s / max s). β is picked on training matches; the test score is the mean
log-likelihood of the actual end cell minus that of a uniform guess (nats per pass: higher is
better, 0 = no better than uniform), plus the end cell's mean rank within its frame.

V3 (does space now predict danger soon?): for each surface's frame total and best cell, the
AUC for a shot / box entry within 10 s in the same possession, and — the real test — what it
adds to ball position: logistic regressions on training matches with ball features only
(xT, x, |y|) and with ball features + the surface, compared on test log loss and AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from xspace.validation.ablations import BETAS, SURFACES
from xspace.validation.pass_model import auc, log_loss

BALL_FEATURES = ("ball_xt", "ball_x", "abs_ball_y")


def v1_scores(passes: pd.DataFrame, train: np.ndarray, test: np.ndarray,
              subsets: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """One row per (subset, surface): best β on train, test gain over uniform, mean rank."""
    subsets = subsets or {"all": np.ones(len(passes), dtype=bool)}
    uniform = f"ll_{SURFACES[0]}_{BETAS[0]:g}"  # β = 0: the same for every surface
    rows = []
    for name, mask in subsets.items():
        tr, te = train & mask, test & mask
        for s in SURFACES:
            cols = [f"ll_{s}_{b:g}" for b in BETAS]
            ok = passes[cols[0]].notna().to_numpy()
            train_ll = passes.loc[tr & ok, cols].mean().to_numpy()
            j = int(np.nanargmax(train_ll))
            te_ok = te & ok
            rows.append({
                "subset": name, "surface": s, "n_test": int(te_ok.sum()), "beta": BETAS[j],
                "gain_nats": float((passes.loc[te_ok, cols[j]]
                                    - passes.loc[te_ok, uniform]).mean()),
                "mean_rank": float(passes.loc[te_ok, f"rank_{s}"].mean()),
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
