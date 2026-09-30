"""Velocity estimation from raw positions."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

MAX_PLAYER_SPEED = 12.0  # m/s; anything faster is tracking noise


def smooth_velocities(pos: np.ndarray, fps: float, period: np.ndarray,
                      window_s: float = 0.28, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay derivative of positions, computed separately per period.

    pos: (T, P, 2). NaN gaps (player off the pitch) are interpolated for filtering and
    re-masked afterwards, so they never leak into neighbouring frames' velocities.
    """
    window = max(polyorder + 2, int(round(window_s * fps)) | 1)  # odd, > polyorder
    vel = np.full_like(pos, np.nan)
    for p_id in np.unique(period):
        rows = np.flatnonzero(period == p_id)
        if len(rows) < window:
            continue
        seg = pos[rows]  # (t, P, 2)
        for j in range(seg.shape[1]):
            for k in range(2):
                series = seg[:, j, k]
                valid = ~np.isnan(series)
                if valid.sum() < window:
                    continue
                idx = np.arange(len(series))
                filled = np.interp(idx, idx[valid], series[valid])
                d = savgol_filter(filled, window, polyorder, deriv=1, delta=1.0 / fps)
                d[~valid] = np.nan
                vel[rows, j, k] = d

    speed = np.linalg.norm(vel, axis=-1, keepdims=True)
    scale = np.where(speed > MAX_PLAYER_SPEED, MAX_PLAYER_SPEED / np.maximum(speed, 1e-9), 1.0)
    return vel * scale
