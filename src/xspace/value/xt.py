"""Expected Threat (xT) value surface.

v0 uses Karun Singh's public 12x8 xT grid (https://karun.in/blog/expected-threat.html),
bilinearly interpolated onto our grid. The long-term plan is to fit our own possession-value
model; anything exposing `value(points) -> (G,)` can replace this.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from xspace.constants import PITCH_LENGTH, PITCH_WIDTH


@lru_cache(maxsize=1)
def _interpolator() -> RegularGridInterpolator:
    raw = json.loads(resources.files("xspace.value").joinpath("xt_12x8.json").read_text())
    grid = np.asarray(raw)  # (8 rows across the width, 12 columns along the length)
    n_y, n_x = grid.shape
    xs = -PITCH_LENGTH / 2 + (np.arange(n_x) + 0.5) * PITCH_LENGTH / n_x
    ys = -PITCH_WIDTH / 2 + (np.arange(n_y) + 0.5) * PITCH_WIDTH / n_y
    return RegularGridInterpolator((ys, xs), grid, bounds_error=False, fill_value=None)


def xt_value(points: np.ndarray) -> np.ndarray:
    """xT at each point, for a team attacking towards +x. points: (G, 2)."""
    x = np.clip(points[:, 0], -PITCH_LENGTH / 2, PITCH_LENGTH / 2)
    y = np.clip(points[:, 1], -PITCH_WIDTH / 2, PITCH_WIDTH / 2)
    return np.clip(_interpolator()(np.stack([y, x], axis=1)), 0.0, None)
