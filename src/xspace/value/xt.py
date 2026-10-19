"""Expected Threat (xT) value surface.

Grids, by name (`PhysicsParams.value_model` picks one; each is bilinearly interpolated onto our
grid):

| Name | Grid | Source |
|---|---|---|
| singh_12x8 | 12 × 8 | Karun Singh's public grid (https://karun.in/blog/expected-threat.html) |
| own_24x16 | 24 × 16 | fitted here (Phase 6, `value/xt_fit.py`) on StatsBomb Open Data |

Grid files are a bare list of rows (Singh's) or `{"grid": rows, "provenance": ..., ...}`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from xspace.constants import PITCH_LENGTH, PITCH_WIDTH

VALUE_MODELS = {"singh_12x8": "xt_12x8.json", "own_24x16": "xt_own_24x16.json"}
DEFAULT_VALUE_MODEL = "singh_12x8"


def grid_bytes(model: str = DEFAULT_VALUE_MODEL) -> bytes:
    return resources.files("xspace.value").joinpath(VALUE_MODELS[model]).read_bytes()


@lru_cache(maxsize=4)
def _interpolator(model: str) -> RegularGridInterpolator:
    raw = json.loads(grid_bytes(model))
    grid = np.asarray(raw["grid"] if isinstance(raw, dict) else raw)  # (rows across, columns along)
    n_y, n_x = grid.shape
    xs = -PITCH_LENGTH / 2 + (np.arange(n_x) + 0.5) * PITCH_LENGTH / n_x
    ys = -PITCH_WIDTH / 2 + (np.arange(n_y) + 0.5) * PITCH_WIDTH / n_y
    return RegularGridInterpolator((ys, xs), grid, bounds_error=False, fill_value=None)


def xt_value(points: np.ndarray, model: str = DEFAULT_VALUE_MODEL) -> np.ndarray:
    """xT at each point, for a team attacking towards +x. points: (G, 2)."""
    x = np.clip(points[:, 0], -PITCH_LENGTH / 2, PITCH_LENGTH / 2)
    y = np.clip(points[:, 1], -PITCH_WIDTH / 2, PITCH_WIDTH / 2)
    return np.clip(_interpolator(model)(np.stack([y, x], axis=1)), 0.0, None)
