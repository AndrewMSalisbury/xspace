"""Whole-match Expected Space timeline: one row per sampled frame.

Frames are sampled at `TimelineConfig.sample_hz` (every 5th frame at 25 Hz, every 6th at
29.97 fps). Every sampled frame gets a row. xSpace is computed only where `status == "ok"`;
the other rows keep their labels and a reason code, with NaN metrics:

| status | Meaning |
|---|---|
| ok | computed |
| quality | a quality flag in `TimelineConfig.skip_flags` is set (e.g. ball missing) |
| no_possession | no team in possession yet (before the first controlling event of a period) |
| set_piece | inside a set-piece window: the defence isn't in its open-play shape |

Possession comes from events (`label_phases`), not the tracking `ball_owner`.
All x / y columns are in the attacking team's frame (it attacks +x, pitch centred on 0).

Work is split into chunks of frames. Each chunk carries only its own frames' arrays, so
parallel workers never need a whole match in memory.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import Executor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from xspace.config import (
    DEFAULT_PARAMS,
    DEFAULT_TIMELINE,
    PhysicsParams,
    TimelineConfig,
    git_sha,
    params_hash,
    settings_dict,
)
from xspace.io.events import SETPIECE_TYPES
from xspace.io.loaders import MatchTracking
from xspace.metrics.space import ZONES, space_from_arrays
from xspace.phases.possession import THIRDS, PhaseLabels
from xspace.physics.pitch_control import make_grid
from xspace.value.xt import xt_value

STATUSES = ("ok", "quality", "no_possession", "set_piece")
# Metric columns filled for "ok" frames (float32, NaN otherwise).
METRICS = (
    "total", *ZONES,  # area-weighted xSpace, xT·m²
    "best", "best_x", "best_y", "best_zone",  # best single cell
    "mean_control",  # mean attacking pitch control over the whole pitch
    "offside_line", "back_line", "mid_line",  # x of the defending lines
    "block_width",  # lateral extent of the defending outfield players, m
    "compactness",  # back line − mid line, m
    "ball_x", "ball_y", "ball_xt",
    "n_att", "n_def",  # players on the pitch
)


@dataclass
class FrameChunk:
    """Inputs for a block of frames, in pitch coordinates (home attacks +x)."""

    frames: np.ndarray  # (N,) frame indices, for the record only
    side: np.ndarray  # (N,) team in possession
    ball: np.ndarray  # (N, 2)
    home_pos: np.ndarray  # (N, P_home, 2)
    home_vel: np.ndarray
    away_pos: np.ndarray  # (N, P_away, 2)
    away_vel: np.ndarray
    home_gk: np.ndarray  # (N,) roster slot, -1 if none
    away_gk: np.ndarray


@lru_cache(maxsize=4)
def _grid(cell_size: float) -> np.ndarray:
    return make_grid(cell_size)[2]


def compute_chunk(chunk: FrameChunk, cell_size: float,
                  params: PhysicsParams = DEFAULT_PARAMS) -> dict[str, np.ndarray]:
    """Metrics for every frame of a chunk. Top-level so worker processes can import it."""
    grid = _grid(cell_size)
    n = len(chunk.frames)
    out = {name: np.full(n, np.nan, dtype=np.float32) for name in METRICS}
    for i in range(n):
        side = int(chunk.side[i])
        teams = ((chunk.home_pos, chunk.home_vel, chunk.home_gk),
                 (chunk.away_pos, chunk.away_vel, chunk.away_gk))
        att_pos, att_vel, _ = teams[side]
        def_pos, def_vel, def_gk = teams[1 - side]
        gk = int(def_gk[i])
        fs = space_from_arrays(att_pos[i], att_vel[i], def_pos[i], def_vel[i], chunk.ball[i],
                               gk if gk >= 0 else None, side, grid, params)
        best = int(np.argmax(fs.xspace))
        ball = chunk.ball[i] if side == 0 else -chunk.ball[i]
        row = {
            **{k: fs.totals[k] for k in ("total", *ZONES, "best")},
            "best_x": grid[best, 0], "best_y": grid[best, 1], "best_zone": fs.zone[best],
            "mean_control": fs.control.attack.mean(),
            "offside_line": fs.shape.offside_line,
            "back_line": fs.shape.back_line,
            "mid_line": fs.shape.mid_line,
            "block_width": fs.shape.block_y[1] - fs.shape.block_y[0],
            "compactness": fs.shape.back_line - fs.shape.mid_line,
            "ball_x": ball[0], "ball_y": ball[1], "ball_xt": xt_value(ball[None])[0],
            "n_att": (~np.isnan(att_pos[i, :, 0])).sum(),
            "n_def": (~np.isnan(def_pos[i, :, 0])).sum(),
        }
        for k, v in row.items():
            out[k][i] = v
    return out


def sample_frames(match: MatchTracking, sample_hz: float) -> np.ndarray:
    """Every k-th frame, k = round(frame_rate / sample_hz)."""
    step = max(1, round(match.frame_rate / sample_hz))
    return np.arange(0, match.n_frames, step)


def frame_status(phases: PhaseLabels, flags: np.ndarray, frames: np.ndarray,
                 skip_flags: int) -> np.ndarray:
    """Index into STATUSES for each frame. Quality wins, then possession, then set piece."""
    status = np.zeros(len(frames), dtype=np.int8)
    status[~phases.open_play[frames]] = STATUSES.index("set_piece")
    status[phases.possession_side[frames] < 0] = STATUSES.index("no_possession")
    status[(flags[frames] & skip_flags) != 0] = STATUSES.index("quality")
    return status


def make_chunks(match: MatchTracking, frames: np.ndarray, side: np.ndarray,
                size: int) -> Iterable[FrameChunk]:
    for start in range(0, len(frames), size):
        f = frames[start:start + size]
        yield FrameChunk(
            frames=f, side=side[start:start + size], ball=match.ball[f],
            home_pos=match.home_pos[f], home_vel=match.home_vel[f],
            away_pos=match.away_pos[f], away_vel=match.away_vel[f],
            home_gk=match.home_gk[f], away_gk=match.away_gk[f],
        )


def build_timeline(match: MatchTracking, phases: PhaseLabels, flags: np.ndarray,
                   config: TimelineConfig = DEFAULT_TIMELINE,
                   params: PhysicsParams = DEFAULT_PARAMS,
                   executor: Executor | None = None) -> pd.DataFrame:
    """One row per sampled frame. With an `executor`, chunks run in parallel; results are
    identical to the serial run (each frame is computed independently)."""
    frames = sample_frames(match, config.sample_hz)
    status = frame_status(phases, flags, frames, config.skip_flags)
    ok = frames[status == 0]
    side = phases.possession_side[ok]
    chunks = list(make_chunks(match, ok, side, config.chunk_size))
    if executor is None:
        results = [compute_chunk(c, config.cell_size, params) for c in chunks]
    else:
        n = len(chunks)
        results = list(executor.map(compute_chunk, chunks, [config.cell_size] * n,
                                    [params] * n))

    df = pd.DataFrame({
        "frame": frames.astype(np.int32),
        "period": match.period[frames].astype(np.int8),
        "time_s": match.timestamp[frames],
        "status": pd.Categorical.from_codes(status, STATUSES),
        "possession_side": phases.possession_side[frames].astype(np.int8),
        "possession_id": phases.possession_id[frames].astype(np.int32),
        "setpiece": pd.Categorical.from_codes(phases.setpiece[frames], SETPIECE_TYPES),
        "transition": phases.transition[frames],
        "third": pd.Categorical.from_codes(phases.third[frames], THIRDS),
        "flags": flags[frames].astype(np.int16),
    })
    for name in METRICS:
        col = np.full(len(frames), np.nan, dtype=np.float32)
        if results:
            col[status == 0] = np.concatenate([r[name] for r in results])
        df[name] = col
    return df


def timeline_metadata(source: str, match: MatchTracking,
                      config: TimelineConfig = DEFAULT_TIMELINE,
                      params: PhysicsParams = DEFAULT_PARAMS, **extra) -> dict[str, str]:
    settings = settings_dict(params, config)
    return {
        "source": source,
        "match_id": match.match_id,
        "home": match.home.name,
        "away": match.away.name,
        "frame_rate": str(match.frame_rate),
        "git_sha": git_sha(),
        "params_hash": params_hash(settings),
        "settings": json.dumps(settings, sort_keys=True),
        **{k: str(v) for k, v in extra.items()},
    }


def write_timeline(df: pd.DataFrame, path: Path, metadata: dict[str, str]) -> Path:
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = {**(table.schema.metadata or {}),
            **{f"xspace.{k}".encode(): v.encode() for k, v in metadata.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(meta), path, compression="zstd")
    return path


def read_metadata(path: Path) -> dict[str, str]:
    meta = pq.read_schema(path).metadata or {}
    return {k.decode()[len("xspace."):]: v.decode() for k, v in meta.items()
            if k.startswith(b"xspace.")}


def read_timeline(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    return pd.read_parquet(path), read_metadata(path)
