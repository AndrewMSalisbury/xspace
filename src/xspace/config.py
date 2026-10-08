"""One place for every tunable setting: physics parameters, grids, sampling and paths.

Anything that changes computed outputs belongs here, so `params_hash` can stamp every output
file with the exact settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PFF_DIR = RAW_DIR / "pff"
CACHE_DIR = DATA_DIR / "processed"  # loaded-match pickles and reports
TIMELINE_DIR = CACHE_DIR / "timeline"
ACTIONS_DIR = CACHE_DIR / "actions"
POSSESSIONS_DIR = CACHE_DIR / "possessions"


@dataclass(frozen=True)
class PhysicsParams:
    reaction_time: float = 0.7  # s before a player can change course
    max_speed: float = 5.0  # m/s, average max running speed
    tti_sigma: float = 0.45  # s, uncertainty in arrival time (same sigma as Pressing Intensity)
    lambda_att: float = 4.3  # 1/s, rate of gaining control once at the ball
    kappa_def: float = 1.0  # defender advantage multiplier on lambda
    lambda_gk_factor: float = 3.0  # goalkeepers can handle the ball
    ball_speed: float = 15.0  # m/s, average ground-pass speed
    int_dt: float = 0.04  # s, integration step
    max_int_time: float = 10.0  # s
    convergence_tol: float = 0.01
    lane_samples: int = 12  # points sampled along a pass to test interception
    # Attackers more than this far beyond the offside line can't receive a pass, so they get
    # no pitch control (level is onside; the margin absorbs ~0.5 m tracking noise).
    # Set to float('inf') to switch offside off.
    offside_margin: float = 0.5  # m


DEFAULT_PARAMS = PhysicsParams()


@dataclass(frozen=True)
class TimelineConfig:
    cell_size: float = 2.0  # m; ~1,800 cells. Figures use FIGURE_CELL_SIZE.
    sample_hz: float = 5.0  # every 5th frame at 25 Hz, every 6th at 29.97 fps
    # Frames with any of these quality-flag bits are skipped (see phases/quality.py).
    # Default: all of them.
    skip_flags: int = 0b1111
    chunk_size: int = 256  # frames per parallel task


DEFAULT_TIMELINE = TimelineConfig()
FIGURE_CELL_SIZE = 1.0


@dataclass(frozen=True)
class ExploitationConfig:
    """Phase 3 action metrics (metrics/exploitation.py). Thresholds are starting values."""

    cell_size: float = 1.0  # m; fine enough to read xSpace at a single target point
    skip_flags: int = 0b1111  # as TimelineConfig: release frames with these flags are skipped
    # exploited = completed, into a cell in the top (1 - exploit_rank) of the frame's positive
    # xSpace, worth at least exploit_min (≈ the median frame's best cell)
    exploit_rank: float = 0.9
    exploit_min: float = 0.005
    # missed = the frame's best cell was big (≈ top 5% of frames) but the choice was in the
    # bottom `missed_rank` of the frame's positive xSpace
    missed_best_min: float = 0.02
    missed_rank: float = 0.5
    chunk_size: int = 64  # actions per parallel task


DEFAULT_EXPLOITATION = ExploitationConfig()


def settings_dict(physics: PhysicsParams = DEFAULT_PARAMS,
                  timeline: TimelineConfig = DEFAULT_TIMELINE,
                  exploitation: ExploitationConfig | None = None) -> dict:
    """Every setting that affects timeline outputs (and, if given, action outputs), as plain
    JSON-able values."""
    import inspect

    from xspace.io import sync
    from xspace.phases import possession, quality
    from xspace.value import xt

    xt_grid = (Path(xt.__file__).parent / "xt_12x8.json").read_bytes()
    extra = {}
    if exploitation is not None:
        refine = inspect.signature(sync.refine_release_frames).parameters
        extra = {
            "exploitation": {k: v for k, v in asdict(exploitation).items()
                             if k != "chunk_size"},
            "release_frames": {k: v.default for k, v in refine.items()
                               if v.default is not inspect.Parameter.empty},
        }
    return {
        "physics": asdict(physics),
        "timeline": {k: v for k, v in asdict(timeline).items() if k != "chunk_size"},
        "phases": {
            "setpiece_window_s": possession.SETPIECE_WINDOW_S,
            "transition_s": possession.TRANSITION_S,
            "stoppage_gap_s": possession.STOPPAGE_GAP_S,
            "restart_fill_s": possession.RESTART_FILL_S,
        },
        "quality": {
            "min_players": quality.MIN_PLAYERS,
            "max_ball_speed": quality.MAX_BALL_SPEED,
            "max_player_speed": quality.MAX_PLAYER_SPEED,
        },
        "xt_grid_sha256": hashlib.sha256(xt_grid).hexdigest()[:12],
        **extra,
    }


def params_hash(settings: dict) -> str:
    """Short, stable hash of a settings dict."""
    blob = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def git_sha() -> str:
    """Current commit, with '-dirty' if tracked files have uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty.strip() else "")
