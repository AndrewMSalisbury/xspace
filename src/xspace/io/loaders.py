"""Load tracking data from any kloppy-supported provider into dense numpy arrays.

Everything downstream works on `MatchTracking`: fixed-shape arrays in metres, pitch centred
on (0, 0), with the home team attacking +x for the whole match.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from kloppy.domain import Orientation, TrackingDataset

from xspace.physics.kinematics import smooth_velocities


@dataclass
class Team:
    team_id: str
    name: str
    player_ids: list[str]
    jersey_numbers: list[int]
    positions: list[str]
    gk_index: int | None


@dataclass
class MatchTracking:
    """Dense tracking arrays. T = frames, P = players on a team's full roster."""

    match_id: str
    frame_rate: float
    period: np.ndarray  # (T,) int
    timestamp: np.ndarray  # (T,) seconds since period start
    ball: np.ndarray  # (T, 2) metres
    ball_owner: np.ndarray  # (T,) 0 = home, 1 = away, -1 = unknown
    home: Team
    away: Team
    home_pos: np.ndarray  # (T, P_home, 2), NaN when player is off the pitch
    away_pos: np.ndarray  # (T, P_away, 2)
    home_vel: np.ndarray  # (T, P_home, 2) m/s
    away_vel: np.ndarray  # (T, P_away, 2)
    # Per-frame goalkeeper slot (T,), -1 if none on the pitch. Derived on load (not cached) by
    # `assign_goalkeepers`, so GK substitutions and red cards are handled.
    home_gk: np.ndarray | None = None
    away_gk: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        return len(self.period)

    def gk_at(self, side: int, frame: int) -> int | None:
        """Slot of side 0 (home) / 1 (away)'s goalkeeper at `frame`, or None."""
        gk = self.home_gk if side == 0 else self.away_gk
        slot = int(gk[frame]) if gk is not None else -1
        return None if slot < 0 else slot

    def team_arrays(self, side: int) -> tuple[np.ndarray, np.ndarray]:
        """Positions and velocities for side 0 (home) or 1 (away)."""
        return (self.home_pos, self.home_vel) if side == 0 else (self.away_pos, self.away_vel)


def _team(kloppy_team) -> Team:
    players = kloppy_team.players
    positions = [p.starting_position.code if p.starting_position else "UNK" for p in players]
    gk = next((i for i, code in enumerate(positions) if code == "GK"), None)
    return Team(
        team_id=kloppy_team.team_id,
        name=kloppy_team.name,
        player_ids=[p.player_id for p in players],
        jersey_numbers=[int(p.jersey_no) if p.jersey_no is not None else -1 for p in players],
        positions=positions,
        gk_index=gk,
    )


def _stack_xy(df, player_ids: list[str]) -> np.ndarray:
    out = np.full((len(df), len(player_ids), 2), np.nan)
    for j, pid in enumerate(player_ids):
        if f"{pid}_x" in df.columns:
            out[:, j, 0] = df[f"{pid}_x"].to_numpy(dtype=float, na_value=np.nan)
            out[:, j, 1] = df[f"{pid}_y"].to_numpy(dtype=float, na_value=np.nan)
    return out


def _fix_goalkeepers(team: Team, pos: np.ndarray, attacks_positive_x: bool) -> None:
    """Pick the GK who actually played the most; if none is tagged, the deepest player.

    Squads list bench goalkeepers too, so the first 'GK' in the roster may never appear.
    This is the match-level GK; per-frame GKs (substitutions, red cards) come from
    `assign_goalkeepers`.
    """
    presence = (~np.isnan(pos[:, :, 0])).sum(axis=0)
    tagged = [i for i, code in enumerate(team.positions) if code == "GK" and presence[i] > 0]
    if tagged:
        team.gk_index = max(tagged, key=lambda i: presence[i])
        return
    team.gk_index = None
    if not presence.any():
        return
    mean_x = np.nanmean(np.where(presence > 0, pos[:, :, 0], np.nan), axis=0)
    if np.all(np.isnan(mean_x)):
        return
    team.gk_index = int(np.nanargmin(mean_x) if attacks_positive_x else np.nanargmax(mean_x))


def _goalkeeper_per_frame(team: Team, pos: np.ndarray) -> np.ndarray:
    """GK slot per frame: the GK-tagged player on the pitch (most minutes wins ties); if none is
    tagged/present, the team's match-level GK when on the pitch; else -1."""
    present = ~np.isnan(pos[:, :, 0])
    tagged = np.array([code == "GK" for code in team.positions], dtype=bool)
    minutes = present.sum(axis=0)
    score = np.where(present & tagged, minutes + 1, 0)  # +1 so a tagged player always beats 0
    gk = np.where(score.max(axis=1) > 0, score.argmax(axis=1), -1)
    if team.gk_index is not None:
        fallback = (gk < 0) & present[:, team.gk_index]
        gk[fallback] = team.gk_index
    return gk.astype(np.int64)


def assign_goalkeepers(match: MatchTracking) -> MatchTracking:
    """Fill `home_gk` / `away_gk` (per-frame goalkeeper slots) in place and return the match."""
    match.home_gk = _goalkeeper_per_frame(match.home, match.home_pos)
    match.away_gk = _goalkeeper_per_frame(match.away, match.away_pos)
    return match


def from_kloppy(dataset: TrackingDataset, match_id: str = "unknown") -> MatchTracking:
    """Convert a kloppy TrackingDataset (any provider) into a MatchTracking."""
    # Second Spectrum's system: metres, centre spot at (0, 0). Real pitch sizes vary slightly
    # (100-110 x 64-75); v0 treats them as 105 x 68.
    dataset = dataset.transform(
        to_orientation=Orientation.STATIC_HOME_AWAY,
        to_coordinate_system="secondspectrum",
    )
    home_k, away_k = dataset.metadata.teams
    home, away = _team(home_k), _team(away_k)

    df = dataset.to_df(engine="pandas")
    period = df["period_id"].to_numpy(dtype=int)
    timestamp = df["timestamp"].dt.total_seconds().to_numpy()
    ball = np.stack(
        [df["ball_x"].to_numpy(dtype=float, na_value=np.nan),
         df["ball_y"].to_numpy(dtype=float, na_value=np.nan)],
        axis=1,
    )
    owner_ids = df["ball_owning_team_id"].astype(str).to_numpy()
    ball_owner = np.where(owner_ids == home.team_id, 0, np.where(owner_ids == away.team_id, 1, -1))

    home_pos = _stack_xy(df, home.player_ids)
    away_pos = _stack_xy(df, away.player_ids)
    _fix_goalkeepers(home, home_pos, attacks_positive_x=True)
    _fix_goalkeepers(away, away_pos, attacks_positive_x=False)

    fps = float(dataset.metadata.frame_rate)
    return assign_goalkeepers(MatchTracking(
        match_id=match_id,
        frame_rate=fps,
        period=period,
        timestamp=timestamp,
        ball=ball,
        ball_owner=ball_owner,
        home=home,
        away=away,
        home_pos=home_pos,
        away_pos=away_pos,
        home_vel=smooth_velocities(home_pos, fps, period),
        away_vel=smooth_velocities(away_pos, fps, period),
    ))


CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"


def load_idsse(match_id: str = "J03WMX", limit: int | None = None,
               use_cache: bool = True) -> MatchTracking:
    """Load one of the 7 open IDSSE Bundesliga matches (CC BY 4.0, Bassek et al. 2025).

    Full matches are cached under data/processed/ after the first (slow) download.
    """
    from kloppy import sportec

    cache = CACHE_DIR / f"idsse_{match_id}.pkl"
    if use_cache and cache.exists():
        with cache.open("rb") as f:
            match = assign_goalkeepers(pickle.load(f))
        return match if limit is None else _head(match, limit)

    ds = sportec.load_open_tracking_data(match_id=match_id, limit=limit, only_alive=True)
    match = from_kloppy(ds, match_id=match_id)
    if use_cache and limit is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as f:
            pickle.dump(match, f, protocol=pickle.HIGHEST_PROTOCOL)
    return match


PFF_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "pff"


def load_pff(match_id: str | int, limit: int | None = None,
             use_cache: bool = True) -> MatchTracking:
    """Load a PFF FC 2022 World Cup match (download first with scripts/download_pff.py).

    PFF tracking is broadcast-derived at ~30 fps; players off camera are estimated.
    """
    from kloppy import pff

    match_id = str(match_id)
    cache = CACHE_DIR / f"pff_{match_id}.pkl"
    if use_cache and cache.exists():
        with cache.open("rb") as f:
            match = assign_goalkeepers(pickle.load(f))
        return match if limit is None else _head(match, limit)

    tracking = PFF_DIR / "Tracking Data" / f"{match_id}.jsonl.bz2"
    if not tracking.exists():
        raise FileNotFoundError(
            f"{tracking} missing — run: uv run python scripts/download_pff.py --tracking {match_id}"
        )
    ds = pff.load_tracking(
        meta_data=PFF_DIR / "Metadata" / f"{match_id}.json",
        roster_meta_data=PFF_DIR / "Rosters" / f"{match_id}.json",
        raw_data=tracking,
        limit=limit,
        only_alive=True,
    )
    match = from_kloppy(ds, match_id=match_id)
    if use_cache and limit is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as f:
            pickle.dump(match, f, protocol=pickle.HIGHEST_PROTOCOL)
    return match


def _head(match: MatchTracking, n: int) -> MatchTracking:
    arrays = {k: v[:n] for k, v in vars(match).items() if isinstance(v, np.ndarray)}
    return replace(match, **arrays)


IDSSE_MATCHES = {
    "J03WMX": "1. FC Köln vs. FC Bayern München",
    "J03WN1": "VfL Bochum 1848 vs. Bayer 04 Leverkusen",
    "J03WPY": "Fortuna Düsseldorf vs. 1. FC Nürnberg",
    "J03WOH": "Fortuna Düsseldorf vs. SSV Jahn Regensburg",
    "J03WQQ": "Fortuna Düsseldorf vs. FC St. Pauli",
    "J03WOY": "Fortuna Düsseldorf vs. F.C. Hansa Rostock",
    "J03WR9": "Fortuna Düsseldorf vs. 1. FC Kaiserslautern",
}


def load_match(source: str, match_id: str, **kwargs) -> MatchTracking:
    """Dispatch to a loader by source name: 'idsse' or 'pff'."""
    loaders = {"idsse": load_idsse, "pff": load_pff}
    if source not in loaders:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(loaders)}")
    return loaders[source](match_id, **kwargs)
