"""Build Phase 5 ratings over every match: team-matches, team profiles, leak maps, players.

    uv run python scripts/build_ratings.py

Reads the per-match timeline, action, possession and player files (all must be built with the
current settings: run build_timeline.py, build_actions.py and build_possessions.py first) and
the player directory (names and positions; PFF rosters, IDSSE metadata via kloppy, cached in
data/processed/ratings/directory.parquet). Writes data/processed/ratings/{team_match,
team_profile,leak_map,player}.parquet, each with a .csv copy, stamped with the git SHA, the
input files' settings hashes and the RatingsConfig. Takes about a minute; no physics.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from xspace.config import (
    ACTIONS_DIR,
    DEFAULT_EXPLOITATION,
    DEFAULT_PARAMS,
    PLAYERS_DIR,
    POSSESSIONS_DIR,
    RATINGS_DIR,
    TIMELINE_DIR,
    git_sha,
    params_hash,
    settings_dict,
)
from xspace.io.lineups import player_directory
from xspace.metrics.possessions import possessions_settings
from xspace.metrics.ratings import (
    DEFAULT_RATINGS,
    action_table,
    leak_map,
    player_ratings,
    possession_table,
    team_match_ratings,
    team_profiles,
)
from xspace.metrics.timeline import read_metadata, read_timeline
from xspace.pipeline import match_ids


def load_directory(jobs: list[tuple[str, str]], refresh: bool) -> pd.DataFrame:
    """Names and positions per match, cached: IDSSE's come from a (slow) kloppy load."""
    path = RATINGS_DIR / "directory.parquet"
    cached = pd.read_parquet(path) if path.exists() and not refresh else None
    have = set() if cached is None else set(zip(cached["source"], cached["match_id"], strict=True))
    new = [player_directory(s, m).assign(source=s, match_id=m)
           for s, m in jobs if (s, m) not in have]
    df = pd.concat([d for d in (cached, *new) if d is not None], ignore_index=True)
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return df


def write(df: pd.DataFrame, name: str, metadata: dict[str, str]) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = {**(table.schema.metadata or {}),
            **{f"xspace.{k}".encode(): v.encode() for k, v in metadata.items()}}
    pq.write_table(table.replace_schema_metadata(meta), RATINGS_DIR / f"{name}.parquet",
                   compression="zstd")
    df.to_csv(RATINGS_DIR / f"{name}.csv", index=False, float_format="%.6g")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all", choices=["idsse", "pff", "all"])
    ap.add_argument("--refresh-directory", action="store_true",
                    help="reload player names and positions")
    args = ap.parse_args()

    start = time.perf_counter()
    sources = ["idsse", "pff"] if args.source == "all" else [args.source]
    jobs = [(s, m) for s in sources for m in match_ids(s)]
    hashes = {
        TIMELINE_DIR: params_hash(settings_dict(DEFAULT_PARAMS)),
        ACTIONS_DIR: params_hash(settings_dict(DEFAULT_PARAMS,
                                               exploitation=DEFAULT_EXPLOITATION)),
        POSSESSIONS_DIR: params_hash(possessions_settings()),
        PLAYERS_DIR: params_hash(possessions_settings()),
    }

    poss, acts, players, rated, skipped = [], [], [], [], []
    for source, match_id in jobs:
        name = f"{source}_{match_id}.parquet"
        stale = [d.name for d, h in hashes.items()
                 if not (d / name).exists() or read_metadata(d / name).get("params_hash") != h]
        if stale:
            skipped.append(f"{source} {match_id} ({', '.join(stale)})")
            continue
        timeline, meta = read_timeline(TIMELINE_DIR / name)
        teams = (meta["home"], meta["away"])
        ids = {"source": source, "match_id": match_id}
        p = possession_table(pd.read_parquet(POSSESSIONS_DIR / name), timeline, teams)
        poss.append(p.assign(**ids))
        acts.append(action_table(pd.read_parquet(ACTIONS_DIR / name), teams).assign(**ids))
        pl = pd.read_parquet(PLAYERS_DIR / name)
        pl["team"] = pl["team_side"].map({0: teams[0], 1: teams[1]})
        players.append(pl.assign(**ids))
        rated.append((source, match_id))
    if skipped:
        print(f"SKIPPED {len(skipped)} matches with missing or out-of-date inputs:")
        print("  " + "\n  ".join(skipped))
    if not poss:
        raise SystemExit("nothing to rate")
    poss = pd.concat(poss, ignore_index=True)
    acts = pd.concat(acts, ignore_index=True)
    players = pd.concat(players, ignore_index=True)
    directory = load_directory(rated, args.refresh_directory)

    team_match = team_match_ratings(poss, acts)
    profiles = team_profiles(poss, acts, team_match)
    leaks = leak_map(acts)
    rated_players = player_ratings(players, acts, directory)

    metadata = {
        "git_sha": git_sha(),
        "inputs": json.dumps({d.name: h for d, h in hashes.items()}, sort_keys=True),
        "ratings": json.dumps(asdict(DEFAULT_RATINGS), sort_keys=True),
        "matches": str(len(rated)),
    }
    RATINGS_DIR.mkdir(parents=True, exist_ok=True)
    for df, name in ((team_match, "team_match"), (profiles, "team_profile"),
                     (leaks, "leak_map"), (rated_players, "player")):
        write(df, name, metadata)
    print(f"{len(rated)} matches: {len(team_match)} team-matches, {len(profiles)} teams, "
          f"{int(rated_players['carrier'].sum())} rated carriers, "
          f"{int(rated_players['runner'].sum())} rated runners "
          f"in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
