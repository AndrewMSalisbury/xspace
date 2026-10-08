"""The Phase 1 data layer for one match, in one call: tracking + synced events + labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xspace.io.events import load_events, pff_match_ids
from xspace.io.lineups import remove_sent_off_players
from xspace.io.loaders import IDSSE_MATCHES, MatchTracking, load_match
from xspace.io.sync import SyncReport, synchronise
from xspace.phases.possession import PhaseLabels, label_phases
from xspace.phases.quality import quality_flags


@dataclass
class PreparedMatch:
    source: str
    match: MatchTracking  # sent-off players already removed
    events: pd.DataFrame  # synced (has `frame`)
    sync: SyncReport
    phases: PhaseLabels
    flags: np.ndarray  # (T,) quality bit flags


def prepare_match(source: str, match_id: str) -> PreparedMatch:
    """Load, synchronise and label one match. Sync problems are reported, not raised."""
    match = load_match(source, match_id)
    events, report = synchronise(load_events(source, match_id), match, check=False,
                                 refine=source == "idsse")
    remove_sent_off_players(match, events)
    return PreparedMatch(source, match, events, report, label_phases(match, events),
                         quality_flags(match))


def match_ids(source: str) -> list[str]:
    return list(IDSSE_MATCHES) if source == "idsse" else pff_match_ids()
