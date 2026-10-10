# Methodology

Working notes on how Expected Space (xSpace) is computed, what it assumes, and what's next.
Every tunable setting (physics, grids, sampling, paths) lives in `src/xspace/config.py`.

## 1. Preprocessing

- Any kloppy provider is converted to the Second Spectrum coordinate system (metres, centre spot at
  the origin) with the home team attacking +x for the whole match.
- Per frame, everything is rotated so the **team in possession attacks +x**.
- Velocities: Savitzky–Golay derivative (0.28 s window, order 2), per period, speeds clipped at
  12 m/s. Gaps (substitutions) are interpolated for filtering, then re-masked.
- v0 assumes a 105 × 68 m pitch for every stadium.
- PFF extra time: kloppy 3.19 leaves periods 3–4 shifted by a whole pitch length and width
  (x ≈ ±105, y ≈ ±68 m off; a pure translation, checked with the keepers' positions).
  `recentre_periods` moves any period whose median position is off the pitch back on load.

### Events and synchronisation

- Events from both providers become one table (`MatchEvents`, `src/xspace/io/events.py`) in the
  tracking coordinate frame, with times in seconds since period start on the tracking clock.
- **PFF**: one row per possession event. Each row's own `eventTime` is used, not the shared
  game-event `startTime` (PFF splits a touch-then-pass into `IT` + `PA` rows of one game event).
  End locations aren't provided, so a ball-moving action ends where the ball is at the next
  event (usually the reception) within 10 s. Raw coordinates are rotated 180° in the periods
  where the home team attacks −x. Extra-time matches have no period start times in metadata;
  the period's kick-off event is used instead (it equals `startPeriodN` where both exist).
- **IDSSE**: DFL events via kloppy; pass end = receiver location (missing for incomplete passes).
- **Sync** (`src/xspace/io/sync.py`): per period, find the time shift that minimises the median
  distance between the tracking ball and the acting player at on-ball events, apply it, then
  map each event to the nearest tracking frame (±0.1 s). The report records the residual shift,
  which should be ≈ 0, and the remaining actor–ball distances.
  - PFF needs no shift: tracking frames are tagged with their event ids. But PFF's *smoothed*
    ball (what kloppy loads) is pulled onto the player at tagged events, so actor–ball distance
    is ~0 by construction there. Event freeze frames use the *raw* ball, which differs by ~2.4 m
    median (more when the ball is in the air).
  - IDSSE event clocks are off by −1.2 to +1.3 s, in either direction, and differ between the
    halves of one match, so offsets must be estimated per period. After the shift, all 7
    matches have a median actor–ball distance of 1.5–2.6 m (J03WMX: 5.4 m → 1.7 m), but only
    52–62% of on-ball events are within 3 m: DFL event timing is noisy per event (± 1 s).
  - **Release frames** (`refine_release_frames`, IDSSE only; in the spirit of ETSY, Van Roy
    et al. 2021): for passes, crosses, shots and clearances, search ± 2 s around the synced
    frame for the kick: the ball within 2 m of the actor, then the biggest jump in the ball's
    speed away from the actor (next 0.32 s vs previous 0.32 s, counting only movement away,
    so receptions and the flight don't qualify), with a 1 m/s-per-second penalty for shifting.
    Releases stay in event order. Stored as `release_frame`; `frame` (and so possession labels
    and timelines) is unchanged. Across the 7 matches, the share of passes whose ball heads
    toward the recorded end location goes from 48–57% to 77–87%, and the share with the
    passer within 2 m of the ball from 50–64% to 88–96%. Run on PFF as a check, it lands
    within 2 frames of PFF's tagged frame for 87–91% of passes; PFF keeps its tags.

### Phases of play

Per-frame labels from `label_phases` (`src/xspace/phases/possession.py`):

- **Possession** comes from events, not the tracking `ball_owner` field. PFF's field is derived
  from events anyway (99.9% agreement); IDSSE's flickers during duels (a change every ~3 s, 92%
  agreement). The team in possession is the team of the latest *controlling* event (pass, cross,
  shot, carry, reception, recovery). Challenges, clearances and touches don't change it.
  After a stoppage, possession is back-filled from the first live frame to the restart event.
- **Restart snapping**: restart events are often stamped up to ~1.5 s before the first live
  tracking frame (dead frames are dropped), so sync snaps them forward by up to 3 s.
- **Set-piece windows** run from the restart for 8 s (corners, free kicks, penalties), 4 s
  (throw-ins, goal kicks, kick-offs) or 2 s (drop balls), ending early if possession changes.
  These are starting values, to be tuned by inspection.
- **Transition**: the first 10 s of a possession won in open play.
- **Thirds** by ball x in the attacking direction of the team in possession.

On 3812 (PFF) and J03WMX (IDSSE), open play is 84% and 86% of ball-in-play time, transitions
28% in both.

Goalkeepers are assigned **per frame** (`assign_goalkeepers`): the GK-tagged player on the
pitch. Three PFF matches change keeper mid-game (3813 injury, 10507 substitution, 3828 red card).
PFF's smoothed tracking keeps a **sent-off player's track** after the card (3828: Hennessey
"stays on" beside his replacement), so `remove_sent_off_players` (`src/xspace/io/lineups.py`)
blanks dismissed players from their card onwards using event data. Substituted players already
disappear correctly.

### Quality flags

`quality_flags` (`src/xspace/phases/quality.py`) marks each frame with bit flags: ball missing,
a team with fewer than 10 players tracked, the ball moving faster than 45 m/s, or a player faster
than 13 m/s between consecutive frames. On 3812 / 3828 (PFF) 93–96% of frames are clean (mostly
missing ball, 4–6%); J03WMX (IDSSE) is 99.9% clean. PFF's per-player "estimated" visibility isn't
exposed through kloppy, so it isn't flagged yet.
All 71 matches use a 105 × 68 m pitch, so per-stadium dimensions aren't needed for this data.

## 2. Time to intercept

Shared by every component, and the same quantity Pressing Intensity is built on. A player keeps
moving on their current velocity for a reaction time (0.43 s), then runs straight at 5.27 m/s:

```
r_react = r + v · t_react
T(player → target) = t_react + |target − r_react| / v_max
```

Every physics parameter marked "fitted" in sections 2–4 was fitted to 45k World Cup pass
outcomes in Phase 4 ([validation.md](validation.md), V2). Phases 0–3 used Spearman's published
values (reaction 0.7 s, 5 m/s, σ 0.45 s, λ 4.3 s⁻¹, ball 15 m/s); they are kept as
`UNCALIBRATED_PARAMS`.

## 3. Pitch control

Spearman (2018), following Shaw's implementation, vectorised over all cells at once. A player's
probability of having arrived by time *t* is logistic in `t − T` (σ = 0.49 s, fitted). Control
accumulates from the moment the ball arrives (ground pass at 25.8 m/s, fitted) at rate
λ = 9.83 s⁻¹ (fitted; defenders ×1.04, the defending goalkeeper ×3) until attack + defence
control ≥ 0.99, then is normalised to sum to 1.

**Lofted balls** (off by default). A lofted pass takes `1.32 s + distance / 20.8 m/s`
(fitted) to land. With `PhysicsParams.air_speed > 0`, control is computed a second time with
that arrival time and each cell keeps whichever ball, ground or lofted, is likelier to arrive
and be received (section 4): `FrameSpace.receive` is the control under that ball and
`FrameSpace.air` marks the lofted cells. The lofted ball is part of the fitted pass-completion
model, but xSpace leaves it off: taking the better ball rated long balls into space far above
how often they're played or completed, and the ground-only surface predicts where passes go and
danger as well or better ([validation.md](validation.md#the-lofted-ball-in-xspace)). With it
off, `receive` is `control.attack`.

Per-player control shares are kept, for attributing space to runners later.

**Implementation.** Both teams are integrated together as one (players × cells) block in
float32. The arrival probability is `1 / (1 + E)` with `E = exp(−k (t − T))`, so each time step
multiplies `E` by the constant `exp(−k Δt)` instead of evaluating `exp` again; converged cells
are dropped from the working arrays as the integration proceeds. This is ~6× faster than
the direct version and matches a float64 reference to ~1e-6 (a cell sitting exactly at the
convergence tolerance can stop one step apart: ≤ 1e-3). `tests/test_timeline.py` keeps the
direct float64 implementation as an oracle.

## 4. Pass reachability

For each cell, sample 12 points along the straight passing lane. At each point, a defender
gets there in time with logistic probability in `(ball arrival time − defender T)`, and having
got there, cuts the ball out with probability 0.152 (`intercept_factor`, fitted). Each defender
gets one chance, at their best point on the lane; defenders are combined like Pressing
Intensity combines pressers:

```
p_d        = 0.152 · max over lane points of P(defender d there in time)
reach      = ∏_d (1 − p_d)
```

Phases 0–3 treated every (defender, lane point) pair as an independent chance, so a defender
beside the lane counted up to 12 times; long passes were rated nearly unreachable (see
[validation.md](validation.md), V2). The low intercept factor also absorbs tracking noise and
the fact that players rarely try lanes that are really shut, so reach penalises lanes only
lightly.

A **lofted** ball (when switched on, section 3) can't be cut out in flight (reach 1) but
arrives later, so its control is lower; the cell then uses the ball with the larger
`control × reach`.

## 5. Value

`xT_gained(cell) = max(xT(cell) − xT(ball), 0)` using Karun Singh's public 12 × 8 xT grid,
bilinearly interpolated. Using raw xT let the large, safe, low-value area in a team's own half
dominate every total; measuring the gain keeps the focus on space that advances the attack.

## 6. Defensive shape and zones

Outfield defenders are split into up to three lines at the largest gaps in x, with at least two
players per line (so a lone defender tracking a runner deep isn't mistaken for the back line).
Line positions are medians.

| Zone | Rule |
|---|---|
| behind | x > back line |
| between | mid line < x ≤ back line, inside the block's width |
| wide | mid line < x ≤ back line, outside the block's width |
| in front | x ≤ mid line |

The legal offside line (second-deepest defender, incl. GK; not behind halfway or the ball) is
stored separately, and it is applied to the attackers: see *Offside* below.

### Offside

Attackers more than `offside_margin` (0.5 m) beyond the offside line are left out of pitch
control for that frame. They can't legally receive a pass played now, so the space they would
own goes to the next player, usually a defender. Level counts as onside, and the margin absorbs
tracking noise (~0.5 m). The player nearest the ball (within 3 m) is never offside. Offside
attackers still matter through the defenders they hold deep, which the defenders' positions
already reflect, so a frame's xSpace reads as "space reachable by a pass played right now".
`n_offside` records how many were left out (timeline and action rows). Set pieces where offside
doesn't apply (throw-ins, corners, goal kicks) are already excluded.

Check: on 4 matches, only ~1% of completed passes had a receiver more than 0.5 m beyond our
line at the release frame, so the line matches what the referees saw. Before this rule,
10–17% of computed frames had an offside attacker, and in those frames the attackers'
behind-space was overstated by about 2× (total by ~25–30%).

## 7. Aggregation

Frame totals are area-weighted sums (`Σ xSpace × cell area`, units xT·m²), overall and per zone,
plus the single best cell.

## 8. Match timeline

`build_timeline` (`src/xspace/metrics/timeline.py`; CLI `scripts/build_timeline.py`) computes
xSpace for every match at 5 Hz (every 5th frame at 25 Hz, every 6th at 29.97 fps) on a 2 m grid.
Every sampled frame gets a row; metrics are computed only when `status == "ok"`:

| status | Meaning |
|---|---|
| ok | computed |
| quality | a quality flag is set (default: any flag) |
| no_possession | no team in possession yet (start of a period) |
| set_piece | inside a set-piece window |

The attacking side is the **event-based** possession from `label_phases`. Each row holds the
labels (period, time, possession, set piece, transition, third, flags) and the metrics: total
and per-zone xSpace, the best cell (value, x, y, zone), mean attacking control, offside / back /
mid line x, block width, compactness (back − mid line), ball position and xT, and players on the
pitch per team. All x / y values are in the attacking team's frame (attacking +x).

Output: `data/processed/timeline/{source}_{match}.parquet`, with the git SHA and a hash of every
setting that affects results (`config.settings_dict`: physics, timeline, phase windows, quality
thresholds, xT grid) in the file metadata. The build skips files whose hash is current.
Frames are spread over a process pool in chunks; each chunk carries only its own frames'
arrays, and a parallel run equals the serial run exactly (tested).

### First results (all 71 matches)

From `notebooks/timeline_sanity.ipynb`, with the calibrated physics: 82–86% of sampled frames
are computed (the rest are set-piece windows, and on PFF ~6% quality flags, mostly a missing
ball). IDSSE and PFF give xSpace on the same scale (team-match median 5.2 and 5.4 xT·m²).
Calibrated control and reach are both higher than in Phases 0–3 (median ≈ 1.15), so xSpace
values aren't comparable across physics versions. The best cell is on the far touchline in 30%
of frames: with lanes cheap to pass through, open flank space often wins.

Transitions keep a higher, less compact block than settled play in every third (back line
3–6 m further from goal, back-to-mid distance 1–2 m larger). But with the calibrated physics
they have *less* xSpace in every zone, including behind (PFF middle third 2.70 against 4.52),
because the team that just won the ball controls less of the pitch (mean control 0.51 against
0.61) and the faster calibrated control rate makes that count for more. With the Phases 0–3
physics, space behind rose for ~8 s after a turnover; that finding doesn't survive calibration.

## 9. Exploitation: was the space used?

`build_actions` (`src/xspace/metrics/exploitation.py`; CLI `scripts/build_actions.py`) gives one
row per open-play pass, cross and carry. At the action's **release frame** (section 1; refined
per event for IDSSE) it computes the frame's xSpace on a **1 m grid** for the acting team and
compares it with the point the ball was sent to:

| Column | Meaning |
|---|---|
| `available`, per zone | frame total xSpace (as in the timeline) |
| `best` (+ x, y, zone) | the frame's best cell |
| `chosen` (+ x, y, zone) | xSpace at the chosen point's cell; `chosen_zone` is the zone targeted |
| `chosen_rank` | share of the frame's positive-xSpace cells worth less than the chosen one |
| `decision_gap` | `best − chosen` (≥ 0) |
| `xt_gained` | xT(end) − xT(origin) if completed, −xT(origin) if lost |
| `exploited` | completed, `chosen_rank` ≥ 0.9 and `chosen` ≥ 0.01 (≈ the median frame's best cell) |
| `missed` | `best` ≥ 0.045 (≈ top 5% of timeline frames) and `chosen_rank` < 0.5 |
| `owner_id`, `best_owner_id` | attacker with the largest pitch-control share at the chosen / best cell |

The **chosen point** (`chosen_source`) is the end location for completed actions; for failed
PFF passes, the intended target (`targetPlayerId`) projected along their velocity for the
ball's flight time; otherwise the ball at the next event, which for a cut-out pass is where it
was intercepted (so it understates the intent). xSpace counts only forward value
(section 5), so backward and square passes score `chosen = 0`; that is a choice, not a bug: the
metric asks whether *valuable* space was used. Thresholds live in `ExploitationConfig`
(`config.py`) and are starting values; they join the params hash for action files only.

### First results (all 71 matches)

From `notebooks/moments.ipynb`: 68,464 open-play actions, 96–99% computed. For completed passes
the attacker owning the chosen cell is the actual receiver 85% of the time (PFF; 76% IDSSE).
The thresholds were rescaled with the calibrated physics to keep their meaning (median and
95th percentile of the frame's best cell): exploited 0.8% (PFF) / 1.2% (IDSSE) of actions,
missed 1.9% / 3.4%. 39% of actions go into zero xSpace (backward or square; 92% completed).
Completion falls as the chosen cell's rank rises (91% in the middle ranks, 45% in the top
10%), and xT gained by completed actions is highest in the top 10% (0.017). With the Phases
0–3 physics it was highest in the *lowest* ranks: long ground passes, completed 55% of the time
although reach rated them nearly unreachable. Calibrating reach (Phase 4) removed that
inversion. Exploited moments look right; missed moments are dominated by best cells near the
six-yard box, where borrowed xT is very high.

## 10. Possessions

`metrics/possessions.py` rolls the timeline and action files up to one row per possession
(`PhaseLabels.possession_id`: a new possession on a change of team or a restart). No new
physics. Per possession: how it started (`regain` in open play, a set-piece type, or `restart`)
and ended (`turnover` in play or `dead_ball`); mean, peak and time-integrated xSpace and peak
behind-space from the 5 Hz timeline, with the seconds to the peak; actions, exploited, missed
and xT gained from the action file, with the seconds to the first exploited action; and whether
the ball reached the final third or the opponent's penalty area, and whether the team shot or
scored. Shots are matched through their synced frame, so shots that never synced (section 1)
are missing: 25% of IDSSE shots, about 5% of PFF's. Seen from the defending team, the same rows are xSpace conceded.

### First results (all 71 matches)

17,937 possessions (median 116 per team-match IDSSE, 125 PFF); 14,006 have at least 2 s of
computed open play. Peak xSpace sorts possessions by outcome: the top fifth ends in a shot
18–20% of the time, the bottom fifth 1–3%, and the gap holds for possessions of the same
length (5–15 s: 11–21% against 1–3%). Peak xSpace is partly a consequence of where the ball got
to, so a cleaner test uses only the **first 2 s** of possessions lasting over 4 s. On PFF, the
top fifth of early xSpace shoots 12% of the time against 6.5% for the bottom fifth; IDSSE shows
no clear trend. Holding the starting third fixed, the effect is small: final third 18% → 22%
from the lowest to the highest tercile, middle 9% → 11%. With the Phases 0–3 physics it was
larger (final third 15% → 26%), in line with V3 ([validation.md](validation.md)).

After an open-play regain, peak xSpace comes a median 3–4 s in; 2–4% of regains contain an
exploited action, and those end in a shot 39–45% of the time against 5–6% for the rest. Teams
differ in how much they concede: from 3.5 xT·m² per possession (Spain) to 5.8 (Costa Rica)
among World Cup teams with at least 3 matches. Conceded xSpace correlates with shots conceded
per possession (r = 0.55, 32 teams).

## 11. Players: space held

`metrics/players.py` gives one row per player per match, from the action file. Minutes count
live-ball frames only, so "per 90" below means per 90 *live* minutes (a World Cup match has
about 55–60). Each player gets their own on-ball numbers (actions, exploited, missed, xT gained,
mean decision gap) and two kinds of off-ball credit:

- **space received**: completed team-mate actions into a cell this player owned (largest
  pitch-control share), whether or not they were the receiver, and the xSpace there;
- **space held**: team-mate actions whose *best* cell (≥ 0.01, the `exploit_min`) this player
  owned, how often the ball went into it (`found`), and how often the action was `missed`.

This counts moments; it doesn't say whether the player's run *created* the space (the
"freeze the runner" counterfactual stays a v2 idea).

### First results (all 71 matches)

Space held per 90 orders the positions sensibly: wingers 85–121 (IDSSE / PFF), strikers 79–92,
attacking midfielders 29–33, full-backs 15–20, central midfielders ~7, defensive midfielders
~4, centre-backs ~3, goalkeepers 0. The ball goes into the held space 8–9% of the time (15–24%
with the Phases 0–3 physics: best cells are now often on the far touchline, section 8). Among PFF
players with at least 270 live minutes the leaders are Olivier Giroud (147 per 90), Julián
Álvarez, Ousmane Dembélé, Kylian Mbappé, Andrej Kramarić and Ivan Perišić, with Denzel Dumfries,
Nahuel Molina and Josip Juranović the top defenders. With the Phases 0–3 physics the rate was
stable within a player (split-half r = 0.85 over 31 players); that hasn't been re-measured.
Much of it is position; a within-position check needs more minutes per player than one
tournament gives.

## Known limitations

- **Independence**: defenders are treated as acting independently (same caveat as Pressing
  Intensity). Real defences cover for each other.
- **Ground passes only**: xSpace has no lofted ball, which undercounts "behind" space that only
  a ball over the top reaches. A fitted lofted ball exists (section 3) and predicts pass
  completion, but taking the better of the two balls per cell made xSpace worse at predicting
  where passes go and danger; a version that weights the lofted ball by how often it's chosen
  is an open option.
- **Offside** is a hard cut at 0.5 m beyond the line; a soft weighting by P(offside) given
  tracking noise is a Phase 4 option.
- **Reach is weak**: the fitted intercept factor (0.152) means lanes cost little. Fitted on
  attempted passes only, it can't see lanes nobody tried; a joint fit that also scored where
  passes went didn't change the picture enough to adopt ([validation.md](validation.md)).
- **One set of physical parameters** for every player, fitted to World Cup passes; could be
  fit per player from tracking data.
- **Set pieces**: corners and free kicks pack the box, so defensive lines are meaningless there.
  These phases need to be filtered out (or modelled separately) using event data.
- **Broadcast tracking** (PFF): off-camera players are estimated, so far-side space is less
  reliable. Compare against IDSSE's optical tracking to quantify this.
- **Validation** ([validation.md](validation.md)) covers where passes go, pass completion,
  danger in the next 10 s and team stability; not yet ratings against results (Phase 5).

## Roadmap

1. ~~**Match timeline**~~ — done (section 8).
2. ~~**Exploited vs. available**~~ — done (sections 9–11). Next: threshold tuning by inspection.
3. ~~**Validation**~~ — done ([validation.md](validation.md)): V1–V4 against ablations, and the
   physics fitted to pass outcomes. V5 (ratings vs results) follows the ratings. The old,
   heavier lane blocking predicted danger (V3) better; kept as a known trade-off, to revisit
   with the own value model.
4. **Ratings** — match ratings per team, team profiles, player ratings (carriers, runners).
5. **PFF 2022 World Cup** — scale to 64 matches / 32 teams.
6. **Own value model** — replace borrowed xT with a possession-value model fit on this data.
7. **Web app** — match scrubber, team and player pages.
