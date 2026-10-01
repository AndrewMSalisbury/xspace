# Methodology

Working notes on how Expected Space (xSpace) is computed, what it assumes, and what's next.
Parameters live in `PhysicsParams` (`src/xspace/physics/pitch_control.py`).

## 1. Preprocessing

- Any kloppy provider is converted to the Second Spectrum coordinate system (metres, centre spot at
  the origin) with the home team attacking +x for the whole match.
- Per frame, everything is rotated so the **team in possession attacks +x**.
- Velocities: Savitzky–Golay derivative (0.28 s window, order 2), per period, speeds clipped at
  12 m/s. Gaps (substitutions) are interpolated for filtering, then re-masked.
- v0 assumes a 105 × 68 m pitch for every stadium.

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
    52–62% of on-ball events are within 3 m. DFL event timing is noisy per event, so a per-event refinement (ETSY-style,
    Van Roy et al. 2021) is a candidate improvement.

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
All 71 matches use a 105 × 68 m pitch, so per-stadium dimensions aren't needed for this data.

## 2. Time to intercept

Shared by every component, and the same quantity Pressing Intensity is built on. A player keeps
moving on their current velocity for a reaction time (0.7 s), then runs straight at 5 m/s:

```
r_react = r + v · t_react
T(player → target) = t_react + |target − r_react| / v_max
```

## 3. Pitch control

Spearman (2018), following Shaw's implementation, vectorised over all cells at once. A player's
probability of having arrived by time *t* is logistic in `t − T` (σ = 0.45 s). Control accumulates
from the moment the ball arrives (ground pass at 15 m/s) at rate λ = 4.3 s⁻¹ (×3 for the
defending goalkeeper) until attack + defence control ≥ 0.99, then is normalised to sum to 1.

Per-player control shares are kept, for attributing space to runners later.

## 4. Pass reachability

For each cell, sample 12 points along the straight passing lane. At each point, each defender
intercepts with logistic probability in `(ball arrival time − defender T)`. Combined over all
defenders and points like Pressing Intensity combines pressers:

```
P(blocked) = 1 − ∏ (1 − p_intercept)
reach      = 1 − P(blocked)
```

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
stored separately.

## 7. Aggregation

Frame totals are area-weighted sums (`Σ xSpace × cell area`, units xT·m²), overall and per zone,
plus the single best cell.

## Known limitations

- **Independence**: defenders are treated as acting independently (same caveat as Pressing
  Intensity). Real defences cover for each other.
- **Ground passes only**: lofted balls over the line aren't modelled, which undercounts "behind"
  space. Planned: a second, slower, higher trajectory that can't be intercepted mid-flight.
- **Offside** is not yet applied to receivers.
- **Fixed physical parameters** for every player; could be fit per player from tracking data.
- **Set pieces**: corners and free kicks pack the box, so defensive lines are meaningless there.
  These phases need to be filtered out (or modelled separately) using event data.
- **Broadcast tracking** (PFF): off-camera players are estimated, so far-side space is less
  reliable. Compare against IDSSE's optical tracking to quantify this.
- **No validation yet** — see roadmap.

## Roadmap

1. **Match timeline** — compute every frame at 5 Hz across a match (parallelised; GPU later).
2. **Exploited vs. available** — link to events: value actually gained by the next pass/carry,
   and the *decision gap* (best reachable option − chosen option).
3. **Validation** — does xSpace at *t* predict the next pass target, pass success, box entries,
   and xG in the next 10 s? Compare against plain pitch control and OBSO as baselines.
4. **Ratings** — match ratings per team, team profiles, player ratings (carriers, runners).
5. **PFF 2022 World Cup** — scale to 64 matches / 32 teams.
6. **Own value model** — replace borrowed xT with a possession-value model fit on this data.
7. **Web app** — match scrubber, team and player pages.
