# Validation

Phase 4: does xSpace mean something, and are its physics honest? Every number here comes from
a script, so it can be regenerated:

```
python scripts/build_pass_set.py      # one frame snapshot per open-play pass (~1 min)
python scripts/calibrate.py           # V2: fit the physics to pass outcomes (~20 min)
python scripts/fit_joint.py           # joint fit to destinations + outcomes (~4 h; not adopted)
python scripts/build_ablations.py     # V1 / V3 surfaces with the calibrated physics (~12 min)
python scripts/build_ablations.py --default   # ... and with the Phases 0-3 physics
python scripts/validate.py            # V1, V3, V4 tables (--default for the old physics)
```

**Split.** Everything fitted (physics parameters, softmax temperatures, logistic regressions)
is fitted on 48 of the 64 PFF World Cup matches and scored on the other 16 (every 4th match by
id); nothing is scored on a match it was fitted on. The 7 IDSSE Bundesliga matches are never
used for fitting: they are an out-of-source check (another league, optical rather than
broadcast tracking, another event provider).

| Task | Question | Status |
|---|---|---|
| V1 | Does xSpace predict *where* the next pass goes? | [Done](#v1-where-does-the-pass-go) |
| V2 | Does reachability predict pass *success*? (and calibration) | [Done](#v2-pass-success-and-calibration) |
| V3 | Does xSpace now predict danger in the next 10 s? | [Done, with a known trade-off](#decision-keep-the-calibrated-physics): old physics predicts it better |
| V4 | Is the team signal stable? | [Done](#v4-is-the-team-signal-stable) |
| V5 | Do ratings relate to results? | After Phase 5 (ratings) |

## V2: pass success and calibration

**Data** (`validation/passes.py`). 64,783 open-play passes and crosses (59,483 PFF, 5,300
IDSSE) with a usable release frame and a known outcome. For each, the frame at the kick with
every player's position and velocity in the passer's frame of reference, who was offside, and
the intended receiver (PFF `targetPlayerId`).

**Model** (`validation/pass_model.py`). The same two factors that weight every cell of
xSpace, evaluated at one target per pass:

    P(pass completed) = control_att(target) × reach(target)

The batched implementation reproduces `pitch_control` / `pass_reachability` to < 1e-6 (tested),
but scores all 65k passes in ~2 s instead of computing whole grids. The target is the
**intended receiver**, run on at their current velocity until the ball meets them. For every
pass, completed or not, so both outcomes are scored at the same kind of point. (Scoring
completed passes where they ended and failed ones where they were cut out would leak the
outcome into the target.)

### Three things the data showed before any fitting

1. **One interception chance per defender, not per lane point.** `pass_reachability` sampled 12
   points along each lane and treated every (defender, point) pair as an independent chance to
   cut the ball out, so a defender beside the lane was counted up to 12 times. Long passes
   were rated nearly unreachable (reach 0.11 at 40–60 m) while PFF ground passes of that length
   were completed 92% of the time. Taking each defender's best point once
   (`lane_combine="max"`) cut held-out log loss from 0.633 to 0.474 without changing the
   ranking (AUC 0.760 → 0.807, all from better calibration of ground passes).
2. **Lofted balls are a different game.** PFF's `highPointType` (the ball's peak height in
   flight) splits passes cleanly: ground passes (`G`, `L`) are completed 91% of the time at
   every distance up to 60 m, lofted ones (`A`, `M`, `H`) about 50% at every distance. A
   ground-only model can't fit both. (PFF's other height field, `ballHeightType`, is the
   height at *contact*: headers are "above head" there however short they are.) The pass
   model now has a lofted ball: it can't be cut out in flight, takes `air_time + d / air_speed`
   to arrive, and is contested where it lands. (xSpace itself stays ground-only: see
   [the lofted ball in xSpace](#the-lofted-ball-in-xspace).)
3. **Short completed passes look contested.** Even where completed short passes were received,
   attacking control was below 0.5 for 10% of them: the model's players were too slow to
   claim the ball. Fitting moved the control rate up (below).

### Fitting

Maximum likelihood on the training matches (Nelder-Mead in log-parameter space), with each
PFF pass scored under its own trajectory. Ten parameters: reaction time, top speed, arrival
uncertainty σ, control rate λ, defender advantage κ, ground ball speed, the share of
in-time defenders who actually cut the ball out (`intercept_factor`), and the lofted ball's
speed, hang time and control-rate factor.

| Fit | Held-out log loss | Brier | AUC | IDSSE log loss | Parameters at a bound |
|---|---|---|---|---|---|
| Phases 0–3 physics | 0.633 | 0.178 | 0.760 | 0.544 | — |
| + one chance per defender | 0.474 | 0.137 | 0.807 | 0.541 | — |
| Unbounded fit | 0.339 | 0.107 | 0.860 | 0.494 | — (but see below) |
| Bounded, no intercept factor | 0.368 | 0.115 | 0.846 | 0.486 | ball speed, hang time |
| **Bounded, with intercept factor (adopted)** | **0.351** | **0.109** | **0.856** | **0.497** | none |

(16 held-out PFF matches, 14,444 passes with an intended receiver, 82% completed. IDSSE has no
intended receivers, so it is scored where the ball ended up.)

**The unbounded fit is a trap.** It scores best, by making the ball effectively instant
(8,070 m/s), reactions 0.06 s and players 34 m/s. Then nothing is ever cut out and control
reduces to "is a defender nearer the receiver than the receiver is?": a good marking statistic
that predicts completion well, but not physics, and as xSpace it would make reach 1 everywhere.
So the fit is bounded to plausible ranges (`calibrate.BOUNDS`). Bounded, it pinned the ground
ball at 30 m/s and the lofted hang time at 2 s: the speed was standing in for something the
model lacked. Adding `intercept_factor` (a defender who gets to the lane in time doesn't always
win the ball) un-pinned both.

**Adopted parameters** (`config.PhysicsParams`; the old ones are `UNCALIBRATED_PARAMS`):

| Parameter | Phases 0–3 | Fitted | Bounds |
|---|---|---|---|
| reaction time | 0.7 s | 0.43 s | 0.3–1.0 |
| top speed | 5.0 m/s | 5.27 m/s | 4–7 |
| arrival σ | 0.45 s | 0.49 s | 0.2–1.0 |
| control rate λ | 4.3 /s | 9.83 /s | 1–10 |
| defender advantage κ | 1.0 | 1.04 | 0.5–2 |
| ground ball speed | 15 m/s | 25.8 m/s | 10–30 |
| intercept factor | 1 | 0.152 | 0.1–1 |
| lofted ball (pass model only) | — | 1.32 s + d / 20.8 m/s | 0.2–2 s, 8–30 m/s |
| lofted control factor | — | 1.00 | 0.3–1.5 |
| lanes | product | one chance per defender | |

Player kinematics barely moved from Spearman's published values: a reassuring sign for the
model and the tracking. The lofted ball takes about 3.2 s over 40 m, which is about right.

Held out, by distance (adopted fit): 0–10 m observed 0.83 / predicted 0.78; 10–20 m 0.86 /
0.85; 20–30 m 0.82 / 0.81; 30–40 m 0.70 / 0.69; 40–60 m 0.63 / 0.58. The reliability curve
is close to the diagonal above p = 0.3 and under-predicts below it (passes rated < 0.2 are
completed 30–50% of the time: mostly intended receivers the model thinks are well marked).

### Caveats

- **`intercept_factor` = 0.15 is low.** It means a defender standing in the lane cuts out a
  30 m pass only ~7% of the time (with a 25.8 m/s ball, even a defender on the line needs to
  react in time). It absorbs more than deflection luck: tracking noise, and selection (players
  rarely try lanes that are really shut, so attempted passes rarely test the blocked case).
  Consequence for xSpace: reach now penalises lanes only lightly, and most of the work is
  done by control. V1 and V3 below check whether reach still earns its place.
- `lambda_att` sits just under its bound (9.83 of 10); a faster control rate didn't help much.
- **Picking the better ball over-rates lofted passes.** Without PFF's tag, the model doesn't
  know which ball the passer will play, so it takes the better one. Scored that way
  (`best_of`), the passes PFF tags as lofted are predicted at 0.70 against 0.53 observed: when a
  player goes long, the ground option the model likes was often not really there. Over all
  passes the best-of score still improves (log loss 0.396 → 0.377, AUC 0.817 → 0.822), but in
  xSpace the same best-of made things worse (below), so xSpace is ground-only.
- The fit uses PFF's broadcast tracking. IDSSE (optical) improves too (0.544 → 0.497), so the
  calibration isn't only fitting broadcast quirks.

## V1: where does the pass go?

For every open-play pass, the frame at the kick is turned into a probability for each 2 m cell,
and the score is the log-likelihood of the cell the ball actually went to, minus a uniform
guess over the pitch (nats per pass; higher is better):

    P(cell) ∝ exp(β · s(cell) / max s − γ · distance from the ball / 10 m)

where `s` is one of the ablation surfaces (`validation/ablations.py`): xT gained by moving the
ball there (`value`), ground-pass pitch control (`control`), `control × value`, xSpace
(`control × reach × value`), xSpace with the lofted ball (`xspace_lofted`, below) and the
probability a pass there arrives (`pass_prob`, no value). β and γ are picked on the training
matches.

**Distance comes first.** Passes are mostly short whatever the space looks like, so distance
alone (β = 0) is already worth 0.96 nats per pass, more than any surface on its own. Scored
without it (γ = 0), as this task first was, the Phases 0–3 physics looked far better (forward
passes: 0.65 nats against 0.30) because its slow ball (15 m/s) and slow reactions made distant
cells worth little: it was standing in for the distance prior. With the prior in every model,
the question becomes what the surface adds to it.

Held-out gain over uniform, with the distance prior:

| Surface | PFF, forward (> 5 m) | IDSSE, forward | PFF, all | IDSSE, all |
|---|---|---|---|---|
| distance only | 0.669 | 0.479 | 0.959 | 0.802 |
| value (xT) | 0.693 | 0.506 | 0.963 | 0.808 |
| control | 0.696 | 0.455 | **1.138** | 0.818 |
| control × value | 1.114 | 0.695 | 1.072 | 0.855 |
| **xSpace** | **1.137** | **0.712** | 1.075 | **0.857** |
| xSpace, lofted ball | 1.040 | 0.699 | 1.028 | 0.855 |
| pass probability | 0.688 | 0.452 | 1.120 | 0.813 |
| *xSpace, Phases 0–3 physics* | *1.074* | *0.804* | *1.049* | *0.870* |

(PFF: 6,437 forward passes and 14,535 in all from the 16 held-out matches; IDSSE: 2,500 and
5,300.)

- On **forward passes**, xSpace beats every ablation on both sources, and each factor earns its
  place: value adds most (0.70 → 1.11 nats over control alone), reach a little (1.114 → 1.137).
- Over **all passes**, plain control (or pass probability) beats xSpace on PFF: by design
  xSpace values only forward progress (methodology §5), so it can't tell apart the backward and
  square passes that make up 40% of the total. That's a question xSpace chose not to ask.
- The calibrated physics beats the Phases 0–3 physics on PFF (1.137 against 1.074) but not on
  IDSSE (0.712 against 0.804). The fit used only PFF passes; IDSSE is a different league with
  optical tracking.

### The lofted ball in xSpace

A lofted ball can reach any cell without being cut out, so xSpace with the lofted ball switched
on (each cell takes whichever ball is likelier to arrive) adds a lot of long, open space.
Passers rarely use it: on every source and subset it predicts destinations worse than
ground-only xSpace (forward passes 1.040 against 1.137 on PFF), even for the passes PFF tags as
lofted (0.879 against 0.971). It's no better at danger either (V3). The pass-completion model
already showed why: picking the better ball over-rates long balls (predicted 0.70, observed
0.53). So **xSpace is ground-only** (`PhysicsParams.air_speed = 0`); the fitted lofted ball
stays in the pass-completion model. Space only a ball over the top can reach is therefore
undercounted (a known limitation; weighting the lofted ball by how often it's chosen is an open
option).

### The joint fit (not adopted)

Fitting the physics to completion alone learns only from passes players chose to attempt, and
they rarely attempt lanes that are really shut. That looked like why the fitted
`intercept_factor` came out at 0.15 and why xSpace lost so much of its V1 score. So the physics
was also fitted jointly (`validation/joint.py`): completion log loss plus the destination
log-likelihood of 8,000 training forward passes, with β fitted alongside (300 Nelder-Mead
iterations; nothing at a bound).

| Physics | V2 completion log loss | V1 gain, surface alone | V1 gain, with distance |
|---|---|---|---|
| Phases 0–3 | 0.633 | 0.58 | 1.07 |
| Completion only (adopted) | **0.351** | 0.18 | 1.13 |
| Joint | 0.371 | 0.25 | **1.14** |

(Held out; V1 on 3,000 forward passes, 3 m grid, ground-only surfaces, β and γ on a coarse
grid.) The joint fit kept lanes more meaningful (`intercept_factor` 0.49, reaction 0.66 s, top
speed 4.38 m/s, λ 8.6 s⁻¹, ground ball 29.4 m/s) but recovered little of the old V1 score: the
lost signal was mostly the missing distance prior, not selection bias. With the prior, the
joint fit gains 0.01 nats on destinations for 0.02 worse completion log loss, so the
completion-only fit stays.

## V3: danger in the next 10 s

211,361 open-play frames sampled at 1 Hz; 4.3% are followed by a shot in the same possession
within 10 s, 18% by a box entry. Ball position alone already predicts both well (AUC 0.82 for
shots and 0.86 for box entries on PFF), so the test is what a surface adds to it: logistic
regressions on the training matches with ball features (xT, x, |y|) only, and with ball
features plus the surface's frame total (log scale), scored on held-out log loss.

Held-out log-loss improvement over ball position alone (%; higher is better):

| Frame total of | PFF shot | PFF box | IDSSE shot | IDSSE box |
|---|---|---|---|---|
| value (xT) | −0.01 | 0.06 | 0.06 | 0.01 |
| control | −0.02 | −0.04 | 0.22 | −0.01 |
| control × value | 0.70 | 0.27 | 0.92 | −0.12 |
| **xSpace** | **0.75** | **0.33** | 0.95 | −0.13 |
| xSpace, lofted ball | 0.64 | 0.17 | **1.02** | −0.13 |
| pass probability | 0.01 | −0.03 | 0.35 | 0.00 |
| *xSpace, Phases 0–3 physics* | *1.24* | *1.16* | *1.82* | *0.70* |
| *control × value, Phases 0–3* | *0.91* | *0.46* | *1.18* | *−0.02* |

(The frame's *best* cell adds less than the total, and on IDSSE it hurts: 0.71 / 0.16 / −0.46 /
−0.93 for xSpace.)

- **Within the calibrated physics, xSpace is the best surface** on PFF, and close to it on IDSSE
  shots, where the lofted variant leads by 0.07 points. Space only matters once it's valued:
  control alone adds nothing.
- **Nothing calibrated predicts IDSSE box entries** beyond ball position.
- **But the Phases 0–3 physics predicts danger better, everywhere**, and the difference is
  reach. With the old physics, reach adds 0.3–0.7 points over `control × value`; with the
  calibrated physics, 0.05. Blocking lanes heavily misjudges which passes get through (V2), but
  it carries information about danger that the calibrated reach, which barely blocks anything
  (`intercept_factor` 0.15), doesn't.

**Hypothesis.** The completion fit sees only lanes players chose to try, so it learns that lanes
seldom matter. Danger depends on how open the defence is in general, including the lanes nobody
tried. A reach fitted to something beyond attempted passes (V3 itself, or the joint fit's
`intercept_factor` of 0.49, not yet scored on V3) may get both. Until then, the honest summary
is: the calibrated physics is better at saying whether a pass will arrive and (on PFF) where it
goes; the old physics is better at saying whether a frame is dangerous.

### Decision: keep the calibrated physics

Decided 2026-10-09: xSpace keeps the calibrated, ground-only physics everywhere (timelines,
actions, possessions, players and the ratings built on them). The V3 gap is accepted for now
and not chased further before Phase 5.

- **One physics, fitted to something observable.** The calibrated parameters are fitted to
  held-out pass outcomes and are close to Spearman's published kinematics. The old values were
  never fitted, and their reach misjudges which passes arrive (V2 log loss 0.633 against 0.351).
  Picking physics per task would mean xSpace measures different things in different places.
- **The calibrated physics still earns its place.** It wins V1 on PFF forward passes and V2
  everywhere, and within it xSpace is the best surface for danger on PFF.
- **What's given up.** xSpace adds less to ball position for predicting danger (PFF shots
  +0.75% against +1.24% with the old physics). So Phase 5 ratings describe space and
  progression; they are not presented as a danger or chance-quality measure.
- **When to revisit.** Phase 6 (own value model) re-runs V1–V3 anyway, which is the natural
  point to try the two untested options: score the joint fit's physics (`intercept_factor`
  0.49, `data/processed/validation/joint.json`) on V3, and fit reach to V3 directly.

## V4: is the team signal stable?

Each team's possessions in a match are split into odd and even in time order, the metric is
computed on each half, and the two halves are correlated across teams. The Spearman-Brown
reliability `2r / (1 + r)` estimates the correlation for the whole sample. At team-match level a
team's "created" is its opponent's "conceded", so only created is shown there; per team over
the tournament (32 PFF teams, all with at least 3 matches), both are.

| Level | Metric | Created r | Reliability | Conceded r | Reliability |
|---|---|---|---|---|---|
| team-match (128) | mean xSpace in possession | 0.88 | **0.93** | | |
| team-match | mean peak xSpace per possession | 0.83 | 0.91 | | |
| team-match | exploitation rate | 0.23 | 0.37 | | |
| team (32) | mean xSpace | 0.91 | **0.96** | 0.88 | **0.94** |
| team | mean peak xSpace | 0.89 | 0.94 | 0.89 | 0.94 |
| team | exploitation rate | 0.37 | 0.54 | 0.37 | 0.54 |

Team xSpace, created and conceded, is very stable, so it describes something persistent about
how a team plays or defends. The exploitation rate isn't: exploited actions are rare (~1%), so
half a match or half a tournament holds too few to rank teams. Ratings (Phase 5) should lean on
xSpace and peak xSpace, and pool exploitation over more matches or shrink it towards the mean.

(Split-half reliability measures noise, not meaning: a stable metric could still be stable for
the wrong reason, e.g. tracking quality per stadium. V3 and V5 are the checks on meaning.)

## Summary against the plan

The plan's definition of done: the full model beats every ablation on held-out matches for
V1–V3, and anything that doesn't is reported with a hypothesis.

- **V1:** done. Ground-only xSpace beats every ablation on forward passes, on both sources, once
  every model has the distance prior. Over all passes plain control wins, because xSpace ignores
  backward and square passes by design.
- **V2:** done. The fitted physics cuts held-out log loss from 0.633 to 0.351 (AUC 0.76 → 0.86),
  and also improves on IDSSE, which wasn't used for fitting.
- **V3:** done, with a known trade-off. Within the calibrated physics, xSpace beats its
  ablations on PFF and is level on IDSSE shots; nothing beats ball position for IDSSE box
  entries. The Phases 0–3 physics predicts danger better everywhere. That is accepted for now
  ([decision](#decision-keep-the-calibrated-physics)) and revisited in Phase 6.
- **V4:** done. Team xSpace is stable; the exploitation rate is too noisy per team.
