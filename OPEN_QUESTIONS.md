# OPEN_QUESTIONS.md

Decisions that could not be settled on Night 1 because they need **real player
data** or **your judgement**. Each entry states the question, what the code does
today, why that placeholder was chosen, how to change it, and what evidence
would settle it.

Nothing here is a bug. Every one of these is a deliberate, reversible
placeholder behind a stable interface.

**Settled and removed.** The league is **winner-take-all**. Championship equity
is `P(win week 17)` and there is no payout vector to weight. This was previously
listed as an open question; it is not one.

**Where these get answered.** Most of section A is blocked on the same piece of
work: importing the existing player model (median stat outcomes, injury odds,
boom/bust grades) behind a versioned input contract, with each field's exact
meaning and provenance written down. `HANDOFF.md` §12 is that plan. A1, A2 and
A3 are largely answerable from those three inputs once their meanings are
pinned; A4, A5, A6 and the `season_sd` / `weekly_state_sd` split are not, and
should be carried explicitly as uncalibrated rather than quietly defaulted.

**The ingestion layer is now built** (`ceauction.realdata`,
`docs/INGESTION_AUDIT.md`). It loads, joins and validates the sources, and it
carries every unresolved question in the payload rather than assuming past one.
What it does *not* do is answer any of the questions below. Section D records
the ones that now block populating the engine.

---

## D. Settled, and blocking the mapping

These were settled by the user during the ingestion phase and are recorded here
so the settlement itself is auditable.

### D0. What the source actually is — RESOLVED against official documentation

Read `winwithodds.com/about` and `/season_long_full_stats`, 2026-09-04. Three
things this project had wrong or unknown are now settled, and one of them
reverses a claim carried in an earlier phase.

**Games basis: 17, and for the right reason.** Season totals are built from
preseason season-long sportsbook props covering the full NFL regular season, so
a total is spread over 17 played games. An earlier revision justified 17 as
"14 regular-season weeks plus a 3-week bracket" — the right number by
coincidence, and it would have broken if either the league's shape or the NFL's
changed.

**The fantasy horizon is shorter than the projection.** This engine simulates
weeks 1–17. The NFL season runs 18 calendar weeks with one bye in weeks 5–14,
so of a player's 17 games only **16 fall inside the horizon**. The 17th is in
week 18 and must never contribute to championship equity. The engine already
produced this; the smoke test now asserts it.

**The total is not a median.** The source derives its categories two ways:
continuous categories (yards, receptions) take the over/under line, which it
calls "an accurate median amount"; discrete categories (touchdowns,
interceptions) are devigged into implied probabilities and turned into
probability-weighted expectations. A total summed from both is a **hybrid
market-location estimate** — neither a proven mean nor a proven median. Both
readings are carried as calibration targets and neither is asserted.

**The source is not full-health.** It states projections "do not fully capture
a player's current health, so an injured player can look more valuable than the
market treats him", and injury designations are applied manually — so a known
injury may already have depressed a given projection. Recorded as
`partially_health_agnostic`. Neither availability reading is safe to assume.

---

### D1. The target league is the 12-team superflex league — SETTLED

`SPEC.md` is definitive. The previous model's 10-team non-superflex
configuration must never enter the CE engine. Enforced in two places:
`build_contract` refuses any `n_teams` other than 12, and the semantic validator
rejects a `league_config_id` that looks like the old configuration.

### D2. Central tendency is `hybrid_market_location` — SETTLED as a hybrid

**This supersedes an earlier entry that recorded the central tendency as
`median` with the provenance "user asserted; vendor documentation not located".
Both halves of that were wrong.** The vendor's published methodology *was*
subsequently located and is now cited on every player row, and what it
describes is not a median.

The source builds its categories two ways: continuous categories (yards,
receptions) take the over/under line, which the vendor calls an accurate median
amount; discrete categories (touchdowns, interceptions) are devigged into
implied probabilities and converted to probability-weighted expectations. A
total summed from both is neither — it is recorded as
**`hybrid_market_location`**, with the published provenance attached.

**Both readings are carried as calibration targets and neither is asserted.**
`median_target` and `mean_target` are swept, and the mapping calibrates the
per-game level against whichever is selected rather than dividing a total by 17
and hoping.

**What the sweep found.** Under `full_health` every modelled weekly component is
symmetric, so a full-season total is symmetric too and the two targets agree to
0.02% — the difference is calibration Monte Carlo noise, not a modelling
choice. The paired CE difference between them is **+0.00013 ± 0.00018**, which
does not exclude zero. Under `availability_adjusted` they stop agreeing: absences
truncate the lower tail only, so the two targets separate by 1.8% at the median
player and 12.5% at the worst. **The equivalence is a property of the
full-health configuration, not a general fact**, and it must be re-checked
whenever any asymmetric component is populated.

**What would still settle it outright:** confirmation from the vendor of a
single estimator per category, or a category-level comparison of published lines
against realised distributions.

### D3. The vendor fantasy total is never used — SETTLED

Points are always recomputed from components under the target league's scoring.
The vendor column solves to full PPR. Enforced structurally: the loader does not
read that column, and `scoring_source` is pinned to `recomputed_from_components`.

### D4. Expert grades never become a distribution — SETTLED

UPSIDE/BUST are preserved as optional source metadata with
`may_derive_dispersion` pinned false. No mapping to variance, standard
deviation, ceiling, floor or spike probability exists or may be created.

### D5. `players_provisional.csv` is never imported — SETTLED

Refused by column signature rather than by filename, because the danger is
precisely that it can be renamed and still look like a real board.

### D6. Injury fields stay separate — SETTLED, and now jointly inverted

`injury_prob` is season-level injury risk; `proj_games_missed` is projected
games missed this season. They are different quantities, preserved separately
in the contract, never combined into one.

**This supersedes an earlier entry stating that no weekly injury process had
been derived.** One now has been, and by inversion rather than by arithmetic
convenience: `weekly_injury_hazard` and `injury_mean_weeks` are solved jointly
against both supplied targets by inverting the engine's own availability
process, and both residual errors are reported per player. The two targets
interact — an absence blocks new onsets — so they are matched together rather
than one after the other, and a pair with no solution is reported as infeasible
instead of quietly fitted.

**The horizon matters and is stated.** Both vendor figures describe the full NFL
season, so the solve runs over **18 calendar weeks containing 17 scheduled games
and one bye** and matches both targets there. The fitted per-week parameters
then carry unchanged into fantasy Weeks 1–17, which contain 16 scheduled games,
and the expected absence over *that* window is reported separately as a
consequence. An earlier pass scaled projected games missed by 16/17 while
leaving injury probability on the full-season basis, asking one fit to reproduce
two targets defined on different spans; that is corrected.

Current achieved error across 240 individually profiled players: median 0.003 on
the full-season injury probability and 0.010 on full-season games missed, with 2
players reported infeasible.

### D7. The availability treatment is unresolved — TWO READINGS, NO CHOICE

The source may or may not already discount for expected absence. Both readings
are computed for every player and **neither is preferred**:

| | Reading | Formula |
|---|---|---|
| **A** | full-health projection | `season_points / 17` |
| **B** | availability-adjusted projection | `season_points / (17 − projected games missed)` |

`active_rate.preferred` is pinned `null`, and the validator rejects any payload
that sets it. Choosing wrongly either double-counts injuries or ignores them.

Note the two are computed over different populations — B only exists for the
300 players carrying a games-missed figure — so their aggregate medians are not
comparable. `docs/INGESTION_AUDIT.md` says so explicitly.

**Both readings are now modelled, not merely computed.**
`PlayerSpecMappingConfig.projection_availability_interpretation` selects between
them and both enter the sensitivity sweep:

* **`full_health`** calibrates the scoring level with no injury process and
  applies absences afterwards, so unconditional season output lands *below* the
  source total — a median shortfall of 10.9 points across the top 300, up to
  44.3. That shortfall is computed and reported rather than left implicit.
* **`availability_adjusted`** puts the fitted process inside the solve, so the
  simulated full-season total reproduces the source total after absences and the
  per-active-game level is correspondingly higher.

**This is currently the largest unresolved axis in the model.** Its paired CE
effect is **+0.04869 [+0.04390, +0.05347]** at 16,000 seasons — larger than any
other assumption swept, and larger than the two uncalibrated variance scenarios
that previously topped the list.

**What would settle it:** vendor documentation, or a comparison of the
projections against realised per-game rates for players with known absences.

### D8. The `Fumbles` column is unresolved — EXCLUDED BY DEFAULT

This league scores −2 for a fumble **lost**; the column is named `Fumbles`. Its
points are excluded from the primary total and the omitted contribution is
reported per player. `lost` and `total` are selectable but never defaulted.

Measured size of the question: 22.8% of players carry a fumble figure, and for
them the excluded contribution ranges from −8.0 to −2.0 points of season
scoring, mean −3.28.

### D9. Missing categories are absent, not zero — SETTLED

The three two-point conversions and the individual special-teams touchdown have
no source column. They are recorded as `treated_as: "absent"`, which the schema
now makes the only legal value.

---

## A. Needs real data

### A1. What are the real per-week means, and what shape is replacement level?

**Today.** `synthetic.PositionProfile` gives each position an exponential decay
from a top value to a floor (`QB 22.0 -> 9.0`, `RB 17.0 -> 4.0`,
`WR 16.5 -> 4.0`, `TE 13.0 -> 3.0`). Invented. Since the zero floor was removed,
`base_mean` is unambiguously **expected fantasy points**, so a real projection
set drops in without reinterpretation.

**Why it matters more than it looks.** The *curvature* of that decay, not its
level, is what determines whether concentrating projection in one player beats
spreading it (experiment `concentration`). A flat replacement level makes studs
cheap; a steep one makes them essential.

**To change.** Populate `PlayerSpec.base_mean` from real projections. Nothing
else moves.

**What settles it.** A real projection set, plus last season's actual weekly
half-PPR distributions to check the tail shape at each position.

---

### A2. How wide is real weekly variance, and does it scale with the mean?

**Today.** A single total weekly dispersion per position (`QB 6.0`, `RB 6.8`,
`WR 7.2`, `TE 5.8`), constant across ranks, of which a fixed part
(`QB 2.5`, `RB 3.0`, `WR 3.0`, `TE 2.4`) is carved out as *forecastable*
(`weekly_state_sd`, SPEC §4.7) and the remainder left as unforecastable
`week_sd`. Constancy across ranks is almost certainly wrong: a 4 pt/week WR5
does not have the same weekly SD as a 17 pt/week WR1.

**The split is a second, separate guess.** How much of a player's week-to-week
movement is knowable before kickoff decides how much of the roster is a genuine
weekly decision at all, and therefore how much a deep bench is worth. The
current 40%-ish forecastable share is invented.

**To change.** `PlayerSpec.week_sd` and `PlayerSpec.weekly_state_sd` are both
already per player. The likely real form for the total is `a + b * base_mean`;
the interface supports any function of the player.

**What settles it.** For the total: weekly half-PPR scores by player for 2-3
seasons, regressed on preseason projection. For the split: the R² of a weekly
projection against the actual weekly score, within player. That R² *is* the
forecastable share.

---

### A3. What are the real injury hazards and absence lengths?

**Today.** A per-week hazard (`RB 0.045`, `WR/TE 0.030`, `QB 0.020`) with a
mean absence of 2.0-2.6 weeks, independent across players, memoryless, with no
age, position-detail or injury-history structure.

**Open sub-questions:** does hazard rise after a first injury? Is a "questionable"
tag worth modelling as partial availability rather than a binary?

**To change.** `weekly_injury_hazard` and `injury_mean_weeks` are already per
player. A richer process (re-injury, multi-state) would need a new function in
`worlds._draw_availability`; the rest of the engine does not care.

**What settles it.** Historical games-missed data by position and age.

---

### A4. What is the real correlation structure?

**Today.** Team shocks with invented betas (`QB 1.3`, `WR 0.9`, `RB 0.7`,
`TE 0.7`) plus a passing-game stack group on each NFL team's first QB/WR/TE.
Both magnitudes are guesses.

**Explicitly not modelled yet:** opponent correlation (you vs the rival starting
the other side of the same game), backfield timeshares as negative correlation,
game-script effects.

**To change.** `PlayerSpec.shock_loadings` accepts any number of named groups
with any betas, so an arbitrary factor structure drops straight in. Negative
correlation already works (`test_negative_beta_creates_negative_correlation`).

**What settles it.** A residual correlation matrix from historical weekly scores
after removing each player's own mean.

---

### A5. How informative is real usage data, and how good are published projections?

**Today.** Two uncalibrated numbers, doing two different jobs.

`signal_noise_sd` (SPEC §4.6) is the precision of the **observable-information
channel** — the thing that stands for snap share, route participation, target or
carry share and depth-chart reporting. It defaults to `week_sd`, meaning "one
week of observed usage tells you about as much about a player's true level as
one observed score would have". That is a placeholder chosen to be conservative,
not an estimate, and it is the single least-grounded parameter in the model.

`proj_noise_sd` (1.0–1.3 points) is analyst noise on top: how wrong the published
number is relative to the conditional mean the model already knows.

**Why it matters.** Everything about the value of forecastability (experiments
`spikes`, `aggregate-lineup-spot`) is measured relative to how good the forecast
is. And `signal_noise_sd` sets how fast a genuinely improved player becomes
startable, which is most of what makes a mid-season breakout worth anything.
Together they make the simulated manager *quite* good — arguably better than a
real one, because he never overreacts and never has a hunch.

**To change.** Both are per-player fields. Separately,
`PlayerSpec.weekly_projection_override` accepts a real published weekly
projection series and bypasses the modelled projection entirely
(`test_projection_override_replaces_the_model`).

**What settles it.** For `proj_noise_sd`: a season of archived weekly projections
joined to actual scores — the RMSE of that join *is* the parameter. For
`signal_noise_sd`: regress a player's rest-of-season scoring on his usage through
week *k*; the residual spread of that relationship is the parameter.

---

### A6. How often do observable role changes happen, and how big are they?

**Today.** 10-18% of players per season, a normal-sized shift revealed one week
after it takes effect.

**Open sub-question:** is a one-week reveal lag right? Some role changes are
announced before they show up in a box score (a trade, a benching); those are
`role_reveal_lag = 0`, which the interface supports and which is tested. A truly
*negative* lag — known to be coming before it takes effect — is not supported,
and would be the natural extension if hand-tagging shows it is common.

**What settles it.** Hand-tagging a season of depth-chart changes.

---

## B. Needs your judgement

### B1. Should the simulated manager be as good as this one?

The engine assumes all 12 managers play the exact projection-optimal lineup
every week and never make an error. That is a modelling choice with a direction:
it *understates* the value of a roster that is easy to set (obvious starters)
and *overstates* the value of a deep bench that requires correct weekly
decisions.

**Options.** (a) Keep perfect play — clean, and arguably the right benchmark for
*your* team. (b) Give opponents a start/sit error rate. (c) Give opponents
perfect play but yourself a noisier projection.

**Recommendation:** keep perfect play for now; it is the conservative choice for
valuing your own depth. Revisit only if opponent modelling becomes a goal.

---

### B2. Should the model include in-season roster change?

There are no waivers, no FAAB and no trades. The drafted 15 are the 15 in week
17. This is the single largest structural simplification in the model.

Its effects do **not** point one way, and an earlier version of this document
claimed they did. The bias depends on what kind of roster spot you are asking
about:

* **Static drafted depth and handcuffs are overvalued as injury protection.**
  The model treats a drafted backup as the only thing standing between you and
  an empty slot, because in the model it is. In reality a comparable
  replacement is usually available on waivers the Tuesday after the injury, so
  much of what the model prices as "insurance" is insurance against a risk you
  could have covered later for free.
* **Churnable lottery-ticket roster spots are undervalued.** A speculative
  bench player's real value includes the option to cut him in week 4 and use
  the spot on whatever emerges next. The model has no such option: a failed
  bet occupies a roster spot for seventeen weeks. Spots held for optionality
  are therefore worth *more* in reality than here.
* **The damage from injuries is overstated.** When a starter goes down, the
  model can only fall back on players you already own, so an injury costs the
  full gap between the starter and your bench. In reality the gap closes
  partly through acquisition.

Those three are not the same claim, and they do not net out to a single
direction. A handcuff and a lottery ticket are biased in *opposite* directions
by the same simplification.

**Question for you:** does the pricing layer need to be right about *draft-day*
value only, or about value net of the waiver wire? Those are different numbers,
and the gap is largest exactly at the bottom of the roster — which is where
$1-$3 auction decisions live.

---

### B3. How should the median result interact with lineup strategy?

The dual head-to-head + median format rewards *consistency* far more than a
pure head-to-head league does — you need to clear the median 14 times. The
engine reproduces this automatically, but it raises a strategy question the
model does not currently answer: should a team that is clearly out of playoff
contention by week 10 start tail-seeking? Today every manager maximises
projected points every week regardless of standings.

---

### B4. What tolerance do you want on CE differences?

At 16,000 seasons a paired CE difference has a standard error of roughly
0.001-0.003 depending on the size of the change. A $1 auction decision may well
be worth less CE than that. Before pricing is built, you need to decide what
resolution the pricing layer requires, because that sets the simulation budget
per candidate roster — and therefore whether a live-auction tool is feasible at
all.

---

## C. Known modelling gaps not yet parameterised

| Gap | Effect on CE | Interface exists? |
|---|---|---|
| Real NFL bye schedule (byes are currently drawn per synthetic team) | small but systematic; real byes cluster | yes, `PlayerSpec.bye_week` |
| Real matchup/volume data behind `weekly_state` (§4.7) | sets how much of the roster is a live weekly decision | yes, `weekly_state_sd` / `weekly_state_pattern` |
| Rival roster composition as an input to your own CE (`rival-fit`) | measurable and non-zero when rivals differ in fit | partly — needs a rival-roster model, not just a pool |
| Playoff-week resting starters | overstates week 17 reliability of players on locked teams | no |
| Opponent-correlation (your player vs your opponent's player in the same game) | affects head-to-head variance, not the median | partly — needs signed cross-roster groups |
| Position-scaled variance (A2) | changes stud vs depth tradeoff | yes, per-player `week_sd` |
| Multi-week "questionable" states | injuries are binary today | no |
| Kickers / DST | not in this league | n/a |
