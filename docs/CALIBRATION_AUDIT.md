# CALIBRATION_AUDIT.md

Result of mapping the normalized real-player contract into `PlayerSpec` and
running the CE engine on the result.

> **This repository is public.** Everything below is an aggregate: counts,
> coverage rates, calibration error summaries, and CE deltas for twelve
> arbitrary integration rosters. There are no player rows, no projection values
> and no local paths. The mapped specs and the unresolved-identity list stay in
> a git-ignored `local_data/`.

Reproduce with:

```bash
ce-lab ingest    --projections … --fantasypros … --injuries … --fits … \
                 --aliases data/player_aliases.json \
                 --contract-out local_data/real_player_contract_v1.json
ce-lab calibrate --contract local_data/real_player_contract_v1.json \
                 --contract-fumbles-lost local_data/contract_fumbles_lost.json \
                 --fits … --limit 300 --sims 400 --sensitivity
```

`--sensitivity` runs at 16,000 seasons per scenario by default and refuses to
run lower without `--allow-underpowered`.

---

## 0. What this revision corrected

This document supersedes an earlier version of itself. Five things in that
version were wrong, and the numbers it reported are withdrawn rather than
edited, so the record of what changed survives.

| Defect | Consequence | Now |
|---|---|---|
| `player_id` / `crn_key` were the row index of a points-sorted list | Reordering the pool gave a player someone else's random streams; every "paired" comparison compared two different people | Ids derive from the canonical player key; collisions raise |
| Rosters were re-dealt under each scenario | An assumption that moved `base_mean` also reshuffled the league, and the reshuffle was attributed to the assumption | The twelve rosters are built once from baseline identities and reused verbatim |
| No availability-treatment axis existed | The largest single assumption in the model was not being swept at all | `projection_availability_interpretation`, both readings, neither preferred |
| Games missed was scaled 16/17 while injury probability was not | One fit was asked to reproduce two targets defined on different spans | Both matched directly over 18 calendar weeks / 17 scheduled games |
| `signal_noise_sd = None` silently became `week_sd` | Learning speed was a hidden function of scoring noise, contaminating the `season_sd` result | Always explicit, from a stated `signal_quality` scenario |
| CE shifts were quoted against a standard error borrowed from a different 16,000-season experiment | Effects were called real on evidence that was not about them | Every delta is paired season-by-season and carries its own standard error |

### Numbers withdrawn

* **"Median versus mean moves CE by 0.0005."** Withdrawn. That figure came
  from differencing two unpaired 2,000-season runs whose rosters had been
  re-dealt. The corrected paired estimate is **+0.00013, standard error
  0.00009, 95% CI [−0.00005, +0.00030]** — an interval that includes zero. The
  right statement is that this run **cannot resolve** the difference, not that
  the difference is 0.0005.
* **"Fumbles move CE by 0.0100."** Withdrawn. The corrected paired estimate is
  **+0.00544 [+0.00161, +0.00926]** — resolved, but roughly half the withdrawn
  figure.
* **"Every shift except `target` is outside sampling noise."** The conclusion
  survives, but its previous justification did not: it rested on a standard
  error imported from `docs/example_curve_output.txt`, a different experiment
  on synthetic data. It is now re-established from each axis's own paired
  interval, and the magnitudes it was based on have all changed.

Also withdrawn: **the claim that the median and mean targets are equivalent**,
as a general statement. It holds under `full_health` and fails under
`availability_adjusted`; see §3.

---

## 1. One current description of the source

This is the single source of truth. Where any other document disagrees, it is
stale.

* **Continuous prop categories are market medians.** Yards and receptions take
  the over/under line, which the vendor describes as an accurate median amount.
* **Discrete categories are probability-weighted expectations.** Touchdowns and
  interceptions are devigged into implied probabilities and converted to an
  expectation.
* **The combined total is therefore `hybrid_market_location`** — neither a
  proven mean nor a proven median. Both readings are carried as calibration
  targets and neither is asserted.
* **The projection covers 17 NFL games**, the full regular season, from
  preseason season-long props.
* **Health treatment is partial and uncertain.** The vendor states projections
  "do not fully capture a player's current health" and that injury designations
  are applied manually, so a known injury may already have depressed a given
  projection. Recorded as `partially_health_agnostic`. Neither the full-health
  nor the availability-adjusted reading is safe to assume, and both are swept.

The published methodology this rests on is cited in every player row's
`central_tendency_provenance` and `health_treatment_provenance`.

---

## 2. Coverage by depth band

The full-pool injury match rate of 54.5% was never the decision-relevant
number. A 12-team, 15-man league consumes 180 players and in-season churn
reaches perhaps 300; everything past that is a tail nobody drafts.

```
  band             n   team%    bye%  injury%   disp%  unresolved
  ---------------------------------------------------------------
  top_180        180   100.0   100.0     91.7   100.0           0
  top_240        240   100.0    99.6     88.3   100.0           1
  top_300        300   100.0    98.3     80.0   100.0           5
  full_pool      549    95.6    87.6     54.5   100.0          68
```

Identity and bye coverage is essentially complete where it matters. Two
top-240 identities were unresolved before this pass: one closed via a reviewed
alias recorded in `data/player_aliases.json`, one documented as genuinely
unresolvable (a free agent has no team and therefore no bye; supplying one
would be fabrication). Fuzzy matching is deliberately not used anywhere — it
would have resolved both and would also silently join two different players
with similar names, with no way to notice afterwards.

---

## 3. Calibration error

300 specs, 21.8s under `full_health` and 5.6s under `availability_adjusted`
(the injury solve is cached across interpretations).

### Identity

Ids come from the canonical player key, so the same player carries the same id
and the same random streams across every fumble interpretation, projection
interpretation, ranking, pool limit and sensitivity scenario. Collisions are
detected and raised; none occurred.

### Projection level

```
  level_abs_error   n=300   max 0.0      median 0.0      mean 0.0
  level_mc_se       n=300   max 0.00739  median 0.00290  mean 0.00295   (full_health)
  level_mc_se       n=300   max 0.04367  median 0.01333  mean 0.01492   (availability_adjusted)
```

The residual is zero by construction. A simulated season total is exactly
homogeneous in the per-game level — every dispersion term is expressed as a
fraction of it, and availability is drawn independently of it — so the
calibration is a division by a simulated unit-level statistic rather than a
search. What is *not* zero is the Monte Carlo error of that statistic, which is
now propagated onto the level and reported per player. The
`availability_adjusted` figure is larger because that statistic depends on the
player's own fitted hazard, duration and bye, so it cannot be cached per
dispersion shape and runs with fewer draws.

### The two projection questions are independent, and only one is settled

| | Question | Options | Paired CE effect |
|---|---|---|---|
| Statistic | what the total reports | `median_target` / `mean_target` | **+0.00013 [−0.00005, +0.00030]** — unresolved |
| Health state | what the total describes | `full_health` / `availability_adjusted` | **+0.04869 [+0.04390, +0.05347]** — resolved, and the largest axis in the model |

**Under `full_health`, median and mean agree — and that is a property of this
configuration, not a general fact.** Every weekly component the source can
populate is symmetric: the idiosyncratic draw and the season shift are both
normal, and the skewed component (spikes) has no source and stays unpopulated.
A sum of symmetric variables is symmetric, so the two targets return the same
level, agreeing to **0.017%** across all 300 players.

**Under `availability_adjusted` they separate, and substantially.** Absences
remove whole weeks of production from the lower tail only, so the unconditional
season total is no longer symmetric:

```
  |median-target level - mean-target level| / mean-target level, n=300

  full_health            median 0.000173   mean 0.000168   max 0.000197
  availability_adjusted  median 0.018345   mean 0.027787   max 0.124995
```

That is a hundredfold increase at the median player and a 12.5% disagreement at
the worst. **The equivalence claim is scoped to `full_health` and must be
re-checked whenever any asymmetric component is populated.**

### What each availability reading does

* **`full_health`** calibrates the scoring level with no injury process, then
  applies absences. Unconditional season output therefore lands **below** the
  source total, by design: median shortfall **10.91** points, mean 13.21, max
  44.33. This is now computed and reported rather than left implicit.
* **`availability_adjusted`** includes the injury process when solving the
  active-game level, so the calibrated statistic reproduces the source total
  after absences and the per-active-game level is higher for anyone projected
  to miss time. Its *expected* output still sits 2.19 points below the total at
  the median player — because the statistic being matched is the median while
  the shortfall is measured against the mean, and those are now different
  numbers. That gap is the asymmetry above, restated.

Neither is preferred. Both enter sensitivity.

### Injury parameters

Both vendor figures — season injury probability and projected games missed —
describe the full NFL season, so both are matched there: **18 calendar weeks
containing 17 scheduled games and one bye**. Neither target is rescaled.

```
  injury_source_counts               individual 240,  positional_all_cause 60

  FULL SEASON (18 weeks / 17 games) -- what is fitted
  injury_prob_abs_error       n=240   median 0.00325  mean 0.00428  max 0.07475
  games_missed_abs_error      n=240   median 0.01000  mean 0.01292  max 0.17750
  games_missed_achieved       n=300   median 1.90625  mean 1.89490  max 4.08350

  FANTASY WEEKS 1-17 (16 games) -- what is reported, never fitted
  games_missed_expected       n=300   median 1.76875  mean 1.76149  max 3.70700

  infeasible_calibrations     2 of 240
```

The fitted hazard and duration are per-week rates, so they carry into the
fantasy simulation unchanged; the third block is a consequence of the first two
and is reported alongside them rather than substituted for them. A week that is
both a bye and an absence costs no scheduled game, in the calibration and in
the engine alike.

**Two of 240 could not be jointly reproduced** and are reported rather than
quietly fitted. The engine's process ties frequency and duration together — an
absence blocks new onsets — so a high injury probability paired with very few
projected games missed has no solution. Both cases are of that shape; the
closest available parameters are used and both errors are carried on the player.

**60 of 300 players have no individual profile** and fall back to the fitted
positional rate. That rate is labelled **all-cause availability**, not injury:
it counts benching, rest and trades, and the mapping warns about it on every
run. Silently treating an unprofiled player as healthy is not an option the
configuration offers.

### Signal quality

`PlayerSpec.signal_noise_sd` defaults to `None`, which the engine reads as
`week_sd`. Left in place, that ties how fast managers learn a latent change to
how noisy scoring is — which is not a modelling claim anyone made, and it
contaminated the earlier `season_sd` result. Real specs now always set the
field explicitly from `signal_quality`:

| Scenario | `signal_noise_sd` | Meaning |
|---|---|---|
| `none` | infinite | usage never reveals the latent level; the projection stays at consensus |
| `week_sd` | `week_sd` | one week of usage is about as informative as one observed score |
| `2x_week_sd` | `2 × week_sd` | usage is half as precise as that |

`none` is represented exactly, not by a large number: the posterior is
identically zero. It is deliberately distinct from `season_sd = 0`, which says
there is *nothing* to learn rather than that nobody learns it.

---

## 4. Real-data CE smoke test

Twelve deterministic, legal, disjoint rosters built by a snake over the real
draftable pool, **solely so the engine has teams to simulate**. Not an auction
allocation; the CE levels are not advice.

```
REAL-DATA CE SMOKE TEST  ok=True
  [PASS] hall_condition_holds
  [PASS] lineups_never_exceed_eight
  [PASS] only_available_players_start
  [PASS] lineups_are_maximal
  [PASS] eight_slots_filled_when_possible  (97.7% of team-weeks fill all eight; min 6 when byes cluster)
  [PASS] non_qb_superflex_allowed  (14.1% of team-weeks)
  [PASS] one_bye_per_player_with_a_bye
  [PASS] sixteen_scheduled_games_in_horizon  (17 weeks minus one bye = 16)
  [PASS] no_week_18  (array has 17 weeks)
  [PASS] unavailable_players_score_zero
  [PASS] exactly_one_champion_per_season
  [PASS] deterministic
  [PASS] chunk_invariant
  [PASS] ce_is_a_distribution  (sum 1.000000)
  [PASS] no_synthetic_players  (sources ['REAL:winwithodds+draftsharks+nflverse-fits'])
  [PASS] all_specs_marked_real
  [PASS] base_mean_is_a_per_game_level  (max 20.35)
  [PASS] no_grade_derived_distribution
```

Two deserve comment.

**`eight_slots_filled_when_possible` is not "always eight".** Real byes cluster,
and a team can legitimately have too few startable players in a given week.
What must hold is that the lineup is **maximal** — no available player could
legally have been added — and that is what is asserted.

**Sixteen scheduled games inside a seventeen-week horizon.** The projection
spans 17 NFL games; the fantasy window holds 16 of them because of the bye, and
week 18 does not exist in the engine at all.

---

## 5. Sensitivity

**This is a model sensitivity diagnostic. It is not a player-value analysis and
not a price band.** The twelve rosters are a deterministic snake over the mapped
pool, built once from the baseline mapping and reused verbatim in every
scenario. What the table measures is how far CE moves when an unresolved
assumption is made differently — and whether that movement is resolvable at all.

### Method

* **One fixed cast.** Every scenario simulates the same players on the same
  teams. Player ids derive from the canonical key, so a player unchanged by a
  scenario draws bit-identical random numbers in both arms.
* **Paired, season by season.** The reported quantity is the mean of
  `1{team wins in the scenario} − 1{team wins in the baseline}` over 16,000
  common seasons. Its standard error comes from that same difference, so
  seasons the scenario did not change contribute an exact zero.
* **Resolved means its own interval excludes zero.** No standard error is
  imported from another experiment, and a larger shift is not thereby a
  significant one.
* **Discordance is reported** — the fraction of seasons whose champion differs.
  A large discordance beside a near-zero delta means the assumption changed the
  world a great deal without favouring anyone, which a delta alone cannot say.

Full output: `docs/example_sensitivity_output.txt`.

### What each axis demonstrated, at 16,000 seasons

Each row is the largest-moving team on that axis, with that team's own paired
interval.

| Axis | max &#124;ΔCE&#124; | 95% CI | Resolved | Basis for the assumption |
|---|---:|---|:---:|---|
| `availability_interpretation` | **0.04869** | [+0.04390, +0.05347] | yes | vendor states health treatment is partial |
| `forecastable_share` | **0.02444** | [+0.01810, +0.03077] | yes | **none — pure scenario** |
| `season_sd_x_signal` | 0.02244 | [−0.02916, −0.01572] | yes | **none — pure scenario** |
| `season_sd` | 0.02169 | [−0.02843, −0.01494] | yes | **none — pure scenario** |
| `injury_model` | 0.01356 | [−0.02006, −0.00706] | yes | individual fit vs all-cause fallback |
| `signal_quality` | 0.01006 | [−0.01632, −0.00380] | yes | **none — pure scenario** |
| `fumbles` | 0.00544 | [+0.00161, +0.00926] | yes | column meaning unresolved |
| `target` | 0.00013 | [−0.00005, +0.00030] | **no** | hybrid; both readings carried |

**The ordering changed, and the new leader is the axis that did not previously
exist.** How the source treats health moves CE roughly twice as far as any
other assumption, and it is the one axis on this list where the vendor has told
us the answer is genuinely in between rather than at either end.

**Median versus mean is the one axis this run cannot resolve.** Its interval
includes zero and its discordance is 0.0004 — the two mappings produce nearly
the same world. That is a stronger and narrower claim than the withdrawn
"moves CE by 0.0005", and it is scoped to `full_health`; §3 shows the two
targets separate by up to 12.5% once availability is inside the solve, and this
run does not measure the CE consequence of that.

**Two of the three largest axes still have no empirical basis at all.**
`forecastable_share` and `season_sd` are labelled scenarios precisely because
nothing in the inventory measures them.

### `season_sd` cannot be reported without stating the learning speed

`season_sd` says how much latent uncertainty there is; `signal_quality` says how
fast anyone detects it. Reading the interaction off two overlapping
baseline-relative intervals would be the same error this pass exists to remove,
so the contrast gets its own paired run.

| Contrast | max &#124;ΔCE&#124; | se | discordance | Resolved |
|---|---:|---:|---:|:---:|
| ssd=0.10: `none` vs `week_sd` | −0.00481 | 0.00201 | 0.4554 | **yes** |
| ssd=0.10: `2x_week_sd` vs `week_sd` | −0.00344 | 0.00210 | 0.3389 | no |
| ssd=0.20: `none` vs `week_sd` | −0.01063 | 0.00253 | 0.5211 | **yes** |
| ssd=0.20: `2x_week_sd` vs `week_sd` | +0.00556 | 0.00212 | 0.4321 | **yes** |

Three of four resolve. **The `season_sd` sensitivity depends on how quickly
managers learn the latent change**, and the dependence grows with the size of
the change: at `ssd = 0.20` the effect of switching off learning entirely
(−0.01063) is about half the size of the `season_sd` effect itself (−0.02169).
Neither parameter can be estimated or reported in isolation from the other.

---

## 6. What this does not do

No dollar value, no opening bid, no live bid, no auction inflation, no
opponent bidding, no roster-completion optimisation. The twelve rosters exist to
give the engine legal teams. Nothing here prices anything, and nothing here is a
statement about any real player's value.
